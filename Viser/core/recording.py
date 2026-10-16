"""Append-only per-run archives; disk I/O stays outside the controller thread."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import queue
import shutil
import sys
import threading
import time
import uuid

import numpy as np

from core.robot import ROOT
from core.policy import output_metadata
from core.joint_tracking import metadata as joint_metadata

MAX_SESSIONS = 5


def retain_recent_sessions(directory, limit=MAX_SESSIONS):
    """Keep only completed/started robot-run directories in this archive root."""
    parent = Path(directory)
    candidates = sorted(
        (path for path in parent.iterdir() if path.is_dir() and not path.is_symlink()
         and (path / 'metadata.json').is_file()
         and json.loads((path / 'metadata.json').read_text()).get('has_robot_data', False)),
        key=lambda path: path.name,
    ) if parent.exists() else []
    for stale in candidates[:-limit]:
        # Session directories contain only recorder output; removing one is safe
        # and prevents old diagnostics or model snapshots from accumulating.
        shutil.rmtree(stale)
    return candidates[-limit:]


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class SessionRecorder:
    def __init__(self, config, arguments, directory=None):
        if arguments['mode'] not in ('observe', 'control') and directory is None:
            raise ValueError('Synthetic recordings require an explicit temporary directory')
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.started = time.monotonic()
        tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.path = Path(directory or ROOT / 'logs/sessions') / (tag + '_' + arguments['mode'] + '_' + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True, exist_ok=False)
        metadata = dict(schema=18, has_robot_data=False, config_token=config.token,
                        preparation_poses={key: str(path) for key, path in config.upper_pose_paths.items()},
                        joint_comparison=joint_metadata(),
                        dex3_pd=dict(config_path=str(config.hand_pd_path),
                                     active=config.hand_pd.copy(), stopped=dict(kp=0., kd=.1),
                                     scope='All 14 Dex3 joints; loaded once at startup. '
                                           'Successful write transitions are logged as DEX3 PD in bridge.log.'),
                        policy_outputs=output_metadata(config),
                        timing=dict(
                            inference_ms='Balance/Homie inference calls plus student ONNX time when sampled',
                            observation_to_action_ms='Python observation preparation through inference completion; excludes prior feedback age and command send',
                            state_age_ms='age since C++ received body feedback at Python read',
                            state_to_command_ms='C++ body-feedback receipt through Python command send; null when unarmed or observe; excludes subsequent DDS/motor response'),
                        policy_switching={key: config.deployment[key] for key in (
                            'model_switch_command_norm', 'command_filter_tau_s',
                            'profile_transition_s', 'policy_target_speed_rad_s')},
                        motion_settings=config.motion_settings,
                        started_utc=self.started_utc, monotonic_origin=self.started,
                        arguments=arguments, python=sys.version, executable=sys.executable,
                        data='robot-body telemetry, raw Balance/Homie actions, Homie estimator outputs '
                             'and run events at each Python policy tick (~50 Hz); '
                             'no source code, URDF, meshes, deployment files or model snapshots are archived',
                        completion='summary.json is written only after recording stops; missing summary means interrupted run')
        if getattr(config, 'high_policy_metadata', None) is not None:
            metadata['high_policy'] = config.high_policy_metadata
            metadata['high_policy']['upper_pd'] = dict(
                file=str(config.deploy / 'config/pushcart_upper_pd.json'),
                transition_s=config.deployment['pushcart_upper_transition_s'],
                target_interpolation_s=config.deployment['pushcart_upper_target_interpolation_s'],
                target_interpolation='linear at 200 Hz from last emitted target; requested_target is the endpoint, '
                                     'target/joint_comparison.command_rad are sampled bridge outputs; '
                                     'upper_target_selected=false cancels motion while retaining pushcart PD',
                scope='waist yaw and arms only; feedback upper_pd_mode and applied_kp/kd record execution')
            metadata['policy_outputs']['observation_history']['previous_action'] = (
                'previous selected output plus the applied high-level leg residual, after training clipping; '
                'Balance receives no leg residual')
        (self.path / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
        self.metadata = metadata
        self.streams = {name: gzip.open(self.path / (name+'.jsonl.gz'), 'xt', encoding='utf-8', compresslevel=1)
                        for name in ('telemetry', 'events')}
        self.pending = queue.Queue(maxsize=4096)
        self.stopping = threading.Event()
        self.error = ''
        self.dropped = 0
        self.counts = dict(telemetry=0, events=0)
        self.thread = threading.Thread(target=self._write, name='g1-recorder', daemon=True)
        self.thread.start()

    def record(self, stream, data):
        try:
            self.pending.put_nowait((stream, dict(monotonic=time.monotonic(), **data)))
        except queue.Full:
            self.dropped += 1

    def _write(self):
        last_flush = time.monotonic()
        try:
            while not self.stopping.is_set() or not self.pending.empty():
                try:
                    name, value = self.pending.get(timeout=.1)
                except queue.Empty:
                    name = None
                if name is not None:
                    self.streams[name].write(json.dumps(value, default=json_value, ensure_ascii=False)+'\n')
                    self.counts[name] += 1
                    if name == 'telemetry' and self.counts[name] == 1:
                        self.metadata['has_robot_data'] = True
                        (self.path / 'metadata.json').write_text(
                            json.dumps(self.metadata, indent=2, ensure_ascii=False)+'\n')
                        retain_recent_sessions(self.path.parent)
                if time.monotonic() - last_flush >= .5:
                    for stream in self.streams.values():
                        stream.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            self.error = repr(exc)
            print('RECORDING ERROR: ' + self.error, file=sys.stderr, flush=True)
        finally:
            for stream in self.streams.values():
                try:
                    stream.close()
                except Exception as exc:
                    self.error = repr(exc)

    def close(self, error=None):
        self.stopping.set()
        self.thread.join(timeout=10)
        summary = dict(started_utc=self.started_utc, ended_utc=datetime.now(timezone.utc).isoformat(),
                       duration_s=time.monotonic()-self.started, records=self.counts,
                       dropped_records=self.dropped, recorder_error=self.error,
                       application_error=error, writer_finished=not self.thread.is_alive())
        if not self.counts['telemetry'] and not self.thread.is_alive() and not self.error:
            shutil.rmtree(self.path)
            return summary
        (self.path / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
        retain_recent_sessions(self.path.parent)
        return summary

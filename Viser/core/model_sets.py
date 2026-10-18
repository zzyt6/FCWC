"""Preload complete walking/student pairs before the robot can be prepared."""
from copy import copy

from core.policy import Policy
from core.pushcart import CONFIG, load_feature
from hybrid_deploy.policy import read_bundle


MODEL_SET_LABELS = {
    'v81': '60000 轮 Homie + V8.1 高层',
    'v82': '68000 轮 Homie + V8.2 高层',
}
MODEL_MANIFESTS = {'v81': 'models/manifest.json', 'v82': 'models/manifest_v82.json'}
DEFAULT_MODEL_SET = 'v81'


class ModelSets:
    def __init__(self, config, mode, interface, path=CONFIG, policy_mode='auto'):
        self.entries = {}
        self.camera_owner = None
        # The bridge token covers the common servo/PD settings. Model selection
        # may only change artifacts, never silently change those control settings.
        artifact_keys = {'model_files', 'contract_files', 'model_manifest'}
        shared = {k: v for k, v in config.deployment.items() if k not in artifact_keys}
        try:
            for key, label in MODEL_SET_LABELS.items():
                cfg, contracts, token = read_bundle(config.deploy, MODEL_MANIFESTS[key])
                if token != config.token or shared != {k: v for k, v in cfg.items() if k not in artifact_keys}:
                    raise ValueError('Model sets must share the same servo configuration')
                view = copy(config)
                view.deployment, view.contracts = cfg, contracts
                view.policy_paths = {name: config.deploy / file for name, file in cfg['model_files'].items()}
                high = load_feature(view, mode, interface, path, model_version=key,
                                    camera=self.camera_owner.camera if self.camera_owner else None,
                                    camera_process=self.camera_owner.camera_process if self.camera_owner else None)
                if high is not None and self.camera_owner is None:
                    self.camera_owner = high
                policy = Policy(config.deploy, cfg, policy_mode,
                                pushcart_homie=high.homie_model if high else None)
                self.entries[key] = dict(config=view, policy=policy, high=high)
                if view.pushcart_error:
                    print(label + '：推车功能未加载，普通行走仍可用：' + view.pushcart_error, flush=True)
            config.model_sets_metadata = {
                key: dict(label=MODEL_SET_LABELS[key],
                          manifest_path=str(config.deploy / MODEL_MANIFESTS[key]),
                          model_paths={name: str(file) for name, file in entry['config'].policy_paths.items()},
                          model_sha256=entry['policy'].source_sha256.copy(),
                          high_policy=entry['high'].metadata if entry['high'] else None,
                          pushcart_error=entry['config'].pushcart_error)
                for key, entry in self.entries.items()
            }
            self.activate(config, DEFAULT_MODEL_SET)
        except BaseException:
            self.close()
            raise

    def activate(self, config, key):
        entry = self.entries[key]
        view = entry['config']
        config.model_set = key
        config.deployment, config.contracts = view.deployment, view.contracts
        config.policy_paths = view.policy_paths
        config.pushcart_error = view.pushcart_error
        config.high_policy_metadata = entry['high'].metadata if entry['high'] else None
        return entry['policy'], entry['high']

    def close(self):
        # All students share one receiver/process; close those resources once.
        if self.camera_owner is not None:
            self.camera_owner.close()
            self.camera_owner = None

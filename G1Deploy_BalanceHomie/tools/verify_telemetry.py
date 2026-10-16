"""Replay saved inputs through both local models and check completed PD transitions."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from hybrid_deploy.configuration import load_config
from hybrid_deploy.policy import Policy,read_bundle
from hybrid_deploy import protocol as p


def verify(path):
    cfg,_,token=read_bundle(ROOT);_,gains,_=load_config(ROOT)
    policy=Policy(ROOT,cfg)
    count=policy_count=0;pd_counts={'balance':0,'homie':0};errors={'balance':0.,'homie':0.}
    times=[];inference_ms=[];selected=[];transitions=0;previous=None
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line);count+=1
            if row['config_token']!=token:raise ValueError('Current configuration differs from recorded run')
            times.append(row['sample_monotonic_ns'])
            if row['inference_ms'] is not None:inference_ms.append(row['inference_ms'])
            name=row['selected_model'];selected.append(name)
            if row['policy_mode']=='auto':
                feedback=SimpleNamespace(stage=p.STAGE_NAMES.index(row['stage']),
                    flags=int(row['healthy']), session=row['bridge_session'], sequence=row['state_sequence'],
                    received_ns=row['lowstate_received_monotonic_ns'],
                    dq=np.asarray(row['dq_sdk29_rad_s']), omega=np.asarray(row['imu_angular_velocity_rad_s']))
                assert name==policy.select(row['velocity_command'],feedback)
            else:assert name==row['policy_mode']
            if row['model_observations'] is not None:
                for model in ('balance','homie'):
                    obs=np.asarray(row['model_observations'][model],dtype=np.float32).reshape(1,-1)
                    action=policy.infer(model,obs)
                    error=float(np.max(np.abs(action-row['model_actions'][model])))
                    errors[model]=max(errors[model],error)
                    np.testing.assert_allclose(action,row['model_actions'][model],atol=1e-5,rtol=1e-5)
                np.testing.assert_array_equal(row['policy_action'],row['model_actions'][name])
            if row['stage']=='POLICY':
                policy_count+=1
                if previous is not None and row['pd_profile']!=previous:transitions+=1
                previous=row['pd_profile']
                if row['pd_transition_blend']==1:
                    profile=row['pd_profile'];pd_counts[profile]+=1
                    np.testing.assert_allclose(row['applied_kp_sdk29'],gains[profile]['kp_sdk29'],atol=1e-5)
                    np.testing.assert_allclose(row['applied_kd_sdk29'],gains[profile]['kd_sdk29'],atol=1e-5)
    if count<2:raise ValueError('Need at least two samples')
    dt=np.diff(np.asarray(times,dtype=np.int64))*1e-6
    assert (dt>0).all()
    return dict(passed=True,samples=count,policy_samples=policy_count,selected_models=sorted(set(selected)),
                completed_pd_profile_samples=pd_counts,pd_switches=transitions,max_action_replay_error=errors,
                mean_loop_hz=float(1000/dt.mean()),loop_dt_ms_p50_p99_max=np.percentile(dt,[50,99,100]).tolist(),
                dual_inference_ms_p50_p99_max=np.percentile(inference_ms,[50,99,100]).tolist() if inference_ms else None,
                scope='Replay and recorded profile consistency; not physical stability validation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=verify(args.log)
    text=json.dumps(result,indent=2)+'\n'
    if args.report:args.report.write_text(text)
    print(text)

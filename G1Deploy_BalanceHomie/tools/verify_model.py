"""Original fixed-vector checks and dual-inference timing; never connects to DDS."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from hybrid_deploy.policy import Policy, read_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional report path; default prints only')
    args=parser.parse_args()
    cfg,_,_=read_bundle(ROOT); policy=Policy(ROOT,cfg)
    errors={};observations={};counts={}
    for name in ('balance','homie'):
        with np.load(ROOT/'models'/name/'verification_vectors.npz',allow_pickle=False) as data:
            obs_key,action_key=('inputs','outputs') if name=='balance' else ('observations','actions')
            obs=data[obs_key];expected=data[action_key]
            actual=np.asarray([policy.infer(name,x.reshape(1,-1)) for x in obs])
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
            errors[name]=float(np.max(np.abs(actual-expected)));counts[name]=len(obs)
            observations[name]=obs[0].reshape(1,-1).copy()
    times=[]
    for _ in range(300):
        started=time.perf_counter();policy(observations,[0,0,0]);times.append((time.perf_counter()-started)*1000)
    report=dict(passed=True,checked_at_utc=datetime.now(timezone.utc).isoformat(),machine=platform.machine(),
                model_sha256=policy.source_sha256,model_bytes_unchanged=True,fixed_vector_samples=counts,
                fixed_vector_max_abs_errors=errors,dual_inference_ms_p50_p99_max=np.percentile(times,[50,99,100]).tolist(),
                scope='CPU inference only; no DDS, physics or motor commands.')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

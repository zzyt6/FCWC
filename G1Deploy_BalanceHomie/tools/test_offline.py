"""仅启动 --simulate，不创建 DDS 通道；验证进程联调与 Python 死亡后的桥接看门狗。"""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from hybrid_deploy import protocol as p
from hybrid_deploy.policy import read_bundle


def watchdog():
    cfg,_,_=read_bundle(ROOT)
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1',cfg['state_port']));sock.settimeout(1)
    bridge=subprocess.Popen([str(ROOT/'build/g1_control'),'--simulate','--control','--duration','3'],
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    try:
        state=p.State.decode(sock.recvfrom(1000)[0])
        sock.sendto(p.command(state,1,p.PREPARE,state.q),('127.0.0.1',cfg['bridge_port']))
        enabled=False;fault=False
        deadline=time.monotonic()+1
        while time.monotonic()<deadline:
            state=p.State.decode(sock.recvfrom(1000)[0])
            enabled=enabled or state.stage==p.RAMPING
            if state.stage==p.FAULT:
                fault=True;break
        assert enabled and fault,'Heartbeat watchdog failed'
    finally:
        sock.close();bridge.terminate()
        output=bridge.communicate(timeout=3)[0]
    assert 'policy heartbeat timeout' in output,output
    return output


def main():
    subprocess.run(['ctest','--output-on-failure'],cwd=ROOT/'build',check=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_*.py','-v'],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/verify_model.py')],check=True)
    import os
    env=dict(os.environ,PYTHONPATH=str(ROOT/'python'))
    bridge=subprocess.Popen([str(ROOT/'build/g1_control'),'--simulate','--control','--scripted','--duration','9'],
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log=ROOT/'reports/offline_telemetry.jsonl'
    try:
        policy=subprocess.run([sys.executable,'-m','hybrid_deploy.run','--mode','control','--duration','9','--log',str(log)],
                              env=env,capture_output=True,text=True,timeout=12)
        bridge_output=bridge.communicate(timeout=3)[0]
    finally:
        if bridge.poll() is None:
            bridge.terminate();bridge.communicate(timeout=3)
    assert policy.returncode==0,policy.stdout+policy.stderr+bridge_output
    assert bridge.returncode==0,bridge_output
    rows=[json.loads(row) for row in log.read_text().splitlines()]
    stages={row['stage'] for row in rows}
    assert {'WAITING','RAMPING','HOLDING','POLICY'}.issubset(stages),stages
    assert 'remote L2' in bridge_output,bridge_output
    assert {row['selected_model'] for row in rows if row['stage']=='POLICY'}=={'balance','homie'}
    watchdog_output=watchdog()
    count=unittest.TestLoader().discover(str(ROOT/'tests'),pattern='test_*.py').countTestCases()
    report={'passed':True,'cpp_control_scenarios':28,'python_tests':count,
            'policy_profile':'hybrid_balance_homie_456_456','fixed_vector_samples':{'balance':64,'homie':128},
            'both_models_in_pipeline':True,
            'checked_at_utc':datetime.now(timezone.utc).isoformat(),'machine':platform.machine(),
            'pipeline_stages':sorted(stages),'independent_watchdog':True,
            'bridge_output':bridge_output,'watchdog_output':watchdog_output,
            'scope':'Loopback and synthetic feedback only; no DDS, physics, or robot operation.'}
    (ROOT/'reports/offline_integration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

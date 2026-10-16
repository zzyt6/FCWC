"""Read-only pelvis heading replay/plots. Never imports a bridge or opens a socket.

Use --synthetic for a labelled signal exercise, or --session ARCHIVE for replay.
Plots require the existing matplotlib installation; --no-plot needs only NumPy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import numpy as np
from core.heading import HeadingControl, load_settings
from core.pelvis_imu import ImuFault

ROOT=Path(__file__).resolve().parents[1]


def synthetic():
    """Prescribed sensor signals, NOT a physics model or closed-loop validation."""
    yaw=0.; baseline=np.zeros(3); rng=np.random.default_rng(7)
    sequence=[(1.,[0,0,0]),(4.,[.2,0,0]),(3.,[-.2,0,0]),(3.,[0,.2,0]),
              (3.,[.2,-.1,0]),(3.,[0,0,.2]),(3.,[.2,.1,0]),(2.,[0,0,0])]
    t=1.; n=0
    for duration,user in sequence:
        for _ in range(round(duration/.02)):
            dt=float(rng.uniform(.018,.022));t+=dt;n+=1
            rate=user[2]+(.012 if np.linalg.norm(user)>0 else 0.)
            yaw+=rate*dt
            baseline+=np.clip(np.asarray(user)-baseline,-np.array([.3,.2,.4])*dt,np.array([.3,.2,.4])*dt)
            yield dict(monotonic=t, stage='POLICY',healthy=True,input_velocity=baseline.tolist(),
                       velocity=baseline.tolist(),heading=dict(vx_user=user[0],vy_user=user[1],wz_user=user[2],command_source='MANUAL'),
                       state=dict(quat=[np.cos(yaw/2),0,0,np.sin(yaw/2)],omega=[0,0,rate+rng.normal(0,.006)],
                                  tick=[1,n],received_ns=round((t-.001)*1e9)),synthetic=True)


def archive_rows(path):
    with gzip.open(path/'telemetry.jsonl.gz','rt') as stream:
        for line in stream:yield json.loads(line)


def continuous(values):
    # Unwrap finite segments independently; missing reference remains missing.
    a=np.asarray([np.nan if v is None else v for v in values],dtype=float)
    valid=np.isfinite(a);starts=np.where(valid & ~np.r_[False,valid[:-1]])[0]
    for start in starts:
        end=start+1
        while end<len(a) and valid[end]:end+=1
        a[start:end]=np.unwrap(a[start:end])
    return a


def plot(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.array([r['control_time'] for r in rows]);t-=t[0]
    def values(name):return [r.get(name) for r in rows]
    fig,ax=plt.subplots(6,1,figsize=(13,14),sharex=True)
    yaw=continuous(values('yaw_used'))
    refs=np.array([np.nan if r.get('yaw_ref') is None else r['yaw_ref'] for r in rows])
    # Draw the reference on the same continuous branch as measured yaw.
    refs=yaw+np.arctan2(np.sin(refs-yaw),np.cos(refs-yaw))
    ax[0].plot(t,np.rad2deg(yaw),label='pelvis yaw');ax[0].plot(t,np.rad2deg(refs),label='reference')
    ax[0].set_ylabel('heading deg')
    ax[1].plot(t,np.rad2deg(np.asarray([np.nan if v is None else v for v in values('heading_error')],dtype=float)),label='heading error')
    ax[1].set_ylabel('error deg')
    for key in ['body_z_rate','yaw_rate_for_heading','filtered_yaw_rate_for_heading']:ax[2].plot(t,values(key),label=key)
    ax[2].set_ylabel('rad/s')
    for key in ['wz_user','wz_suggested','wz_final_applied','recorded_wz']:ax[3].plot(t,values(key),label=key)
    ax[3].set_ylabel('command rad/s')
    states=list(dict.fromkeys(values('heading_state')))
    ax[4].step(t,[states.index(r['heading_state']) for r in rows],where='post',label='heading state')
    ax[4].set_yticks(range(len(states)));ax[4].set_yticklabels(states,fontsize=8)
    for key in ['imu_age','imu_dt','control_dt']:
        ax[5].plot(t,1000*np.asarray([np.nan if v is None else v for v in values(key)],dtype=float),label=key+' ms')
    limited=np.array([r.get('slew_limit_flag',False) or any(r.get('saturation_flags',{}).values()) for r in rows])
    ax[5].scatter(t[limited],np.zeros(sum(limited)),s=6,color='red',label='limit event')
    ax[5].set_ylabel('host timing ms');ax[5].set_xlabel('control time since start (s)')
    for a in ax:a.grid(alpha=.2);a.legend(loc='upper right',fontsize=8)
    fig.suptitle('OFFLINE replay: measured yaw is prescribed, not a counterfactual robot response')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(path/'heading.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--synthetic',action='store_true');group.add_argument('--session',type=Path)
    parser.add_argument('--heading-mode',choices=['disabled','shadow','active'],default='shadow')
    parser.add_argument('--output',type=Path);parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args()
    path=args.output or Path(tempfile.gettempdir())/('viser_replay_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    path.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT.parent/'G1Deploy_BalanceHomie/config/deploy.json').read_text())
    settings=load_settings(run_mode=args.heading_mode)
    control=HeadingControl(settings,cfg['command_limits'],cfg['command_slew_per_s'])
    records=[];last_time=None;last_session=None;faults=[];inferred=0
    for row in synthetic() if args.synthetic else archive_rows(args.session):
        state=row.get('state');old=row.get('heading',{})
        if not state or 'input_velocity' not in row:continue
        if row.get('stage') in ('DAMPED','FAULT'):continue
        now=old.get('control_time',row['monotonic']);key=tuple(state['tick'])
        if key[0]!=last_session:
            control.clear();last_time=None;last_session=key[0]
        dt=1/cfg['policy_hz'] if last_time is None else now-last_time
        last_time=now
        baseline=row['input_velocity']
        if all(k in old for k in ('vx_user','vy_user','wz_user')):
            user=[old[k] for k in ('vx_user','vy_user','wz_user')]
        else:
            user=baseline;inferred+=1
        source=old.get('command_source','MANUAL' if row.get('source') in ('web','remote') else 'UNKNOWN')
        active=row.get('stage')=='POLICY' and row.get('healthy',False) and (np.any(user) or np.any(baseline) or row.get('decelerating',False))
        try:
            _,h=control.update(state['quat'],baseline,dt,active,omega=state['omega'],sample_key=key,
                               received_time=state['received_ns']*1e-9,control_time=now,raw_command=user,
                               command_source=source,smooth_zero=row.get('decelerating',False))
        except ImuFault as exc:
            faults.append(dict(control_time=now,reason=str(exc)))
            control.clear();continue
        h['recorded_wz']=row['velocity'][2]
        records.append(h)
    if not records:raise ValueError('No usable samples')
    with (path/'heading.jsonl').open('x') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    summary=dict(samples=len(records),input='synthetic prescribed signals' if args.synthetic else str(args.session),
                 run_mode=args.heading_mode,settings=settings,states=dict(Counter(r['heading_state'] for r in records)),
                 faults=faults,inferred_raw_command_samples=inferred,
                 limitation='Offline signal replay only. No DDS, no hardware, no counterfactual robot response. '
                            'Legacy raw intent is approximated from recorded smoothed input; sample_time is unavailable.')
    (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not args.no_plot:plot(records,path)
    print(path)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('settings','faults')},indent=2))

if __name__=='__main__':main()

import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from hybrid_deploy import protocol as p
from hybrid_deploy.policy import Policy, Observations, Commands, read_bundle, targets
from hybrid_deploy.configuration import load_config


def state(cfg):
    return p.State(42,1,1000000000,1,0,1,0,1,
                   np.asarray(cfg['default_q_sdk29'],dtype=np.float32),np.zeros(29,dtype=np.float32),
                   np.zeros(3,dtype=np.float32),np.array([1,0,0,0],dtype=np.float32),
                   np.zeros(3,dtype=np.float32),np.zeros(4,dtype=np.float32),
                   np.asarray(cfg['default_q_sdk29'],dtype=np.float32))


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg,cls.contract,_=read_bundle(ROOT)
        cls.policy=Policy(ROOT,cls.cfg)

    def setUp(self):
        self.policy.reset_selection()

    def test_profile_wire_and_pd_feedback_roundtrip(self):
        self.assertEqual(p.STATE.size,1120);self.assertEqual(p.COMMAND.size,248)
        s=state(self.cfg)
        data=p.STATE.pack(p.STATE_MAGIC,p.VERSION,s.session,s.sequence,s.received_ns,s.config_token,
                          s.stage,s.flags,s.buttons,s.mode_machine,*np.arange(101),1,.5,
                          *np.arange(29),*np.ones(29), 63, .25, *np.arange(42),
                          1200, 1000, 1001, 1002, 1100, 1101, (1 << 43)-1, *np.arange(43), 0, -1, 1, .75)
        decoded=p.State.decode(data)
        np.testing.assert_array_equal(decoded.q,np.arange(29))
        np.testing.assert_array_equal(decoded.applied_q,np.arange(72,101))
        np.testing.assert_array_equal(decoded.applied_kp,np.arange(29))
        self.assertEqual(decoded.profile,1);self.assertEqual(decoded.gain_blend,.5)
        self.assertEqual(decoded.upper_mode,1);self.assertEqual(decoded.upper_gain_blend,.75)
        self.assertEqual(decoded.hand_flags,63);self.assertEqual(decoded.hand_progress,.25)
        np.testing.assert_array_equal(decoded.hand_q,np.arange(14))
        np.testing.assert_array_equal(decoded.hand_target,np.arange(28,42))
        self.assertEqual(decoded.bridge_ns,1200)
        self.assertEqual(decoded.sent_ns,(1000,1001,1002))
        self.assertEqual(decoded.hand_received_ns,(1100,1101))
        self.assertEqual(decoded.position_mask,(1<<43)-1)
        np.testing.assert_array_equal(decoded.sent_q,np.arange(43))
        for index, value in [(1,4), (216,1201), (221,1<<43), (222,float('nan')), (265,1<<14), (266,14),
                             (267,2), (268,float('nan')), (268,1.1)]:
            invalid=list(p.STATE.unpack(data));invalid[index]=value
            with self.assertRaises(ValueError):p.State.decode(p.STATE.pack(*invalid))
        values=p.COMMAND.unpack(p.command(s,3,p.RUN,s.q,profile=1))
        self.assertEqual(values[7:9],(p.RUN,1))
        with self.assertRaises(ValueError):p.command(s,4,p.RUN,s.q,profile=2)
        self.assertEqual(p.COMMAND.unpack(p.command(s,3,p.RUN,s.q,hand_pose=2))[38],2)
        self.assertEqual(p.COMMAND.unpack(p.command(s,4,p.RUN,s.q,hand_pose=3))[38],3)
        self.assertEqual(p.COMMAND.unpack(p.command(s,5,p.RUN,s.q,hand_pose=4))[38],4)
        with self.assertRaises(ValueError):p.command(s,6,p.RUN,s.q,hand_pose=5)
        compensated = list(p.STATE.unpack(data)); compensated[171] = 119
        self.assertEqual(p.State.decode(p.STATE.pack(*compensated)).hand_flags,119)
        recovery = list(p.STATE.unpack(data)); recovery[171] = 161; recovery[266] = 8
        self.assertEqual(p.State.decode(p.STATE.pack(*recovery)).hand_recovery_joint,8)
        self.assertEqual(p.COMMAND.unpack(p.command(s,7,p.WAIT,s.q,hand_recover=8))[39],8)
        edited = p.COMMAND.unpack(p.command(s,8,p.RUN,s.q,hand_pose=2,
                                            hand_grasp=np.arange(14)*.01, hand_live=True))
        self.assertEqual(edited[40],2)
        np.testing.assert_allclose(edited[41:55],np.arange(14)*.01)
        self.assertEqual(edited[55],0)
        self.assertEqual(p.COMMAND.unpack(p.command(s,9,p.RUN,s.q,upper_mode=1))[55],1)
        self.assertEqual(p.COMMAND.unpack(p.command(s,9,p.RUN,s.q,upper_mode=1))[56],0)
        self.assertEqual(p.COMMAND.unpack(p.command(s,9,p.RUN,s.q,upper_mode=1,
                                                   upper_target_active=True))[55:],(1,1))
        with self.assertRaises(ValueError):p.command(s,9,p.RUN,s.q,upper_target_active=True)
        for active in (1, -1, None):
            with self.assertRaises(ValueError):p.command(s,9,p.RUN,s.q,upper_mode=1,upper_target_active=active)
        for mode in (-1,2,True):
            with self.assertRaises(ValueError):p.command(s,9,p.RUN,s.q,upper_mode=mode)
        for joint in (-2,14,True):
            with self.assertRaises(ValueError): p.command(s,8,p.WAIT,s.q,hand_recover=joint)
        for raw in (bytes(460),bytes(700)):
            with self.assertRaises(ValueError):p.State.decode(raw)

    def test_model_selection_boundary_and_forced_modes(self):
        for cmd,expected in [([0,0,0],'balance'),([.049,0,0],'balance'),([.05,0,0],'homie'),
                             ([0,-.051,0],'homie'),([0,0,.051],'homie'),([.049,0,0],'balance'),
                             ([0,0,0],'balance')]:
            self.assertEqual(self.policy.select(cmd),expected)
        for mode in ('balance','homie'):
            policy=Policy(ROOT,self.cfg,policy_mode=mode)
            for cmd in ([0,0,0],[.3,-.3,.5]):self.assertEqual(policy.select(cmd),mode)
        with self.assertRaises(ValueError):Policy(ROOT,self.cfg,policy_mode='walk')
        with self.assertRaises(ValueError):self.policy.select([0,np.inf,0])

    def test_immediate_return_at_threshold_while_body_still_moves(self):
        s=state(self.cfg);s.stage=p.POLICY;s.dq.fill(2.);s.omega.fill(.6)
        self.assertEqual(self.policy.select([.2,0,0],s),'homie')
        self.assertEqual(self.policy.select([.04,.04,0],s),'homie')
        self.assertEqual(self.policy.select([.049,0,0],s),'balance')
        self.assertEqual(self.policy.selection_reason,'below_command_threshold')
        self.assertEqual(self.policy.select([0,0,.05],s),'homie')
        self.assertEqual(self.policy.select([0,0,0],s),'balance')

    def test_training_filter_reaches_switch_in_49_ticks_from_point_six(self):
        cmd=Commands(self.cfg);cmd.value[0]=.6
        feedback=state(self.cfg);feedback.buttons=p.L1
        for n in range(1,50):
            velocity=cmd.update(feedback,.02)
            self.assertAlmostEqual(velocity[0],.6*.95**n,places=6)
            self.assertEqual(self.policy.select(velocity),'homie' if n<49 else 'balance')
        self.assertEqual(self.cfg['command_filter_tau_s'],.4)

    def test_selection_and_inference_use_the_same_threshold(self):
        s=state(self.cfg);s.stage=p.POLICY
        obs=Observations(self.contract).push(s,[.049,0,0],.74,[0,0,0])
        self.policy.select([.049,0,0],s)
        self.policy(obs,[.049,0,0],s)
        self.assertEqual(self.policy.selected,'balance')

    def test_both_models_match_original_export_vectors(self):
        for name in ('balance','homie'):
            with np.load(ROOT/'models'/name/'verification_vectors.npz',allow_pickle=False) as data:
                obs_key, action_key = ('inputs','outputs') if name=='balance' else ('observations','actions')
                for obs,action in zip(data[obs_key],data[action_key]):
                    np.testing.assert_allclose(self.policy.infer(name,obs.reshape(1,-1)),action,rtol=1e-5,atol=1e-5)

    def test_training_history_tracks_selected_actions_across_both_switch_directions(self):
        actual=Observations(self.contract);rng=np.random.default_rng(17)
        exported=json.loads((ROOT/'models/balance/config.json').read_text())
        indices=[self.cfg['joint_names_sdk29'].index(n) for n in exported['joint_names']]
        # Oracle follows training compute_observations: append measured state
        # with the previously selected/clipped action, never the shadow action.
        history=np.zeros((6,76),dtype=np.float32)
        previous=np.zeros(12,dtype=np.float32)
        for i in range(40):
            feedback=state(self.cfg)
            feedback.q+=rng.normal(0,.02,29).astype(np.float32)
            feedback.dq=rng.normal(0,.1,29).astype(np.float32)
            feedback.omega=rng.normal(0,.1,3).astype(np.float32)
            feedback.applied_q+=.7  # Must not be used to infer previous action.
            velocity=np.array([0,0,0] if i%8<4 else [.12,-.08,.1],dtype=np.float32)
            selected=self.policy.select(velocity)
            obs=actual.push(feedback,velocity,.74,[0,0,0])
            frame=np.zeros(76,dtype=np.float32);frame[3]=.74;frame[9]=-1
            frame[4:7]=feedback.omega*.5
            frame[10:37]=feedback.q[indices]-np.array(exported['default_joint_pos'],dtype=np.float32)
            frame[37:64]=feedback.dq[indices]*.05;frame[64:]=previous
            expected={}
            for name in ('balance','homie'):
                candidate=frame.copy()
                candidate[:3]=0 if name=='balance' else velocity*[2,2,.5]
                expected[name]=np.vstack((history[1:],candidate))
                np.testing.assert_allclose(obs[name].reshape(6,76),expected[name],atol=1e-7)
            actions={name:rng.uniform(-150,150,12).astype(np.float32) for name in ('balance','homie')}
            saved={name:value.copy() for name,value in obs.items()}
            actual.record_action(actions,selected)
            history=expected[selected];previous=np.clip(actions[selected],-100,100)
            for name,model in actual.models.items():
                np.testing.assert_array_equal(model.previous_action,previous)
                np.testing.assert_allclose(model.history,history,atol=1e-7)
                np.testing.assert_array_equal(obs[name],saved[name])  # commit cannot rewrite inference inputs
        actual.reset();self.assertEqual(actual.frames,0)
        for model in actual.models.values():
            self.assertFalse(model.previous_action.any());self.assertFalse(model.history.any())

    def test_both_models_skip_fixed_waist_and_preserve_upper_targets(self):
        s=state(self.cfg);s.q[13:15]=[.2,-.2]
        obs=Observations(self.contract).push(s,[0,0,0],.74,[0,0,0])
        for name in ('balance','homie'):
            np.testing.assert_array_equal(obs[name][0,-76+10:-76+37],np.zeros(27))
        s.applied_q[12:]=.12
        homie=targets(np.zeros(12),s,self.cfg,self.contract,'homie')
        balance=targets(np.zeros(12),s,self.cfg,self.contract,'balance')
        np.testing.assert_array_equal(homie[12:],s.applied_q[12:])
        np.testing.assert_array_equal(balance[12:],s.applied_q[12:])
        np.testing.assert_array_equal(homie[:12],np.array(self.cfg['default_q_sdk29'][:12],dtype=np.float32))

    def test_balance_zero_command_raw_history_clipping_and_reset(self):
        model=Observations(self.contract).models['balance'];s=state(self.cfg)
        s.applied_q[:12]+=.2
        first=model.push(s,[.8,-.4,.6],.5,[.1,.2,.3]).reshape(6,76).copy()
        np.testing.assert_array_equal(first[:-1],0)
        np.testing.assert_array_equal(first[-1,:3],0)
        self.assertAlmostEqual(first[-1,3],.74,places=6)
        np.testing.assert_array_equal(first[-1,64:],0)
        raw=np.array([150.,-150.]+[.7]*10,dtype=np.float32)
        model.record_action(raw)
        s.dq[:]=3000
        second=model.push(s,[0,0,0],.74,[0,0,0]).reshape(6,76)
        np.testing.assert_array_equal(second[-2],first[-1])
        np.testing.assert_array_equal(second[-1,64:],np.clip(raw,-100,100))
        np.testing.assert_array_equal(second[-1,37:64],100)
        np.testing.assert_array_equal(model.previous_action,np.clip(raw,-100,100))
        model.reset()
        np.testing.assert_array_equal(model.history,0)
        np.testing.assert_array_equal(model.previous_action,0)

    def test_balance_pd_matches_export_by_joint_name(self):
        export=json.loads((ROOT/'models/balance/config.json').read_text())
        _,profiles,_=load_config(ROOT)
        indices=[self.cfg['joint_names_sdk29'].index(n) for n in export['joint_names']]
        for key in ('kp','kd'):
            np.testing.assert_array_equal(np.array(profiles['balance'][key+'_sdk29'])[indices],export['pd'][key])

    def test_action_clip_precedes_scaling(self):
        # Wide test-only bounds distinguish training clipping from physical joint limits.
        cfg=dict(self.cfg,joint_lower_sdk29=[-1000.]*29,joint_upper_sdk29=[1000.]*29)
        s=state(self.cfg);s.applied_q[12:]=.12
        for name in ('balance','homie'):
            q=targets(np.full(12,150.),s,cfg,self.contract,name)
            np.testing.assert_allclose(q[:12],np.array(cfg['default_q_sdk29'][:12])+25,atol=1e-6)
            np.testing.assert_array_equal(q[12:],s.applied_q[12:])

    def test_pushcart_upper_pd_matches_confirmed_values_and_token(self):
        indices=[12]+list(range(15,29))
        for key,expected in [('kp',[300]+[100,100,50,50,20,20,20]*2),
                             ('kd',[3]+[2,2,2,2,1,1,1]*2)]:
            values=np.asarray(self.cfg['pushcart_upper_'+key+'_sdk29'])
            np.testing.assert_array_equal(values[indices],expected)
            np.testing.assert_array_equal(values[list(range(12))+[13,14]],0)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copytree(ROOT/'config',root/'config')
            _,_,before=load_config(root)
            path=root/'config/pushcart_upper_pd.json';data=json.loads(path.read_text())
            data['kp'][0]+=1;path.write_text(json.dumps(data))
            self.assertNotEqual(load_config(root)[2],before)

    def test_upper_interpolation_period_is_validated_and_changes_token(self):
        self.assertEqual(self.cfg['pushcart_upper_target_interpolation_s'],.02)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copytree(ROOT/'config',root/'config')
            _,_,before=load_config(root)
            path=root/'config/pushcart_upper_pd.json';data=json.loads(path.read_text())
            data['target_interpolation_s']=0;path.write_text(json.dumps(data))
            cfg,_,token=load_config(root)
            self.assertEqual(cfg['pushcart_upper_target_interpolation_s'],0)
            self.assertNotEqual(token,before)
            for value in (-.02,.01,.04,True,None,float('nan'),float('inf')):
                data['target_interpolation_s']=value;path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):load_config(root)

    def test_independent_pd_edits_change_combined_token(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copytree(ROOT/'config',root/'config')
            cfg,pd,token=load_config(root)
            self.assertNotEqual(pd['balance']['kp_sdk29'],pd['homie']['kp_sdk29'])
            for name in ('balance','homie'):
                path=root/cfg['pd_files'][name];original=path.read_text();data=json.loads(original)
                data['kp_sdk29'][0]+=1;path.write_text(json.dumps(data))
                _,gains,new=load_config(root)
                self.assertNotEqual(new,token)
                other='balance' if name=='homie' else 'homie'
                self.assertEqual(gains[other],pd[other])
                path.write_text(original)
            path=root/cfg['pd_files']['homie'];data=json.loads(path.read_text());data['kd_sdk29'][14]=-1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_config(root)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):self.policy.infer('homie',np.zeros((1,516)))
        with self.assertRaises(ValueError):self.policy.infer('balance',np.zeros((1,516)))
        with self.assertRaises(ValueError):self.policy.infer('balance',np.full((1,456),np.nan))
        with self.assertRaises(ValueError):targets(np.zeros(15),state(self.cfg),self.cfg,self.contract,'balance')
        with self.assertRaises(ValueError):targets(np.zeros(15),state(self.cfg),self.cfg,self.contract,'homie')

    def test_command_filter_and_immediate_deadman(self):
        s=state(self.cfg);s.sticks[:]=[0,1,0,0];cmd=Commands(self.cfg)
        np.testing.assert_array_equal(cmd.update(s,.02),[0,0,0])
        s.buttons=p.L1;out=cmd.update(s,.02);self.assertAlmostEqual(out[0],.06,places=6)
        s.buttons=p.B|p.L1;np.testing.assert_array_equal(cmd.update(s,.02),[0,0,0])


if __name__=='__main__':unittest.main()

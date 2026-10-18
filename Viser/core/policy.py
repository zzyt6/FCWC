"""Expose Homie's estimator from the same inference as its action, in memory only."""
import time
import numpy as np

from core import robot  # Make the sibling deployment package importable.
from hybrid_deploy.policy import Policy as DeploymentPolicy


ENCODER_OUTPUT = '/encoder/encoder.4/Gemm_output_0'
LATENT_OUTPUT = '/Div_output_0'


class Policy(DeploymentPolicy):
    def __init__(self, root, cfg, policy_mode='auto', *, pushcart_homie=None):
        # Retain the deployment's artifact hashes and original interface checks.
        super().__init__(root, cfg, policy_mode)
        self.pushcart_homie = pushcart_homie
        self.pushcart_active = False
        self.homie_estimator_source = None
        import onnx
        import onnxruntime as ort

        model = onnx.shape_inference.infer_shapes(
            onnx.load(str(self.root / cfg['model_files']['homie'])))
        tensors = {value.name: value for value in model.graph.value_info}
        for name, size in ((ENCODER_OUTPUT, 35), (LATENT_OUTPUT, 32)):
            value = tensors.get(name)
            if (value is None or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                    or len(value.type.tensor_type.shape.dim) != 2
                    or value.type.tensor_type.shape.dim[-1].dim_value != size):
                raise ValueError('Unsupported Homie estimator tensor: ' + name)
            model.graph.output.append(value)
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        # No derived ONNX file is written or copied to the log directory.
        self.sessions['homie'] = ort.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider'])
        self.homie_estimator = None
        # Trigger ORT's first-run work before a bridge can be armed. These dummy
        # inputs never enter the robot's observation history or action stream.
        started = time.perf_counter()
        for _ in range(2):
            for name in ('balance', 'homie'):
                self.infer(name, np.zeros((1, 456), np.float32))
        self.warmup_ms = (time.perf_counter() - started) * 1000
        self.warmed = True
        self.homie_estimator = None
        self.homie_estimator_source = None

    def __call__(self, observations, velocity, state=None, *, pushcart=False):
        if pushcart and self.pushcart_homie is None:
            raise ValueError('Paired pushcart HIM estimator was not loaded')
        self.pushcart_active = bool(pushcart)
        return super().__call__(observations, velocity, state)

    def infer(self, name, observations):
        if name != 'homie':
            return super().infer(name, observations)
        self.homie_estimator = None
        self.homie_estimator_source = None
        if self.pushcart_active:
            action, self.homie_estimator = self.pushcart_homie.infer(observations)
            self.homie_estimator_source = 'pushcart_finetuned'
            return action
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (1, 456) or not np.isfinite(obs).all():
            raise ValueError('Invalid observation: homie')
        session = self.sessions['homie']
        action, encoder, latent = session.run(
            ['actions', ENCODER_OUTPUT, LATENT_OUTPUT], {session.get_inputs()[0].name: obs})
        for value, shape in ((action, (1, 12)), (encoder, (1, 35)), (latent, (1, 32))):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid Homie action/estimator output')
        self.homie_estimator = dict(velocity_raw=encoder[0, :3].copy(),
                                    velocity_original_raw=encoder[0, :3].copy(),
                                    latent_raw=encoder[0, 3:].copy(),
                                    latent_normalized=latent[0].copy())
        self.homie_estimator_source = 'homie_original'
        return action[0]


def output_metadata(config):
    """Small analysis contract; raw estimates are not measured robot velocity."""
    import hashlib
    return dict(
        model_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for name, path in config.policy_paths.items()},
        homie_model_selection='homie_estimator_source selects the original artifact above or '
                              'the paired artifact in high_policy.homie; model_actions.homie is before leg residuals',
        model_actions={name: dict(
            shape=[config.contracts[name]['output']['shape'][-1]],
            sdk_joint_names=config.deployment['joint_names_sdk29'][:config.contracts[name]['output']['shape'][-1]],
            representation='raw network output before action clipping/scaling and joint limits',
            action_scale=config.contracts[name]['output']['action_scale'])
            for name in ('balance', 'homie')},
        observation_history=dict(
            frames=6, frame_dim=76, source='selected_policy_execution',
            previous_action='previous selected output after training clipping, before joint/PD limits',
            candidate_commands='only newest frame differs: Balance zero, Homie filtered velocity; '
                               'older frames retain selected commands',
            preparation='real feedback sampled throughout PREPARING and retained into HOLDING/POLICY',
            commit='only after RUN is sent; startup/observe shadow outputs are not committed'),
        homie_estimator=dict(
            velocity_raw=dict(shape=[3], axes=['x', 'y', 'z'],
                              units='actor input in training-target units; not measured velocity',
                              source='homie_estimator_source: homie_original or pushcart_finetuned',
                              tensor='original encoder[:3] or selected model set paired Homie vel_used'),
            velocity_original_raw=dict(shape=[3], tensor=ENCODER_OUTPUT, indices=[0, 3],
                                       scope='original HIM velocity on the same history, including during pushcart'),
            latent_raw=dict(shape=[32], tensor=ENCODER_OUTPUT, indices=[3, 35]),
            latent_normalized=dict(shape=[32], tensor=LATENT_OUTPUT,
                                   normalization='L2, denominator clamped to >= 1e-12'),
            scope='network estimate from the same Homie inference as model_actions.homie; not sensor ground truth'),
        alignment='same telemetry row: state.tick, state.received_ns, policy_steps, policy_inference_monotonic; '
                  'model_actions and homie_estimator are null when no new inference occurred; '
                  'selected_model identifies the selected output; bridge targets may lag the requested target')

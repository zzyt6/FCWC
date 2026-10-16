"""The student's paired HIM velocity estimator and frozen Homie actor.

Use Viser's 456-value execution history. Only the actor's three velocity inputs
come from the tuned encoder; the 32-value latent stays with the original HIM.
"""
import hashlib
from pathlib import Path
import time

import numpy as np


ENCODER = '/encoder/encoder.4/Gemm_output_0'
LATENT = '/Div_output_0'


class PushcartHomie:
    def __init__(self, path, original_path, threads=1):
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        self.path = Path(path)
        model = onnx.shape_inference.infer_shapes(onnx.load(str(self.path)))
        original = onnx.load(str(original_path))
        weights = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
        # The packaged graph must preserve the exact locomotion actor/encoder.
        # A different base policy cannot be treated as just velocity adaptation.
        for value in original.graph.initializer:
            if value.name not in weights or not np.array_equal(numpy_helper.to_array(value), weights[value.name]):
                raise ValueError('Pushcart Homie base weight mismatch: ' + value.name)
        values = {v.name: v for v in model.graph.value_info}
        for name, size in ((ENCODER, 35), (LATENT, 32)):
            value = values.get(name)
            if (value is None or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                    or len(value.type.tensor_type.shape.dim) != 2
                    or value.type.tensor_type.shape.dim[-1].dim_value != size):
                raise ValueError('Unsupported pushcart HIM tensor: ' + name)
            model.graph.output.append(value)
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # Extra diagnostics are in-memory outputs, not another model artifact.
        self.session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
        if ([(v.name, v.shape, v.type) for v in self.session.get_inputs()] !=
                [('obs_history', [1, 456], 'tensor(float)'), ('use_finetuned', [1, 1], 'tensor(float)')]
                or [(v.name, v.shape, v.type) for v in self.session.get_outputs()[:3]] !=
                [('leg_actions', [1, 12], 'tensor(float)'), ('vel_used', [1, 3], 'tensor(float)'),
                 ('vel_homie', [1, 3], 'tensor(float)')]):
            raise ValueError('Pushcart Homie ONNX interface mismatch')
        self.use_finetuned = np.ones((1, 1), np.float32)
        started = time.perf_counter()
        for _ in range(2):
            self.infer(np.zeros((1, 456), np.float32))
        self.warmup_ms = (time.perf_counter() - started) * 1000
        self.warmed = True
        self.metadata = dict(path=str(self.path), sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                             velocity='paired fine-tuned HIM encoder, first 3 outputs',
                             latent='original HIM encoder, last 32 outputs, L2 normalized',
                             actor='unchanged; checked against the regular Homie artifact',
                             input='Viser current Homie candidate observation and selected execution history')

    def infer(self, observations):
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (1, 456) or not np.isfinite(obs).all():
            raise ValueError('Invalid observation: pushcart Homie')
        action, velocity, original_velocity, encoder, latent = self.session.run(
            ['leg_actions', 'vel_used', 'vel_homie', ENCODER, LATENT],
            dict(obs_history=obs, use_finetuned=self.use_finetuned))
        for value, shape in ((action, (1, 12)), (velocity, (1, 3)), (original_velocity, (1, 3)),
                             (encoder, (1, 35)), (latent, (1, 32))):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid pushcart Homie action/estimator output')
        return action[0], dict(velocity_raw=velocity[0].copy(),
                               velocity_original_raw=original_velocity[0].copy(),
                               latent_raw=encoder[0, 3:].copy(), latent_normalized=latent[0].copy())

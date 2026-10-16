"""Compile limits, locomotion PD and the independent push-cart upper PD into the bridge."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from hybrid_deploy.configuration import load_config


def generate(source, output):
    cfg, gains, token = load_config(Path(source).resolve().parents[1])
    lines = ['#pragma once', '#include <array>', '#include <cstdint>',
             'namespace decoupled { namespace config {',
             'constexpr uint64_t token = %dULL;' % token]
    values = dict(cfg)
    for name, pd in gains.items():
        for key in ('kp_sdk29','kd_sdk29'):
            values[name+'_'+key] = pd[key]
    for name, value in values.items():
        if isinstance(value, list) and len(value) == 29 and all(isinstance(v,(int,float)) for v in value):
            lines.append('constexpr std::array<float,29> %s = {%s};' %
                         (name, ','.join(format(float(v), '.10f')+'f' for v in value)))
        elif isinstance(value, bool):
            lines.append('constexpr bool %s = %s;' % (name, str(value).lower()))
        elif isinstance(value, (int,float)):
            lines.append('constexpr double %s = %.10f;' % (name,value))
    lines.append('} }')
    target = Path(output)
    target.parent.mkdir(parents=True,exist_ok=True)
    text = '\n'.join(lines)+'\n'
    if not target.exists() or target.read_text() != text:
        target.write_text(text)


if __name__ == '__main__':
    generate(sys.argv[1], sys.argv[2])

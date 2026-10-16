"""检查本机架构、SDK 库和 ONNX 环境；绝不连接 DDS 或操作电机。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    arch={'arm64':'aarch64','AMD64':'x86_64'}.get(platform.machine(),platform.machine())
    if arch not in ('aarch64','x86_64'): raise RuntimeError('Unsupported architecture: '+arch)
    if sys.version_info < (3,10): raise RuntimeError('Python >=3.10 required for deployment')
    sdk=ROOT/'vendor/unitree_sdk2'
    for name in ['libddsc.so','libddscxx.so']:
        path=sdk/'thirdparty/lib'/arch/name
        header=path.read_bytes()[:20]
        expected=183 if arch=='aarch64' else 62
        if header[:4]!=b'\x7fELF' or struct.unpack('<H',header[18:20])[0]!=expected:
            raise RuntimeError('SDK ELF architecture mismatch: '+str(path))
    if not (sdk/'lib'/arch/'libunitree_sdk2.a').is_file(): raise RuntimeError('Missing SDK static library')
    import numpy, onnxruntime
    report=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),machine=arch,
                python=platform.python_version(),python_executable=sys.executable,
                numpy=numpy.__version__,onnxruntime=onnxruntime.__version__,
                providers=onnxruntime.get_available_providers(),sdk_elf_architecture_matches=True,
                os_release=Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else None,
                l4t_release=Path('/etc/nv_tegra_release').read_text() if Path('/etc/nv_tegra_release').exists() else None,
                scope='Local dependency/ELF inspection only. No DDS, factory mode release, or motor commands.')
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/platform_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

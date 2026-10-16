"""无第三方依赖；解压后核对部署包固定文件的完整性。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / 'SHA256SUMS.json'
    if not manifest.exists():
        raise RuntimeError('Run tools/package.py before checking a source folder')
    records = json.loads(manifest.read_text())
    failed = []
    for name, expected in records.items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failed.append(name)
    if failed:
        raise RuntimeError('Missing/modified deployment files: '+', '.join(failed))
    print('Deployment integrity passed: %d files' % len(records))


if __name__ == '__main__':
    main()

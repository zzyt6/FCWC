"""打包全部部署依赖，排除原电脑的构建目录、虚拟环境及缓存。"""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'build', '.venv', 'dist', '__pycache__', '.git', 'reports'}


def payload():
    return sorted(path for path in ROOT.rglob('*') if path.is_file()
                  and not any(part in SKIP for part in path.relative_to(ROOT).parts)
                  and path.suffix != '.pyc')


def main():
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in payload()
              if path.name != 'SHA256SUMS.json' and path.relative_to(ROOT).parts[0] != 'reports'}
    # reports 是运行时可重写的结果；固定源码、配置、模型和库参与传输完整性检查。
    (ROOT / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2, sort_keys=True)+'\n')
    destination = ROOT / 'dist' / (ROOT.name+'.tar.gz')
    destination.parent.mkdir(exist_ok=True)
    with tarfile.open(destination, 'w:gz') as archive:
        for path in payload():
            archive.add(path, arcname=ROOT.name+'/'+str(path.relative_to(ROOT)), recursive=False)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix+'.sha256').write_text(digest+'  '+destination.name+'\n')
    print(str(destination))
    print('SHA256 '+digest)


if __name__ == '__main__':
    main()

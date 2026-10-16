"""Run the single outer Viser integration test (synthetic feedback, no DDS)."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    script = Path(__file__).resolve().parents[2] / 'Viser/test/test_web.py'
    raise SystemExit(subprocess.call([sys.executable, str(script)]))

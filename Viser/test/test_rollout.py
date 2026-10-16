"""Compatibility entry point for the hybrid offline integration test.

The old physics rollout depended on the removed RL-gym policy. This checks
synthetic bridge feedback and model switching, not physical walking stability.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_web import main

if __name__ == '__main__':
    main()

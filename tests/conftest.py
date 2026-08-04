import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

for path in (REPO_ROOT, BENCHMARKS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

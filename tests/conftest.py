from __future__ import annotations

import sys
from pathlib import Path

# Ensure `import src...` works when pytest uses importlib import mode.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


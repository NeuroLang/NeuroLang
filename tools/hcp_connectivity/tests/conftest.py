import sys
from pathlib import Path

# the package is run from tools/ (python -m hcp_connectivity)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

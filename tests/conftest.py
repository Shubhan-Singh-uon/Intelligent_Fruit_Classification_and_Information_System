import sys
from pathlib import Path

# Make `from src.xxx import ...` work when pytest is started from any directory,
# and let the UI tests import the shared fakes in this folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# ShotCut TikTok Automator source package
# Makes src/ a proper Python package for relative imports

import sys
import os

# Ensure the project root is on the path for absolute imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
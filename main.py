import sys
import os
from pathlib import Path

# Ensure the root directory is in the path so we can import production_system, etc.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.app import App

if __name__ == "__main__":
    app = App()
    app.run()

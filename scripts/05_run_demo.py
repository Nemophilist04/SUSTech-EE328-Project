from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(project_root / "app.py")], check=True)


if __name__ == "__main__":
    main()

"""
server/app.py — Entry point required by openenv validate.

openenv validate expects:
  - server/app.py to exist
  - [project.scripts] server = "server.app:main" in pyproject.toml

This file imports the FastAPI app from the root app.py and exposes
a main() function that starts the uvicorn server.
"""

from __future__ import annotations

import os
import sys

# Make sure the root directory is on the path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from root app.py
from app import app  # noqa: F401  (re-exported for openenv)


def main() -> None:
    """Entry point called by `server` script defined in pyproject.toml."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

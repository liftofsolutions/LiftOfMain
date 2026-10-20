"""Shared by every suite: tests read `test.env`, never `server.env`.

`setdefault`, so `RAG_ENV=server uv run pytest` can still point a run at
the server's variables on purpose.
"""

import os

os.environ.setdefault("RAG_ENV", "test")

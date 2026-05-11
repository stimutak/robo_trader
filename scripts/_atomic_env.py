"""Atomic .env writer to prevent partial-write corruption on interrupt.

Used by helper scripts (`_apply_security_env.py`, `_set_dashboard_password.py`,
`_disable_auth_for_dev.py`) to ensure that a SIGKILL / power loss / Ctrl-C in
the middle of writing the .env file leaves the original file intact rather
than truncated.

Rationale: `Path.write_text` opens the target in `'w'` mode which truncates
the file before any new bytes are written. A process kill in that window
destroys the user's secrets. `os.replace` is atomic on POSIX file systems
when source and target are on the same filesystem.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_env_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` atomically with mode 0600.

    Steps:
      1. Create a temp file in the same directory (so os.replace stays atomic).
      2. Write + fsync the contents.
      3. chmod 0600 before publishing.
      4. os.replace() onto the final name.

    On any error before the replace, the temp file is removed and the original
    file is untouched.
    """
    path = Path(path)
    parent = path.parent if str(path.parent) else Path(".")
    fd, tmp_name = tempfile.mkstemp(
        dir=str(parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

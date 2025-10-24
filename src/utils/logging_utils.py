import json
from pathlib import Path
from datetime import datetime
import subprocess

class JsonlLogger:
    def __init__(self, path, extra=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.extra = extra or {}
        self._fh = self.path.open("a", buffering=1, encoding="utf-8")
    def info(self, event, **kwargs):
        rec = {"ts": datetime.utcnow().isoformat() + "Z", "level": "INFO", "event": event}
        rec.update(self.extra)
        rec.update(kwargs)
        self._fh.write(json.dumps(rec) + "\n")
    def close(self):
        self._fh.close()

def get_git_commit(default="notgit"):
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        return sha
    except Exception:
        return default
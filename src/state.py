import threading
import time


class ServiceState:
    """Minimal thread-safe status registry.

    Replaces the old global HEALTH_STATUS dict: model state is written from
    the worker thread, per-task status from the event loop, all under one lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.model_state = "loading"  # loading | ready | error
        self.model_error: str | None = None
        self.processed = 0
        self.errors = 0
        # record_id -> {"status": queued|processing|done|error, "error": str|None}
        self._tasks: dict[str, dict] = {}

    def set_model_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self.model_state = state
            self.model_error = error

    def set_task(self, record_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            self._tasks[record_id] = {"status": status, "error": error}

    def get_task(self, record_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(record_id)
            return dict(task) if task else None

    def all_tasks(self) -> dict[str, dict]:
        with self._lock:
            return {rid: dict(t) for rid, t in self._tasks.items()}

    def bump(self, ok: bool = True) -> None:
        with self._lock:
            if ok:
                self.processed += 1
            else:
                self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            pending = sum(
                1 for t in self._tasks.values() if t["status"] in ("queued", "processing")
            )
            return {
                "model": self.model_state,
                "model_error": self.model_error,
                "uptime_seconds": int(time.time() - self.started_at),
                "processed": self.processed,
                "errors": self.errors,
                "pending": pending,
            }
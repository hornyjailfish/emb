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
        self.db_connected = False
        self.db_error: str | None = None
        self.processed = 0
        self.errors = 0
        # record_id -> {status, error, embedding?, kind?}
        self._tasks: dict[str, dict] = {}

    def set_model_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self.model_state = state
            self.model_error = error

    def model_ready(self) -> bool:
        with self._lock:
            return self.model_state == "ready"

    def set_db_status(self, connected: bool, error: str | None = None) -> None:
        with self._lock:
            self.db_connected = connected
            self.db_error = error

    def set_task(
        self,
        record_id: str,
        status: str,
        error: str | None = None,
        *,
        embedding: list[float] | None = None,
        kind: str | None = None,
        clear_embedding: bool = False,
    ) -> None:
        with self._lock:
            prev = self._tasks.get(record_id) or {}
            task: dict = {
                "status": status,
                "error": error,
            }
            # Preserve kind / embedding across status transitions unless cleared.
            k = kind if kind is not None else prev.get("kind")
            if k is not None:
                task["kind"] = k
            if clear_embedding:
                pass
            elif embedding is not None:
                task["embedding"] = embedding
            elif "embedding" in prev and status == "done":
                task["embedding"] = prev["embedding"]
            self._tasks[record_id] = task

    def get_task(self, record_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(record_id)
            return dict(task) if task else None

    def all_tasks(self) -> dict[str, dict]:
        with self._lock:
            # Omit bulky embeddings from the full dump — fetch per-task for those.
            out: dict[str, dict] = {}
            for rid, t in self._tasks.items():
                row = {k: v for k, v in t.items() if k != "embedding"}
                if "embedding" in t:
                    row["has_embedding"] = True
                    row["dim"] = len(t["embedding"]) if t["embedding"] is not None else 0
                out[rid] = row
            return out

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
                "db_connected": self.db_connected,
                "db_error": self.db_error,
                "uptime_seconds": int(time.time() - self.started_at),
                "processed": self.processed,
                "errors": self.errors,
                "pending": pending,
            }
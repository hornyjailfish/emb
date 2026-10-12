import logging
import math
import threading
import uuid
from dataclasses import dataclass
from queue import Empty, Queue

from src.config import Settings, get_settings
from src.state import ServiceState

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when a query arrives before the model has finished loading."""


class QueryError(RuntimeError):
    """Raised when the model fails to encode a query."""


@dataclass
class ModelJob:
    record_id: str
    text: str | None
    image_url: str | None


@dataclass
class ModelResult:
    record_id: str
    text_embedding: list[float] | None
    image_embedding: list[float] | None
    error: str | None


@dataclass
class QueryJob:
    query_id: str
    text: str | None
    image_url: str | None
    normalize: bool = False


@dataclass
class QueryResult:
    text_embedding: list[float] | None
    image_embedding: list[float] | None
    error: str | None


def _to_list(vec) -> list[float] | None:
    if vec is None:
        return None
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


def _normalize(vec: list[float] | None) -> list[float] | None:
    """L2-normalize to unit length so cosine similarity == dot product."""
    if vec is None:
        return None
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class ModelWorker:
    """Owns the torch model and runs inference far from the event loop.

    The model is loaded lazily inside the thread so startup is instant and the
    web server keeps responding while weights load. Jobs are taken strictly one
    at a time (multimodal models encode images per-sample, so no batching).
    """

    def __init__(self, state: ServiceState):
        self.state = state
        self.settings: Settings = get_settings()
        self.jobs: Queue[ModelJob] = Queue()
        self.results: Queue[ModelResult] = Queue()
        self.query_jobs: Queue[QueryJob] = Queue()
        self._query_results: dict[str, QueryResult] = {}
        self._query_cond = threading.Condition()
        self._model = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="model-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker loop to exit so the thread can wind down cleanly."""
        self._stop.set()

    def _load_model(self):
        # Imported here so heavyweight deps load only inside this thread.
        from sentence_transformers import SentenceTransformer

        # Re-assert after the library imports (some libs touch logging during
        # import), so the download itself stays quiet.
        from src.utils import quiet_third_party_loggers

        quiet_third_party_loggers()

        return SentenceTransformer(
            self.settings.model,
            trust_remote_code=True,
            model_kwargs={"modality": self.settings.modalities},
        )

    def _run(self) -> None:
        self.state.set_model_state("loading")
        try:
            self._model = self._load_model()
        except Exception as e:
            logger.exception("Model failed to load")
            self.state.set_model_state("error", str(e))
            return

        self.state.set_model_state("ready")
        logger.info("Model ready")

        while not self._stop.is_set():
            # Query jobs jump the document queue so interactive search requests
            # aren't starved behind a large batch of record embeddings.
            try:
                self._handle_query(self.query_jobs.get_nowait())
                continue
            except Empty:
                pass

            # Timed poll keeps the loop responsive to query jobs even when no
            # document jobs are queued.
            try:
                job = self.jobs.get(timeout=0.1)
            except Empty:
                continue
            self._handle_document(job)

    def _handle_document(self, job: ModelJob) -> None:
        try:
            result = ModelResult(
                record_id=job.record_id,
                text_embedding=_to_list(self._model.encode_document(job.text)) if job.text else None,
                image_embedding=_to_list(self._model.encode_document(job.image_url)) if job.image_url else None,
                error=None,
            )
        except Exception as e:
            result = ModelResult(
                record_id=job.record_id,
                text_embedding=None,
                image_embedding=None,
                error=str(e),
            )
        self.results.put(result)

    def _handle_query(self, job: QueryJob) -> None:
        try:
            text_embedding = _to_list(self._model.encode_query(job.text)) if job.text else None
            image_embedding = _to_list(self._model.encode_query(job.image_url)) if job.image_url else None
            if job.normalize:
                text_embedding = _normalize(text_embedding)
                image_embedding = _normalize(image_embedding)
            result = QueryResult(
                text_embedding=text_embedding,
                image_embedding=image_embedding,
                error=None,
            )
        except Exception as e:
            result = QueryResult(
                text_embedding=None,
                image_embedding=None,
                error=str(e),
            )
        with self._query_cond:
            self._query_results[job.query_id] = result
            self._query_cond.notify_all()

    def encode_query(
        self,
        text: str | None,
        image_url: str | None,
        timeout: float = 60.0,
        normalize: bool = False,
    ) -> tuple[list[float] | None, list[float] | None]:
        """Block the caller until the worker encodes a query and returns it.

        Runs on the worker thread but is safe to call from the event loop via
        `asyncio.to_thread`. Returns (text_embedding, image_embedding); each is
        None when that modality wasn't supplied. `normalize` L2-normalizes the
        returned vectors.
        """
        if self._stop.is_set():
            raise ModelNotReadyError("model worker is stopped")
        if not self.state.model_ready():
            raise ModelNotReadyError(f"model not ready (state: {self.state.model_state})")

        query_id = uuid.uuid4().hex
        with self._query_cond:
            self._query_results.pop(query_id, None)
            self.query_jobs.put(QueryJob(query_id=query_id, text=text, image_url=image_url, normalize=normalize))
            ok = self._query_cond.wait_for(
                lambda: query_id in self._query_results,
                timeout=timeout,
            )
            result = self._query_results.pop(query_id, None)

        if not ok or result is None:
            raise TimeoutError("query encoding timed out")
        if result.error:
            raise QueryError(result.error)
        return result.text_embedding, result.image_embedding
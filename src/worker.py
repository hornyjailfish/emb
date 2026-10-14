import io
import logging
import math
import threading
import uuid
from dataclasses import dataclass
from queue import Empty, Queue

from PIL import Image

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
    image: Image.Image | str | None  # prepared PIL image, or a source the model loads


@dataclass
class ModelResult:
    record_id: str
    embedding: list[float] | None
    error: str | None


@dataclass
class QueryJob:
    query_id: str
    text: str | None
    image: Image.Image | str | None  # prepared PIL image, or a source the model loads
    normalize: bool = False


@dataclass
class QueryResult:
    embedding: list[float] | None
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


def _fused_input(
    text: str | None, image: Image.Image | str | None
) -> str | Image.Image | tuple | None:
    """Build the model input.

    A (text, image) tuple fuses both modalities into ONE embedding in a single
    forward pass; a lone modality is passed through unchanged.
    """
    if text and image:
        return (text, image)
    return text or image


def prepare_image(data: bytes, max_pixels: int) -> Image.Image:
    """Decode image bytes and downscale them (aspect ratio kept) for encoding.

    The model's vision processor turns images into 256-1280 vision tokens
    (262144..1310720 px), so CPU cost scales with image size. Capping at
    `max_pixels` keeps inference fast without distorting the picture. The
    config default (`image_max_pixels`, 262144 px ~= 512x512) is the model's
    floor — the fastest CPU setting. Returns a PIL Image, which the model
    accepts directly.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


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
            # self._model = self._load_model()
            self._model = None
        except Exception as e:
            logger.exception("Model failed to load")
            self.state.set_model_state("error", str(e))
            return

        # Surface a failed vision/audio processor early (e.g. missing torchvision)
        # instead of a late, confusing "'NoneType' object is not callable" on the
        # first image job.
        module = None
        try:
            module = self._model[0] if len(self._model) else None
        except (TypeError, KeyError, IndexError, AttributeError):
            module = None
        if module is not None and getattr(module, "processor", None) is None:
            logger.error(
                "Model multimodal processor failed to initialize; image/audio "
                "embeddings will fail. Install the required modality extras "
                "(e.g. torchvision) and restart the service."
            )

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

    def _encode(self, method, content) -> tuple[list[float] | None, str | None]:
        """Encode one input (text, image, or a fused tuple). Returns (vector, error);
        never raises.

        Malformed or missing images are already dropped before this point (see
        prepare_image / main._prepare_image), so a bad image cannot discard a
        valid text embedding.
        """
        if content is None:
            return None, None
        try:
            return _to_list(method(content)), None
        except Exception as e:
            msg = f"encode failed: {e}"
            logger.warning("%s", msg)
            return None, msg

    def _handle_document(self, job: ModelJob) -> None:
        logger.info(job)
        embedding, error = self._encode(
            self._model.encode_document, _fused_input(job.text, job.image)
        )
        self.results.put(
            ModelResult(record_id=job.record_id, embedding=embedding, error=error)
        )

    def _handle_query(self, job: QueryJob) -> None:
        embedding, error = self._encode(
            self._model.encode_query, _fused_input(job.text, job.image)
        )
        if job.normalize:
            embedding = _normalize(embedding)
        logger.info("Query done: %s", job.query_id)
        with self._query_cond:
            self._query_results[job.query_id] = QueryResult(
                embedding=embedding, error=error
            )
            self._query_cond.notify_all()

    def encode_query(
        self,
        text: str | None,
        image: Image.Image | str | None,
        timeout: float = 60.0,
        normalize: bool = False,
    ) -> list[float] | None:
        """Block the caller until the worker encodes a query and returns it.

        Runs on the worker thread but is safe to call from the event loop via
        `asyncio.to_thread`. Returns the single fused query embedding (None if
        no input was given). `normalize` L2-normalizes the returned vector.
        """
        if self._stop.is_set():
            raise ModelNotReadyError("model worker is stopped")
        if not self.state.model_ready():
            raise ModelNotReadyError(f"model not ready (state: {self.state.model_state})")

        query_id = uuid.uuid4().hex
        with self._query_cond:
            self._query_results.pop(query_id, None)
            self.query_jobs.put(QueryJob(query_id=query_id, text=text, image=image, normalize=normalize))
            ok = self._query_cond.wait_for(
                lambda: query_id in self._query_results,
                timeout=timeout,
            )
            result = self._query_results.pop(query_id, None)

        if not ok or result is None:
            raise TimeoutError("query encoding timed out")
        if result.error:
            raise QueryError(result.error)
        return result.embedding

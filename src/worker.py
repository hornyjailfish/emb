import logging
import threading
from dataclasses import dataclass
from queue import Queue

from src.config import Settings, get_settings
from src.state import ServiceState

logger = logging.getLogger(__name__)


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


def _to_list(vec) -> list[float] | None:
    if vec is None:
        return None
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


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
        self._model = None
        self._thread = threading.Thread(target=self._run, name="model-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _load_model(self):
        # Imported here so heavyweight deps load only inside this thread.
        from sentence_transformers import SentenceTransformer

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

        while True:
            job = self.jobs.get()
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
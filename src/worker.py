"""Single-thread model worker.

Loads ONE backend at a time (resource constraint) — either MegaLoc (VPR,
raw torch) or Jina (SentenceTransformer). Jobs and queries share that thread.

MegaLoc path:
  - image-only descriptors, L2-normalized, dim 8448
  - geometry via ``src.preprocess`` (squash / letterbox / center_crop)
  - ImageNet normalize + torch tensor on the configured device

Jina path (kept, activated with MODEL_BACKEND=jina):
  - fused text+image via encode_document / encode_query
  - prepare_image downscale (max_pixels)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import torch
from PIL import Image

from src.config import get_settings
from src.preprocess import TransformMode, open_image, transform_image

logger = logging.getLogger(__name__)

# ImageNet stats used by MegaLoc eval preprocessing.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Shared job / result types
# ---------------------------------------------------------------------------


@dataclass
class ModelJob:
    """Queued record embed (Jina or VPR — image-only when backend is megaloc)."""

    record_id: str
    text: str | None
    # Prepared PIL image, raw bytes, or a source the Jina model loads.
    image: Image.Image | bytes | str | None
    # Record `version` sampled at dispatch time; written back as
    # embedding_queue.version_snapshot on successful persist (both backends
    # when the job is marker_views-backed).
    version: Any = None
    # Optional per-job VPR geometry override (else settings.vpr_transform).
    transform: str | None = None


@dataclass
class ModelResult:
    record_id: str
    embedding: list[float] | None
    error: str | None = None
    version: Any = None


@dataclass
class QueryJob:
    """Synchronous on-demand encode (encode_query / encode_vpr)."""

    text: str | None
    image: Image.Image | bytes | str | None
    normalize: bool = False
    # VPR geometry override for this call.
    transform: str | None = None
    # Filled under Condition once the worker finishes this job.
    result: list[float] | None = None
    error: str | None = None
    done: threading.Condition | None = None


class ModelNotReadyError(RuntimeError):
    pass


class QueryError(RuntimeError):
    pass


class BackendMismatchError(RuntimeError):
    """Raised when a caller asks for a backend that is not the loaded one."""


# ---------------------------------------------------------------------------
# Image helpers (Jina)
# ---------------------------------------------------------------------------


def prepare_image(data: bytes, max_pixels: int) -> Image.Image:
    """Decode bytes → RGB PIL and downscale if ``w*h > max_pixels``.

    Aspect ratio kept; never upscales. Used by the Jina path.
    """
    if max_pixels < 1:
        raise ValueError(f"max_pixels must be >= 1, got {max_pixels}")
    with Image.open(BytesIO(data)) as im:
        img = im.convert("RGB")
    w, h = img.size
    pixels = w * h
    if pixels <= max_pixels:
        return img
    scale = (max_pixels / pixels) ** 0.5
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.BICUBIC)


def _fused_input(
    text: str | None, image: Image.Image | bytes | str | None
) -> str | Image.Image | bytes | tuple | None:
    """Build the single input SentenceTransformer encode_* expects."""
    has_t = bool(text)
    has_i = image is not None
    if has_t and has_i:
        return (text, image)
    if has_t:
        return text
    if has_i:
        return image
    return None


def _to_list(emb: Any) -> list[float]:
    """Coerce a model output row into a plain Python float list."""
    if emb is None:
        return []
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().float().cpu().reshape(-1)
        return emb.tolist()
    if hasattr(emb, "tolist"):
        out = emb.tolist()
        # Nested row from encode_* batch dim.
        if out and isinstance(out[0], list):
            return list(out[0])
        return list(out)
    return list(emb)


# ---------------------------------------------------------------------------
# MegaLoc preprocessing + inference
# ---------------------------------------------------------------------------


def image_to_megaloc_tensor(
    image: Image.Image | bytes,
    *,
    size: int = 322,
    mode: TransformMode | str = TransformMode.SQUASH,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """PIL/bytes → ``[1, 3, size, size]`` ImageNet-normalized float tensor.

    Geometry is applied first (``src.preprocess``), then ToTensor + Normalize
    matching the MegaLoc evaluation recipe.
    """
    import torchvision.transforms.functional as TF

    img = open_image(image)
    img = transform_image(img, size, mode)
    t = TF.to_tensor(img)  # [3,H,W] float 0..1
    t = TF.normalize(t, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
    return t.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class ModelWorker:
    """Background thread that owns the model and runs one job at a time."""

    def __init__(self, state):
        self.state = state
        self.settings = get_settings()
        self.backend = (self.settings.model_backend or "megaloc").strip().lower()
        self.jobs: queue.Queue[ModelJob] = queue.Queue()
        self.results: queue.Queue[ModelResult] = queue.Queue()
        self._query_jobs: queue.Queue[QueryJob] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.model = None
        self.device = self.settings.device

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="model-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # Unblock queues so the loop can exit.
        try:
            self.jobs.put_nowait(
                ModelJob(record_id="__stop__", text=None, image=None)
            )
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            self.model = self._load_model()
            self.state.set_model_state("ready")
            logger.info(
                "Model ready backend=%s device=%s",
                self.backend,
                self.device,
            )
        except Exception as e:
            logger.exception("Model load failed")
            self.state.set_model_state("error", str(e))
            return

        while not self._stop.is_set():
            # Prefer synchronous query jobs (latency-sensitive encode endpoints).
            try:
                qj = self._query_jobs.get_nowait()
            except queue.Empty:
                qj = None
            if qj is not None:
                self._handle_query(qj)
                continue

            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job.record_id == "__stop__":
                break
            self._handle_job(job)

    # -- loading -------------------------------------------------------------

    def _load_model(self):
        backend = self.backend
        if backend in ("megaloc", "vpr"):
            return self._load_megaloc()
        if backend in ("jina", "sentence-transformers", "st"):
            return self._load_jina()
        raise ValueError(
            f"unknown model_backend={backend!r}; expected megaloc|jina"
        )

    def _load_megaloc(self):
        """Load MegaLoc via torch.hub (raw torch, not SentenceTransformer)."""
        s = self.settings
        device = s.device
        logger.info(
            "Loading MegaLoc repo=%s entry=%s device=%s transform=%s size=%d",
            s.megaloc_repo,
            s.megaloc_entry,
            device,
            s.vpr_transform,
            s.vpr_image_size,
        )
        t0 = time.perf_counter()
        # trust_repo / skip_validation keep torch.hub quiet on recent torch.
        try:
            model = torch.hub.load(
                s.megaloc_repo,
                s.megaloc_entry,
                trust_repo=True,
                skip_validation=True,
            )
        except TypeError:
            # Older torch.hub without trust_repo.
            model = torch.hub.load(s.megaloc_repo, s.megaloc_entry)
        model = model.to(device)
        model.eval()
        logger.info(
            "MegaLoc loaded on %s (%.1fs)",
            device,
            time.perf_counter() - t0,
        )
        return model

    def _load_jina(self):
        """Load the Jina SentenceTransformer (kept for MODEL_BACKEND=jina)."""
        from sentence_transformers import SentenceTransformer

        s = self.settings
        logger.info(
            "Loading Jina model=%s device=%s modalities=%s",
            s.model,
            s.device,
            s.modalities,
        )
        t0 = time.perf_counter()
        model = SentenceTransformer(
            s.model,
            trust_remote_code=True,
            device=s.device,
            model_kwargs={"modality": s.modalities},
        )
        logger.info(
            "Jina loaded on %s (%.1fs)",
            s.device,
            time.perf_counter() - t0,
        )
        return model

    # -- encode helpers ------------------------------------------------------

    def _require_ready(self) -> None:
        if not self.state.model_ready() or self.model is None:
            err = self.state.model_error or "model not ready"
            raise ModelNotReadyError(err)

    def _require_backend(self, *allowed: str) -> None:
        if self.backend not in allowed and not (
            self.backend == "vpr" and "megaloc" in allowed
        ):
            raise BackendMismatchError(
                f"backend is {self.backend!r}; this call needs one of {allowed}"
            )

    def _vpr_encode_image(
        self,
        image: Image.Image | bytes,
        *,
        transform: str | None = None,
    ) -> list[float]:
        """Forward one image through MegaLoc → L2-normalized float list."""
        s = self.settings
        mode = transform or s.vpr_transform
        tensor = image_to_megaloc_tensor(
            image,
            size=s.vpr_image_size,
            mode=mode,
            device=self.device,
        )
        with torch.inference_mode():
            desc = self.model(tensor)
        return _to_list(desc)

    def _jina_encode(
        self,
        text: str | None,
        image: Image.Image | bytes | str | None,
        *,
        query: bool,
        normalize: bool = False,
    ) -> list[float]:
        value = _fused_input(text, image)
        if value is None:
            raise QueryError("no text or image to encode")
        if query:
            emb = self.model.encode_query(value, normalize_embeddings=normalize)
        else:
            emb = self.model.encode_document(value, normalize_embeddings=normalize)
        return _to_list(emb)

    # -- job handlers --------------------------------------------------------

    def _handle_job(self, job: ModelJob) -> None:
        rid = job.record_id
        t0 = time.perf_counter()
        try:
            if self.backend in ("megaloc", "vpr"):
                if job.image is None:
                    raise QueryError("VPR job requires an image")
                if isinstance(job.image, str):
                    raise QueryError(
                        "VPR job image must be PIL or bytes (got path/url string)"
                    )
                emb = self._vpr_encode_image(job.image, transform=job.transform)
            else:
                emb = self._jina_encode(
                    job.text, job.image, query=False, normalize=False
                )
            self.results.put(
                ModelResult(
                    record_id=rid,
                    embedding=emb,
                    version=job.version,
                )
            )
            logger.info(
                "Job done %s dim=%d backend=%s (%.0fms)",
                rid,
                len(emb),
                self.backend,
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            logger.warning("Job failed %s: %s", rid, e)
            self.results.put(
                ModelResult(
                    record_id=rid,
                    embedding=None,
                    error=str(e),
                    version=job.version,
                )
            )

    def _handle_query(self, job: QueryJob) -> None:
        t0 = time.perf_counter()
        try:
            if self.backend in ("megaloc", "vpr"):
                if job.image is None:
                    raise QueryError("VPR encode requires an image")
                if isinstance(job.image, str):
                    raise QueryError(
                        "VPR encode image must be PIL or bytes (got path/url string)"
                    )
                emb = self._vpr_encode_image(job.image, transform=job.transform)
            else:
                emb = self._jina_encode(
                    job.text,
                    job.image,
                    query=True,
                    normalize=job.normalize,
                )
            job.result = emb
            logger.info(
                "Query done dim=%d backend=%s (%.0fms)",
                len(emb),
                self.backend,
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            job.error = str(e)
            logger.warning("Query failed: %s", e)
        finally:
            if job.done is not None:
                with job.done:
                    job.done.notify_all()

    # -- public API ----------------------------------------------------------

    def encode_query(
        self,
        text: str | None,
        image: Image.Image | bytes | str | None,
        timeout: float = 0.0,
        *,
        normalize: bool = False,
        transform: str | None = None,
    ) -> list[float]:
        """Block until the worker encodes a query. Raises on failure/timeout."""
        self._require_ready()
        cond = threading.Condition()
        job = QueryJob(
            text=text,
            image=image,
            normalize=normalize,
            transform=transform,
            done=cond,
        )
        with cond:
            self._query_jobs.put(job)
            if timeout and timeout > 0:
                ok = cond.wait(timeout=timeout)
                if not ok and job.result is None and job.error is None:
                    raise TimeoutError(f"query encode timed out after {timeout}s")
            else:
                while job.result is None and job.error is None:
                    cond.wait(timeout=0.5)
                    if self._stop.is_set():
                        raise QueryError("worker stopped")
        if job.error:
            raise QueryError(job.error)
        if job.result is None:
            raise QueryError("empty query result")
        return job.result

    def encode_vpr(
        self,
        image: Image.Image | bytes,
        timeout: float = 0.0,
        *,
        transform: str | None = None,
    ) -> list[float]:
        """Image-only MegaLoc descriptor. Requires megaloc backend."""
        self._require_ready()
        self._require_backend("megaloc", "vpr")
        return self.encode_query(
            text=None,
            image=image,
            timeout=timeout,
            transform=transform,
        )

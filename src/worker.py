"""Single-thread MegaLoc (VPR) model worker.

One backend only — raw torch via torch.hub. Jobs and queries share the thread.

  - image-only descriptors, L2-normalized, dim 8448
  - geometry via ``src.preprocess`` (squash / letterbox / center_crop)
  - ImageNet normalize + torch tensor on the configured device
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
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
    """Queued VPR embed (image-only)."""

    record_id: str
    image: Image.Image | bytes
    # marker_views.image_hash sampled at dispatch; written back as
    # embedding_queue.hash_snapshot on successful persist (marker_views jobs).
    image_hash: str | None = None
    # Optional per-job VPR geometry override (else settings.vpr_transform).
    transform: str | None = None


@dataclass
class ModelResult:
    record_id: str
    embedding: list[float] | None
    error: str | None = None
    image_hash: str | None = None


@dataclass
class QueryJob:
    """Synchronous on-demand encode (encode_vpr)."""

    image: Image.Image | bytes
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_list(emb: Any) -> list[float]:
    """Coerce a model output row into a plain Python float list."""
    if emb is None:
        return []
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().float().cpu().reshape(-1)
        return emb.tolist()
    if hasattr(emb, "tolist"):
        out = emb.tolist()
        # Nested row from batch dim.
        if out and isinstance(out[0], list):
            return list(out[0])
        return list(out)
    return list(emb)


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
    """Background thread that owns MegaLoc and runs one job at a time."""

    def __init__(self, state):
        self.state = state
        self.settings = get_settings()
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
                ModelJob(record_id="__stop__", image=b"")
            )
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            self.model = self._load_megaloc()
            self.state.set_model_state("ready")
            logger.info("Model ready backend=megaloc device=%s", self.device)
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

    def _load_megaloc(self):
        """Load MegaLoc via torch.hub (raw torch)."""
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

    # -- encode helpers ------------------------------------------------------

    def _require_ready(self) -> None:
        if not self.state.model_ready() or self.model is None:
            err = self.state.model_error or "model not ready"
            raise ModelNotReadyError(err)

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

    # -- job handlers --------------------------------------------------------

    def _handle_job(self, job: ModelJob) -> None:
        rid = job.record_id
        t0 = time.perf_counter()
        try:
            if job.image is None:
                raise QueryError("VPR job requires an image")
            if isinstance(job.image, str):
                raise QueryError(
                    "VPR job image must be PIL or bytes (got path/url string)"
                )
            emb = self._vpr_encode_image(job.image, transform=job.transform)
            self.results.put(
                ModelResult(
                    record_id=rid,
                    embedding=emb,
                    image_hash=job.image_hash,
                )
            )
            logger.info(
                "Job done %s dim=%d (%.0fms)",
                rid,
                len(emb),
                (time.perf_counter() - t0) * 1000,
            )
        except Exception as e:
            logger.warning("Job failed %s: %s", rid, e)
            self.results.put(
                ModelResult(
                    record_id=rid,
                    embedding=None,
                    error=str(e),
                    image_hash=job.image_hash,
                )
            )

    def _handle_query(self, job: QueryJob) -> None:
        t0 = time.perf_counter()
        try:
            if job.image is None:
                raise QueryError("VPR encode requires an image")
            if isinstance(job.image, str):
                raise QueryError(
                    "VPR encode image must be PIL or bytes (got path/url string)"
                )
            emb = self._vpr_encode_image(job.image, transform=job.transform)
            job.result = emb
            logger.info(
                "Query done dim=%d (%.0fms)",
                len(emb),
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

    def encode_vpr(
        self,
        image: Image.Image | bytes,
        timeout: float = 0.0,
        *,
        transform: str | None = None,
    ) -> list[float]:
        """Block until the worker encodes an image. Raises on failure/timeout."""
        self._require_ready()
        cond = threading.Condition()
        job = QueryJob(
            image=image,
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

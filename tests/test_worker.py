"""Synthetic tests for the model worker's query path and the /encode endpoint.

No DB and no real model: the worker's `_load_model` is stubbed with a fake
model, and `get_settings` is patched so instantiating `ModelWorker` needs no
`.env` / SurrealDB config. The `/encode` endpoint is exercised with a patched
module-level worker.
"""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

import src.main as main_module
from src.state import ServiceState
from src.worker import (
    ModelJob,
    ModelNotReadyError,
    ModelWorker,
    QueryError,
    _fused_input,
    _to_list,
)


def _jina_settings(**overrides):
    base = dict(
        model_backend="jina",
        model="fake",
        modalities="vision",
        device="cpu",
        megaloc_repo="gmberton/MegaLoc",
        megaloc_entry="get_trained_model",
        vpr_image_size=322,
        vpr_transform="squash",
        query_timeout=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _megaloc_settings(**overrides):
    return _jina_settings(model_backend="megaloc", **overrides)


class FakeModel:
    """Deterministic stand-in for the SentenceTransformer model."""

    def __init__(self, emb=None, error=None):
        self.emb = emb if emb is not None else [3.0, 4.0]
        self.error = error
        self.calls = []

    def _encode(self, value, normalize_embeddings=False):
        self.calls.append(value)
        if self.error:
            raise RuntimeError(self.error)
        emb = list(self.emb)
        if normalize_embeddings:
            n = sum(x * x for x in emb) ** 0.5 or 1.0
            emb = [x / n for x in emb]
        return emb

    def encode_query(self, value, normalize_embeddings=False):
        return self._encode(value, normalize_embeddings=normalize_embeddings)

    def encode_document(self, value, normalize_embeddings=False):
        return self._encode(value, normalize_embeddings=normalize_embeddings)


class FakeMegaLoc:
    """Stand-in nn.Module-like: callable on a batch tensor → descriptor."""

    def __init__(self, emb=None, error=None):
        self.emb = emb if emb is not None else [0.0] * 8
        self.emb[0] = 1.0
        self.error = error
        self.calls = 0

    def __call__(self, tensor):
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        import torch

        return torch.tensor([self.emb], dtype=torch.float32)

    def eval(self):
        return self

    def to(self, device):
        return self


def _make_worker(model, *, settings=None):
    settings = settings or _jina_settings()
    with mock.patch("src.worker.get_settings", return_value=settings):
        state = ServiceState()
        worker = ModelWorker(state)
    worker._load_model = lambda: model
    worker.start()
    _wait_ready(state)
    return worker, state


def _wait_ready(state, timeout=5.0):
    deadline = time.time() + timeout
    while not state.model_ready() and time.time() < deadline:
        time.sleep(0.01)
    assert state.model_ready(), f"worker model never became ready: {state.model_error}"


class FusedInputTests(unittest.TestCase):
    def test_both_modalities_become_one_tuple(self):
        self.assertEqual(_fused_input("text", "image"), ("text", "image"))

    def test_single_modality_passes_through(self):
        self.assertEqual(_fused_input("text", None), "text")
        self.assertEqual(_fused_input(None, "image"), "image")

    def test_no_input(self):
        self.assertIsNone(_fused_input(None, None))


class ToListTests(unittest.TestCase):
    def test_tensor_row(self):
        import torch

        self.assertEqual(_to_list(torch.tensor([[1.0, 2.0]])), [1.0, 2.0])

    def test_nested_list(self):
        class Arr:
            def tolist(self):
                return [[1.5, 2.5]]

        self.assertEqual(_to_list(Arr()), [1.5, 2.5])


class WorkerQueryTests(unittest.TestCase):
    def test_text_only(self):
        model = FakeModel()
        worker, _ = _make_worker(model)
        try:
            emb = worker.encode_query("text", None, timeout=2.0)
            self.assertEqual(emb, [3.0, 4.0])
            self.assertEqual(model.calls, ["text"])
        finally:
            worker.stop()

    def test_image_only(self):
        model = FakeModel()
        worker, _ = _make_worker(model)
        try:
            emb = worker.encode_query(None, "image", timeout=2.0)
            self.assertEqual(emb, [3.0, 4.0])
            self.assertEqual(model.calls, ["image"])
        finally:
            worker.stop()

    def test_fused_tuple_is_one_encode_call(self):
        model = FakeModel()
        worker, _ = _make_worker(model)
        try:
            emb = worker.encode_query("text", "image", timeout=2.0)
            self.assertEqual(emb, [3.0, 4.0])
            self.assertEqual(model.calls, [("text", "image")])
        finally:
            worker.stop()

    def test_normalize(self):
        model = FakeModel()
        worker, _ = _make_worker(model)
        try:
            emb = worker.encode_query("text", None, timeout=2.0, normalize=True)
            self.assertAlmostEqual(emb[0], 0.6)
            self.assertAlmostEqual(emb[1], 0.8)
        finally:
            worker.stop()

    def test_empty_query_raises(self):
        model = FakeModel()
        worker, _ = _make_worker(model)
        try:
            with self.assertRaises(QueryError):
                worker.encode_query(None, None, timeout=2.0)
            self.assertEqual(model.calls, [])
        finally:
            worker.stop()

    def test_model_not_ready(self):
        with mock.patch("src.worker.get_settings", return_value=_jina_settings()):
            state = ServiceState()
            state.set_model_state("loading")
            worker = ModelWorker(state)
        with self.assertRaises(ModelNotReadyError):
            worker.encode_query("text", None, timeout=1.0)

    def test_timeout_when_no_worker(self):
        with mock.patch("src.worker.get_settings", return_value=_jina_settings()):
            state = ServiceState()
            state.set_model_state("ready")
            worker = ModelWorker(state)
            worker.model = FakeModel()
        with self.assertRaises(TimeoutError):
            worker.encode_query("text", None, timeout=0.05)

    def test_encode_error(self):
        worker, _ = _make_worker(FakeModel(error="boom"))
        try:
            with self.assertRaises(QueryError):
                worker.encode_query("text", None, timeout=2.0)
        finally:
            worker.stop()

    def test_document_is_one_fused_embedding(self):
        model = FakeModel()
        worker, _ = _make_worker(model)
        try:
            worker.jobs.put(
                ModelJob(record_id="marker_views:1", text="text", image="image")
            )
            result = worker.results.get(timeout=2.0)
            self.assertIsNone(result.error)
            self.assertEqual(result.embedding, [3.0, 4.0])
            self.assertEqual(model.calls, [("text", "image")])
        finally:
            worker.stop()

    def test_document_without_image_uses_text_alone(self):
        model = FakeModel()
        worker, _ = _make_worker(model)
        try:
            worker.jobs.put(
                ModelJob(record_id="marker_views:2", text="text", image=None)
            )
            result = worker.results.get(timeout=2.0)
            self.assertIsNone(result.error)
            self.assertEqual(result.embedding, [3.0, 4.0])
            self.assertEqual(model.calls, ["text"])
        finally:
            worker.stop()

    def test_document_encode_failure_sets_error(self):
        worker, _ = _make_worker(FakeModel(error="boom"))
        try:
            worker.jobs.put(
                ModelJob(record_id="marker_views:3", text="text", image=None)
            )
            result = worker.results.get(timeout=2.0)
            self.assertIsNotNone(result.error)
            self.assertIn("boom", result.error)
            self.assertIsNone(result.embedding)
        finally:
            worker.stop()

    def test_stop_terminates_thread(self):
        worker, _ = _make_worker(FakeModel())
        thread = worker._thread
        self.assertTrue(thread.is_alive())
        worker.stop()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())


class VprWorkerTests(unittest.TestCase):
    def test_encode_vpr_image(self):
        model = FakeMegaLoc(emb=[1.0, 0.0, 0.0, 0.0])
        worker, _ = _make_worker(model, settings=_megaloc_settings())
        try:
            img = Image.new("RGB", (64, 48), (10, 20, 30))
            emb = worker.encode_vpr(img, timeout=5.0, transform="squash")
            self.assertEqual(len(emb), 4)
            self.assertEqual(model.calls, 1)
        finally:
            worker.stop()

    def test_vpr_job_requires_image(self):
        worker, _ = _make_worker(FakeMegaLoc(), settings=_megaloc_settings())
        try:
            worker.jobs.put(ModelJob(record_id="t1", text=None, image=None))
            result = worker.results.get(timeout=2.0)
            self.assertIsNotNone(result.error)
            self.assertIn("image", result.error.lower())
        finally:
            worker.stop()


class EncodeEndpointTests(unittest.TestCase):
    def test_root_encode_uses_jina_when_configured(self):
        fake_emb = [0.1, 0.2, 0.3]
        with mock.patch.object(main_module, "_is_jina", return_value=True), mock.patch.object(
            main_module, "_is_megaloc", return_value=False
        ), mock.patch.object(main_module, "worker") as fake_worker:
            fake_worker.encode_query.return_value = fake_emb
            client = TestClient(main_module.app, raise_server_exceptions=False)
            resp = client.post("/encode", json={"text": "hello", "normalize": True})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.json(),
            {"embedding": fake_emb, "dim": 3, "backend": "jina"},
        )

    def test_jina_alias_encode(self):
        fake_emb = [0.4, 0.5]
        with mock.patch.object(main_module, "_is_jina", return_value=True), mock.patch.object(
            main_module, "_is_megaloc", return_value=False
        ), mock.patch.object(main_module, "worker") as fake_worker:
            fake_worker.encode_query.return_value = fake_emb
            client = TestClient(main_module.app, raise_server_exceptions=False)
            resp = client.post("/jina/encode", json={"text": "hi"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["backend"], "jina")
        self.assertEqual(resp.json()["embedding"], fake_emb)

    def test_jina_encode_requires_input(self):
        with mock.patch.object(main_module, "_is_jina", return_value=True), mock.patch.object(
            main_module, "_is_megaloc", return_value=False
        ):
            client = TestClient(main_module.app, raise_server_exceptions=False)
            resp = client.post("/encode", json={})
        self.assertEqual(resp.status_code, 400)

    def test_jina_alias_rejected_on_megaloc_backend(self):
        with mock.patch.object(main_module, "_is_jina", return_value=False), mock.patch.object(
            main_module, "_backend", return_value="megaloc"
        ):
            client = TestClient(main_module.app, raise_server_exceptions=False)
            resp = client.post("/jina/encode", json={"text": "hello"})
        self.assertEqual(resp.status_code, 503)

    def test_vpr_encode_rejected_on_jina_backend(self):
        with mock.patch.object(main_module, "_is_megaloc", return_value=False), mock.patch.object(
            main_module, "_backend", return_value="jina"
        ):
            client = TestClient(main_module.app, raise_server_exceptions=False)
            resp = client.post("/vpr/encode", json={"image_url": "data:,"})
        self.assertEqual(resp.status_code, 503)


if __name__ == "__main__":
    unittest.main()

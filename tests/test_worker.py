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

import src.main as main_module
from src.state import ServiceState
from src.worker import (
    ModelJob,
    ModelNotReadyError,
    ModelWorker,
    QueryError,
    _fused_input,
)


class FakeModel:
    """Deterministic stand-in for the SentenceTransformer model."""

    def __init__(self, emb=None, error=None):
        self.emb = emb if emb is not None else [3.0, 4.0]
        self.error = error
        self.calls = []

    def _encode(self, value):
        self.calls.append(value)
        if self.error:
            raise RuntimeError(self.error)
        return self.emb

    def encode_query(self, value):
        return self._encode(value)

    def encode_document(self, value):
        return self._encode(value)


def _make_worker(model):
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
    assert state.model_ready(), "worker model never became ready"


class FusedInputTests(unittest.TestCase):
    def test_both_modalities_become_one_tuple(self):
        self.assertEqual(_fused_input("text", "image"), ("text", "image"))

    def test_single_modality_passes_through(self):
        self.assertEqual(_fused_input("text", None), "text")
        self.assertEqual(_fused_input(None, "image"), "image")

    def test_no_input(self):
        self.assertIsNone(_fused_input(None, None))


class WorkerQueryTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "src.worker.get_settings",
            return_value=SimpleNamespace(model="fake", modalities="vision"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_text_only(self):
        worker, _ = _make_worker(FakeModel())
        emb = worker.encode_query("text", None, timeout=2.0)
        self.assertEqual(emb, [3.0, 4.0])
        self.assertEqual(worker._model.calls, ["text"])

    def test_image_only(self):
        worker, _ = _make_worker(FakeModel())
        emb = worker.encode_query(None, "image", timeout=2.0)
        self.assertEqual(emb, [3.0, 4.0])
        self.assertEqual(worker._model.calls, ["image"])

    def test_fused_tuple_is_one_encode_call(self):
        worker, _ = _make_worker(FakeModel())
        emb = worker.encode_query("text", "image", timeout=2.0)
        self.assertEqual(emb, [3.0, 4.0])
        self.assertEqual(worker._model.calls, [("text", "image")])

    def test_normalize(self):
        worker, _ = _make_worker(FakeModel())
        emb = worker.encode_query("text", None, timeout=2.0, normalize=True)
        # L2 norm of [3, 4] is 5 -> [0.6, 0.8]
        self.assertEqual(emb, [0.6, 0.8])

    def test_empty_query_returns_none(self):
        worker, _ = _make_worker(FakeModel())
        self.assertIsNone(worker.encode_query(None, None, timeout=2.0))
        self.assertEqual(worker._model.calls, [])

    def test_model_not_ready(self):
        state = ServiceState()
        state.set_model_state("loading")
        worker = ModelWorker(state)  # thread not started
        with self.assertRaises(ModelNotReadyError):
            worker.encode_query("text", None, timeout=1.0)

    def test_timeout_when_no_worker(self):
        state = ServiceState()
        state.set_model_state("ready")
        worker = ModelWorker(state)  # no thread consuming the query queue
        with self.assertRaises(TimeoutError):
            worker.encode_query("text", None, timeout=0.05)

    def test_encode_error(self):
        worker, _ = _make_worker(FakeModel(error="boom"))
        with self.assertRaises(QueryError):
            worker.encode_query("text", None, timeout=2.0)

    def test_document_is_one_fused_embedding(self):
        worker, _ = _make_worker(FakeModel())
        worker.jobs.put(ModelJob(record_id="marker_views:1", text="text", image="image"))
        result = worker.results.get(timeout=2.0)
        self.assertIsNone(result.error)
        self.assertEqual(result.embedding, [3.0, 4.0])
        self.assertEqual(worker._model.calls, [("text", "image")])

    def test_document_without_image_uses_text_alone(self):
        worker, _ = _make_worker(FakeModel())
        worker.jobs.put(ModelJob(record_id="marker_views:2", text="text", image=None))
        result = worker.results.get(timeout=2.0)
        self.assertIsNone(result.error)
        self.assertEqual(result.embedding, [3.0, 4.0])
        self.assertEqual(worker._model.calls, ["text"])

    def test_document_encode_failure_sets_error(self):
        worker, _ = _make_worker(FakeModel(error="boom"))
        worker.jobs.put(ModelJob(record_id="marker_views:3", text="text", image=None))
        result = worker.results.get(timeout=2.0)
        self.assertIsNotNone(result.error)
        self.assertIn("encode failed", result.error)
        self.assertIsNone(result.embedding)

    def test_stop_terminates_thread(self):
        worker, _ = _make_worker(FakeModel())
        thread = worker._thread
        self.assertTrue(thread.is_alive())
        worker.stop()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())

    def test_encode_query_after_stop(self):
        worker, _ = _make_worker(FakeModel())
        worker.stop()
        worker._thread.join(timeout=2.0)
        with self.assertRaises(ModelNotReadyError):
            worker.encode_query("text", None, timeout=1.0)


class EncodeEndpointTests(unittest.TestCase):
    def test_returns_embedding(self):
        fake_emb = [0.1, 0.2, 0.3]
        with mock.patch.object(main_module, "worker") as fake_worker:
            fake_worker.encode_query.return_value = fake_emb
            client = TestClient(main_module.app)
            resp = client.post("/encode", json={"text": "hello", "normalize": True})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"embedding": fake_emb})

    def test_requires_input(self):
        client = TestClient(main_module.app)
        resp = client.post("/encode", json={})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()

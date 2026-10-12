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
from src.worker import ModelNotReadyError, ModelWorker, QueryError


class FakeModel:
    """Deterministic stand-in for the SentenceTransformer model."""

    def __init__(self, text_emb=None, image_emb=None, error=None):
        self.text_emb = text_emb if text_emb is not None else [3.0, 4.0]
        self.image_emb = image_emb if image_emb is not None else [6.0, 8.0]
        self.error = error
        self.calls = []

    def encode_query(self, value):
        self.calls.append(value)
        if self.error:
            raise RuntimeError(self.error)
        if value == "text":
            return self.text_emb
        if value == "image":
            return self.image_emb
        return [1.0]


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
        text_emb, image_emb = worker.encode_query("text", None, timeout=2.0)
        self.assertEqual(text_emb, [3.0, 4.0])
        self.assertIsNone(image_emb)

    def test_image_only(self):
        worker, _ = _make_worker(FakeModel())
        text_emb, image_emb = worker.encode_query(None, "image", timeout=2.0)
        self.assertIsNone(text_emb)
        self.assertEqual(image_emb, [6.0, 8.0])

    def test_both_modalities(self):
        worker, _ = _make_worker(FakeModel())
        text_emb, image_emb = worker.encode_query("text", "image", timeout=2.0)
        self.assertEqual(text_emb, [3.0, 4.0])
        self.assertEqual(image_emb, [6.0, 8.0])

    def test_normalize(self):
        worker, _ = _make_worker(FakeModel())
        text_emb, _ = worker.encode_query("text", None, timeout=2.0, normalize=True)
        # L2 norm of [3, 4] is 5 -> [0.6, 0.8]
        self.assertEqual(text_emb, [0.6, 0.8])

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
    def test_returns_embeddings(self):
        fake_text = [0.1, 0.2, 0.3]
        with mock.patch.object(main_module, "worker") as fake_worker:
            fake_worker.encode_query.return_value = (fake_text, None)
            client = TestClient(main_module.app)
            resp = client.post("/encode", json={"text": "hello", "normalize": True})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"text_embedding": fake_text})

    def test_requires_input(self):
        client = TestClient(main_module.app)
        resp = client.post("/encode", json={})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
"""Synthetic tests for the model worker's query path and the /encode endpoint.

No DB and no real model: the worker's `_load_megaloc` is stubbed with a fake
model, and `get_settings` is patched so instantiating `ModelWorker` needs no
`.env` / SurrealDB config.
"""

import time
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient
from PIL import Image

import src.main as main_module
from src.state import ServiceState
from src.worker import ModelJob, ModelNotReadyError, ModelWorker, QueryError, _to_list


def _settings(**overrides):
    base = dict(
        megaloc_repo="gmberton/MegaLoc",
        megaloc_entry="get_trained_model",
        megaloc_model_label="gmberton/MegaLoc",
        device="cpu",
        vpr_image_size=322,
        vpr_transform="squash",
        query_timeout=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


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
    settings = settings or _settings()
    with mock.patch("src.worker.get_settings", return_value=settings):
        state = ServiceState()
        worker = ModelWorker(state)
    worker._load_megaloc = lambda: model
    worker.start()
    _wait_ready(state)
    return worker, state


def _wait_ready(state, timeout=5.0):
    deadline = time.time() + timeout
    while not state.model_ready() and time.time() < deadline:
        time.sleep(0.01)
    assert state.model_ready(), f"worker model never became ready: {state.model_error}"


class ToListTests(unittest.TestCase):
    def test_tensor_row(self):
        import torch

        self.assertEqual(_to_list(torch.tensor([[1.0, 2.0]])), [1.0, 2.0])

    def test_nested_list(self):
        class Arr:
            def tolist(self):
                return [[1.5, 2.5]]

        self.assertEqual(_to_list(Arr()), [1.5, 2.5])

    def test_none_is_empty(self):
        self.assertEqual(_to_list(None), [])


class VprWorkerTests(unittest.TestCase):
    def test_encode_vpr_image(self):
        model = FakeMegaLoc(emb=[1.0, 0.0, 0.0, 0.0])
        worker, _ = _make_worker(model)
        try:
            img = Image.new("RGB", (64, 48), (10, 20, 30))
            emb = worker.encode_vpr(img, timeout=5.0, transform="squash")
            self.assertEqual(len(emb), 4)
            self.assertEqual(model.calls, 1)
        finally:
            worker.stop()

    def test_vpr_job_requires_image_bytes_or_pil(self):
        worker, _ = _make_worker(FakeMegaLoc())
        try:
            # ModelJob.image is required at type level; pass empty bytes and
            # ensure encode still runs (empty bytes fail open_image).
            worker.jobs.put(
                ModelJob(record_id="t1", image=b"", image_hash="abc")
            )
            result = worker.results.get(timeout=2.0)
            self.assertIsNotNone(result.error)
        finally:
            worker.stop()

    def test_job_success_keeps_image_hash(self):
        model = FakeMegaLoc(emb=[1.0, 0.0])
        worker, _ = _make_worker(model)
        try:
            img = Image.new("RGB", (32, 32), (1, 2, 3))
            worker.jobs.put(
                ModelJob(record_id="marker_views:1", image=img, image_hash="deadbeef")
            )
            result = worker.results.get(timeout=5.0)
            self.assertIsNone(result.error)
            self.assertEqual(result.image_hash, "deadbeef")
            self.assertEqual(len(result.embedding or []), 2)
        finally:
            worker.stop()

    def test_model_not_ready(self):
        with mock.patch("src.worker.get_settings", return_value=_settings()):
            state = ServiceState()
            state.set_model_state("loading")
            worker = ModelWorker(state)
        img = Image.new("RGB", (8, 8), (0, 0, 0))
        with self.assertRaises(ModelNotReadyError):
            worker.encode_vpr(img, timeout=1.0)

    def test_timeout_when_no_worker(self):
        with mock.patch("src.worker.get_settings", return_value=_settings()):
            state = ServiceState()
            state.set_model_state("ready")
            worker = ModelWorker(state)
            worker.model = FakeMegaLoc()
        img = Image.new("RGB", (8, 8), (0, 0, 0))
        with self.assertRaises(TimeoutError):
            worker.encode_vpr(img, timeout=0.05)

    def test_encode_error(self):
        worker, _ = _make_worker(FakeMegaLoc(error="boom"))
        try:
            img = Image.new("RGB", (16, 16), (5, 5, 5))
            # First encode pays torchvision/preprocess import cost; keep headroom.
            with self.assertRaises(QueryError):
                worker.encode_vpr(img, timeout=30.0)
        finally:
            worker.stop()

    def test_stop_terminates_thread(self):
        worker, _ = _make_worker(FakeMegaLoc())
        thread = worker._thread
        self.assertTrue(thread.is_alive())
        worker.stop()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())


class EncodeEndpointTests(unittest.TestCase):
    def test_root_encode_requires_image(self):
        client = TestClient(main_module.app, raise_server_exceptions=False)
        resp = client.post("/encode", json={})
        self.assertEqual(resp.status_code, 400)

    def test_vpr_encode_alias_requires_image(self):
        client = TestClient(main_module.app, raise_server_exceptions=False)
        resp = client.post("/vpr/encode", json={"image_url": ""})
        self.assertEqual(resp.status_code, 400)

    def test_encode_with_mocked_worker(self):
        fake_emb = [0.1, 0.2, 0.3]
        png = Image.new("RGB", (4, 4), (1, 2, 3))
        import io
        import base64

        buf = io.BytesIO()
        png.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        with mock.patch.object(main_module, "worker") as fake_worker, mock.patch.object(
            main_module, "_load_vpr_image", new=mock.AsyncMock(return_value=buf.getvalue())
        ):
            fake_worker.encode_vpr.return_value = fake_emb
            # Need app.state.conn for _encode_vpr path
            main_module.app.state.conn = mock.Mock()
            client = TestClient(main_module.app, raise_server_exceptions=False)
            resp = client.post("/encode", json={"image_url": data_url})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["embedding"], fake_emb)
        self.assertEqual(body["dim"], 3)
        self.assertEqual(body["backend"], "megaloc")


if __name__ == "__main__":
    unittest.main()

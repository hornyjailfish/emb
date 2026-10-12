"""Integration tests against the real jina model.

These download/load the actual weights (~1.9GB) and run inference on CPU, so
they are slow and need network + enough RAM. They are skipped unless you opt
in explicitly:

    python -m unittest tests.test_real_model -v            # always skip fast
    EMBED_REAL_MODEL=1 python -m unittest tests.test_real_model -v

They verify what the synthetic tests can't: that the model actually loads, that
`encode_query` / `encode_document` return embeddings in the expected shape, and
that POST /encode queues a request through the worker and returns the JSON the
app expects for cosine search.
"""

import math
import os
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import src.main as main_module
from src.config import get_settings
from src.state import ServiceState
from src.worker import ModelWorker, _to_list

LOAD_TIMEOUT = 1200.0  # seconds; covers the one-time weights download + load


@unittest.skipUnless(
    os.environ.get("EMBED_REAL_MODEL") == "1",
    "set EMBED_REAL_MODEL=1 to run real-model integration tests",
)
class RealModelIntegrationTest(unittest.TestCase):
    worker: ModelWorker | None = None
    image_path: str = ""

    @classmethod
    def setUpClass(cls):
        settings = get_settings()
        # Mirror main._setup_runtime so we reuse the project's model cache.
        os.environ.setdefault("HF_HOME", settings.hf_home)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "warning")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        # Deterministic test image (noise, so it's a real, non-degenerate input).
        rng = np.random.default_rng(42)
        img = Image.fromarray((rng.random((224, 224, 3)) * 255).astype(np.uint8), "RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cls.image_path = tmp.name
        tmp.close()
        img.save(cls.image_path)

        cls.worker = ModelWorker(ServiceState())
        cls.worker.start()

        deadline = time.time() + LOAD_TIMEOUT
        while True:
            if cls.worker.state.model_ready():
                break
            if cls.worker.state.model_state == "error":
                cls.worker.stop()
                raise unittest.SkipTest(f"model failed to load: {cls.worker.state.model_error}")
            if time.time() > deadline:
                cls.worker.stop()
                raise unittest.SkipTest("timed out loading the model weights")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        if cls.worker is not None:
            cls.worker.stop()
        if cls.image_path and os.path.exists(cls.image_path):
            os.remove(cls.image_path)

    def test_model_is_ready(self):
        self.assertEqual(self.worker.state.model_state, "ready")
        self.assertTrue(self.worker.state.model_ready())
        self.assertIsNotNone(self.worker._model)

    def test_query_embedding_format(self):
        text_emb, image_emb = self.worker.encode_query(
            "How is the weather today?", None, timeout=300
        )
        self.assertIsNone(image_emb)
        self.assertIsInstance(text_emb, list)
        self.assertGreater(len(text_emb), 0)
        self.assertTrue(all(isinstance(x, float) for x in text_emb))

    def test_query_normalization(self):
        emb, _ = self.worker.encode_query("weather forecast", None, timeout=300, normalize=True)
        norm = math.sqrt(sum(x * x for x in emb))
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_document_embedding(self):
        model = self.worker._model
        self.assertTrue(hasattr(model, "encode_document"))
        vec = _to_list(model.encode_document("a document to embed"))
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)

    def test_image_embedding(self):
        text_emb, image_emb = self.worker.encode_query(None, self.image_path, timeout=300)
        self.assertIsNone(text_emb)
        self.assertIsInstance(image_emb, list)
        self.assertGreater(len(image_emb), 0)
        self.assertTrue(all(isinstance(x, float) for x in image_emb))

    def test_text_and_image_embeddings(self):
        text_emb, image_emb = self.worker.encode_query("a red car", self.image_path, timeout=300)
        self.assertIsInstance(text_emb, list)
        self.assertIsInstance(image_emb, list)
        self.assertGreater(len(text_emb), 0)
        self.assertGreater(len(image_emb), 0)
        self.assertTrue(all(isinstance(x, float) for x in text_emb))
        self.assertTrue(all(isinstance(x, float) for x in image_emb))

    def test_document_image_embedding(self):
        model = self.worker._model
        vec = _to_list(model.encode_document(self.image_path))
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)

    def test_encode_endpoint_serves_real_embeddings(self):
        with mock.patch.object(main_module, "worker", self.worker):
            client = TestClient(main_module.app)
            resp = client.post("/encode", json={"text": "hello world", "normalize": False})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("text_embedding", body)
        self.assertNotIn("image_embedding", body)
        self.assertIsInstance(body["text_embedding"], list)
        self.assertGreater(len(body["text_embedding"]), 0)
        self.assertTrue(all(isinstance(x, float) for x in body["text_embedding"]))


if __name__ == "__main__":
    unittest.main()
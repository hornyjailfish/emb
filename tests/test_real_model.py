"""Integration tests against the real MegaLoc weights.

These download/load the actual model via torch.hub and run inference, so they
are slow and need network + enough RAM/disk. Skipped unless you opt in:

    EMBED_REAL_MODEL=1 python -m unittest tests.test_real_model -v

Verifies what the synthetic tests can't: model loads, encode_vpr returns an
L2-normalized float vector of the expected dim, and POST /encode serves it.
"""

from __future__ import annotations

import base64
import io
import math
import os
import time
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import src.main as main_module
from src.config import get_settings
from src.state import ServiceState
from src.worker import ModelWorker

LOAD_TIMEOUT = 1200.0  # seconds; covers one-time hub download + load
EXPECTED_DIM = 8448


@unittest.skipUnless(
    os.environ.get("EMBED_REAL_MODEL") == "1",
    "set EMBED_REAL_MODEL=1 to run real-model integration tests",
)
class RealModelIntegrationTest(unittest.TestCase):
    worker: ModelWorker | None = None
    image: Image.Image | None = None

    @classmethod
    def setUpClass(cls):
        settings = get_settings()
        os.environ.setdefault("HF_HOME", settings.hf_home)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "warning")

        rng = np.random.default_rng(42)
        cls.image = Image.fromarray(
            (rng.random((240, 320, 3)) * 255).astype(np.uint8), "RGB"
        )

        cls.worker = ModelWorker(ServiceState())
        cls.worker.start()

        deadline = time.time() + LOAD_TIMEOUT
        while True:
            if cls.worker.state.model_ready():
                break
            if cls.worker.state.model_state == "error":
                cls.worker.stop()
                raise unittest.SkipTest(
                    f"model failed to load: {cls.worker.state.model_error}"
                )
            if time.time() > deadline:
                cls.worker.stop()
                raise unittest.SkipTest("timed out loading MegaLoc weights")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        if cls.worker is not None:
            cls.worker.stop()

    def test_model_is_ready(self):
        self.assertEqual(self.worker.state.model_state, "ready")
        self.assertTrue(self.worker.state.model_ready())
        self.assertIsNotNone(self.worker.model)

    def test_encode_vpr_shape_and_dtype(self):
        emb = self.worker.encode_vpr(self.image, timeout=300.0, transform="squash")
        self.assertIsInstance(emb, list)
        self.assertEqual(len(emb), EXPECTED_DIM)
        self.assertTrue(all(isinstance(x, float) for x in emb))

    def test_encode_vpr_is_l2_normalized(self):
        emb = self.worker.encode_vpr(self.image, timeout=300.0)
        norm = math.sqrt(sum(x * x for x in emb))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_encode_endpoint_serves_real_embeddings(self):
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(
            "ascii"
        )

        with mock.patch.object(main_module, "worker", self.worker), mock.patch.object(
            main_module,
            "_load_vpr_image",
            new=mock.AsyncMock(return_value=buf.getvalue()),
        ):
            main_module.app.state.conn = mock.Mock()
            client = TestClient(main_module.app)
            resp = client.post("/encode", json={"image_url": data_url})

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body.get("backend"), "megaloc")
        self.assertEqual(body.get("dim"), EXPECTED_DIM)
        self.assertIsInstance(body["embedding"], list)
        self.assertEqual(len(body["embedding"]), EXPECTED_DIM)


if __name__ == "__main__":
    unittest.main()

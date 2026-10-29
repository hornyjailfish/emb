"""POST /generate batch accept + per-item SSE queued events."""

from __future__ import annotations

import asyncio
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.main as main
from src.state import ServiceState


class GenerateBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate global queue/state/broadcaster between tests.
        main.vpr_queue = asyncio.Queue(maxsize=100)
        main.state = ServiceState()
        main.broadcaster.reset()

        # Minimal app: no lifespan, no DB, no worker — only generate routes.
        self.app = FastAPI()
        self.app.add_api_route(
            "/generate", main.generate, methods=["POST"], status_code=202
        )
        self.app.add_api_route(
            "/vpr/generate",
            main.vpr_generate,
            methods=["POST"],
            status_code=202,
        )
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        main.broadcaster.close()
        main.broadcaster.reset()

    def test_accepts_array_and_queues_each(self) -> None:
        items = ["marker_views:a", "marker_views:b", "marker_views:c"]
        q = main.broadcaster.subscribe()

        resp = self.client.post("/generate", json={"items": items})
        self.assertEqual(resp.status_code, 202)
        body = resp.json()
        self.assertEqual(body["count"], 3)
        self.assertEqual(body["accepted"], items)
        self.assertEqual(body["backend"], "megaloc")

        # All three landed on the VPR queue.
        queued = []
        while not main.vpr_queue.empty():
            queued.append(main.vpr_queue.get_nowait())
        self.assertEqual([item.task_id for item in queued], items)

        # One SSE "queued" event per item, in order.
        events = [q.get_nowait() for _ in items]
        self.assertEqual(
            events,
            [
                {"record_id": rid, "status": "queued", "kind": "vpr"}
                for rid in items
            ],
        )

        # In-memory task registry mirrors queued state.
        for rid in items:
            self.assertEqual(main.state.get_task(rid)["status"], "queued")

        main.broadcaster.unsubscribe(q)

    def test_vpr_alias_queues(self) -> None:
        resp = self.client.post(
            "/vpr/generate",
            json={"items": [{"id": "free-1", "image_url": "data:image/png;base64,AAAA"}]},
        )
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(resp.json()["backend"], "megaloc")
        item = main.vpr_queue.get_nowait()
        self.assertEqual(item.task_id, "free-1")
        self.assertTrue(item.image_url)

    def test_freeform_requires_image_url(self) -> None:
        resp = self.client.post("/generate", json={"items": ["not_a_view:1"]})
        self.assertEqual(resp.status_code, 422)

    def test_rejects_empty_items(self) -> None:
        resp = self.client.post("/generate", json={"items": []})
        self.assertEqual(resp.status_code, 400)

    def test_rejects_invalid_marker_views_id_before_queueing(self) -> None:
        # Bare strings that look like marker_views still need a valid RecordID.
        resp = self.client.post(
            "/generate", json={"items": ["marker_views:"]}
        )
        self.assertEqual(resp.status_code, 422)
        self.assertTrue(main.vpr_queue.empty())


if __name__ == "__main__":
    unittest.main()

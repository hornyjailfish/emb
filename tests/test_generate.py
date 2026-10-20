"""POST /generate batch accept + per-item SSE queued events."""

from __future__ import annotations

import asyncio
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from surrealdb import RecordID

import src.main as main
from src.state import ServiceState


class GenerateBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate global queue/state/broadcaster between tests.
        main.job_queue = asyncio.Queue(maxsize=100)
        main.state = ServiceState()
        main.broadcaster.reset()

        # Minimal app: no lifespan, no DB, no worker — only /generate.
        self.app = FastAPI()
        self.app.add_api_route(
            "/generate", main.generate, methods=["POST"], status_code=202
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

        # All three landed on the dispatch queue as RecordIDs.
        queued = []
        while not main.job_queue.empty():
            queued.append(main.job_queue.get_nowait())
        self.assertEqual([str(r) for r in queued], items)
        self.assertTrue(all(isinstance(r, RecordID) for r in queued))

        # One SSE "queued" event per item, in order.
        events = [q.get_nowait() for _ in items]
        self.assertEqual(
            events,
            [{"record_id": rid, "status": "queued"} for rid in items],
        )

        # In-memory task registry mirrors queued state.
        for rid in items:
            self.assertEqual(main.state.get_task(rid)["status"], "queued")

        main.broadcaster.unsubscribe(q)

    def test_rejects_empty_items(self) -> None:
        resp = self.client.post("/generate", json={"items": []})
        self.assertEqual(resp.status_code, 400)

    def test_rejects_invalid_id_before_queueing(self) -> None:
        resp = self.client.post("/generate", json={"items": ["not a record"]})
        self.assertEqual(resp.status_code, 422)
        self.assertTrue(main.job_queue.empty())


if __name__ == "__main__":
    unittest.main()

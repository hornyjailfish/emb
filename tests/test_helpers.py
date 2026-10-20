"""Synthetic tests for pure helpers (no DB, no model).

Covers `_to_list` / `_normalize` (worker) and the URL/segment/record-id helpers
from main. None of these touch SurrealDB or load the model.
"""

import unittest

from fastapi import HTTPException
from surrealdb import File, RecordID

from src.main import (
    _as_file,
    _build_proxy_url,
    _parse_record_id,
    _queue_rid,
    _safe_literal_segment,
    settings,
)
from src.worker import _normalize, _to_list, prepare_image


class ToListTests(unittest.TestCase):
    def test_none_is_none(self):
        self.assertIsNone(_to_list(None))

    def test_list_becomes_floats(self):
        self.assertEqual(_to_list([1, 2, 3]), [1.0, 2.0, 3.0])

    def test_tolist_object(self):
        class Vec:
            def tolist(self):
                return [1, 2, 3]

        self.assertEqual(_to_list(Vec()), [1.0, 2.0, 3.0])


class NormalizeTests(unittest.TestCase):
    def test_none_is_none(self):
        self.assertIsNone(_normalize(None))

    def test_unit_length(self):
        self.assertEqual(_normalize([3.0, 4.0]), [0.6, 0.8])

    def test_zero_norm_unchanged(self):
        self.assertEqual(_normalize([0.0, 0.0]), [0.0, 0.0])


class BuildProxyUrlTests(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertIsNone(_build_proxy_url(None))
        self.assertIsNone(_build_proxy_url(""))

    def test_http_passthrough(self):
        self.assertEqual(_build_proxy_url("http://x/y.png"), "http://x/y.png")
        self.assertEqual(_build_proxy_url("https://x/y.png"), "https://x/y.png")

    def test_file_pointer(self):
        expected = f"http://127.0.0.1:{settings.fastapi_port}/files/bkt/path/img.png"
        self.assertEqual(_build_proxy_url("bkt:/path/img.png"), expected)

    def test_file_pointer_double_slash_path(self):
        expected = f"http://127.0.0.1:{settings.fastapi_port}/files/bkt/path"
        self.assertEqual(_build_proxy_url("bkt://path"), expected)

    def test_native_file(self):
        expected = f"http://127.0.0.1:{settings.fastapi_port}/files/bkt/path/img.png"
        self.assertEqual(_build_proxy_url(File("bkt", "/path/img.png")), expected)

    def test_no_path(self):
        self.assertIsNone(_build_proxy_url("bkt:"))

    def test_no_bucket(self):
        self.assertIsNone(_build_proxy_url(":path"))

    def test_no_colon_returns_raw(self):
        self.assertEqual(_build_proxy_url("plain"), "plain")


class AsFileTests(unittest.TestCase):
    def test_native_passthrough(self):
        f = File("images", "/a.png")
        self.assertIs(_as_file(f), f)

    def test_string_forms(self):
        self.assertEqual(_as_file("bkt:/path/img.png"), File("bkt", "/path/img.png"))
        self.assertEqual(_as_file("bkt://path"), File("bkt", "/path"))
        self.assertEqual(_as_file('f"bkt:/path"'), File("bkt", "/path"))

    def test_http_and_junk(self):
        self.assertIsNone(_as_file("http://x/y.png"))
        self.assertIsNone(_as_file("plain"))
        self.assertIsNone(_as_file(None))
        self.assertIsNone(_as_file(123))


class SafeLiteralSegmentTests(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(_safe_literal_segment("img-123.png"))
        self.assertTrue(_safe_literal_segment("folder/sub"))

    def test_invalid(self):
        self.assertFalse(_safe_literal_segment(""))
        self.assertFalse(_safe_literal_segment('a"b'))
        self.assertFalse(_safe_literal_segment("a\\b"))
        self.assertFalse(_safe_literal_segment("a b"))
        self.assertFalse(_safe_literal_segment("a\tb"))
        self.assertFalse(_safe_literal_segment("a\nb"))


class ParseRecordIdTests(unittest.TestCase):
    def test_valid(self):
        rid = _parse_record_id("marker_views:abc123")
        self.assertIsInstance(rid, RecordID)
        self.assertEqual(str(rid), "marker_views:abc123")

    def test_invalid_raises_422(self):
        with self.assertRaises(HTTPException) as ctx:
            _parse_record_id("invalid")
        self.assertEqual(ctx.exception.status_code, 422)


class QueueRidTests(unittest.TestCase):
    def test_maps_marker_views_key_to_embedding_queue(self):
        # The add_embedding_task event keys the queue row by record::id($after.id),
        # so the queue row shares the marker_views record's key.
        self.assertEqual(str(_queue_rid("marker_views:abc123")), "embedding_queue:abc123")

    def test_preserves_string_key(self):
        q = _queue_rid("marker_views:abc123")
        self.assertEqual(getattr(q, "table_name", None) or getattr(q, "table", None), "embedding_queue")
        self.assertEqual(q.id, "abc123")


class PrepareImageTests(unittest.TestCase):
    def _png_bytes(self, w, h):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (w, h), (200, 100, 50)).save(buf, format="PNG")
        return buf.getvalue()

    def test_small_image_unchanged(self):
        img = prepare_image(self._png_bytes(100, 80), max_pixels=262144)
        self.assertEqual(img.size, (100, 80))

    def test_large_image_downscaled_keeping_aspect(self):
        img = prepare_image(self._png_bytes(2000, 1000), max_pixels=262144)
        w, h = img.size
        self.assertLessEqual(w * h, 262144)
        self.assertAlmostEqual(w / h, 2.0, delta=0.05)

    def test_no_upscale(self):
        img = prepare_image(self._png_bytes(50, 40), max_pixels=262144)
        self.assertEqual(img.size, (50, 40))


if __name__ == "__main__":
    unittest.main()
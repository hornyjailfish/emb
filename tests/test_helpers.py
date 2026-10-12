"""Synthetic tests for pure helpers (no DB, no model).

Covers `_to_list` / `_normalize` (worker) and the URL/segment/record-id helpers
from main. None of these touch SurrealDB or load the model.
"""

import unittest

from fastapi import HTTPException
from surrealdb import RecordID

from src.main import _build_proxy_url, _parse_record_id, _safe_literal_segment, settings
from src.worker import _normalize, _to_list


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

    def test_no_path(self):
        self.assertIsNone(_build_proxy_url("bkt:"))

    def test_no_bucket(self):
        self.assertIsNone(_build_proxy_url(":path"))

    def test_no_colon_returns_raw(self):
        self.assertEqual(_build_proxy_url("plain"), "plain")


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


if __name__ == "__main__":
    unittest.main()
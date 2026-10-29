"""Synthetic tests for pure helpers (no DB, no model).

Covers URL/segment/record-id helpers from main. None of these touch SurrealDB
or load the model.
"""

import unittest

from fastapi import HTTPException
from surrealdb import File, RecordID

from src.ids import parse_record_id, record_key
from src.main import (
    _as_file,
    _build_proxy_url,
    _decode_base64_image,
    _image_ref_log,
    _marker_views_key_from_queue_row,
    _parse_record_id,
    _queue_rid,
    _safe_literal_segment,
    settings,
)
from src.worker import _to_list


class ToListTests(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(_to_list(None), [])

    def test_list_becomes_floats(self):
        self.assertEqual(_to_list([1, 2, 3]), [1.0, 2.0, 3.0])

    def test_tolist_object(self):
        class Vec:
            def tolist(self):
                return [1, 2, 3]

        self.assertEqual(_to_list(Vec()), [1.0, 2.0, 3.0])


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

    def test_data_url_is_not_a_file(self):
        # Regression: data:image/... must not become bucket "data".
        self.assertIsNone(_as_file("data:image/jpeg;base64,AAAA"))
        self.assertIsNone(_as_file("data:image/png;base64,iVBOR"))

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

    def test_digit_leading_id_stays_bare(self):
        # Surreal str(RecordID) wraps these in ⟨⟩; our key/parse must not.
        raw = "marker_views:1yhf83tq1chseep1ts1s"
        rid = _parse_record_id(raw)
        self.assertEqual(rid.id, "1yhf83tq1chseep1ts1s")
        self.assertEqual(record_key(rid), raw)
        # SDK display form must round-trip through our parser, not RecordID.parse.
        display = str(rid)
        self.assertTrue(display.startswith("marker_views:"))
        self.assertEqual(display[13], "\u27e8")  # ⟨
        self.assertEqual(display[-1], "\u27e9")  # ⟩
        self.assertNotEqual(rid.id, RecordID.parse(display).id)
        self.assertEqual(RecordID.parse(display).id, "\u27e81yhf83tq1chseep1ts1s\u27e9")
        self.assertEqual(parse_record_id(display).id, "1yhf83tq1chseep1ts1s")
        self.assertEqual(parse_record_id(display), rid)

    def test_accepts_already_escaped_client_input(self):
        rid = _parse_record_id("marker_views:\u27e81yhf83tq1chseep1ts1s\u27e9")
        self.assertEqual(rid.id, "1yhf83tq1chseep1ts1s")
        self.assertEqual(record_key(rid), "marker_views:1yhf83tq1chseep1ts1s")

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

    def test_digit_leading_and_escaped_forms(self):
        bare = "marker_views:1yhf83tq1chseep1ts1s"
        escaped = "marker_views:\u27e81yhf83tq1chseep1ts1s\u27e9"
        for src in (bare, escaped, str(RecordID.parse(bare))):
            with self.subTest(src=src.encode("unicode_escape")):
                q = _queue_rid(src)
                self.assertEqual(q.table_name, "embedding_queue")
                self.assertEqual(q.id, "1yhf83tq1chseep1ts1s")


class QueueRowKeyTests(unittest.TestCase):
    def test_from_data_field(self):
        key = _marker_views_key_from_queue_row(
            {"id": RecordID("embedding_queue", "abc"), "data": RecordID("marker_views", "abc")}
        )
        self.assertEqual(key, "marker_views:abc")

    def test_from_queue_id_fallback(self):
        key = _marker_views_key_from_queue_row(
            {"id": RecordID("embedding_queue", "xyz")}
        )
        self.assertEqual(key, "marker_views:xyz")


class Base64ImageTests(unittest.TestCase):
    def _png_b64(self, w=64, h=48):
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (w, h), (10, 20, 30)).save(buf, format="PNG")
        raw = buf.getvalue()
        return raw, base64.b64encode(raw).decode("ascii")

    def test_raw_base64(self):
        raw, b64 = self._png_b64()
        self.assertEqual(_decode_base64_image(b64), raw)

    def test_data_url(self):
        raw, b64 = self._png_b64()
        self.assertEqual(
            _decode_base64_image(f"data:image/png;base64,{b64}"), raw
        )

    def test_log_never_dumps_body(self):
        _, b64 = self._png_b64()
        label = _image_ref_log(b64)
        self.assertTrue(label.startswith("base64(len="))
        self.assertNotIn(b64[:32], label)
        data_label = _image_ref_log(f"data:image/jpeg;base64,{b64}")
        self.assertTrue(data_label.startswith("data_url("))
        self.assertNotIn(b64[:32], data_label)

    def test_rejects_garbage(self):
        self.assertIsNone(_decode_base64_image("not-base64!!!"))
        self.assertIsNone(_decode_base64_image("draft:/x.png"))


if __name__ == "__main__":
    unittest.main()

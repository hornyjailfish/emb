"""Unit tests for search ranking, filters, and query building."""

from __future__ import annotations

import unittest

from surrealdb import RecordID

import math

from src.search import (
    VIEW_COUNT_ALPHA,
    FilterDef,
    SearchTarget,
    _parse_filter_record_id,
    aggregate_by_parent,
    bind_filters,
    build_candidate_query,
    candidate_limit,
    filter_top_views,
    get_target,
    parent_score,
    unwrap_query_result,
)


class FilterTopViewsTests(unittest.TestCase):
    def test_applies_min_score_and_top_k_on_views(self) -> None:
        rows = [
            {"id": "marker_views:1", "marker": "markers:a", "score": 0.9},
            {"id": "marker_views:2", "marker": "markers:b", "score": 0.6},
            {"id": "marker_views:3", "marker": "markers:c", "score": 0.4},
            {"id": "marker_views:4", "marker": "markers:d", "score": 0.3},
        ]
        # cutoff drops c/d; top_k keeps only the first view above cutoff
        got = filter_top_views(rows, top_k=1, min_score=0.55)
        self.assertEqual([r["id"] for r in got], ["marker_views:1"])

    def test_stops_once_score_falls_below_cutoff(self) -> None:
        rows = [
            {"id": "a", "score": 0.8},
            {"id": "b", "score": 0.5},
            {"id": "c", "score": 0.9},  # would pass cutoff but sort contract broken
        ]
        got = filter_top_views(rows, top_k=10, min_score=0.55)
        self.assertEqual([r["id"] for r in got], ["a"])


class AggregateTests(unittest.TestCase):
    def test_parent_score_is_peak_plus_log_boost(self) -> None:
        rows = [
            {"id": "marker_views:1", "marker": "markers:a", "score": 0.9},
            {"id": "marker_views:2", "marker": "markers:a", "score": 0.5},
            {"id": "marker_views:3", "marker": "markers:b", "score": 1.0},
        ]
        hits = aggregate_by_parent(rows, parent_field="marker")
        # b: 1.0 + α·log(2); a: 0.9 + α·log(3) — peak still decides
        self.assertEqual([h["id"] for h in hits], ["markers:b", "markers:a"])
        self.assertAlmostEqual(hits[0]["score"], parent_score(1.0, 1))
        self.assertAlmostEqual(hits[1]["score"], parent_score(0.9, 2))
        self.assertAlmostEqual(hits[0]["peak"], 1.0)
        self.assertAlmostEqual(hits[1]["peak"], 0.9)
        self.assertEqual(len(hits[1]["views"]), 2)
        self.assertEqual(hits[1]["views"][0]["id"], "marker_views:1")

    def test_many_weak_views_do_not_outrank_one_strong(self) -> None:
        rows = [
            {"id": "marker_views:strong", "marker": "markers:best", "score": 0.8},
            {"id": "marker_views:w1", "marker": "markers:spam", "score": 0.56},
            {"id": "marker_views:w2", "marker": "markers:spam", "score": 0.56},
            {"id": "marker_views:w3", "marker": "markers:spam", "score": 0.56},
        ]
        hits = aggregate_by_parent(rows, parent_field="marker")
        self.assertEqual(hits[0]["id"], "markers:best")
        self.assertAlmostEqual(hits[0]["score"], parent_score(0.8, 1))
        self.assertAlmostEqual(hits[1]["score"], parent_score(0.56, 3))
        # log boost must not let 3×0.56 beat a lone 0.8 at default α
        self.assertGreater(hits[0]["score"], hits[1]["score"])

    def test_multi_hit_boosts_over_equal_peak(self) -> None:
        rows = [
            {"id": "marker_views:1", "marker": "markers:solo", "score": 0.7},
            {"id": "marker_views:2", "marker": "markers:multi", "score": 0.7},
            {"id": "marker_views:3", "marker": "markers:multi", "score": 0.65},
        ]
        hits = aggregate_by_parent(rows, parent_field="marker")
        self.assertEqual(hits[0]["id"], "markers:multi")
        self.assertGreater(hits[0]["score"], hits[1]["score"])
        self.assertAlmostEqual(
            hits[0]["score"] - hits[1]["score"],
            VIEW_COUNT_ALPHA * (math.log(3) - math.log(2)),
        )

    def test_record_id_objects_group_correctly(self) -> None:
        rows = [
            {
                "id": RecordID.parse("marker_views:1"),
                "marker": RecordID.parse("markers:x"),
                "score": 0.7,
            },
            {
                "id": RecordID.parse("marker_views:2"),
                "marker": RecordID.parse("markers:x"),
                "score": 0.2,
            },
        ]
        hits = aggregate_by_parent(rows, parent_field="marker")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "markers:x")
        self.assertAlmostEqual(hits[0]["peak"], 0.7)
        self.assertAlmostEqual(hits[0]["score"], parent_score(0.7, 2))


class FilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = SearchTarget(
            name="markers",
            source_table="marker_views",
            parent_field="marker",
            filters=(FilterDef(key="level", expr="marker.level", kind="record"),),
        )

    def test_bind_level_filter(self) -> None:
        clauses, params = bind_filters(self.target, {"level": "levels:ground"})
        self.assertEqual(clauses, ["marker.level = $f_level"])
        self.assertEqual(params["f_level"], RecordID.parse("levels:ground"))

    def test_bind_numeric_level_id_as_int(self) -> None:
        # DB stores levels:-1 with int id; string "-1" must not be used.
        _, params = bind_filters(self.target, {"level": "levels:-1"})
        rid = params["f_level"]
        self.assertEqual(rid.table_name, "levels")
        self.assertEqual(rid.id, -1)
        self.assertIsInstance(rid.id, int)

    def test_rejects_unknown_filter(self) -> None:
        with self.assertRaises(ValueError):
            bind_filters(self.target, {"zone": "zones:1"})

    def test_rejects_bad_record_id(self) -> None:
        with self.assertRaises(ValueError):
            bind_filters(self.target, {"level": "not-a-record"})


class ParseFilterRecordIdTests(unittest.TestCase):
    def test_numeric_negative(self) -> None:
        rid = _parse_filter_record_id("levels:-1")
        self.assertEqual(rid.id, -1)
        self.assertIsInstance(rid.id, int)

    def test_numeric_positive(self) -> None:
        rid = _parse_filter_record_id("levels:3")
        self.assertEqual(rid.id, 3)

    def test_string_id_unchanged(self) -> None:
        rid = _parse_filter_record_id("levels:lobby")
        self.assertEqual(rid.id, "lobby")

    def test_record_id_input_coerced(self) -> None:
        rid = _parse_filter_record_id(RecordID("levels", "-1"))
        self.assertEqual(rid.id, -1)


class QueryBuildTests(unittest.TestCase):
    def test_candidate_sql_includes_cosine_and_filters(self) -> None:
        target = get_target("markers")
        clauses, _ = bind_filters(target, {"level": "levels:1"})
        sql = build_candidate_query(
            target, filter_clauses=clauses, limit=40, min_score=0.55
        )
        self.assertIn("vector::similarity::cosine(embedding, $query)", sql)
        self.assertIn("FROM marker_views", sql)
        self.assertIn("marker.level = $f_level", sql)
        self.assertIn(
            "vector::similarity::cosine(embedding, $query) >= $min_score", sql
        )
        self.assertIn("ORDER BY score DESC", sql)
        self.assertIn("LIMIT 40", sql)
        self.assertIn("marker", sql)

    def test_candidate_limit_is_view_level_top_k(self) -> None:
        target = get_target("markers")
        self.assertEqual(candidate_limit(target, 1), 1)
        self.assertEqual(candidate_limit(target, 10), 10)


class RegistryTests(unittest.TestCase):
    def test_markers_target_registered(self) -> None:
        t = get_target("markers")
        self.assertEqual(t.source_table, "marker_views")
        self.assertEqual(t.parent_field, "marker")
        self.assertIn("level", [f.key for f in t.filters])

    def test_unknown_target(self) -> None:
        with self.assertRaises(KeyError):
            get_target("nope")


class UnwrapTests(unittest.TestCase):
    def test_nested_statement_rows(self) -> None:
        raw = [[{ "id": "a", "score": 1.0 }, { "id": "b", "score": 0.5 }]]
        rows = unwrap_query_result(raw)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], "a")

    def test_flat_list_passthrough(self) -> None:
        raw = [{ "id": "a" }]
        self.assertEqual(unwrap_query_result(raw), raw)

    def test_single_dict(self) -> None:
        self.assertEqual(unwrap_query_result({"id": "a"}), [{"id": "a"}])

    def test_empty(self) -> None:
        self.assertEqual(unwrap_query_result(None), [])
        self.assertEqual(unwrap_query_result([]), [])
        self.assertEqual(unwrap_query_result([[]]), [])


if __name__ == "__main__":
    unittest.main()

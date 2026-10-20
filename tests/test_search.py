"""Unit tests for search ranking, filters, and query building."""

from __future__ import annotations

import unittest

from surrealdb import RecordID

from src.search import (
    FilterDef,
    SearchTarget,
    aggregate_by_parent,
    bind_filters,
    build_candidate_query,
    candidate_limit,
    get_target,
)


class AggregateTests(unittest.TestCase):
    def test_sums_scores_across_views_of_same_marker(self) -> None:
        rows = [
            {"id": "marker_views:1", "marker": "markers:a", "score": 0.9},
            {"id": "marker_views:2", "marker": "markers:a", "score": 0.5},
            {"id": "marker_views:3", "marker": "markers:b", "score": 1.0},
        ]
        hits = aggregate_by_parent(rows, parent_field="marker", top_k=10)
        # a = 1.4, b = 1.0 → a first
        self.assertEqual([h["id"] for h in hits], ["markers:a", "markers:b"])
        self.assertAlmostEqual(hits[0]["score"], 1.4)
        self.assertEqual(len(hits[0]["views"]), 2)
        self.assertAlmostEqual(hits[1]["score"], 1.0)

    def test_top_k_cuts_parents_not_views(self) -> None:
        rows = [
            {"id": "marker_views:1", "marker": "markers:a", "score": 0.3},
            {"id": "marker_views:2", "marker": "markers:b", "score": 0.2},
            {"id": "marker_views:3", "marker": "markers:c", "score": 0.1},
        ]
        hits = aggregate_by_parent(rows, parent_field="marker", top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["id"], "markers:a")

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
        hits = aggregate_by_parent(rows, parent_field="marker", top_k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "markers:x")
        self.assertAlmostEqual(hits[0]["score"], 0.9)


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

    def test_rejects_unknown_filter(self) -> None:
        with self.assertRaises(ValueError):
            bind_filters(self.target, {"zone": "zones:1"})

    def test_rejects_bad_record_id(self) -> None:
        with self.assertRaises(ValueError):
            bind_filters(self.target, {"level": "not-a-record"})


class QueryBuildTests(unittest.TestCase):
    def test_candidate_sql_includes_cosine_and_filters(self) -> None:
        target = get_target("markers")
        clauses, _ = bind_filters(target, {"level": "levels:1"})
        sql = build_candidate_query(target, filter_clauses=clauses, limit=40)
        self.assertIn("vector::similarity::cosine(embedding, $query)", sql)
        self.assertIn("FROM marker_views", sql)
        self.assertIn("marker.level = $f_level", sql)
        self.assertIn("ORDER BY score DESC", sql)
        self.assertIn("LIMIT 40", sql)
        self.assertIn("marker", sql)

    def test_candidate_limit_scales_with_top_k(self) -> None:
        target = get_target("markers")
        self.assertEqual(candidate_limit(target, 1), target.min_candidate_limit)
        self.assertEqual(
            candidate_limit(target, 10),
            max(target.min_candidate_limit, 10 * target.candidate_multiplier),
        )


class RegistryTests(unittest.TestCase):
    def test_markers_target_registered(self) -> None:
        t = get_target("markers")
        self.assertEqual(t.source_table, "marker_views")
        self.assertEqual(t.parent_field, "marker")
        self.assertIn("level", [f.key for f in t.filters])

    def test_unknown_target(self) -> None:
        with self.assertRaises(KeyError):
            get_target("nope")


if __name__ == "__main__":
    unittest.main()

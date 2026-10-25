"""Extensible embedding search targets and cosine ranking.

Add a new searchable collection by registering a ``SearchTarget`` in
``TARGETS`` — no endpoint changes required. Each target names:

- which table holds the vectors (``source_table`` / ``embedding_field``);
- how hits collapse onto a parent entity (``parent_field``), e.g. many
  ``marker_views`` → one ``markers`` row;
- which request ``filters`` map onto SurrealQL predicates;
- which parent columns to hydrate in the response (``parent_fields``).

Ranking: take the top-k **views** above ``min_score``, then collapse onto
parents. Parent score is peak cosine plus a small multi-hit boost::

    score = max(sim) + α · log(1 + n_views)

so one strong match stays on top, while genuine multi-view support still
nudges a marker up. Tune α via ``VIEW_COUNT_ALPHA`` (default 0.07).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from surrealdb import RecordID
from surrealdb.errors import InvalidRecordIdError

from src.ids import parse_record_id, record_key

# Multi-hit boost on parent score: max(sim) + α * log(1 + n_views).
# Keep small so raw cosine still dominates; bump if multi-view support
# should matter more.
VIEW_COUNT_ALPHA: float = 0.04


@dataclass(frozen=True)
class FilterDef:
    """One request-side filter → SurrealQL predicate.

    ``key`` is the JSON filters map key (e.g. ``\"level\"``).
    ``expr`` is the left-hand SurrealQL path (e.g. ``\"marker.level\"``).
    ``kind`` is ``\"record\"`` (parsed via ``RecordID.parse``) or ``\"value\"``
    (passed through as-is).
    """

    key: str
    expr: str
    kind: str = "record"  # record | value


@dataclass(frozen=True)
class SearchTarget:
    name: str
    source_table: str
    embedding_field: str = "embedding"
    # SurrealQL expression projected AS id (default: row id).
    # Use e.g. ``data`` when vectors live on ``embeddings`` and the view id
    # should be the linked ``marker_views`` record.
    id_expr: str = "id"
    # Collapse source rows onto this link field (None = no grouping).
    # ``parent_expr`` is the SurrealQL path; result column is ``parent_field``.
    parent_field: str | None = None
    parent_expr: str | None = None
    parent_table: str | None = None
    # Columns to SELECT from parent_table when hydrating hits.
    parent_fields: tuple[str, ...] = ()
    filters: tuple[FilterDef, ...] = ()
    # Extra headroom when fetching views (usually 1 — top_k is view-level).
    candidate_multiplier: int = 1
    min_candidate_limit: int = 1
    # Which model backend encodes the query for this target.
    # ``jina`` → text and/or image; ``megaloc`` → image-only VPR.
    query_backend: str = "jina"


# ---------------------------------------------------------------------------
# Registry — add targets here to expose them on POST /search
# ---------------------------------------------------------------------------

# Vectors live on ``embeddings`` (not marker_views). Each row links back via
# ``data → marker_views``; parent marker is ``data.marker``. View ids in the
# response are the marker_views records (``data AS id``).
_MARKERS_COMMON = dict(
    source_table="embeddings",
    id_expr="data",
    parent_field="marker",
    parent_expr="data.marker",
    parent_table="markers",
    parent_fields=("level", "zone", "closest_shop", "geometry"),
    filters=(
        FilterDef(key="level", expr="data.marker.level", kind="record"),
        # zone reserved for later — enable when the app needs it:
        # FilterDef(key="zone", expr="data.marker.zone", kind="record"),
    ),
)

TARGETS: dict[str, SearchTarget] = {
    "markers": SearchTarget(
        name="markers",
        embedding_field="embedding_jina",
        query_backend="jina",
        **_MARKERS_COMMON,
    ),
    "markers_vpr": SearchTarget(
        name="markers_vpr",
        embedding_field="embedding_megaloc",
        query_backend="megaloc",
        **_MARKERS_COMMON,
    ),
}


def get_target(name: str) -> SearchTarget:
    target = TARGETS.get(name)
    if target is None:
        known = ", ".join(sorted(TARGETS)) or "(none)"
        raise KeyError(f"unknown search target {name!r}; known: {known}")
    return target


def unwrap_query_result(result: Any) -> list[Any]:
    """Normalize SDK 3 ``query()`` payloads into a flat list of row values.

    A single SurrealQL statement comes back as ``[[row, row, ...]]`` (outer list
    = statements, inner = rows). Older/simpler responses may already be a flat
    list of dicts or a lone dict. Without unwrapping, search sees the inner list
    as a "row", drops it (not a dict with ``score``), and always returns empty.
    """
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    if not isinstance(result, list):
        return []
    if not result:
        return []
    # One or more statement result sets: take the first statement's rows.
    if isinstance(result[0], list):
        rows = result[0]
        return list(rows) if isinstance(rows, list) else []
    return result


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _as_id_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, RecordID):
        # Bare table:id — str(RecordID) wraps digit-leading ids in ⟨⟩.
        return record_key(value)
    if isinstance(value, dict):
        # Surreal sometimes returns nested record objects
        tb = value.get("tb") or value.get("table")
        rid = value.get("id")
        if tb is not None and rid is not None:
            return record_key(f"{tb}:{rid}")
    s = str(value)
    if ":" in s:
        try:
            return record_key(s)
        except (InvalidRecordIdError, ValueError, TypeError):
            return s
    return s


def filter_top_views(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    min_score: float,
) -> list[dict[str, Any]]:
    """Keep scored rows at/above ``min_score``, capped at ``top_k``.

    Assumes ``rows`` are already ordered by score descending (as produced by
    ``build_candidate_query``). Once a row falls below the cutoff, the rest
    are skipped.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or "score" not in row:
            continue
        try:
            score = float(row["score"])
        except (TypeError, ValueError):
            continue
        if score < min_score:
            # Query is ORDER BY score DESC — remaining rows cannot qualify.
            break
        out.append(row)
        if len(out) >= max(0, top_k):
            break
    return out


def parent_score(peak: float, n_views: int, *, alpha: float = VIEW_COUNT_ALPHA) -> float:
    """Combine peak view cosine with a diminishing multi-hit boost."""
    n = max(0, int(n_views))
    return float(peak) + float(alpha) * math.log(1 + n)


def aggregate_by_parent(
    rows: list[dict[str, Any]],
    *,
    parent_field: str,
    alpha: float = VIEW_COUNT_ALPHA,
) -> list[dict[str, Any]]:
    """Group already-selected view hits by parent; rank by peak + multi-hit.

    Caller is expected to pass **top-k views** (after ``min_score``). Each row
    needs ``parent_field`` + ``score``. Optional ``id`` is kept under ``views``.

    Parent score::

        max(view_sim) + α * log(1 + n_views)

    Peak keeps a single confident hit ahead of many weak ones; the log term
    gives a small bump when several views of the same marker all cleared the
    view gate. ``alpha`` defaults to ``VIEW_COUNT_ALPHA``.
    """
    groups: dict[str, dict[str, Any]] = {}

    for row in rows:
        parent = _as_id_str(row.get(parent_field))
        if not parent:
            continue
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if parent not in groups:
            groups[parent] = {
                "id": parent,
                "peak": score,
                "views": [],
            }
        g = groups[parent]
        if score > g["peak"]:
            g["peak"] = score
        view_id = row.get("id")
        g["views"].append(
            {
                "id": _as_id_str(view_id) if view_id is not None else None,
                "score": score,
            }
        )

    # Best view first within each parent; finalize peak+boost score.
    for g in groups.values():
        g["views"].sort(key=lambda v: v["score"], reverse=True)
        g["score"] = parent_score(g["peak"], len(g["views"]), alpha=alpha)

    ranked = sorted(groups.values(), key=lambda g: g["score"], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


def _parse_filter_record_id(value: Any) -> RecordID:
    """Parse a filter value into a ``RecordID``, coercing numeric ids to ``int``.

    Surreal stores ``levels:-1`` with an integer id ``-1``. String parse keeps
    the segment as ``"-1"``, and ``levels:-1`` (str) never equals ``levels:-1``
    (int) — filters then match nothing. Coerce plain integer segments (optional
    leading ``-``) so equality works. Also strips SurrealQL ``⟨⟩`` escapes.
    """
    rid = parse_record_id(value if isinstance(value, RecordID) else str(value).strip())

    id_part = rid.id
    if isinstance(id_part, str):
        s = id_part.strip()
        # Plain decimal int only — leave strings/UUIDs alone.
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return RecordID(rid.table_name, int(s))
    return rid


def bind_filters(
    target: SearchTarget, raw_filters: dict[str, Any] | None
) -> tuple[list[str], dict[str, Any]]:
    """Map request filters onto WHERE clauses + query params.

    Unknown filter keys raise ``ValueError`` so typos fail loudly instead of
    being silently ignored.
    """
    raw_filters = raw_filters or {}
    allowed = {f.key: f for f in target.filters}
    unknown = set(raw_filters) - set(allowed)
    if unknown:
        raise ValueError(
            f"unsupported filters for target {target.name!r}: "
            f"{sorted(unknown)}; allowed: {sorted(allowed) or 'none'}"
        )

    clauses: list[str] = []
    params: dict[str, Any] = {}
    for key, value in raw_filters.items():
        if value is None:
            continue
        fdef = allowed[key]
        param = f"f_{key}"
        if fdef.kind == "record":
            try:
                params[param] = _parse_filter_record_id(value)
            except (InvalidRecordIdError, ValueError, TypeError) as e:
                raise ValueError(f"invalid record id for filter {key!r}: {value}") from e
        else:
            params[param] = value
        clauses.append(f"{fdef.expr} = ${param}")
    return clauses, params


def build_candidate_query(
    target: SearchTarget,
    *,
    filter_clauses: list[str],
    limit: int,
    min_score: float | None = None,
) -> str:
    """SurrealQL that scores source rows by cosine similarity to ``$query``.

    When ``min_score`` is set, the cosine expression is also applied in WHERE
    against ``$min_score`` so LIMIT counts only qualifying views.
    """
    score_expr = f"vector::similarity::cosine({target.embedding_field}, $query)"
    where = [f"{target.embedding_field} IS NOT NONE", *filter_clauses]
    if min_score is not None:
        where.append(f"{score_expr} >= $min_score")
    where_sql = " AND ".join(where)
    id_sel = f"{target.id_expr} AS id" if target.id_expr != "id" else "id"
    parent_sel = ""
    if target.parent_field:
        p_expr = target.parent_expr or target.parent_field
        if p_expr == target.parent_field:
            parent_sel = f", {target.parent_field}"
        else:
            parent_sel = f", {p_expr} AS {target.parent_field}"
    return (
        f"SELECT {id_sel}{parent_sel}, "
        f"{score_expr} AS score "
        f"FROM {target.source_table} "
        f"WHERE {where_sql} "
        f"ORDER BY score DESC "
        f"LIMIT {int(limit)}"
    )


def candidate_limit(target: SearchTarget, top_k: int) -> int:
    """How many source views to pull — ``top_k`` is view-level."""
    return max(target.min_candidate_limit, int(top_k) * target.candidate_multiplier)


async def hydrate_parents(
    db: Any,
    target: SearchTarget,
    hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach parent record fields onto ranked hits (best-effort)."""
    if not hits or not target.parent_table or not target.parent_fields:
        return hits

    ids: list[RecordID] = []
    for h in hits:
        try:
            ids.append(RecordID.parse(h["id"]))
        except (InvalidRecordIdError, ValueError, KeyError):
            continue
    if not ids:
        return hits

    fields = ", ".join(("id", *target.parent_fields))
    # $ids is an array of record ids; Surreal resolves them.
    sql = f"SELECT {fields} FROM $ids"
    try:
        raw = await db.query(sql, {"ids": ids})
    except Exception:
        return hits

    rows = unwrap_query_result(raw)
    by_id = {_as_id_str(r.get("id")): r for r in rows if isinstance(r, dict)}
    out: list[dict[str, Any]] = []
    for h in hits:
        merged = dict(h)
        parent = by_id.get(h["id"])
        if parent:
            for col in target.parent_fields:
                if col in parent:
                    val = parent[col]
                    merged[col] = _as_id_str(val) if col in ("level", "zone", "closest_shop") else val
        out.append(merged)
    return out

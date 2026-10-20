"""Extensible embedding search targets and cosine ranking.

Add a new searchable collection by registering a ``SearchTarget`` in
``TARGETS`` — no endpoint changes required. Each target names:

- which table holds the vectors (``source_table`` / ``embedding_field``);
- how hits collapse onto a parent entity (``parent_field``), e.g. many
  ``marker_views`` → one ``markers`` row;
- which request ``filters`` map onto SurrealQL predicates;
- which parent columns to hydrate in the response (``parent_fields``).

Ranking: each parent’s score is the **sum** of its matching view cosine
scores, so extra views that also match the query add points naturally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from surrealdb import RecordID
from surrealdb.errors import InvalidRecordIdError


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
    # Collapse source rows onto this link field (None = no grouping).
    parent_field: str | None = None
    parent_table: str | None = None
    # Columns to SELECT from parent_table when hydrating hits.
    parent_fields: tuple[str, ...] = ()
    filters: tuple[FilterDef, ...] = ()
    # How many source rows to pull before grouping (top_k * multiplier).
    candidate_multiplier: int = 8
    min_candidate_limit: int = 32


# ---------------------------------------------------------------------------
# Registry — add targets here to expose them on POST /search
# ---------------------------------------------------------------------------

TARGETS: dict[str, SearchTarget] = {
    "markers": SearchTarget(
        name="markers",
        source_table="marker_views",
        embedding_field="embedding",
        parent_field="marker",
        parent_table="markers",
        parent_fields=("level", "zone", "closest_shop", "geometry"),
        filters=(
            FilterDef(key="level", expr="marker.level", kind="record"),
            # zone reserved for later — enable when the app needs it:
            # FilterDef(key="zone", expr="marker.zone", kind="record"),
        ),
    ),
}


def get_target(name: str) -> SearchTarget:
    target = TARGETS.get(name)
    if target is None:
        known = ", ".join(sorted(TARGETS)) or "(none)"
        raise KeyError(f"unknown search target {name!r}; known: {known}")
    return target


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _as_id_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, RecordID):
        return str(value)
    if isinstance(value, dict):
        # Surreal sometimes returns nested record objects
        tb = value.get("tb") or value.get("table")
        rid = value.get("id")
        if tb is not None and rid is not None:
            return f"{tb}:{rid}"
    return str(value)


def aggregate_by_parent(
    rows: list[dict[str, Any]],
    *,
    parent_field: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Group source hits by parent and rank by summed cosine scores.

    Each row needs at least ``parent_field`` and ``score``. Optional ``id`` is
    kept under ``views`` so the client can see which embeddings contributed.

    Multiple views of the same parent **add** their scores — extra matching
    views push the parent higher.
    """
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []

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
                "score": 0.0,
                "views": [],
            }
            order.append(parent)
        g = groups[parent]
        g["score"] += score
        view_id = row.get("id")
        g["views"].append(
            {
                "id": _as_id_str(view_id) if view_id is not None else None,
                "score": score,
            }
        )

    ranked = sorted(groups.values(), key=lambda g: g["score"], reverse=True)
    # Stable tie-break by first-seen order for equal scores
    return ranked[: max(0, top_k)]


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


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
                params[param] = RecordID.parse(str(value).strip())
            except (InvalidRecordIdError, ValueError) as e:
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
) -> str:
    """SurrealQL that scores source rows by cosine similarity to ``$query``."""
    where = [f"{target.embedding_field} IS NOT NONE", *filter_clauses]
    where_sql = " AND ".join(where)
    parent_sel = f", {target.parent_field}" if target.parent_field else ""
    return (
        f"SELECT id{parent_sel}, "
        f"vector::similarity::cosine({target.embedding_field}, $query) AS score "
        f"FROM {target.source_table} "
        f"WHERE {where_sql} "
        f"ORDER BY score DESC "
        f"LIMIT {int(limit)}"
    )


def candidate_limit(target: SearchTarget, top_k: int) -> int:
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
        rows = await db.query(sql, {"ids": ids})
    except Exception:
        return hits

    if not isinstance(rows, list):
        return hits

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

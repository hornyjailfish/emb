"""RecordID helpers that survive SurrealQL identifier escaping.

``str(RecordID)`` is the SurrealQL *display* form. Ids that start with a digit
(or contain non-bare characters) are wrapped in ``⟨...⟩`` (or backticks).
``RecordID.parse`` does **not** reverse that escaping — so:

    RecordID.parse("marker_views:1abc")       # id = "1abc"
    str(_)                                    # "marker_views:⟨1abc⟩"
    RecordID.parse(str(_))                    # id = "⟨1abc⟩"  ← WRONG

UPDATE/MERGE on the corrupted id targets a different (usually missing) record.
Surreal ``UPDATE`` of a missing row is a silent no-op, so the service can log
"saved" while the real row never changes.

Always:
- talk to the DB with a ``RecordID`` whose ``.id`` is the raw value;
- use :func:`record_key` for task keys / logs / SSE (round-trip safe);
- parse inbound strings with :func:`parse_record_id` (strips ``⟨⟩`` / backticks).
"""

from __future__ import annotations

from surrealdb import RecordID
from surrealdb.errors import InvalidRecordIdError

_ANGLE_OPEN = "\u27e8"  # ⟨
_ANGLE_CLOSE = "\u27e9"  # ⟩


def unescape_ident(identifier: str) -> str:
    """Strip SurrealQL identifier delimiters (``⟨...⟩`` or backtick-quoted)."""
    s = identifier.strip()
    if len(s) >= 2 and s[0] == _ANGLE_OPEN and s[-1] == _ANGLE_CLOSE:
        return s[1:-1]
    if len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        inner = s[1:-1]
        out: list[str] = []
        i = 0
        while i < len(inner):
            if inner[i] == "\\" and i + 1 < len(inner):
                out.append(inner[i + 1])
                i += 2
                continue
            out.append(inner[i])
            i += 1
        return "".join(out)
    return s


def parse_record_id(raw: str | RecordID) -> RecordID:
    """Parse ``table:id``, undoing SurrealQL escapes on either half.

    Always yields a string id (same contract as ``RecordID.parse``). Prefer this
    over ``RecordID.parse(str(rid))`` round-trips.
    """
    if isinstance(raw, RecordID):
        table = (
            unescape_ident(raw.table_name)
            if isinstance(raw.table_name, str)
            else raw.table_name
        )
        ident = raw.id
        if isinstance(ident, str):
            ident = unescape_ident(ident)
        if table == raw.table_name and ident == raw.id:
            return raw
        return RecordID(table, ident)

    s = str(raw).strip()
    if ":" not in s:
        raise InvalidRecordIdError(
            'invalid string provided for parse. the expected string format is '
            '"table_name:record_id"'
        )
    table, ident = s.split(":", 1)
    table = unescape_ident(table)
    ident = unescape_ident(ident)
    if not table or not ident:
        raise InvalidRecordIdError(
            'invalid string provided for parse. the expected string format is '
            '"table_name:record_id"'
        )
    return RecordID(table, ident)


def record_key(rid: str | RecordID) -> str:
    """Stable ``table:id`` without SurrealQL escapes — safe to parse back."""
    if not isinstance(rid, RecordID):
        rid = parse_record_id(rid)
    return f"{rid.table_name}:{rid.id}"

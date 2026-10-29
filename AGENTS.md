# Embeddings / VPR Service

Internal service that runs **MegaLoc** VPR on a single worker thread.
Connects to SurrealDB namespace/database configured via env (typically
`main` / `vision`).

| | |
|---|---|
| Model | [gmberton/MegaLoc](https://github.com/gmberton/MegaLoc) via `torch.hub` |
| Input | image only |
| Output | L2-normalized VPR descriptor (dim 8448) |
| Vector field | `embeddings.embedding` |
| Sync key | `marker_views.image_hash` ↔ `embedding_queue.hash_snapshot` |

Requires `surrealdb>=3.0.0b10` (native `File` / CBOR tag 55, builder-style
`update().merge()` / `upsert().merge()`, `db.files.get`).

Vectors are **not** stored on `marker_views`. The service UPSERTs into the
`embeddings` table and mirrors job status onto `embedding_queue`. Free-form
VPR client ids (non-`marker_views`) stay in-memory only.

There is **no text embedding path**. Abstract text search (if needed) belongs
in Surreal full-text / the app, not this service.

## Backend (MegaLoc / VPR)

- Loaded with `torch.hub.load(megaloc_repo, megaloc_entry)` — raw torch /
  torchvision, **not** sentence-transformers.
- CUDA/CPU selection (`settings.device` / `check_cuda()`) and Windows
  below-normal process priority on CPU.
- Eval recipe: geometry → ToTensor → ImageNet normalize → forward.
- Geometry size default **322×322** (`VPR_IMAGE_SIZE`); mode default **squash**
  (`VPR_TRANSFORM`).
- Async jobs via `POST /generate` (and `/vpr/generate`) **or** the pending
  queue poller:
  - **`marker_views:…` ids** — load image from the row (or override
    `image_url`), sample `image_hash`, UPSERT `embeddings.embedding` +
    `data` + `model`, mirror `embedding_queue` (processing → done/failed +
    `hash_snapshot`). Descriptor also kept in `ServiceState` for
    `GET /tasks/{id}`.
  - **Free-form ids** — require `image_url`; status + descriptor in-memory /
    SSE only (no queue / embeddings write).

## Data model

### `marker_views` (SCHEMAFULL) — source records (no vectors)

| Field | Type | Notes |
|---|---|---|
| `marker` | `record<markers>` | REFERENCE |
| `image_url` | `file \| none` | SurrealDB v3 file pointer (bucket `marker_images`) |
| `content_type` | `string \| none` | |
| `image_hash` | `string \| none` | `fn::image_hash(file::get(image_url))` — content fingerprint |
| `created_at` | `datetime` | READONLY default now |

Out-of-sync is **not** stored here. Compare:

```surql
marker_views.image_hash != embedding_queue.hash_snapshot
```

### `embeddings` (SCHEMAFULL) — vectors

Row key matches the marker_views record key (`marker_views:abc` →
`embeddings:abc`). Service UPSERTs via `db.upsert(rid).merge({...})`.

| Field | Type | Notes |
|---|---|---|
| `embedding` | `array<float> \| none` | VPR descriptor (MegaLoc, dim 8448) |
| `model` | `string` | default / label `gmberton/MegaLoc` |
| `data` | `record<marker_views>` | REFERENCE back to the view |

HNSW (DB-side):

```surql
DEFINE INDEX IF NOT EXISTS megaloc_idx
  ON embeddings COLUMNS embedding HNSW DIMENSION 8448 DIST COSINE;
```

### `embedding_queue` (SCHEMAFULL) — job status

Row key matches the marker_views record key (`marker_views:abc` →
`embedding_queue:abc`), created by the app's `add_embedding_task` event.

| Field | Type | Notes |
|---|---|---|
| `data` | `record<marker_views> \| none` | |
| `hash_snapshot` | `string \| none` | `marker_views.image_hash` at queue time; rewritten on **successful** embed |
| `status` | `'pending' \| 'processing' \| 'failed' \| 'done'` | DEFAULT `'pending'`; mirrored by this service |
| `error_message` | `string \| none` | set on failure |
| `updated_at` | computed | `VALUE time::now()` |

Event `add_embedding_task` (re)queues when `image_url` / `image_hash` changes
or the queue row's `hash_snapshot` mismatches the view.

### `markers` (parent)

| Field | Type | Notes |
|---|---|---|
| `level` | `record<basemap_floor>` | |
| `zone` | `record<zones> \| null` | |
| `geometry` | `geometry<point>` | floor SVG space `[x,y]` |
| `label` | `string \| null` | |

## Architecture

Two OS threads; no live queries, no batching.

- **Model worker thread** (`src/worker.py`) — loads MegaLoc lazily, then
  processes jobs one at a time. Query jobs (sync encode endpoints) are
  preferred over queued jobs.
- **Main async loop** (`src/main.py`) — FastAPI + SurrealDB (async SDK) + SSE.
  Lifespan starts:
  - `_vpr_dispatch` + `_vpr_collect` (SSE + embeddings/queue when
    marker_views-backed)
  - `_poll_pending_queue` — drains `embedding_queue WHERE status = 'pending'`
    on an interval (`QUEUE_POLL_INTERVAL`, default 2s; `0` disables)

Bridge: thread-safe `queue.Queue`s (`worker.jobs` / `worker.results` /
`worker._query_jobs`).

### SSE (`GET /queue/stream`)

Lifecycle: `queued` → `processing` → `done` | `error`
(plus `queue_done` when a busy wave drains). Events include `"kind": "vpr"`.
`done` events carry `dim` (not the full vector).

On app shutdown / uvicorn reload, `broadcaster.close()` pushes a shutdown
sentinel to every subscriber. Generators also poll `broadcaster.closed` and
`request.is_disconnected()` on a 1s timeout with SSE comment heartbeats.

## Image handling

### Loading (`main._image_bytes`)

Resolves `data:` / bare base64, `http(s)`, or Surreal `File` / `bucket:/key`
the same way for every endpoint. Never logs payload bodies.

### VPR geometry (`src/preprocess.py`)

All modes produce an exact `size×size` RGB image (default size 322):

| Mode | Env / field value | Behavior |
|---|---|---|
| **squash** | `squash` (default) | Stretch to square; ignore aspect (MegaLoc eval default) |
| **letterbox** | `letterbox` | Fit inside square, pad short axis (ImageNet-mean pad) |
| **center_crop** | `center_crop` | Scale so short side covers size, center-crop to square |

Worker then applies ImageNet normalize and runs MegaLoc. Per-request override
via `transform` on `/encode` and `/generate`. Settings:
`VPR_IMAGE_SIZE`, `VPR_TRANSFORM`.

## Files

- `src/config.py` — pydantic-settings (`Settings`, `get_settings`).
- `src/preprocess.py` — `squash` / `letterbox` / `center_crop`, `transform_image`.
- `src/state.py` — `ServiceState`; VPR tasks may hold `embedding` in memory.
- `src/worker.py` — MegaLoc worker, `image_to_megaloc_tensor`,
  `ModelJob` / `ModelResult` / `QueryJob`.
- `src/search.py` — cosine search target registry / aggregation.
- `src/main.py` — app, endpoints, dispatch/collect/poll loops, SSE, image loading.
- `src/utils.py` — logging, dependency check, Windows priority.
- `run.py` — uvicorn entrypoint (`reload=True`).
- `requirements.txt` — torch/torchvision + HF hub/safetensors; `surrealdb>=3.0.0b10`.

## Settings (env)

| Key | Default | Notes |
|---|---|---|
| `MEGALOC_REPO` | `gmberton/MegaLoc` | `torch.hub` repo |
| `MEGALOC_ENTRY` | `get_trained_model` | hub entrypoint |
| `MEGALOC_MODEL_LABEL` | `gmberton/MegaLoc` | written to `embeddings.model` |
| `VPR_IMAGE_SIZE` | `322` | square side in px |
| `VPR_TRANSFORM` | `squash` | `squash` \| `letterbox` \| `center_crop` |
| `DEVICE` | auto cuda/cpu | |
| `HF_HOME` | `D:/models/` | |
| `QUERY_TIMEOUT` | `0` | 0 = no timeout |
| `QUEUE_POLL_INTERVAL` | `2` | seconds; `0` = disable pending poll |
| `SURREALDB_NAMESPACE` | (required) | typically `main` |
| `SURREALDB_NAME` | (required) | typically `vision` |

## Search

`POST /search` encodes a query image with MegaLoc, runs cosine over a
registered **target**, keeps top-k **views**, collapses onto a parent. Source
table is always `embeddings`; view ids in the response are the linked
`marker_views` records (`SELECT data AS id`).

| Vector field | Query input |
|---|---|
| `embedding` | image only |

Registered targets: `markers`, `markers_vpr` (alias — same field/shape).

- Parent: `data.marker` → `markers`
- Parent hydrate fields: `level`, `zone`, `geometry`, `label`
- `top_k`: max **views** kept (not parents) — main retrieval cap
- `min_score`: **soft-deprecated** optional view cosine **noise floor**
  (default `0` = off for typical scores). Not a relevance/confidence gate;
  prefer client-side score filtering. Response may include `warnings` when
  `min_score > 0`. OpenAPI marks the field `deprecated`.
- Parent score: `max(sim) + α · log(1 + n_views)` (`VIEW_COUNT_ALPHA`, default `0.04`)
- Filter: `filters.level` → `data.marker.level` (`record<basemap_floor>`)

### Extending to more tables

Register another `SearchTarget` in `src/search.py` → `TARGETS`.
`GET /search/targets` lists what is registered.

## Endpoints

- `POST /encode` — sync one-shot vector  
  `{"image_url", "transform"?}` → `{embedding, dim, backend, …}`
- `POST /vpr/encode` — alias of `/encode`
- `POST /generate` — async batch → `202`  
  `{"items": ["marker_views:…"] | {"id", "image_url"?}], "transform"?}`  
  → UPSERT `embeddings.embedding` + queue for marker_views ids; free-form ids
  stay in-memory only
- `POST /vpr/generate` — alias of `/generate`
- `POST /search` — cosine VPR search (image required)
- `GET /search/targets` — registered targets + filter keys
- `GET /health` — device, queue sizes, DB ns/db, MegaLoc fields
- `GET /tasks`, `GET /tasks/{id}` — in-memory status; `done` includes
  `embedding`. List omits full vectors (`has_embedding` + `dim`)
- `GET /queue/stream` — SSE task feed
- `GET /files/{bucket}/{path}` — bucket file proxy

`image_url` accepts `data:` / base64, `http(s)`, or `bucket:/key` (needs DB).

## Conventions

- **Record IDs**: parse with `src.ids.parse_record_id()` (not bare
  `RecordID.parse`) and pass the object to the SDK (parameterized,
  injection-safe). Never interpolate raw IDs into SurrealQL.
  **Never** use `str(RecordID)` as something you parse back: digit-leading
  ids render as `table:⟨id⟩`, and `RecordID.parse` keeps the brackets inside
  `.id`, so `UPDATE` silently no-ops on a missing row. Task keys / logs / SSE
  use `src.ids.record_key(rid)` (`table:raw_id`, no escapes).
- **No regex** for validation. Path/bucket segments for the files proxy reject
  `"`, `\`, whitespace, control chars via `_safe_path_segment`.
- **File pointers**: SurrealDB v3 stores bucket refs as CBOR tag 55. Read with
  `await db.files.get(File(bucket, key))` — never interpolate into SurrealQL.
  `image_url` on a record arrives as a `File`; string forms `bucket:/key` are
  still accepted for the proxy/encode endpoints via `_as_file`.
- **Writes (vectors)**: `await db.upsert(embeddings_rid).merge({embedding,
  model, data})`. Never write vectors onto `marker_views`. Hash snapshot goes
  to `embedding_queue.hash_snapshot` only (on successful embed). Queue status
  via `await db.update(queue_rid).merge(...)`.
- **Writes (free-form ids)**: none — descriptors stay in `ServiceState`.
- **DB**: use the async SDK; it self-connects on first RPC. Connection is
  wrapped in `main.Connection`, which retries on startup and pings liveness so
  an unavailable DB never crashes the app.
- DB access stays on the async loop; model inference stays on the worker thread.

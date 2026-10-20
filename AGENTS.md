# Embeddings Service

Internal service that generates text and image embeddings for SurrealDB
`marker_views` records using `jinaai/jina-embeddings-v5-omni-nano-retrieval`.

The app calls an HTTP API; the service reads a record, fuses its text and image
into ONE embedding (a `(text, image)` tuple → single forward pass), writes it
back to the `embedding` field via `MERGE`, and downscales images into a PIL
image before handing them to the model (see "Image handling" below).

Requires `surrealdb>=3.0.0b10` (native `File` / CBOR tag 55, builder-style
`update().merge()`, `db.files.get`).

## Data model

### `marker_views` (SCHEMAFULL)

| Field | Type | Notes |
|---|---|---|
| `marker` | `record<markers>` | REFERENCE |
| `user_description` | `string \| none` | |
| `generated_description` | computed `string` | `VALUE` joins level/zone/area/shop/aliases/user_description |
| `embedding` | `array<float> \| none` | fused vector written by this service |
| `image_url` | `file \| none` | SurrealDB v3 file pointer (bucket ref, CBOR tag 55) |
| `version` | `int` | `VALUE ($value ?? 0) + 1` — bumps on every record write |

Out-of-sync is **not** stored on this table. Compare
`marker_views.version` with `embedding_queue.version_snapshot`.

### `embedding_queue` (SCHEMAFULL)

Row key matches the marker_views record key (`marker_views:abc` →
`embedding_queue:abc`), created by the app's `add_embedding_task` event.

| Field | Type | Notes |
|---|---|---|
| `data` | `record<marker_views> \| none` | |
| `version_snapshot` | `int` | `marker_views.version` sampled at dispatch, written on **successful** embed |
| `status` | `'pending' \| 'processing' \| 'failed' \| 'done'` | DEFAULT `'pending'`; mirrored by this service |
| `error_message` | `string \| none` | set on failure |
| `updated_at` | computed | `VALUE time::now()` |

Out-of-sync check (app side):

```surql
marker_views.version != embedding_queue.version_snapshot
```

## Architecture

Two OS threads; no live queries, no batching.

- **Model worker thread** (`src/worker.py`) — loads the model lazily in the
  background, then processes jobs one at a time. `encode_document` for record
  embeddings, `encode_query` for on-the-fly search queries. Each job is ONE
  model call: `worker._fused_input` turns text and/or image into a
  `(text, image)` tuple (or the lone modality) and the model returns a single
  fused vector. `ModelJob.version` carries the dispatch-time snapshot through
  to `ModelResult`.
- **Main async loop** (`src/main.py`) — FastAPI + SurrealDB (async SDK) + SSE.

Bridge between them: two thread-safe `queue.Queue`s.
- `_dispatch` (loop): read record → sample `version` → push `ModelJob`.
- `_persist` (loop): receive `ModelResult` → `MERGE embedding` onto
  `marker_views` → on success, set `embedding_queue` to `done` +
  `version_snapshot`.

Both loops also mirror task status onto the matching `embedding_queue` row:
`processing` when dispatched, then `done` / `failed` (+ `error_message`) at
the terminal states.

Queries bypass the DB: `POST /encode` pushes a `QueryJob` and blocks on a
`threading.Condition` until the worker returns the embedding.

### SSE (`GET /queue/stream`)

Lifecycle events: `queued` → `processing` → `done` | `error`.

On app shutdown / uvicorn reload, `broadcaster.close()` pushes a shutdown
sentinel to every subscriber. Generators also poll `broadcaster.closed` and
`request.is_disconnected()` on a 1s timeout with SSE comment heartbeats, so
reload is not blocked waiting for the client to drop the stream.

## Image handling (CPU)

The model runs on CPU, so image size is the main cost driver.

- The model (`jinaai/jina-embeddings-v5-omni-nano-retrieval`) accepts a
  `PIL.Image` directly (also bytes, paths, URLs, ndarray) — no temp files or
  URL round-trip needed.
- Its vision processor turns each image into 256–1280 vision tokens
  (262144–1310720 px), so CPU cost scales with pixel count.
- `worker.prepare_image(data, max_pixels)` downscales (aspect ratio kept, never
  upscales) before encoding. `main._prepare_image` loads the bytes (bucket
  pointer via `_read_file_bytes` → `db.files.get(File)`, or `http(s)` via
  httpx) then calls it, and the resulting PIL image becomes the image part of
  the fused model input. A malformed or missing image degrades to text-only —
  it is dropped here, never sent to the model.
- The size cap is `settings.image_max_pixels` (env `IMAGE_MAX_PIXELS`). Its
  default `262144` px (~512×512) is the model's floor — the fastest CPU
  setting. Raise it for more detail (e.g. `1310720` = 1280 tokens) once you have
  a GPU. This is the single source of truth for the size — don't hardcode it.
- One job runs at a time, so the prepared image is simply used and dropped (no
  cleanup juggling).

## Files

- `src/config.py` — pydantic-settings (`Settings`, `get_settings`).
- `src/state.py` — `ServiceState`, small thread-safe status registry.
- `src/worker.py` — model worker thread, `prepare_image`, `ModelJob` / `ModelResult`.
- `src/main.py` — app, endpoints, dispatch/persist tasks, SSE, image loading.
- `src/utils.py` — logging, dependency check, Windows process priority.
- `run.py` — uvicorn entrypoint (`reload=True`).
- `requirements.txt` — pins `surrealdb>=3.0.0b10`.

## Endpoints

- `POST /generate` `{"items": ["marker_views:..."]}` → `202`
- `POST /encode` `{"text": "..."}` and/or `{"image_url": "..."}` → `{"embedding": [...]}`
- `GET /health`
- `GET /tasks`, `GET /tasks/{record_id}`
- `GET /queue/stream` — SSE task feed (`queued` / `processing` / `done` / `error`)
- `GET /files/{bucket}/{path}` — bucket file proxy

## Conventions

- **Record IDs**: parse with `RecordID.parse()` and pass the object to the SDK
  (parameterized, injection-safe). Never interpolate raw IDs into SurrealQL.
- **No regex** for validation. Path/bucket segments for the files proxy reject
  `"`, `\`, whitespace, control chars via `_safe_path_segment`.
- **File pointers**: SurrealDB v3 stores bucket refs as CBOR tag 55. Read with
  `await db.files.get(File(bucket, key))` — never interpolate into SurrealQL.
  `image_url` on a record arrives as a `File`; string forms `bucket:/key` are
  still accepted for the proxy/encode endpoints via `_as_file`.
- **Writes**: `await db.update(rid).merge({...})` (SDK 3.x has no top-level
  `merge`). Only write `embedding` onto `marker_views` — do not invent fields
  (SCHEMAFULL). Version snapshot goes to `embedding_queue.version_snapshot`.
  Leave `generated_description` (computed) alone.
- **DB**: use the async SDK; it self-connects on first RPC. Connection is
  wrapped in `main.Connection`, which retries on startup and pings liveness so
  an unavailable DB never crashes the app.
- DB access stays on the async loop; model inference stays on the worker thread.

# Embeddings Service

Internal service that generates text and image embeddings for SurrealDB
`marker_views` records using `jinaai/jina-embeddings-v5-omni-nano-retrieval`.

The app calls an HTTP API; the service reads a record, fuses its text and image
into ONE embedding (a `(text, image)` tuple → single forward pass), writes it
back to the `embedding` field via `MERGE`, and downscales images into a PIL
image before handing them to the model (see "Image handling" below).

## Architecture

Two OS threads; no live queries, no batching.

- **Model worker thread** (`src/worker.py`) — loads the model lazily in the
  background, then processes jobs one at a time. `encode_document` for record
  embeddings, `encode_query` for on-the-fly search queries. Each job is ONE
  model call: `worker._fused_input` turns text and/or image into a
  `(text, image)` tuple (or the lone modality) and the model returns a single
  fused vector.
- **Main async loop** (`src/main.py`) — FastAPI + SurrealDB (async SDK) + SSE.

Bridge between them: two thread-safe `queue.Queue`s.
- `_dispatch` (loop): read record → push `ModelJob` to worker.
- `_persist` (loop): receive `ModelResult` → `MERGE` the fused embedding back
  to the DB.

Both loops also mirror task status onto the matching `embedding_queue` row
(`marker_views:abc` → `embedding_queue:abc`): `processing` when dispatched,
then `done` / `failed` (+ `error_message`) at the terminal states.

Queries bypass the DB: `POST /encode` pushes a `QueryJob` and blocks on a
`threading.Condition` until the worker returns the embedding.

## Image handling (CPU)

The model runs on CPU, so image size is the main cost driver.

- The model (`jinaai/jina-embeddings-v5-omni-nano-retrieval`) accepts a
  `PIL.Image` directly (also bytes, paths, URLs, ndarray) — no temp files or
  URL round-trip needed.
- Its vision processor turns each image into 256–1280 vision tokens
  (262144–1310720 px), so CPU cost scales with pixel count.
- `worker.prepare_image(data, max_pixels)` downscales (aspect ratio kept, never
  upscales) before encoding. `main._prepare_image` loads the bytes (bucket
  pointer via `_read_file_bytes`, or `http(s)` via httpx) then calls it, and the
  resulting PIL image becomes the image part of the fused model input. A
  malformed or missing image degrades to text-only — it is dropped here, never
  sent to the model.
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
- `run.py` — uvicorn entrypoint.

## Endpoints

- `POST /generate` `{"items": ["marker_views:..."]}` → `202`
- `POST /encode` `{"text": "..."}` and/or `{"image_url": "..."}` → `{"embedding": [...]}`
- `GET /health`
- `GET /tasks`, `GET /tasks/{record_id}`
- `GET /queue/stream` — SSE task feed
- `GET /files/{bucket}/{path}` — bucket file proxy

## Conventions

- **Record IDs**: parse with `RecordID.parse()` and pass the object to the SDK
  (parameterized, injection-safe). Never interpolate raw IDs into SurrealQL.
- **No regex** for validation. For the file-pointer literal only check that a
  segment can't break out of `f"..."` (no `"`, `\`, whitespace, control chars).
- **File read**: `RETURN f"{bucket}:/{path}".get()` — returns bytes.
- **Write back** with `db.merge(...)`, not `update`, so `version` and the
  computed `generated_description` field are left untouched.
- **DB**: use the async SDK; no manual `connect()` or timeouts (it self-connects
  and has its own client timeout). Connection is wrapped in `main.Connection`,
  which retries on startup and pings liveness so an unavailable DB never crashes
  the app.
- DB access stays on the async loop; model inference stays on the worker thread.
# Embeddings / VPR Service

Internal service that runs **one** model backend on a single worker thread
(resource constraint — never both at once). Selected by `MODEL_BACKEND`:

| Backend | Default | Model | Input | Output |
|---|---|---|---|---|
| `megaloc` | **yes** | [gmberton/MegaLoc](https://github.com/gmberton/MegaLoc) via `torch.hub` | image only | L2-normalized VPR descriptor (dim 8448) |
| `jina` | no | `jinaai/jina-embeddings-v5-omni-nano-retrieval` (SentenceTransformer) | text and/or image | fused embedding |

The default is MegaLoc (VPR). Jina code paths are kept and activated with
`MODEL_BACKEND=jina`.

Requires `surrealdb>=3.0.0b10` (native `File` / CBOR tag 55, builder-style
`update().merge()` / `upsert().merge()`, `db.files.get`).

Vectors are **not** stored on `marker_views`. Both backends UPSERT into the
`embeddings` table (`embedding_jina` / `embedding_megaloc`) and mirror job
status onto `embedding_queue`. Free-form VPR client ids (non-`marker_views`)
still stay in-memory only.

## Backends

### MegaLoc / VPR (`MODEL_BACKEND=megaloc`)

- Loaded with `torch.hub.load(megaloc_repo, megaloc_entry)` — raw torch /
  torchvision, **not** sentence-transformers.
- Reuses the same CUDA/CPU selection (`settings.device` / `check_cuda()`) and
  Windows below-normal process priority on CPU.
- Eval recipe: geometry → ToTensor → ImageNet normalize → forward.
- Geometry size default **322×322** (`VPR_IMAGE_SIZE`); mode default **squash**
  (`VPR_TRANSFORM`).
- Async jobs via `POST /vpr/generate`:
  - **`marker_views:…` ids** — load image from the row (or override
    `image_url`), sample `version`, UPSERT `embeddings.embedding_megaloc` +
    `data` + `model`, mirror `embedding_queue` (processing → done/failed +
    `version_snapshot`). Descriptor also kept in `ServiceState` for
    `GET /tasks/{id}`.
  - **Free-form ids** — require `image_url`; status + descriptor in-memory /
    SSE only (no queue / embeddings write).

### Jina (`MODEL_BACKEND=jina`)

- SentenceTransformer fused `(text, image)` → one vector.
- Reads `marker_views` (`generated_description` + `image_url`), UPSERTs
  `embeddings.embedding_jina` (+ `data` + `model`), mirrors status onto
  `embedding_queue` (see Data model).
- Image downscale via `prepare_image` / `IMAGE_MAX_PIXELS` /
  `DISABLE_IMAGE_TRANSFORM`.

## Data model

### `marker_views` (SCHEMAFULL) — source records (no vectors)

| Field | Type | Notes |
|---|---|---|
| `marker` | `record<markers>` | REFERENCE |
| `user_description` | `string \| none` | |
| `generated_description` | computed `string` | joins level/zone/area/shop/aliases/user_description |
| `image_url` | `file \| none` | SurrealDB v3 file pointer (bucket ref, CBOR tag 55) |
| `version` | `int` | `VALUE ($value ?? 0) + 1` — bumps on every record write |

Out-of-sync is **not** stored here. Compare `marker_views.version` with
`embedding_queue.version_snapshot`.

### `embeddings` (SCHEMAFULL) — vectors (both backends)

Row key matches the marker_views record key (`marker_views:abc` →
`embeddings:abc`). Service UPSERTs via `db.upsert(rid).merge({...})` so one
backend never wipes the other's field.

| Field | Type | Notes |
|---|---|---|
| `embedding_megaloc` | `array<float> \| none` | VPR descriptor (MegaLoc) |
| `embedding_jina` | `array<float> \| none` | fused Jina vector |
| `model` | `string` | last writer label (default hub id / Jina model id) |
| `data` | `record<marker_views>` | REFERENCE back to the view |

### `embedding_queue` (SCHEMAFULL) — job status (both backends)

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

- **Model worker thread** (`src/worker.py`) — loads **one** backend lazily in
  the background (`_load_megaloc` or `_load_jina`), then processes jobs one at
  a time. Query jobs (sync encode endpoints) are preferred over queued jobs.
- **Main async loop** (`src/main.py`) — FastAPI + SurrealDB (async SDK) + SSE.
  Lifespan starts only the loops that match the active backend:
  - megaloc: `_vpr_dispatch` + `_vpr_collect` (SSE + embeddings/queue when
    marker_views-backed)
  - jina: `_dispatch` + `_persist` (embeddings + queue + SSE)

Bridge: thread-safe `queue.Queue`s (`worker.jobs` / `worker.results` /
`worker._query_jobs`).

### SSE (`GET /queue/stream`)

Shared by both backends. Lifecycle: `queued` → `processing` → `done` | `error`
(plus `queue_done` when a busy wave drains). Events may include `"kind": "vpr"`
or `"kind": "jina"`. VPR `done` events carry `dim` (not the full vector).

On app shutdown / uvicorn reload, `broadcaster.close()` pushes a shutdown
sentinel to every subscriber. Generators also poll `broadcaster.closed` and
`request.is_disconnected()` on a 1s timeout with SSE comment heartbeats, so
reload is not blocked waiting for the client to drop the stream.

## Image handling

### Shared loading (`main._image_bytes`)

Resolves `data:` / bare base64, `http(s)`, or Surreal `File` / `bucket:/key`
the same way for every endpoint. Never logs payload bodies.

### VPR geometry (`src/preprocess.py`)

Reusable modes — all produce an exact `size×size` RGB image (default size 322):

| Mode | Env / field value | Behavior |
|---|---|---|
| **squash** | `squash` (default) | Stretch to square; ignore aspect (MegaLoc eval default) |
| **letterbox** | `letterbox` | Fit inside square, pad short axis (ImageNet-mean pad) |
| **center_crop** | `center_crop` | Scale so short side covers size, center-crop to square |

Worker then applies ImageNet normalize and runs MegaLoc. Per-request override
via `transform` on `/vpr/encode` and `/vpr/generate`. Settings:
`VPR_IMAGE_SIZE`, `VPR_TRANSFORM`.

### Jina downscale (`worker.prepare_image`)

- Aspect-preserving max-pixel cap (`IMAGE_MAX_PIXELS`, default `262144`).
- `DISABLE_IMAGE_TRANSFORM=true` passes raw bytes to the model (Jina accepts
  bytes). Not used on the VPR path — MegaLoc always needs geometry + norm.

## Files

- `src/config.py` — pydantic-settings (`Settings`, `get_settings`).
- `src/preprocess.py` — `squash` / `letterbox` / `center_crop`, `transform_image`.
- `src/state.py` — `ServiceState`; VPR tasks may hold `embedding` in memory.
- `src/worker.py` — model worker, MegaLoc + Jina loaders, `prepare_image`,
  `image_to_megaloc_tensor`, `ModelJob` / `ModelResult` / `QueryJob`.
- `src/search.py` — cosine search target registry / aggregation (Jina + VPR).
- `src/main.py` — app, endpoints, dispatch/persist/VPR loops, SSE, image loading.
- `src/utils.py` — logging, backend-aware dependency check, Windows priority.
- `run.py` — uvicorn entrypoint (`reload=True`).
- `requirements.txt` — torch/torchvision + HF hub/safetensors (MegaLoc);
  sentence-transformers (Jina); `surrealdb>=3.0.0b10`.

## Settings (env)

| Key | Default | Notes |
|---|---|---|
| `MODEL_BACKEND` | `megaloc` | `megaloc` \| `jina` |
| `MEGALOC_REPO` | `gmberton/MegaLoc` | `torch.hub` repo |
| `MEGALOC_ENTRY` | `get_trained_model` | hub entrypoint |
| `VPR_IMAGE_SIZE` | `322` | square side in px |
| `VPR_TRANSFORM` | `squash` | `squash` \| `letterbox` \| `center_crop` |
| `MODEL` | jina id | Jina only |
| `MODALITIES` | `vision` | Jina only |
| `IMAGE_MAX_PIXELS` | `262144` | Jina only |
| `DISABLE_IMAGE_TRANSFORM` | `false` | Jina only |
| `DEVICE` | auto cuda/cpu | both |
| `HF_HOME` | `D:/models/` | both |
| `QUERY_TIMEOUT` | `0` | 0 = no timeout |

## Search

`POST /search` encodes a query with the **active** `MODEL_BACKEND`, runs cosine
over a registered **target**, keeps top-k **views** above `min_score`, collapses
onto a parent. Source table is always `embeddings`; view ids in the response are
the linked `marker_views` records (`SELECT data AS id`).

Encoding / vector field follow the live backend (not a hard target lock-in):

| `MODEL_BACKEND` | Vector field | Query input |
|---|---|---|
| `jina` | `embedding_jina` | text and/or image |
| `megaloc` | `embedding_megaloc` | image only — any `text` is stripped with a **warning** |

Registered target names (`markers`, `markers_vpr`) still select filters/parent
shape; if their default field disagrees with the live backend, search swaps the
field to match.

- Parent: `data.marker` → `markers`
- `top_k`: max **views** kept (not parents)
- `min_score`: view cosine cutoff; weak views never reach parent aggregation
- Parent score: `max(sim) + α · log(1 + n_views)` (`VIEW_COUNT_ALPHA` in
  `src/search.py`, default `0.07`)
- Filter: `filters.level` → `data.marker.level`

### Extending to more tables

Register another `SearchTarget` in `src/search.py` → `TARGETS`.
`GET /search/targets` lists what is registered.

## Endpoints

### Root (dispatches by `MODEL_BACKEND`)

- `POST /encode` — sync one-shot vector
  - jina: `{"text"?, "image_url"?, "normalize"?}` → `{embedding, dim, backend}`
  - megaloc: `{"image_url", "transform"?}` → `{embedding, dim, backend, ...}`
- `POST /generate` — async batch → `202` `{accepted, count, backend, ...}`
  - jina: `{"items": ["marker_views:…"]}` → UPSERT `embedding_jina` + queue
  - megaloc: `{"items": ["marker_views:…"] | {"id", "image_url"?}], "transform"?}`
    → UPSERT `embedding_megaloc` + queue for marker_views ids; free-form ids
    stay in-memory only

### Backend-specific aliases (503 if wrong `MODEL_BACKEND`)

- `POST /jina/encode`, `POST /jina/generate` — Jina only
- `POST /vpr/encode`, `POST /vpr/generate` — MegaLoc only

`image_url` accepts `data:` / base64, `http(s)`, or `bucket:/key` (needs DB).

### Search (encoder follows live `MODEL_BACKEND`)

- `POST /search` — cosine search (see Search); text stripped+warned on megaloc
- `GET /search/targets` — registered targets + filter keys + `query_backend`

### Shared

- `GET /health` — includes `backend`, device, queue sizes, VPR or Jina fields
- `GET /tasks`, `GET /tasks/{id}` — in-memory status; VPR `done` includes
  `embedding`. `GET /tasks` lists omit full vectors (`has_embedding` + `dim`)
- `GET /queue/stream` — SSE task feed
- `GET /files/{bucket}/{path}` — bucket file proxy

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
- **Writes (vectors)**: `await db.upsert(embeddings_rid).merge({field,
  model, data})` where `field` is `embedding_jina` or `embedding_megaloc`.
  Never write vectors onto `marker_views`. Leave `generated_description`
  alone. Version snapshot goes to `embedding_queue.version_snapshot` only
  (on successful embed). Queue status via `await db.update(queue_rid).merge(...)`.
- **Writes (VPR free-form ids)**: none — descriptors stay in `ServiceState`.
- **DB**: use the async SDK; it self-connects on first RPC. Connection is
  wrapped in `main.Connection`, which retries on startup and pings liveness so
  an unavailable DB never crashes the app.
- DB access stays on the async loop; model inference stays on the worker thread.
- **One backend at a time** — do not attempt to load Jina and MegaLoc together.

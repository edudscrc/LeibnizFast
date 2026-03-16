# LeibnizFast — Claude Code Project Context

## Project Overview
GPU-accelerated 2D matrix visualization library for the browser. Renders matrices of millions to billions of pixels as interactive heatmaps with zoom, pan, and tooltip inspection.

**Tech stack**: Rust + wgpu → WASM (via wasm-pack) → npm package
**Backend**: WebGPU (primary), WebGL2 fallback (limited — no compute shaders)

## Architecture

```
JS (Float32Array) → WASM (Rust) → GPU buffers
                                  ↓
                    Compute shader: apply colormap → RGBA texture
                                  ↓
                    Render pass: textured quad with camera transform → canvas
```

- Compute shader pre-applies colormap to texture (only re-runs on data/colormap change, not on pan/zoom)
- Camera is UV-space offset/scale in fragment shader
- Events handled in JS, forwarded to Rust via wasm-bindgen
- CPU copy of matrix data for tooltip lookups (avoids async GPU readback)
- Nearest-neighbor sampling for pixel-perfect cells at high zoom

### Chunked upload flow (added for large matrix support)

Both `setData` and the streaming API use the same internal upload path:

```
MatrixView::with_empty_buffer()        ← pre-allocates full STORAGE|COPY_DST buffer
                                         (or staging buffer if data > max_buffer_size)
ChunkedUploader (auto ~16 MB chunks, 16-row aligned)
  └─ MatrixData::append_rows()         ← running min/max on CPU
  └─ MatrixView::write_chunk()         ← queue.write_buffer() at byte offset
MatrixData::finalize()                 ← handles all-NaN edge case
MatrixView::update_params()            ← writes final min/max to uniform
rebuild_pipelines() → render_frame()
```

### Staging buffer mode (for matrices > max_buffer_size)

When the full matrix exceeds the GPU's `max_buffer_size` (e.g. 1 GB on Firefox),
`MatrixView::with_empty_buffer()` automatically creates a **staging buffer** that
holds as many rows as fit within the device limit, aligned to 16 rows.

```
MatrixView::with_empty_buffer()        ← detects data_size > max_buffer_size
  └─ creates staging buffer (staging_rows × cols × 4 bytes)
rebuild_pipelines() (early)            ← sets up compute pipeline for per-chunk dispatch
for each chunk of data:
  └─ write_staging_chunk()              ← overwrites staging buffer at offset 0
  └─ update_chunk_params()              ← sets rows=chunk_rows, row_offset=current_row
  └─ dispatch compute shader            ← writes colormapped pixels to texture region
renderer.colormap_applied = true       ← render_frame skips compute pass
```

Limitations of staging mode:
- **No colormap changes** — raw data is not retained, so `setColormap()` returns an error
- **No tooltip hover** — CPU data uses range-only mode (no `Vec` storage)
- **CPU range-only** — `MatrixData::range_only()` tracks min/max without storing data,
  required because WASM's 4 GB address space can't hold the full CPU-side `Vec<f32>`

### Streaming API (for incremental / large-matrix ingestion)

`LeibnizFast.begin_data(rows, cols)` → `append_chunk(chunk, start_row)` × N → `end_data()`

- Tracked by `PendingUpload` struct on `LeibnizFast` (Rust side)
- GPU buffer pre-allocated at full size in `begin_data`; no reallocation during appends
- `start_row` parameter enforces sequential ordering (reserved for future ring-buffer waterfall)
- `end_data` errors if any rows are missing

### GPU limits
- `Renderer` stores `max_texture_dimension` (from `device.limits().max_texture_dimension_2d`)
- `rebuild_pipelines` validates rows/cols against this limit and returns a clear `Err` if exceeded
- Typical limits: 8192 on WebGPU Chrome, 16384+ on desktop WebGPU/native
- `MatrixView::with_empty_buffer()` auto-detects when data exceeds `max_buffer_size` and creates a staging buffer
- Exposed to JS as `getMaxTextureDimension()` and `getMaxMatrixElements()`

## Build & Test Commands

```bash
# Rust
cargo fmt --check          # Check formatting
cargo clippy -- -D warnings # Lint
cargo test                  # Unit tests (42 tests: camera, colormap, interaction, matrix, chunked_upload)
wasm-pack build --target web # Build WASM

# TypeScript
npx prettier --check js/   # Check formatting
npx eslint js/              # Lint

# Full build
npm run build               # WASM + JS bundle
npm run dev                 # Build + serve example
```

## Coding Style

### Rust
- Doc comments (`///`) on every public item
- Module-level docs (`//!`) at the top of each file
- 4-space indent, snake_case functions, PascalCase types, SCREAMING_SNAKE constants
- Group imports: std → external → crate-internal
- Prefer `Result`/`Option` over panics; use `?` operator

### TypeScript
- JSDoc on every export
- 2-space indent, single quotes, trailing commas (Prettier)
- Strict mode, explicit return types on public functions

### WGSL
- Comment every binding
- Explain workgroup sizes and dispatch logic
- Comment math operations

## Key Patterns

- **TDD**: Write tests before implementation for pure-logic modules
- **DI via traits**: `ColormapProvider` trait allows testing without GPU
- **Factory pattern**: `PipelineFactory` centralizes wgpu pipeline creation
- **State pattern**: `InteractionState` enum for mouse events (Idle/Dragging)
- **Pure/GPU split**: `CameraState`/`Camera`, `MatrixData`/`MatrixView` — pure math is testable, GPU wrapper adds buffers
- **Staging buffer**: `MatrixView` auto-selects between full buffer and staging buffer based on `max_buffer_size`

## File Structure

- `src/lib.rs` — wasm-bindgen entry point, public API, `PendingUpload` struct
- `src/renderer.rs` — wgpu setup, pipelines, render loop, texture dimension check
- `src/camera.rs` — CameraState (pure math) + Camera (GPU uniform)
- `src/colormap.rs` — ColormapProvider trait + ColormapTexture
- `src/colormap_data.rs` — Const 256-entry RGB tables
- `src/matrix.rs` — MatrixData (CPU, supports incremental builds + range-only mode) + MatrixView (GPU buffer, supports chunked writes + staging buffer)
- `src/chunked_upload.rs` — ChunkedUploader: pure-logic chunk boundary computation
- `src/interaction.rs` — InteractionState enum (Idle/Dragging)
- `src/pipeline.rs` — PipelineFactory for compute + render pipelines
- `src/shaders/colormap.wgsl` — Compute shader (supports row_offset for staging)
- `src/shaders/render.wgsl` — Vertex + fragment shader
- `js/index.ts` — TypeScript wrapper (includes beginData/appendChunk/endData/getMaxTextureDimension)
- `js/types.ts` — Type definitions (includes StreamingDataOptions)

## Common Tasks

### Add a new colormap
1. Add const `[[u8; 3]; 256]` to `src/colormap_data.rs`
2. Add name to `COLORMAP_NAMES` and `get_colormap_by_name` match
3. Add test in `src/colormap.rs` tests
4. Add option to `js/types.ts` `ColormapName`

### Add an interaction mode
1. Add variant to `InteractionState` enum in `src/interaction.rs`
2. Add transition logic in `mouse_down`/`mouse_move`/`mouse_up`
3. Add tests for the new transitions
4. Wire up in `src/lib.rs` event handlers

### Extend the streaming API for live/waterfall data
The `start_row` param on `append_chunk` is intentionally exposed for future out-of-order support.
To extend to a rolling window (overwrite old rows):
1. Remove the sequential-ordering assert in `append_chunk`
2. Add a `mode: "static" | "ring"` field to `PendingUpload`
3. In ring mode, skip `end_data` — allow continuous appends that overwrite rows modulo buffer height

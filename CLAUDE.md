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

### Universal staging buffer architecture

All matrix sizes use the same upload path through a **staging buffer** (≤ 64 MB).
No single GPU allocation exceeds 64 MB regardless of matrix size.

**setData flow:**
```
MatrixData::append_rows()             ← CPU-side paged storage (64 MB pages) + running min/max
MatrixData::finalize()                ← handles all-NaN edge case
MatrixView::with_empty_buffer()       ← creates staging buffer (≤ 64 MB, 16-row aligned)
rebuild_pipelines()                   ← per-tile textures, params buffers, bind groups
Renderer::apply_colormap_tiled()      ← iterates tiles → chunks → staging → compute dispatch
render_frame()                        ← draw quads (no compute — colormap already applied)
```

**Streaming flow (beginData/appendChunk/endData):**
```
begin_data()                          ← creates staging MatrixView + builds pipelines early
append_chunk() × N                    ← each chunk: append to PagedStorage + immediate compute dispatch
end_data()                            ← finalize min/max, mark colormap_applied, render
```

### PagedStorage (CPU-side data for tooltips)

CPU-side matrix data is stored in `PagedStorage` — a collection of 64 MB pages
(16M f32 elements each). This allows tooltip hover for matrices up to ~3.5 GB
without any single allocation exceeding 64 MB.

For matrices exceeding ~2 GB, `range_only` mode tracks only min/max (no tooltip).

### setColormap re-dispatch

Since CPU-side data is always retained (in PagedStorage), `setColormap()` can
re-apply the colormap at any matrix size by re-dispatching from paged CPU data
through the staging buffer. The only exception is range-only mode (>2 GB).

### Streaming API (for incremental / large-matrix ingestion)

`LeibnizFast.begin_data(rows, cols)` → `append_chunk(chunk, start_row)` × N → `end_data()`

- Tracked by `PendingUpload` struct on `LeibnizFast` (Rust side)
- Staging buffer created in `begin_data`; pipelines built early for per-chunk compute dispatch
- Each `append_chunk` immediately dispatches compute shader for the chunk's tiles
- `start_row` parameter enforces sequential ordering (reserved for future ring-buffer waterfall)
- `end_data` errors if any rows are missing

### Texture tiling (for matrices > maxTextureDimension2D)

When matrix dimensions exceed the device's `maxTextureDimension2D` (e.g. 8192 on
Chrome WebGPU), the output texture is split into a grid of tiles using `TileGrid`.
Each tile texture is at most `max_dim × max_dim` pixels.

```
TileGrid::new(rows, cols, max_dim)        ← computes tiles_x × tiles_y grid
PipelineFactory::create_tiled_textures()  ← one texture per tile
Per-tile params buffers                   ← row_offset, col_offset, texture_row_offset per tile
Per-tile compute bind groups              ← each tile has its own bind group
Per-tile camera buffers                   ← composed camera transform per tile
render_frame():
  Render pass only: draw one full-screen quad per tile
    └─ discard fragments outside tile's UV region
```

Each tile has its own `MatrixParams` buffer with `col_offset`, `row_offset`, and
`texture_row_offset` for correct data indexing and texture writes. The staging
buffer is shared across all tiles. The render shader receives a per-tile composed
camera uniform that maps screen UV → tile-local UV. Fragments outside the tile's
[0,1] region are discarded.

### GPU limits
- `Renderer` stores `max_texture_dimension` (from `device.limits().max_texture_dimension_2d`)
- Matrices exceeding this limit are automatically tiled — no hard rejection
- Typical limits: 8192 on WebGPU Chrome, 16384+ on desktop WebGPU/native
- GPU staging buffer is always ≤ 64 MB (capped by `MAX_STAGING_BYTES`), further limited by `max_buffer_size`
- No single GPU or CPU allocation exceeds 64 MB regardless of matrix size
- Exposed to JS as `getMaxTextureDimension()` and `getMaxMatrixElements()`

## Build & Test Commands

```bash
# Rust
cargo fmt --check          # Check formatting
cargo clippy -- -D warnings # Lint
cargo test                  # Unit tests (55 tests: camera, colormap, interaction, matrix, chunked_upload, tile_grid)
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
- **Universal staging**: `MatrixView` always uses a ≤64 MB staging buffer — no full-matrix GPU allocation
- **Paged CPU storage**: `PagedStorage` stores data in 64 MB pages for tooltips at any matrix size

## File Structure

- `src/lib.rs` — wasm-bindgen entry point, public API, `PendingUpload` struct
- `src/renderer.rs` — wgpu setup, pipelines, render loop, texture dimension check
- `src/camera.rs` — CameraState (pure math) + Camera (GPU uniform)
- `src/colormap.rs` — ColormapProvider trait + ColormapTexture
- `src/colormap_data.rs` — Const 256-entry RGB tables
- `src/matrix.rs` — PagedStorage (64 MB paged f32 storage), MatrixData (CPU, paged storage + range-only mode), MatrixView (GPU staging buffer ≤ 64 MB)
- `src/chunked_upload.rs` — ChunkedUploader: pure-logic chunk boundary computation
- `src/tile_grid.rs` — TileGrid: pure-logic tile layout for matrices exceeding maxTextureDimension2D
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

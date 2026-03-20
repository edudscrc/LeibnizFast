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
- Matrix data kept in JS heap (Float32Array) — no WASM memory pressure; enables tooltips and colormap at any size

### Key subsystems

**JsDataSource** (`src/matrix.rs`): wraps `js_sys::Float32Array` in JS heap. Reads via `subarray().copy_to()`, single-element via `get_index()`. Bypasses WASM 4 GB limit; tested up to 32000×32000 (~3.81 GB).

**Universal staging buffer** (`MatrixView`): all GPU uploads go through a ≤256 MB staging buffer. No single GPU allocation exceeds 256 MB regardless of matrix size.

**setData flow:**
```
JsDataSource::new()  → MatrixView::with_empty_buffer()  → rebuild_pipelines()
→ apply_colormap_tiled()  → render_frame()
```

**setColormap flow** (zero VRAM spike):
```
set_colormap_internal()   ← replaces ColormapTexture only
rebuild_compute_bind_groups()  ← recreates compute bind groups (cheap); tile textures reused
apply_colormap_tiled()    ← re-dispatches compute from JS heap
```
> Tile textures are NOT recreated on colormap change — only bind groups update. `setRange()` follows the same pattern.

**Streaming flow (initial load):**
```
begin_data()  → append_chunk() × N  → end_data()
```
Each `append_chunk` copies to JS accumulator and dispatches compute immediately.

**Streaming update flow (real-time, same dimensions):**
```
begin_update()  → append_chunk() × N  → end_data()
```
`begin_update()` reuses the existing `JsDataSource` (JS Float32Array) and `MatrixView` (GPU staging buffer) — zero allocation, zero pipeline rebuild. Falls back to `begin_data()` on first call or dimension change. `abort_data()` cancels an in-progress upload and restores resources for reuse (enables frame dropping). `render()` exposes frame rendering independently from data upload for rAF-decoupled rendering.

**Texture tiling**: matrices exceeding `maxTextureDimension2D` are split into a `TileGrid` of tiles. Each tile has its own texture, params buffer, compute bind group, camera buffer, and render bind group. Fragments outside a tile's UV region are discarded in the fragment shader.

### Performance instrumentation

**Debug flag**: `CreateOptions.debug` (JS) / `debug: Option<bool>` (WASM `create()`). When enabled, logs `[perf] label: X.XXms` to the browser console for all key operations. Zero overhead when disabled (single branch per call site).

**PerfTimer** (`src/perf.rs`): WASM-only utility using `web_sys::Performance::now()`. Instantiate with `PerfTimer::new(label, debug)`, consume with `.finish()` or `.finish_with(extra)`. The `debug` flag propagates from `LeibnizFast` → `Renderer` → `PipelineFactory`.

**JS-side**: `LeibnizFast` class stores `debug` and provides `timeSync(label, fn)` helper. The basic example has a "Debug timing" checkbox that controls JS-side logging and passes the flag to WASM on viewer creation.

### GPU limits
- Auto-tiled when matrix dims exceed `max_texture_dimension` (8192 on Chrome, 16384+ on desktop)
- Staging buffer capped at `MAX_STAGING_BYTES` (256 MB), further limited by `max_buffer_size`
- Exposed to JS: `getMaxTextureDimension()`, `getMaxMatrixElements()`

## Build & Test

```bash
cargo fmt --check && cargo clippy -- -D warnings && cargo test  # Rust checks (56 tests)
wasm-pack build --target web                                     # Build WASM
npx prettier --check js/ && npx eslint js/                       # TS checks
npm run build                                                    # Full bundle
npm run dev                                                      # Build + serve
```

## Coding Style

**Rust**: `///` on every public item, `//!` module docs, 4-space indent, snake_case/PascalCase/SCREAMING_SNAKE, `Result`/`Option` over panics.
**TypeScript**: JSDoc on exports, 2-space indent, single quotes, trailing commas, strict mode.
**WGSL**: comment every binding, workgroup size, and math operation.

## Key Patterns

- **TDD**: write tests first for pure-logic modules
- **DI via closures**: `apply_colormap_tiled` accepts `&dyn Fn` for data reading
- **Factory pattern**: `PipelineFactory` centralizes wgpu pipeline creation
- **Pure/GPU split**: `CameraState`/`Camera`, `MatrixData`/`MatrixView` — pure math is testable
- **In-place colormap**: `rebuild_compute_bind_groups()` reuses tile textures; no 2× VRAM spike
- **Zero-alloc streaming updates**: `begin_update()` reuses JsDataSource + MatrixView for same-dimension frames; `abort_data()` restores resources on frame drop

## File Structure

| File | Purpose |
|------|---------|
| `src/lib.rs` | wasm-bindgen entry point, public API, `PendingUpload` |
| `src/renderer.rs` | wgpu setup, pipelines, render loop |
| `src/camera.rs` | `CameraState` (pure) + `Camera` (GPU uniform) |
| `src/colormap.rs` | `ColormapProvider` trait + `ColormapTexture` |
| `src/colormap_data.rs` | Const 256-entry RGB tables |
| `src/matrix.rs` | `JsDataSource`, `PagedStorage`, `MatrixData`, `MatrixView` |
| `src/chunked_upload.rs` | `ChunkedUploader`: pure chunk boundary logic |
| `src/tile_grid.rs` | `TileGrid`: pure tile layout |
| `src/interaction.rs` | `InteractionState` (Idle/Dragging) |
| `src/perf.rs` | `PerfTimer`: debug-gated performance timing (WASM-only) |
| `src/pipeline.rs` | `PipelineFactory` |
| `src/shaders/colormap.wgsl` | Compute shader |
| `src/shaders/render.wgsl` | Vertex + fragment shader |
| `js/index.ts` | TypeScript wrapper |
| `js/types.ts` | Type definitions |
| `examples/waterfall/generator.cpp` | C++ DAS data generator → ZMQ PUSH (waterfall protocol) |
| `examples/waterfall/bridge.py` | ZMQ PULL → WebSocket broadcast bridge (copy of cpp-stream) |
| `examples/waterfall/main.js` | WebSocket waterfall client: FIFO buffer + `setData` fast path |
| `examples/waterfall/index.html` | DAS waterfall example page with controls |

## Common Tasks

### Add a new colormap
1. Add `[[u8; 3]; 256]` const to `src/colormap_data.rs`
2. Add to `COLORMAP_NAMES` and `get_colormap_by_name` match
3. Add test in `src/colormap.rs`
4. Add to `ColormapName` in `js/types.ts`

### Add an interaction mode
1. Add variant to `InteractionState` in `src/interaction.rs`
2. Add transitions in `mouse_down`/`mouse_move`/`mouse_up`
3. Write tests for all new transitions
4. Wire up in `src/lib.rs` event handlers

### Real-time streaming (same-dimension updates)
Use `beginUpdate` → `appendChunk` × N → `endData` instead of `setData` for real-time feeds.
`beginUpdate` reuses GPU resources (zero allocation); `appendChunk` dispatches compute per-chunk
(no full-matrix staging). See `examples/cpp-stream/main.js` for the complete pattern including
frame dropping via `abortData`.

### Waterfall / FIFO streaming (right-to-left scroll)
Use `setData` fast path with a pre-allocated `Float32Array(rows × displayCols)`. On each new batch:
per-row `copyWithin` shifts left, new columns written at right edge. Render decoupled via `requestAnimationFrame`.
See `examples/waterfall/main.js` for the complete pattern including the `WaterfallBuffer` class.
Generator rate is controlled by `--sampling-rate` (MHz) and `--downsample` (factor):
`output_rate = sampling_rate × 4 / downsample` MB/s.

### Extend streaming API for waterfall/ring-buffer
`start_row` on `append_chunk` is reserved for out-of-order support. To add rolling window:
1. Remove sequential assert in `append_chunk`
2. Add `mode: "static" | "ring"` to `PendingUpload`
3. In ring mode, skip `end_data` and overwrite rows modulo buffer height

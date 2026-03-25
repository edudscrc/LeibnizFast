# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LeibnizFast is a GPU-accelerated 2D matrix visualization library for browsers, published as npm package `leibniz-fast`. It combines a Rust/WASM core (wgpu) with a thin TypeScript wrapper. WebGPU is the primary backend; WebGL2 is the fallback.

## Build & Dev Commands

```bash
npm run build:wasm    # wasm-pack build --target web --release → pkg/
npm run build:js      # tsup js/index.ts --format esm --dts → dist/
npm run build         # both WASM + JS
npm run dev           # build + serve at localhost:8080/examples/basic/
```

## Test & Lint

```bash
npm run test:rs       # cargo test (unit tests across Rust modules)
cargo test <test_name>  # run a single Rust test
npm run lint:rs       # cargo fmt --check && cargo clippy -- -D warnings
npm run lint:ts       # prettier --check js/ && eslint js/
npm run lint          # all linting (Rust + TypeScript)
```

## Architecture

### Data Flow

```
JavaScript Float32Array (column-major ring buffer)
  → TypeScript LeibnizFast class (js/index.ts) — event handling, WASM init
    → WASM exports (src/lib.rs, wasm_entry module)
      → Rust core: MatrixView (GPU buffer), ring cursor tracking
        → GPU compute shader (shaders/colormap.wgsl): normalize + colormap → RGBA texture
          → GPU render shader (shaders/render.wgsl): ring unwrap + camera transform → canvas
```

### Rust Modules

**Pure logic (testable natively, no GPU):** `camera`, `chunked_upload`, `colormap`, `colormap_data`, `interaction`, `matrix`, `tile_grid`

**GPU/WASM-only (cfg=wasm32):** `perf`, `pipeline`, `renderer`

Key entry point: `src/lib.rs` contains the `wasm_entry` module with the main `LeibnizFast` struct exported via `#[wasm_bindgen]`.

### TypeScript Layer

`js/index.ts` — thin wrapper that lazy-loads WASM, wraps the WASM instance with a typed API, manages DOM event listeners, and provides callbacks (`onCreate`, `onHover`).

`js/types.ts` — public API types (`ColormapName`, `CreateOptions`, `ChartConfig`, etc.).

`js/axes.ts` — 2D canvas overlay renderer for axes, tick labels, and chart title.

### GPU Resource Patterns

- **Ring buffer streaming:** The primary path for real-time data. The RGBA tile texture is treated as a circular buffer with a `ring_cursor` write index. New columns are written at the cursor; the render shader unwraps the ring visually. Per-frame cost is O(rows × new_cols), independent of total window size — increasing the time window does **not** slow streaming.
- **Ring offset in full-matrix UV space:** `ring_offset = ring_cursor / total_cols` is computed once and kept in full-matrix UV space. The fragment shader applies `fract(full_x + ring_offset)` *before* mapping to tile-local UV. This is critical for correct multi-tile behavior — applying the offset in tile-local space would produce a different pixel shift per tile width, causing ghost images.
- **Column-major JS buffer:** The JS-side ring buffer stores data column-major (`data[col * rows + row]`). New columns from the wire are a direct `TypedArray.set()` memcpy into the ring position — no row-major transpose or O(cols) shift.
- **Compute shader column-major read:** When `col_major = 1` in `MatrixParams`, the colormap compute shader indexes staging data as `col * col_stride + row` instead of the row-major path. This avoids a transpose on the CPU.
- **Chunked upload:** Large matrices split into ~16MB chunks, 16-row aligned (`chunked_upload.rs`).
- **Tiling:** Matrices exceeding `maxTextureDimension2D` are split into a grid of smaller textures (`tile_grid.rs`). Tile textures carry `COPY_SRC | COPY_DST` usage flags.
- **Staging path:** If data exceeds `max_buffer_size`, the compute shader runs per-chunk instead of all-at-once.
- **Camera-only updates:** Pan/zoom only updates a uniform buffer; compute shader does not re-run.
- **Ring cursor desync guard:** `set_data` (full re-render) always calls `reset_ring_cursor()` so the next `setDataScrolled` frame starts at cursor 0, matching the freshly allocated JS `WaterfallBuffer`.

### Render Shader UV Pipeline (render.wgsl)

The fragment shader has two distinct axes:

```
X (ring-aware, full-matrix space):
  full_x = uv_x_offset + uv.x * uv_x_scale   // camera pan/zoom in full-matrix UV
  full_x = fract(full_x + ring_offset)         // ring unwrap (if ring active)
  tile_x = (full_x - tile_x_offset) / tile_x_size  // → tile-local UV, discard if outside

Y (no ring, tile-local space):
  tile_y = uv_y_offset + uv.y * uv_y_scale    // pre-composed on CPU, already tile-local
```

Y is pre-composed on the CPU to avoid extra math per fragment. X stays in full-matrix space until after the ring fract so the same `ring_offset` is correct for every tile.

### WebGL2 Fallback Limitations

No hover tooltips; colormap changes require full data reload; colormapping is CPU-side; ring buffer streaming not available (falls back to full `setData`).

## Toolchain

- **Rust:** Stable channel, `wasm32-unknown-unknown` target (see `rust-toolchain.toml`)
- **wasm-pack:** Compiles Rust → WASM, outputs to `pkg/`
- **tsup:** Bundles TypeScript (ESM), outputs to `dist/`
- **Prettier:** 80 cols, semicolons, 2-space indent (`.prettierrc`)
- **ESLint:** TypeScript-ESLint (`eslint.config.js`)

## Key Dependencies

- `wgpu 24` — WebGPU abstraction for Rust
- `wasm-bindgen` + `web-sys` — Rust↔JS interop
- `bytemuck` — zero-copy type casting for GPU buffers

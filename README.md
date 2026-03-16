# LeibnizFast

GPU-accelerated 2D matrix visualization for the browser.

Render matrices of millions to billions of pixels as interactive heatmaps with smooth zoom, pan, and cell-level tooltip inspection — powered by WebGPU via Rust and WASM.

## Features

- **GPU-accelerated rendering** — compute shader applies colormaps, fragment shader handles zoom/pan
- **Chunked data loading** — large matrices are uploaded to the GPU in ~16 MB slices, avoiding single-shot allocation failures
- **Staging buffer** — matrices exceeding the GPU's `max_buffer_size` (e.g. 1 GB on Firefox) are processed through a smaller staging buffer with chunked compute shader dispatch
- **Streaming API** — `beginData` / `appendChunk` / `endData` lets the caller feed data row-by-row without ever allocating the full matrix in JavaScript
- **Interactive** — scroll to zoom (cursor-anchored), drag to pan, hover for cell values
- **Multiple colormaps** — viridis, inferno, magma, plasma, cividis, grayscale
- **Device-aware limits** — queries and enforces GPU texture dimension and buffer size limits with clear error messages
- **Pixel-perfect** — nearest-neighbor sampling shows individual cells at high zoom
- **TypeScript API** — clean typed wrapper over Rust/WASM core

## Browser Requirements

WebGPU support required for the compute shader path:
- Chrome 113+ / Edge 113+
- Firefox Nightly (behind flag)
- Safari Technology Preview

Falls back to WebGL2 (CPU colormap + texture upload). The streaming API works on both paths.

## GPU Limits

The maximum renderable matrix size depends on the GPU:

| Limit | Typical value | What it means |
|-------|--------------|---------------|
| `max_texture_dimension_2d` | 8 192 (WebGPU Chrome) – 32 768 (desktop) | Max rows **and** cols |
| `max_buffer_size` | 256 MB – 2 GB | Max raw data in a single GPU buffer |

When the matrix exceeds `max_buffer_size`, a **staging buffer** is used automatically. Data is processed chunk-by-chunk through the compute shader. This allows rendering matrices up to `max_texture_dimension²` (e.g. 32 000×32 000 on capable GPUs) regardless of the buffer size limit.

Call `viewer.getMaxTextureDimension()` and `viewer.getMaxMatrixElements()` after `create()` to query the current device. The example app automatically disables size options that exceed the texture limit.

## Installation

```bash
npm install leibniz-fast
```

## Quick Start

```typescript
import { LeibnizFast } from 'leibniz-fast';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const viewer = await LeibnizFast.create(canvas, { colormap: 'viridis' });

// Standard path — one allocation, best for matrices up to ~8192×8192
const data = new Float32Array(1000 * 2000);
// ... fill data ...
viewer.setData(data, { rows: 1000, cols: 2000 });

// Change colormap
viewer.setColormap('inferno');

// Set custom data range
viewer.setRange(0.0, 1.0);

// Hover tooltip
viewer.onHover((row, col, value) => {
  console.log(`[${row}, ${col}] = ${value}`);
});

// Cleanup
viewer.destroy();
```

### Streaming API

Use `beginData` / `appendChunk` / `endData` when the full matrix is too large to hold in a single JavaScript `Float32Array` (roughly > 2 GB = ~23 000×23 000), or when data arrives incrementally (e.g. live waterfall feeds).

```typescript
const rows = 8000, cols = 8000;
const chunkRows = 1000; // generate/upload 1000 rows at a time

viewer.beginData({ rows, cols });

for (let startRow = 0; startRow < rows; startRow += chunkRows) {
  const endRow = Math.min(startRow + chunkRows, rows);
  const chunk = computeRows(startRow, endRow, cols); // Float32Array
  viewer.appendChunk(chunk, startRow);
}

viewer.endData(); // finalises min/max, rebuilds pipelines, renders
```

Key properties:
- Peak JS memory ∘ one chunk, not the full matrix
- GPU buffer is pre-allocated at full size in `beginData` (or as a staging buffer if data exceeds `max_buffer_size`)
- For staged matrices, each chunk is immediately processed through the compute shader
- `appendChunk` validates sequential ordering; out-of-order calls return an error
- `endData` errors if not all rows have been uploaded

> **Note:** Staged matrices (exceeding `max_buffer_size`) have two limitations:
> tooltip hover is unavailable, and colormap changes require reloading the data.

## API Reference

### `LeibnizFast.create(canvas, options?)`
Async factory. Initialises WebGPU/WebGL2 and returns a viewer instance.
- `options.colormap`: `ColormapName` (default `'viridis'`)

### `viewer.setData(data, options)`
Set matrix data in one shot.
- `data`: `Float32Array` row-major
- `options.rows`, `options.cols`: matrix dimensions

Internally uploads in ~16 MB chunks. Uses a staging buffer if data exceeds `max_buffer_size`. Errors if dimensions exceed `max_texture_dimension_2d`.

### `viewer.beginData(options)`
Begin a streaming upload. Allocates GPU buffer (full or staging-sized).
- `options.rows`, `options.cols`: final matrix dimensions

### `viewer.appendChunk(data, startRow)`
Append rows to an in-progress upload.
- `data`: `Float32Array` containing a whole number of rows
- `startRow`: zero-based starting row index (must be sequential)

### `viewer.endData()`
Finalise a streaming upload. Errors if not all rows have been appended.

### `viewer.setColormap(name)`
Change colormap. Options: `'viridis'` `'inferno'` `'magma'` `'plasma'` `'cividis'` `'grayscale'`.

Errors if the current matrix was loaded via staged upload (reload data after changing colormap).

### `viewer.setRange(min, max)`
Override auto-detected data range.

### `viewer.onHover(callback)`
Register `(row, col, value) => void` for hover events.

### `viewer.getMaxTextureDimension(): number`
Maximum rows or cols this device supports. Matrices exceeding this fail at pipeline build time.

### `viewer.getMaxMatrixElements(): number`
Maximum total elements (rows × cols) fitting in a single GPU buffer.

### `viewer.destroy()`
Clean up GPU resources and event listeners.

## Available Colormaps

| Name | Description |
|------|-------------|
| `viridis` | Perceptually uniform, colorblind-friendly (purple → teal → yellow) |
| `inferno` | Dark to bright (black → purple → orange → yellow) |
| `magma` | Dark to bright (black → purple → pink → light yellow) |
| `plasma` | Blue → purple → orange → yellow |
| `cividis` | Optimised for colour vision deficiency (blue → gray-green → yellow) |
| `grayscale` | Simple black to white |

## Development Setup

### 1. Install Prerequisites

**Rust** (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**WASM target + wasm-pack**:
```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

**Node.js 18+** — install via your package manager or [nvm](https://github.com/nvm-sh/nvm).

### 2. Install Node Dependencies

```bash
npm install
```

### 3. Build & Run the Example

```bash
# Build the WASM package
wasm-pack build --target web --release

# Serve from repo root (so ../../pkg/ resolves correctly)
npx http-server . -p 8080 -o /examples/basic/
```

Open `http://localhost:8080/examples/basic/` in a WebGPU-capable browser (Chrome 113+).

### 4. Build Commands

```bash
npm run build          # Full build (WASM + JS bundle)
npm run build:wasm     # WASM only
npm run build:js       # JS/TS only
npm run dev            # Build + serve example at localhost:8080
```

### 5. Test & Lint

```bash
cargo test                        # 42 Rust unit tests
cargo fmt --check                 # Rust formatting
cargo clippy -- -D warnings       # Rust linting
npx prettier --check js/          # TypeScript formatting
npx eslint js/                    # TypeScript linting
```

## Architecture

```
JS (Float32Array) → WASM (Rust) → GPU buffers
                                  ↓
                    Compute shader: apply colormap → RGBA texture
                                  ↓
                    Render pass: textured quad with camera transform → canvas
```

### Chunked upload flow

```
setData(data, rows, cols)
  └─ MatrixView::with_empty_buffer()   ← pre-allocates GPU buffer (full or staging)
  └─ ChunkedUploader (auto ~16 MB chunks, 16-row aligned)
       └─ MatrixData::append_rows()    ← updates running min/max on CPU
       └─ MatrixView::write_chunk()    ← queue.write_buffer() at byte offset
  └─ [staging: apply_colormap_staged() ← per-chunk compute dispatch]
  └─ MatrixData::finalize()            ← NaN edge-case handling
  └─ MatrixView::update_params()       ← final min/max → params uniform
  └─ rebuild_pipelines()
  └─ render_frame()
```

The compute shader only re-runs when data or colormap changes. Pan/zoom only updates a camera uniform in the fragment shader, making viewport changes nearly free.

## License

MIT

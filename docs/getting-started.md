# Getting Started

## Prerequisites

- **Node.js 18+** for bundling your application
- **WebGPU-capable browser** served from HTTPS or localhost. LeibnizFast has no CPU/WebGL fallback.

## Installation

```bash
npm install leibniz-fast
```

## Minimal Example

The following creates a 512×1024 heatmap on a canvas, loads a sine-wave pattern, and prints hover information to the console.

```ts
import { LeibnizFast } from 'leibniz-fast';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;

// 1. Optional support probe for friendly errors.
const support = await LeibnizFast.checkSupport();
if (!support.supported) {
  throw new Error(support.reason);
}

// 2. Initialize — lazy-loads WASM and acquires the WebGPU context.
const viewer = await LeibnizFast.create(canvas, {
  colormap: 'viridis',
});

// 3. Build a Float32Array in row-major order.
const rows = 512;
const cols = 1024;
const data = new Float32Array(rows * cols);
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    data[r * cols + c] = Math.sin((r + c) / 50);
  }
}

// 4. Upload and render.
viewer.setData(data, { rows, cols });

// 5. Register a hover callback.
viewer.onHover((info) => {
  console.log(`[${info.row}, ${info.col}] = ${info.value.toFixed(4)}`);
});

// 6. Clean up when the component unmounts.
// viewer.destroy();
```

## What Happens Under the Hood

On the first `LeibnizFast.create()` call, the library fetches and compiles the WASM module. Subsequent calls reuse the cached module, so initialization is fast after the first viewer. The GPU context is tied to the `<canvas>` element passed in — one viewer per canvas.

`setData()` uploads the Float32Array through a bounded GPU staging buffer into tiled `R32Float` textures and renders the result. The fragment shader applies the active colormap/range. Pan and zoom are handled automatically via pointer and wheel events registered on the canvas.

## Next Steps

| Topic | Description |
|---|---|
| [Initialization](/guide/initialization) | `CreateOptions`, debug mode, device limits, and lifecycle |
| [Static Data](/guide/static-data) | `setData()`, chunked uploads for large matrices |
| [Chart Customization](/guide/chart-customization) | Axes, colormaps, color range, hover tooltips |
| [Streaming: Full Frame](/guide/streaming-full-frame) | rAF-decoupled streaming where each frame replaces the full matrix |
| [Streaming: Waterfall](/guide/streaming-waterfall) | Scrolling ring-buffer waterfall for time-series data |

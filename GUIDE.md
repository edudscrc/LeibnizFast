# LeibnizFast Usage Guide

A step-by-step guide to using LeibnizFast for GPU-accelerated 2D matrix visualization in the browser.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
  - [1. Set Up the HTML](#1-set-up-the-html)
  - [2. Create a Viewer](#2-create-a-viewer)
  - [3. Prepare and Load Matrix Data](#3-prepare-and-load-matrix-data)
  - [4. Add Interactivity](#4-add-interactivity)
  - [5. Clean Up](#5-clean-up)
- [Using the Raw WASM API](#using-the-raw-wasm-api)
- [Working with Large Matrices](#working-with-large-matrices)
- [Colormaps](#colormaps)
- [Custom Data Ranges](#custom-data-ranges)
- [Responding to User Interaction](#responding-to-user-interaction)
- [Resizing the Canvas](#resizing-the-canvas)
- [WebGPU vs WebGL2 Fallback](#webgpu-vs-webgl2-fallback)
- [Full Example](#full-example)

## Overview

LeibnizFast renders a flat `Float32Array` as a colored heatmap on an HTML `<canvas>`. The rendering pipeline runs on the GPU via WebGPU (with a WebGL2 fallback), so matrices with millions of cells remain interactive.

The typical workflow is:

1. Create a canvas element.
2. Initialize a `LeibnizFast` viewer on that canvas.
3. Pass a `Float32Array` plus row/column dimensions.
4. Register event handlers for zoom, pan, and hover.

## Installation

```bash
npm install leibniz-fast
```

## Basic Usage

### 1. Set Up the HTML

Add a `<canvas>` element where the heatmap will render:

```html
<canvas id="canvas" width="800" height="600"></canvas>
```

The `width` and `height` attributes set the rendering resolution. Use CSS to control the display size.

### 2. Create a Viewer

Import `LeibnizFast` and call the async `create()` factory:

```typescript
import { LeibnizFast } from 'leibniz-fast';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const viewer = await LeibnizFast.create(canvas, { colormap: 'viridis' });
```

`create()` initializes the WASM module (on first call), requests a GPU device, and configures the rendering pipeline. The `colormap` option is optional and defaults to `'viridis'`.

### 3. Prepare and Load Matrix Data

LeibnizFast expects a `Float32Array` in **row-major order** — row 0 first, then row 1, etc.

```typescript
const rows = 1000;
const cols = 2000;
const data = new Float32Array(rows * cols);

// Fill the array — index for cell (r, c) is: r * cols + c
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    data[r * cols + c] = Math.sin(c / cols * 10) * Math.cos(r / rows * 10);
  }
}

// Send data to the viewer
viewer.setData(data, { rows, cols });
```

After `setData`, the heatmap renders immediately. The colormap is auto-scaled to the data's min/max values.

### 4. Add Interactivity

The TypeScript wrapper (`LeibnizFast` class from `leibniz-fast`) automatically registers canvas event listeners for:

- **Scroll to zoom** — cursor-anchored, scroll up to zoom in
- **Drag to pan** — click and drag to move the viewport
- **Hover** — moving the mouse triggers the hover callback

To display cell values on hover:

```typescript
viewer.onHover((row, col, value) => {
  console.log(`Cell [${row}, ${col}] = ${value}`);
});
```

### 5. Clean Up

When the viewer is no longer needed, call `destroy()` to release GPU resources and remove event listeners:

```typescript
viewer.destroy();
```

## Using the Raw WASM API

If you import the WASM bindings directly (instead of the TypeScript wrapper), you must register event handlers yourself. This is how the `examples/basic/` demo works:

```javascript
import init, { LeibnizFast } from './pkg/leibniz_fast.js';

await init();
const viewer = await LeibnizFast.create(canvas, 'viridis');

// Set data — raw WASM API takes (data, rows, cols) directly
viewer.setData(data, rows, cols);

// You must wire up DOM events manually:
canvas.addEventListener('mousedown', (e) => {
  const rect = canvas.getBoundingClientRect();
  viewer.onMouseDown(e.clientX - rect.left, e.clientY - rect.top);
});

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  viewer.onMouseMove(e.clientX - rect.left, e.clientY - rect.top);
});

window.addEventListener('mouseup', () => {
  viewer.onMouseUp();
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  viewer.onWheel(e.clientX - rect.left, e.clientY - rect.top, -e.deltaY);
}, { passive: false });

// Hover callback
viewer.onHover((row, col, value) => {
  console.log(`[${row}, ${col}] = ${value}`);
});
```

Key differences from the TypeScript wrapper:

| | TypeScript Wrapper | Raw WASM API |
|---|---|---|
| Import | `import { LeibnizFast } from 'leibniz-fast'` | `import init, { LeibnizFast } from './pkg/leibniz_fast.js'` |
| Init | Automatic on first `create()` | Must call `await init()` first |
| `create()` options | `{ colormap: 'viridis' }` | `'viridis'` (string directly) |
| `setData()` | `(data, { rows, cols })` | `(data, rows, cols)` |
| Event handlers | Automatic | Manual (see above) |

## Working with Large Matrices

LeibnizFast is designed for large matrices. A few things to keep in mind:

**Memory**: A `Float32Array` uses 4 bytes per cell. A 31600x31600 matrix is ~3.7 GB. Make sure the browser tab has enough memory. Chrome's V8 heap limit can be an issue for very large allocations.

**GPU texture limits**: Most GPUs support textures up to 16384x16384. For matrices larger than this, the library may need to tile internally. Check your GPU's `maxTextureDimension2D` limit.

**Data generation**: For very large matrices, generate data in chunks to avoid blocking the main thread:

```typescript
const rows = 31600;
const cols = 31600;
const data = new Float32Array(rows * cols);

// Fill in chunks to avoid long blocking
const chunkSize = 1000;
for (let startRow = 0; startRow < rows; startRow += chunkSize) {
  const endRow = Math.min(startRow + chunkSize, rows);
  for (let r = startRow; r < endRow; r++) {
    for (let c = 0; c < cols; c++) {
      data[r * cols + c] = /* your value */;
    }
  }
}

viewer.setData(data, { rows, cols });
```

## Colormaps

Six built-in colormaps are available:

| Name | Description |
|------|-------------|
| `viridis` | Perceptually uniform, colorblind-friendly (purple to teal to yellow) |
| `inferno` | Dark to bright (black to purple to orange to yellow) |
| `magma` | Dark to bright (black to purple to pink to light yellow) |
| `plasma` | Blue to purple to orange to yellow |
| `cividis` | Optimized for color vision deficiency (blue to gray-green to yellow) |
| `grayscale` | Simple black to white |

Switch colormaps at any time — the change is applied instantly without reloading data:

```typescript
viewer.setColormap('inferno');
```

## Custom Data Ranges

By default, the colormap maps across the full data range (min to max). Override this to focus on a specific range:

```typescript
// Map colormap from 0.0 to 1.0 — values outside are clamped
viewer.setRange(0.0, 1.0);
```

This is useful when:
- Comparing multiple matrices with different ranges on a common scale
- Filtering out outliers that compress the color range
- Focusing on a specific value range of interest

## Responding to User Interaction

The `onHover` callback receives the row index, column index, and raw data value at the cursor position:

```typescript
viewer.onHover((row, col, value) => {
  // Update a tooltip, status bar, or side panel
  document.getElementById('info').textContent =
    `Row: ${row}, Col: ${col}, Value: ${value.toFixed(6)}`;
});
```

The row/col values are zero-based indices into the original matrix. The value is the original `Float32Array` value (not the colormapped color).

## Resizing the Canvas

If using the TypeScript wrapper, the canvas automatically resizes on window resize events (accounting for device pixel ratio).

With the raw WASM API, call `resize()` manually:

```javascript
window.addEventListener('resize', () => {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  viewer.resize(canvas.width, canvas.height);
});
```

## WebGPU vs WebGL2 Fallback

LeibnizFast uses WebGPU when available for maximum performance (compute shaders apply the colormap on the GPU). When WebGPU is unavailable, it falls back to WebGL2 where the colormap is applied on the CPU.

**WebGPU requires a secure context**. This means it only works over:
- `https://` connections
- `localhost` or `127.0.0.1`

If you access your dev server via a LAN IP (e.g., `http://192.168.1.2:8080`), the browser will not expose WebGPU and the library falls back to WebGL2. To use WebGPU over LAN:

- Serve over HTTPS (a self-signed certificate works for development)
- In Chrome: go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add your origin, and restart

You can check which path is active by looking at the browser console — the library logs whether compute shaders are available at initialization.

## Full Example

A complete standalone HTML page:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>LeibnizFast Demo</title>
  <style>
    body { margin: 0; background: #1a1a2e; }
    canvas { display: block; margin: 20px auto; border: 1px solid #333; cursor: crosshair; }
    #tooltip {
      position: fixed; padding: 6px 10px; background: rgba(0,0,0,0.85);
      color: #fff; border-radius: 4px; font: 0.85rem monospace;
      pointer-events: none; display: none;
    }
  </style>
</head>
<body>
  <canvas id="canvas" width="800" height="600"></canvas>
  <div id="tooltip"></div>

  <script type="module">
    import { LeibnizFast } from 'leibniz-fast';

    const canvas = document.getElementById('canvas');
    const tooltip = document.getElementById('tooltip');

    // Create viewer
    const viewer = await LeibnizFast.create(canvas, { colormap: 'viridis' });

    // Generate a 1000x1000 sine-wave matrix
    const rows = 1000, cols = 1000;
    const data = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        data[r * cols + c] =
          Math.sin(c / cols * 20) * Math.cos(r / rows * 20);
      }
    }
    viewer.setData(data, { rows, cols });

    // Tooltip on hover
    viewer.onHover((row, col, value) => {
      tooltip.style.display = 'block';
      tooltip.textContent = `[${row}, ${col}] = ${value.toFixed(4)}`;
    });
    canvas.addEventListener('mousemove', (e) => {
      tooltip.style.left = `${e.clientX + 12}px`;
      tooltip.style.top = `${e.clientY + 12}px`;
    });
    canvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });
  </script>
</body>
</html>
```

This uses the TypeScript wrapper, so zoom, pan, and hover work out of the box with no extra event wiring.

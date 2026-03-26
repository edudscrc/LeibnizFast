---
layout: home

hero:
  name: LeibnizFast
  text: GPU-Accelerated Matrix Visualization
  tagline: WebGPU-native heatmaps with real-time streaming, interactive pan/zoom, and chart overlays. Rust/WASM core with a TypeScript API.
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: API Reference
      link: /api/leibniz-fast

features:
  - title: WebGPU Powered
    details: Rust/WASM core via wgpu renders millions of cells at interactive framerates. Falls back gracefully to WebGL2 for broader browser compatibility.
  - title: Real-Time Streaming
    details: Ring buffer waterfall pattern delivers O(rows × newCols) GPU cost per frame — independent of display window width. Ingest at network rate, render at display rate.
  - title: Chart Overlays
    details: 2D Canvas overlay renders axes, tick marks, axis labels, and a chart title on top of the GPU canvas. Hover callback delivers interpolated axis coordinates per cell.
---

## Installation

```bash
npm install leibniz-fast
```

## Browser Support

| Browser | Version | Notes |
|---|---|---|
| Chrome / Edge | 113+ | Full WebGPU support |
| Firefox | Nightly | Requires `dom.webgpu.enabled` flag |
| Safari | Technology Preview | Full WebGPU support |
| All others | Any | WebGL2 fallback (no hover, no ring streaming) |

::: info WebGL2 Fallback
When WebGPU is unavailable, LeibnizFast automatically falls back to a WebGL2 renderer. The fallback does not support hover tooltips, ring buffer streaming, or live colormap changes without a full data reload.
:::

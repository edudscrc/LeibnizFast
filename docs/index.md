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
    details: Rust/WASM core via wgpu renders millions of cells at interactive framerates. WebGPU is required; missing adapter/device support produces a clear error.
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
| Chrome / Edge | 113+ on supported platforms | WebGPU enabled by default on major desktop targets |
| Firefox | 141+ on Windows | Other platforms are still rolling out |
| Safari | 26+ | macOS/iOS/iPadOS/visionOS support |

::: warning WebGPU Required
LeibnizFast does not provide CPU or WebGL2 fallback rendering. Serve from HTTPS or localhost and use `LeibnizFast.checkSupport()` to show a friendly message when `navigator.gpu`, adapter creation, or device creation fails.
:::

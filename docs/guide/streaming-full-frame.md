# Streaming: Full Frame

Use this pattern when each new data event replaces the entire matrix — for example, a physics simulation frame, a radar sweep, or any source that sends complete snapshots.

## The rAF Decoupling Pattern

Network data arrives at an unpredictable or variable rate. Calling `setData()` directly inside a WebSocket `message` handler blocks the GPU work on the network thread's schedule, which causes jank when frames arrive faster than 60 Hz and wastes GPU work when they arrive slower.

The solution is to decouple ingestion from rendering:

1. The **network callback** writes incoming data into a pre-allocated `Float32Array` and sets a `dirty` flag.
2. The **`requestAnimationFrame` loop** checks `dirty`, calls `setData()` once per display frame, and clears the flag.

```ts
import { LeibnizFast } from 'leibniz-fast';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const viewer = await LeibnizFast.create(canvas, {
  colormap: 'inferno',
  chart: {
    title: 'Wave Simulation',
    xAxis: { label: 'X', unit: 'm', min: 0, max: 10 },
    yAxis: { label: 'Y', unit: 'm', min: 0, max: 10 },
    valueUnit: 'm/s',
  },
});

let rows = 0;
let cols = 0;
let frameBuffer: Float32Array | null = null;
let dirty = false;

const ws = new WebSocket('ws://localhost:8765');
ws.binaryType = 'arraybuffer';

ws.addEventListener('message', (event) => {
  // Parse your binary protocol header here to extract rows/cols.
  const view = new DataView(event.data as ArrayBuffer);
  rows = view.getUint32(0, true);
  cols = view.getUint32(4, true);
  const payload = new Float32Array(event.data, 8);

  if (!frameBuffer || frameBuffer.length !== rows * cols) {
    frameBuffer = new Float32Array(rows * cols);
  }
  frameBuffer.set(payload);
  dirty = true;
});

function renderLoop() {
  if (dirty && frameBuffer) {
    viewer.setData(frameBuffer, { rows, cols });
    dirty = false;
  }
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);
```

::: tip render() for even more control
If you want to decouple rendering from data ingestion without calling `setData()` every frame (for example, when the source is faster than 60 Hz and you want to render the _latest_ complete frame), call `viewer.render()` in the rAF loop instead. `render()` redraws the current GPU texture without re-uploading any data.
:::

## Chunked Streaming for Large Frames

When each frame exceeds ~16 MB, avoid allocating a single large `Float32Array`. Use `beginUpdate()` instead of `beginData()` to reuse GPU staging buffers across frames:

```ts
// Called once per frame inside the rAF loop or network handler.
async function uploadFrame(chunks: AsyncIterable<{ data: Float32Array; startRow: number }>) {
  viewer.beginUpdate({ rows, cols }); // reuses buffers if dims unchanged
  try {
    for await (const chunk of chunks) {
      viewer.appendChunk(chunk.data, chunk.startRow);
    }
    viewer.endData();
  } catch {
    viewer.abortData();
  }
}
```

### `beginUpdate()` vs `beginData()`

| Method | When to use |
|---|---|
| `beginData({ rows, cols })` | First upload, or when matrix dimensions change |
| `beginUpdate({ rows, cols })` | Subsequent frames with the same dimensions — reuses the GPU staging buffer and avoids a pipeline rebuild |

`beginUpdate()` automatically falls back to `beginData()` if the dimensions differ from the previous frame.

## Frame Dropping

When data arrives faster than the display rate, keep only the most recent complete frame. The pattern below discards any in-progress frame assembly when a newer frame header arrives:

```ts
let expectedFrameId = -1;
let accumBuffer: Float32Array | null = null;
let accumReceived = 0;
let totalChunks = 0;

ws.addEventListener('message', (event) => {
  const view = new DataView(event.data as ArrayBuffer);
  const frameId = view.getUint32(8, true);
  const chunkIndex = view.getUint32(12, true);
  const totalChunksInFrame = view.getUint32(16, true);

  if (frameId !== expectedFrameId) {
    // New frame — discard incomplete previous frame.
    expectedFrameId = frameId;
    accumReceived = 0;
    totalChunks = totalChunksInFrame;
    accumBuffer = new Float32Array(rows * cols);
  }

  const chunkData = new Float32Array(event.data, HEADER_BYTES);
  const rowStart = view.getUint32(20, true);
  accumBuffer!.set(chunkData, rowStart * cols);
  accumReceived++;

  if (accumReceived === totalChunks) {
    frameBuffer = accumBuffer;
    dirty = true;
  }
});
```

## See Also

- [API: setData()](/api/leibniz-fast#setdata)
- [API: render()](/api/leibniz-fast#render)
- [API: beginUpdate()](/api/leibniz-fast#beginupdate)
- [API: appendChunk()](/api/leibniz-fast#appendchunk)
- [API: endData()](/api/leibniz-fast#enddata)
- [Guide: Static Data](/guide/static-data) — chunked upload details

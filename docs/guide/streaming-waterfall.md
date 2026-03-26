# Streaming: Waterfall

The waterfall pattern displays a scrolling time-series matrix: the oldest data scrolls off the left edge and new data appears on the right. This is the standard display used for spectrograms, sonar returns, oscilloscopes, and radar waterfall displays.

## `setDataScrolled()`

```ts
setDataScrolled(data: Float32Array, options: ScrolledDataOptions): void
```

Unlike `setData()`, which colorizes the entire matrix every call, `setDataScrolled()` only colorizes the `newCols` newest columns. The GPU cost is **O(rows × newCols)** per frame — independent of the total display window width.

```ts
// After appending 4 new columns to the ring buffer:
viewer.setDataScrolled(buffer.data, {
  rows: 1024,
  cols: 2000,    // total display window width
  newCols: 4,    // only these 4 columns are re-colorized
  xOffset: buffer.totalColumnsReceived,
});
```

::: warning setRange() is required
`setDataScrolled()` does not auto-detect the data range. If no range has been set, it falls back to a full `setData()`. Always call `viewer.setRange(min, max)` before starting the waterfall loop.
:::

## The Ring Buffer Pattern

Pre-allocate a column-major `Float32Array` once. On each new batch of columns, write them at the current cursor position and advance the cursor. No data is ever shifted — the GPU shader handles the visual unwrapping.

**Column-major layout:** element at row `r`, column `c` lives at index `c * rows + r`. This allows a batch of new columns to be written as a single contiguous `TypedArray.set()` — a direct memcpy with no per-element work.

```ts
class WaterfallBuffer {
  readonly data: Float32Array;
  private cursor = 0;
  totalColumnsReceived = 0;

  constructor(
    readonly rows: number,
    readonly cols: number,
  ) {
    // Column-major: col c occupies data[c * rows .. (c+1) * rows]
    this.data = new Float32Array(rows * cols);
  }

  pushColumns(newData: Float32Array, newCols: number): void {
    for (let i = 0; i < newCols; i++) {
      const srcOffset = i * this.rows;
      const dstOffset = this.cursor * this.rows;
      this.data.set(newData.subarray(srcOffset, srcOffset + this.rows), dstOffset);
      this.cursor = (this.cursor + 1) % this.cols;
    }
    this.totalColumnsReceived += newCols;
  }
}
```

::: tip Why column-major?
Each new column from the network is already a contiguous block of `rows` floats. Storing it column-major means writing it into the ring buffer is a single `TypedArray.set()` call — O(rows) — with no transpose or per-element work. Row-major storage would require interleaving with existing data.
:::

## Complete Waterfall Example

```ts
import { LeibnizFast } from 'leibniz-fast';

const ROWS = 1024;
const COLS = 2000;     // display window: 2000 columns
const V_MIN = -0.5;
const V_MAX = 0.5;

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const viewer = await LeibnizFast.create(canvas, {
  colormap: 'viridis',
  chart: {
    title: 'Sonar Returns',
    xAxis: { label: 'Time', unit: 's', unitsPerCell: 0.001 }, // 1 kHz
    yAxis: { label: 'Depth', unit: 'm', min: 0, max: 500 },
    valueUnit: 'dB',
  },
});

// Range must be set before setDataScrolled.
viewer.setRange(V_MIN, V_MAX);

const buffer = new WaterfallBuffer(ROWS, COLS);
let dirty = false;
let pendingData: Float32Array | null = null;
let pendingNewCols = 0;

// Network handler — runs at source rate.
const ws = new WebSocket('ws://localhost:8765');
ws.binaryType = 'arraybuffer';
ws.addEventListener('message', (event) => {
  const view = new DataView(event.data as ArrayBuffer);
  const newCols = view.getUint32(4, true);
  const columnData = new Float32Array(event.data, 16);

  buffer.pushColumns(columnData, newCols);
  pendingData = buffer.data;
  pendingNewCols = newCols;
  dirty = true;
});

// rAF loop — runs at display rate (≤60 Hz).
function renderLoop() {
  if (dirty && pendingData) {
    viewer.setDataScrolled(pendingData, {
      rows: ROWS,
      cols: COLS,
      newCols: pendingNewCols,
      xOffset: buffer.totalColumnsReceived,
    });
    dirty = false;
  }
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);
```

## `ScrolledDataOptions`

| Property | Type | Required | Description |
|---|---|---|---|
| `rows` | `number` | **Yes** | Number of rows in the display window |
| `cols` | `number` | **Yes** | Total columns in the display window (ring buffer width) |
| `newCols` | `number` | **Yes** | Number of new columns written this frame |
| `xOffset` | `number` | No | Total columns received across all frames. Used to keep the streaming X axis label advancing correctly as old columns scroll off. |

## Initial Load

The first call after initialization should be a full `setData()` to populate the ring buffer and establish the display dimensions. After that, switch to `setDataScrolled()`:

```ts
// Initialize with zeros.
viewer.setData(buffer.data, { rows: ROWS, cols: COLS });

// Then enter the streaming loop.
ws.addEventListener('message', ...);
```

## Performance Characteristics

| Operation | GPU cost |
|---|---|
| `setData()` | O(rows × cols) — full matrix colorized |
| `setDataScrolled()` | O(rows × newCols) — only new columns colorized |

Increasing `cols` (the display window width) **does not increase** per-frame GPU work, because only `newCols` new columns are ever re-colorized. The ring offset is a single uniform buffer update. A 1-second window and a 10-second window at the same `newCols` cost exactly the same per frame.

## See Also

- [API: setDataScrolled()](/api/leibniz-fast#setdatascrolled)
- [API: setRange()](/api/leibniz-fast#setrange)
- [API: ScrolledDataOptions](/api/types#scrolleddataoptions)
- [API: StreamingAxisConfig](/api/types#streamingaxisconfig)
- [Guide: Chart Customization](/guide/chart-customization)

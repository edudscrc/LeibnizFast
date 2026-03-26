# Static Data

## `setData()`

```ts
setData(data: Float32Array, options: DataOptions): void
```

The primary method for loading a matrix into the viewer. It uploads `data` to the GPU, runs the colormap compute shader over all cells, and renders the result in a single call.

**Data layout:** Row-major `Float32Array`. Element at row `r`, column `c` lives at index `r * cols + c`.

```ts
const rows = 256;
const cols = 512;
const data = new Float32Array(rows * cols);

for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    data[r * cols + c] = Math.sin(r / 20) * Math.cos(c / 30);
  }
}

viewer.setData(data, { rows, cols });
```

After `setData()` returns:
- The data range is auto-detected from the min/max values in `data`. Override it with [`setRange()`](/api/leibniz-fast#setrange).
- The chart overlay (if configured) redraws tick marks and labels to reflect any new axis configuration.
- The hover callback (if registered) is refreshed to the new data dimensions.

## `DataOptions`

| Property | Type | Required | Description |
|---|---|---|---|
| `rows` | `number` | Yes | Number of rows in the matrix |
| `cols` | `number` | Yes | Number of columns in the matrix |
| `xOffset` | `number` | No | Total columns received so far, used to advance the streaming X axis label. See [Streaming: Waterfall](/guide/streaming-waterfall). |

## Chunked Upload for Large Matrices

For matrices larger than ~1 GB (`rows × cols × 4 bytes`), use the three-step chunked API to avoid allocating a single massive `Float32Array` in JavaScript.

```ts
const rows = 8000;
const cols = 8000;
const chunkRows = 1000;

viewer.beginData({ rows, cols });

for (let startRow = 0; startRow < rows; startRow += chunkRows) {
  const actualRows = Math.min(chunkRows, rows - startRow);
  const chunk = generateChunk(startRow, actualRows, cols); // your data source
  viewer.appendChunk(chunk, startRow);
}

viewer.endData();
```

### Chunked API Methods

| Method | Description |
|---|---|
| `beginData({ rows, cols })` | Allocates GPU buffers for the full matrix. Must be called before `appendChunk`. |
| `appendChunk(data, startRow)` | Uploads a slice of rows. `data` must contain exactly `actualRows × cols` elements. `startRow` is zero-based. |
| `endData()` | Finalizes the upload: auto-detects range, rebuilds the render pipeline, and draws the first frame. |

::: tip Chunk size
Aim for chunks of ~16 MB (`chunkRows = Math.floor(16_000_000 / (cols * 4))`). This keeps each JavaScript allocation small while minimizing the number of GPU staging buffer submissions.
:::

### Aborting an Upload

If an upload needs to be cancelled (e.g., a network error during streaming), call `abortData()`:

```ts
viewer.beginData({ rows, cols });

try {
  for await (const chunk of networkStream) {
    viewer.appendChunk(chunk.data, chunk.startRow);
  }
  viewer.endData();
} catch (err) {
  viewer.abortData(); // releases staging buffers for reuse
}
```

`abortData()` is a no-op if no upload is in progress. After aborting, GPU resources are restored and the next [`beginUpdate()`](/api/leibniz-fast#beginupdate) call can reuse them.

## Auto Range Detection

Both `setData()` and `endData()` automatically scan the uploaded data to find its min and max, then set the colormap range accordingly. You can override this at any time:

```ts
viewer.setData(data, { rows, cols });
viewer.setRange(-1.0, 1.0); // override auto-detected range
```

::: warning Waterfall requirement
`setDataScrolled()` (waterfall streaming) does **not** auto-detect the range. You must call `setRange()` before the first `setDataScrolled()` call, or it will fall back to a full `setData()`. See [Streaming: Waterfall](/guide/streaming-waterfall) for details.
:::

## See Also

- [API: setData()](/api/leibniz-fast#setdata)
- [API: DataOptions](/api/types#dataoptions)
- [API: beginData()](/api/leibniz-fast#begindata) / [appendChunk()](/api/leibniz-fast#appendchunk) / [endData()](/api/leibniz-fast#enddata)
- [API: abortData()](/api/leibniz-fast#abortdata)

# LeibnizFast

The main class exported by `leibniz-fast`. Construct it with the static `create()` method; the constructor is private.

```ts
import { LeibnizFast } from 'leibniz-fast';
```

---

## create()

```ts
static async create(
  canvas: HTMLCanvasElement,
  options?: CreateOptions
): Promise<LeibnizFast>
```

Creates and returns a new `LeibnizFast` viewer attached to the given canvas.

On the first call, the WASM module is fetched, compiled, and cached. Subsequent calls on the same page reuse the cached module. The GPU context is initialized from `canvas` — one viewer per canvas element.

| Parameter | Type | Description |
|---|---|---|
| `canvas` | `HTMLCanvasElement` | The canvas element to render into |
| `options` | [`CreateOptions`](/api/types#createoptions) | Optional configuration: colormap, debug, chart |

**Returns:** `Promise<LeibnizFast>`

**Throws** if neither WebGPU nor WebGL2 is available in the current browser.

```ts
const viewer = await LeibnizFast.create(canvas, {
  colormap: 'viridis',
  debug: false,
  chart: { title: 'My Chart' },
});
```

---

## setData()

```ts
setData(data: Float32Array, options: DataOptions): void
```

Uploads a complete matrix to the GPU, colormaps all cells, and renders the frame.

Data must be in **row-major order**: element at row `r`, column `c` is at `data[r * cols + c]`.

After this call, the data range is auto-detected from the min/max of `data`. Override it with [`setRange()`](#setrange).

| Parameter | Type | Description |
|---|---|---|
| `data` | `Float32Array` | Flat matrix data, row-major order |
| `options` | [`DataOptions`](/api/types#dataoptions) | Matrix dimensions (`rows`, `cols`) and optional `xOffset` |

See [Guide: Static Data](/guide/static-data) for chunked upload patterns for large matrices.

---

## setDataScrolled()

```ts
setDataScrolled(data: Float32Array, options: ScrolledDataOptions): void
```

Efficient waterfall update: only the `newCols` newest columns are colorized on the GPU. GPU cost is O(rows × newCols) per call, independent of the total window width.

Data must be in **column-major order**: element at row `r`, column `c` is at `data[c * rows + r]`.

::: warning
`setRange()` must be called before the first `setDataScrolled()`. Without a range, this method falls back to a full `setData()`.
:::

| Parameter | Type | Description |
|---|---|---|
| `data` | `Float32Array` | Full ring buffer contents, column-major order |
| `options` | [`ScrolledDataOptions`](/api/types#scrolleddataoptions) | Matrix dimensions, `newCols`, and optional `xOffset` |

See [Guide: Streaming: Waterfall](/guide/streaming-waterfall) for the complete ring buffer pattern.

---

## setColormap()

```ts
setColormap(name: ColormapName): void
```

Changes the colormap used to render data values. Takes effect on the next render call. No data reload is needed.

| Parameter | Type | Description |
|---|---|---|
| `name` | [`ColormapName`](/api/types#colormapname) | One of `'viridis'`, `'inferno'`, `'magma'`, `'plasma'`, `'cividis'`, `'grayscale'` |

```ts
viewer.setColormap('plasma');
```

---

## setRange()

```ts
setRange(min: number, max: number): void
```

Sets the data range for colormap mapping. Values at or below `min` render as the first colormap color; values at or above `max` render as the last.

| Parameter | Type | Description |
|---|---|---|
| `min` | `number` | Data value mapping to the first colormap color |
| `max` | `number` | Data value mapping to the last colormap color |

**Required** before [`setDataScrolled()`](#setdatascrolled). Optional after [`setData()`](#setdata) (auto-detected if not set).

```ts
viewer.setRange(-1.0, 1.0);
```

---

## onHover()

```ts
onHover(callback: HoverCallback): void
```

Registers a callback that fires when the pointer moves over a matrix cell. Only one callback is active at a time; calling `onHover()` again replaces the previous callback.

The callback receives a [`HoverInfo`](/api/types#hoverinfo) object enriched with interpolated axis coordinates when a [`ChartConfig`](/api/types#chartconfig) is present.

| Parameter | Type | Description |
|---|---|---|
| `callback` | [`HoverCallback`](/api/types#hovercallback) | Function called with hover details on cell entry |

```ts
viewer.onHover((info) => {
  console.log(`value=${info.value} at row=${info.row} col=${info.col}`);
  if (info.x !== undefined) console.log(`x=${info.x} ${info.xUnit}`);
  if (info.y !== undefined) console.log(`y=${info.y} ${info.yUnit}`);
});
```

---

## beginData()

```ts
beginData(options: StreamingDataOptions): void
```

Begins a chunked matrix upload. Allocates GPU staging buffers for the specified dimensions. Must be followed by one or more [`appendChunk()`](#appendchunk) calls, then [`endData()`](#enddata).

Use [`beginUpdate()`](#beginupdate) instead of `beginData()` in real-time streaming loops where dimensions stay constant — it reuses the existing staging buffer.

| Parameter | Type | Description |
|---|---|---|
| `options` | [`StreamingDataOptions`](/api/types#streamingdataoptions) | Matrix dimensions (`rows`, `cols`) |

---

## beginUpdate()

```ts
beginUpdate(options: StreamingDataOptions): void
```

Begins a streaming update. When called with the same dimensions as the previous frame, reuses the GPU staging buffer and avoids a pipeline rebuild. Falls back to [`beginData()`](#begindata) on the first call or on a dimension change.

Use this instead of `beginData()` in high-frequency rendering loops.

| Parameter | Type | Description |
|---|---|---|
| `options` | [`StreamingDataOptions`](/api/types#streamingdataoptions) | Matrix dimensions (`rows`, `cols`) |

```ts
// Inside the rAF loop:
viewer.beginUpdate({ rows, cols });
for (const chunk of chunks) {
  viewer.appendChunk(chunk.data, chunk.startRow);
}
viewer.endData();
```

---

## appendChunk()

```ts
appendChunk(data: Float32Array, startRow: number): void
```

Appends a block of rows to an in-progress upload started by [`beginData()`](#begindata) or [`beginUpdate()`](#beginupdate).

| Parameter | Type | Description |
|---|---|---|
| `data` | `Float32Array` | A contiguous block of complete rows. Length must equal `chunkRows × cols`. |
| `startRow` | `number` | Zero-based index of the first row in this chunk |

---

## endData()

```ts
endData(): void
```

Finalizes a chunked upload: auto-detects the data range, rebuilds render pipelines, and renders the first frame. Must be called after all [`appendChunk()`](#appendchunk) calls.

---

## abortData()

```ts
abortData(): void
```

Cancels an in-progress upload and releases staging buffers back for reuse by the next [`beginUpdate()`](#beginupdate). No-op if no upload is in progress.

---

## render()

```ts
render(): void
```

Renders a single frame without modifying any data. The current GPU texture is redrawn to the canvas.

Use this in a `requestAnimationFrame` loop when data is being ingested at a different rate than the display refresh:

```ts
function renderLoop() {
  viewer.render();
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);
```

---

## setChart()

```ts
setChart(config: ChartConfig | null): void
```

Updates the chart configuration (axes, title, fonts, colors). Creates the 2D overlay if it does not exist yet. Passing `null` removes the overlay entirely and reverts to raw matrix mode.

| Parameter | Type | Description |
|---|---|---|
| `config` | [`ChartConfig`](/api/types#chartconfig) \| `null` | New configuration, or `null` to remove the overlay |

```ts
// Update the axis range live.
viewer.setChart({
  xAxis: { label: 'Time', unit: 's', min: 0, max: newEndTime },
  yAxis: { label: 'Freq', unit: 'Hz', min: 0, max: 24000 },
});

// Remove overlay.
viewer.setChart(null);
```

---

## setTitle()

```ts
setTitle(title: string): void
```

Updates the chart title. Creates the chart overlay if it does not already exist.

| Parameter | Type | Description |
|---|---|---|
| `title` | `string` | Title text displayed centered above the matrix |

```ts
viewer.setTitle(`Frame ${frameIndex} / ${totalFrames}`);
```

---

## getMaxMatrixElements()

```ts
getMaxMatrixElements(): number
```

Returns the maximum number of `Float32` elements (`rows × cols`) that fit in a single GPU buffer on this device. Typical values: 64 M–256 M on integrated GPUs, 256 M–1 B on discrete GPUs.

**Returns:** `number`

---

## getMaxTextureDimension()

```ts
getMaxTextureDimension(): number
```

Returns the maximum value for a single matrix dimension (`rows` or `cols`). Matrices exceeding this along either dimension are automatically tiled, but individual tiles must fit within this limit.

**Returns:** `number`

---

## resetZoom()

```ts
resetZoom(): void
```

Resets the camera to show the full matrix (both axes at zoom level 1.0). This is the programmatic equivalent of double-clicking on the matrix area.

```ts
viewer.resetZoom();
```

See [Guide: Mouse Interaction](/guide/interaction) for the full set of zoom/pan controls.

---

## destroy()

```ts
destroy(): void
```

Releases all resources held by this viewer:

- Removes all DOM event listeners (pointer, wheel, resize) from the canvas.
- Destroys GPU buffers, textures, bind groups, and pipelines.
- Tears down the chart overlay DOM nodes.
- Nullifies all internal references to allow garbage collection.

Must be called when the viewer is no longer needed to prevent memory leaks, particularly in SPAs where components mount and unmount.

```ts
// Cleanup on component unmount:
viewer.destroy();
```

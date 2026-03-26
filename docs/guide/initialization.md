# Initialization

## `LeibnizFast.create()`

```ts
static async create(
  canvas: HTMLCanvasElement,
  options?: CreateOptions
): Promise<LeibnizFast>
```

`create()` is the only way to construct a `LeibnizFast` instance. It:

1. Lazy-loads and compiles the WASM module (cached after the first call across all viewers on the page).
2. Acquires a WebGPU adapter and device from the canvas. Falls back to WebGL2 if WebGPU is unavailable.
3. Registers pointer, wheel, and resize event listeners on the canvas for pan/zoom interactions.
4. If `options.chart` is provided, creates the 2D Canvas overlay for axes and labels.

```ts
import { LeibnizFast } from 'leibniz-fast';

const canvas = document.getElementById('viz') as HTMLCanvasElement;
const viewer = await LeibnizFast.create(canvas, {
  colormap: 'inferno',
  debug: false,
});
```

## `CreateOptions`

| Property | Type | Default | Description |
|---|---|---|---|
| `colormap` | [`ColormapName`](/api/types#colormapname) | `'viridis'` | Initial colormap applied to all data values |
| `debug` | `boolean` | `false` | Log performance timings to the browser console |
| `chart` | [`ChartConfig`](/api/types#chartconfig) | `undefined` | Axes, title, and label configuration. Omit for a raw matrix view with no overlays. |

## Debug Mode

Setting `debug: true` instructs LeibnizFast to emit timing logs for every major operation. This is useful when profiling frame budgets.

```ts
const viewer = await LeibnizFast.create(canvas, { debug: true });
```

Sample console output:

```
[LeibnizFast] ensureWasmLoaded: 42.3ms
[LeibnizFast] create (WASM init): 8.1ms
[LeibnizFast] setData (256×512): 1.4ms
[LeibnizFast] setColormap: 0.1ms
```

## Device Limits

Before uploading a very large matrix, query the device's GPU limits to avoid runtime errors.

```ts
const maxElements = viewer.getMaxMatrixElements();
const maxDim = viewer.getMaxTextureDimension();

if (rows * cols > maxElements) {
  console.error(`Matrix too large: ${rows * cols} elements, limit is ${maxElements}`);
  return;
}

if (rows > maxDim || cols > maxDim) {
  console.error(`Dimension exceeds limit of ${maxDim}`);
  return;
}
```

| Method | Returns | Description |
|---|---|---|
| `getMaxMatrixElements()` | `number` | Maximum `rows × cols` that fits in a single GPU buffer (typically 256 M–1 B elements) |
| `getMaxTextureDimension()` | `number` | Maximum value for `rows` or `cols` (typically 8192–16384 on integrated, 16384–32768 on discrete GPUs) |

::: tip Tiling
Matrices where `rows` or `cols` exceed `getMaxTextureDimension()` are automatically split into a grid of tiles. The limit is per-tile, not total. For most devices this is not a concern unless a single dimension exceeds ~16 k.
:::

## Lifecycle and Cleanup

Every `LeibnizFast` instance registers event listeners on the canvas element. Call `destroy()` when the viewer is no longer needed — for example, when a SPA component unmounts.

```ts
// React example
useEffect(() => {
  let viewer: LeibnizFast | null = null;
  LeibnizFast.create(canvasRef.current!, { colormap: 'viridis' }).then((v) => {
    viewer = v;
    v.setData(data, { rows, cols });
  });
  return () => {
    viewer?.destroy();
  };
}, []);
```

`destroy()` removes all pointer, wheel, and resize event listeners; frees GPU buffers and textures; tears down the chart overlay DOM nodes; and nullifies internal references to allow garbage collection.

## See Also

- [API: LeibnizFast.create()](/api/leibniz-fast#create)
- [API: CreateOptions](/api/types#createoptions)
- [Guide: Chart Customization](/guide/chart-customization)

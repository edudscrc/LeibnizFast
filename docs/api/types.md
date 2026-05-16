# Types

All types are re-exported from the main package entry point:

```ts
import type {
  ColormapName,
  CreateOptions,
  DataOptions,
  DataChunk,
  ChunkedDataOptions,
  DataRange,
  WebGpuSupport,
  ScrolledDataOptions,
  StreamingDataOptions,
  ChartConfig,
  AxisConfig,
  StreamingAxisConfig,
  LineChartConfig,
  LineSeriesInput,
  LineSeriesUpdate,
  LineXAxisConfig,
  LineYAxisConfig,
  LineDataOptions,
  LineScrolledDataOptions,
  RgbaColor,
  HeatmapHoverInfo,
  LineHoverPoint,
  LineHoverInfo,
  HoverInfo,
  HoverCallback,
} from 'leibniz-fast';
```

---

## ColormapName

```ts
type ColormapName =
  | 'viridis'
  | 'inferno'
  | 'magma'
  | 'plasma'
  | 'cividis'
  | 'grayscale';
```

Union of all available colormap names. Pass to [`CreateOptions.colormap`](#createoptions) or [`LeibnizFast.setColormap()`](/api/leibniz-fast#setcolormap).

| Value         | Character                                                      |
| ------------- | -------------------------------------------------------------- |
| `'viridis'`   | Blue → green → yellow. Default. Perceptually uniform.          |
| `'inferno'`   | Black → purple → orange → yellow. High contrast for dark data. |
| `'magma'`     | Black → purple → pink → white.                                 |
| `'plasma'`    | Blue → purple → orange → yellow. Vivid, high contrast.         |
| `'cividis'`   | Blue-grey → yellow. Colorblind-friendly.                       |
| `'grayscale'` | Black → white.                                                 |

---

## CreateOptions

```ts
interface CreateOptions {
  colormap?: ColormapName;
  debug?: boolean;
  chart?: ChartConfig;
}
```

Options for [`LeibnizFast.create()`](/api/leibniz-fast#create).

| Field      | Type                            | Default     | Description                                                                                           |
| ---------- | ------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------- |
| `colormap` | [`ColormapName`](#colormapname) | `'viridis'` | Initial colormap applied to data values                                                               |
| `debug`    | `boolean`                       | `false`     | Log performance timing to the browser console                                                         |
| `chart`    | [`ChartConfig`](#chartconfig)   | `undefined` | Axes, title, and label configuration. Omit for a raw matrix view. Use `type: 'line'` for line charts. |

---

## DataOptions

```ts
interface DataOptions {
  rows: number;
  cols: number;
  xOffset?: number;
}
```

Options for [`LeibnizFast.setData()`](/api/leibniz-fast#setdata).

| Field     | Type     | Required | Description                                                                                                                 |
| --------- | -------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| `rows`    | `number` | **Yes**  | Number of rows in the matrix                                                                                                |
| `cols`    | `number` | **Yes**  | Number of columns in the matrix                                                                                             |
| `xOffset` | `number` | No       | Total columns received so far. Advances the streaming X axis label. Use with [`StreamingAxisConfig`](#streamingaxisconfig). |

---

## WebGpuSupport

```ts
interface WebGpuSupport {
  supported: boolean;
  reason?: string;
  adapterInfo?: Record<string, string | number | boolean>;
}
```

Returned by [`LeibnizFast.checkSupport()`](/api/leibniz-fast#checksupport).

---

## DataRange

```ts
interface DataRange {
  min: number;
  max: number;
}
```

Explicit colormap range used by streaming and chunked upload options.

---

## DataChunk

```ts
interface DataChunk {
  startRow: number;
  data: Float32Array;
}
```

One row-major chunk for [`setDataChunks()`](/api/leibniz-fast#setdatachunks).

---

## ChunkedDataOptions

```ts
interface ChunkedDataOptions extends DataOptions {
  retainData?: boolean;
  range?: DataRange;
}
```

Options for [`setDataChunks()`](/api/leibniz-fast#setdatachunks).

| Field        | Type                      | Required | Description                                                                  |
| ------------ | ------------------------- | -------- | ---------------------------------------------------------------------------- |
| `rows`       | `number`                  | **Yes**  | Number of rows in the matrix                                                 |
| `cols`       | `number`                  | **Yes**  | Number of columns in the matrix                                              |
| `retainData` | `boolean`                 | No       | Keep CPU-side values for hover callbacks. Defaults to `false`.               |
| `range`      | [`DataRange`](#datarange) | No       | Fixed colormap range. When omitted, chunks are scanned once while uploading. |

---

## ScrolledDataOptions

```ts
interface ScrolledDataOptions extends DataOptions {
  newCols: number;
}
```

Options for [`LeibnizFast.setDataScrolled()`](/api/leibniz-fast#setdatascrolled). Extends [`DataOptions`](#dataoptions).

| Field     | Type     | Required | Description                                                                    |
| --------- | -------- | -------- | ------------------------------------------------------------------------------ |
| `rows`    | `number` | **Yes**  | _(inherited)_ Number of rows                                                   |
| `cols`    | `number` | **Yes**  | _(inherited)_ Total display window width in columns                            |
| `xOffset` | `number` | No       | _(inherited)_ Total columns received for streaming axis                        |
| `newCols` | `number` | **Yes**  | Number of new columns written this frame. GPU re-colorizes only these columns. |

---

## StreamingDataOptions

```ts
interface StreamingDataOptions {
  rows: number;
  cols: number;
  range?: DataRange;
  retainData?: boolean;
}
```

Options for [`LeibnizFast.beginData()`](/api/leibniz-fast#begindata) and [`LeibnizFast.beginUpdate()`](/api/leibniz-fast#beginupdate).

| Field        | Type                      | Required | Description                                                                   |
| ------------ | ------------------------- | -------- | ----------------------------------------------------------------------------- |
| `rows`       | `number`                  | **Yes**  | Number of rows in the matrix                                                  |
| `cols`       | `number`                  | **Yes**  | Number of columns in the matrix                                               |
| `range`      | [`DataRange`](#datarange) | No       | Fixed colormap range                                                          |
| `retainData` | `boolean`                 | No       | Streaming uploads retain data. Use `setDataChunks()` for `retainData: false`. |

---

## ChartConfig

```ts
type ChartConfig = HeatmapChartConfig | LineChartConfig;

interface HeatmapChartConfig {
  type?: 'heatmap';
  title?: string;
  xAxis?: AxisConfig | StreamingAxisConfig;
  yAxis?: AxisConfig;
  valueUnit?: string;
  colorbar?: boolean;
  font?: string;
  titleFont?: string;
  tickColor?: string;
  labelColor?: string;
  backgroundColor?: string;
}

interface LineChartConfig {
  type: 'line';
  title?: string;
  xAxis?: LineXAxisConfig;
  yAxis?: LineYAxisConfig;
  grid?: boolean;
  legend?: boolean;
  font?: string;
  titleFont?: string;
  tickColor?: string;
  labelColor?: string;
  backgroundColor?: string;
}
```

Configuration for heatmap and line chart overlays. Heatmaps are the default when `type` is omitted. Line charts use a WebGPU line renderer plus the same overlay axes and interaction model.

| Field             | Type                                                                         | Default                  | Description                                                                                                 |
| ----------------- | ---------------------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `title`           | `string`                                                                     | —                        | Text centered above the matrix                                                                              |
| `xAxis`           | [`AxisConfig`](#axisconfig) \| [`StreamingAxisConfig`](#streamingaxisconfig) | —                        | X axis configuration                                                                                        |
| `yAxis`           | [`AxisConfig`](#axisconfig)                                                  | —                        | Y axis configuration                                                                                        |
| `valueUnit`       | `string`                                                                     | —                        | Unit string appended to hover tooltip values and used as the default colorbar label (e.g. `'dBFS'`, `'°C'`) |
| `colorbar`        | `boolean`                                                                    | `true`                   | Show the right-side colorbar. Set to `false` to hide it.                                                    |
| `font`            | `string` (CSS font)                                                          | `'12px sans-serif'`      | Font for tick labels                                                                                        |
| `titleFont`       | `string` (CSS font)                                                          | `'bold 16px sans-serif'` | Font for the chart title                                                                                    |
| `tickColor`       | `string` (CSS color)                                                         | `'#999'`                 | Color of tick marks and axis lines                                                                          |
| `labelColor`      | `string` (CSS color)                                                         | `'#ccc'`                 | Color of all text labels                                                                                    |
| `backgroundColor` | `string` (CSS color)                                                         | `'#1a1a1a'`              | Background fill of the margin area                                                                          |

Line-specific fields:

| Field    | Type                                  | Default      | Description                             |
| -------- | ------------------------------------- | ------------ | --------------------------------------- |
| `type`   | `'line'`                              | **required** | Enables line chart rendering            |
| `xAxis`  | [`LineXAxisConfig`](#linexaxisconfig) | index axis   | Shared X coordinates for every series   |
| `yAxis`  | [`LineYAxisConfig`](#lineyaxisconfig) | sticky auto  | Value axis and range behavior           |
| `grid`   | `boolean`                             | `false`      | Draw background grid lines              |
| `legend` | `boolean`                             | `true`       | Show clickable right-side series legend |

See [Guide: Chart Customization](/guide/chart-customization) for full usage examples.

---

## LineSeriesInput

```ts
type RgbaColor = [red: number, green: number, blue: number, alpha: number];

interface LineSeriesInput {
  id?: string;
  name: string;
  color: RgbaColor;
  data: Float32Array;
  visible?: boolean;
}
```

RGB values use `0..255`; alpha uses `0..1`. All series in one chart share the same sample count and X axis.

---

## LineXAxisConfig

```ts
type LineXAxisConfig =
  | { kind?: 'index'; label?: string; unit?: string }
  | { kind?: 'linear'; label?: string; unit?: string; min: number; max: number }
  | {
      kind: 'explicit';
      label?: string;
      unit?: string;
      values: Float32Array | readonly number[];
    }
  | {
      kind: 'streaming';
      label?: string;
      unit?: string;
      start?: number;
      unitsPerSample: number;
    };
```

The default index axis maps samples to `0..N-1`. Linear axes behave like `linspace(min, max, N)`. Explicit X values must be finite, strictly increasing, and shared by every series. Streaming axes advance a scrolling window by `unitsPerSample`.

---

## LineYAxisConfig

```ts
interface LineYAxisConfig {
  label?: string;
  unit?: string;
  rangeMode?: 'stickyAuto' | 'fixed';
  min?: number;
  max?: number;
  paddingRatio?: number;
}
```

`stickyAuto` expands when visible series exceed the current bounds and keeps a small vertical margin. User Y-axis zoom/pan switches to manual until the user double-clicks the plot or Y axis. `fixed` requires finite `min` and `max`.

---

## AxisConfig

```ts
interface AxisConfig {
  label?: string;
  unit?: string;
  min: number;
  max: number;
}
```

Fixed-range axis configuration. Use for static heatmaps and for the Y axis of streaming charts where the physical range does not change.

| Field   | Type     | Required | Description                                                    |
| ------- | -------- | -------- | -------------------------------------------------------------- |
| `label` | `string` | No       | Human-readable axis name shown beside the axis                 |
| `unit`  | `string` | No       | Unit string displayed after the label and in the hover tooltip |
| `min`   | `number` | **Yes**  | Data-space value at the axis origin (bottom for Y, left for X) |
| `max`   | `number` | **Yes**  | Data-space value at the axis far end                           |

---

## StreamingAxisConfig

```ts
interface StreamingAxisConfig {
  label?: string;
  unit?: string;
  unitsPerCell: number;
}
```

Auto-incrementing axis for streaming/waterfall charts. The axis origin is always 0; each column advances the displayed value by `unitsPerCell`. Pass `xOffset` in [`ScrolledDataOptions`](#scrolleddataoptions) to keep the label correct as old columns scroll off.

| Field          | Type     | Required | Description                |
| -------------- | -------- | -------- | -------------------------- |
| `label`        | `string` | No       | Human-readable axis name   |
| `unit`         | `string` | No       | Unit string                |
| `unitsPerCell` | `number` | **Yes**  | Value increment per column |

Example — a 1 kHz stream where each column represents 1 ms:

```ts
const xAxis: StreamingAxisConfig = {
  label: 'Time',
  unit: 's',
  unitsPerCell: 0.001,
};
```

---

## HoverInfo

```ts
interface HoverInfo {
  kind?: 'heatmap';
  row: number;
  col: number;
  value: number;
  valueAvailable: boolean;
  y?: number;
  x?: number;
  yUnit?: string;
  xUnit?: string;
  valueUnit?: string;
  color?: RgbaColor;
}
```

Passed to the [`HoverCallback`](#hovercallback) registered via [`LeibnizFast.onHover()`](/api/leibniz-fast#onhover).

| Field            | Type         | Description                                                                                          |
| ---------------- | ------------ | ---------------------------------------------------------------------------------------------------- |
| `row`            | `number`     | Zero-based row index of the hovered cell                                                             |
| `col`            | `number`     | Zero-based column index of the hovered cell                                                          |
| `value`          | `number`     | Raw data value at `(row, col)`                                                                       |
| `valueAvailable` | `boolean`    | False when data was uploaded with `retainData: false`; then `value` is `NaN`.                        |
| `y`              | `number?`    | Interpolated Y axis value. Present only when `yAxis` is configured in [`ChartConfig`](#chartconfig). |
| `x`              | `number?`    | Interpolated X axis value. Present only when `xAxis` is configured.                                  |
| `yUnit`          | `string?`    | Y axis `unit` string from `AxisConfig`                                                               |
| `xUnit`          | `string?`    | X axis `unit` string                                                                                 |
| `valueUnit`      | `string?`    | Value unit from `ChartConfig.valueUnit`                                                              |
| `color`          | `RgbaColor?` | RGBA color currently used to render the heatmap value, when available                                |

---

## HoverCallback

```ts
type HoverCallback = (info: HoverInfo) => void;
```

Function type for the hover event callback registered via [`LeibnizFast.onHover()`](/api/leibniz-fast#onhover).

```ts
const onHover: HoverCallback = (info) => {
  const value = info.valueAvailable ? info.value.toFixed(3) : 'unavailable';
  displayTooltip(`${value} at [${info.row}, ${info.col}]`);
};
viewer.onHover(onHover);
```

Line charts use a discriminated hover object:

```ts
interface LineHoverPoint {
  seriesId: string;
  seriesName: string;
  seriesIndex: number;
  sampleIndex: number;
  x: number;
  value: number;
  color: RgbaColor;
}

interface LineHoverInfo {
  kind: 'line';
  x: number;
  mouseX: number;
  mouseY: number;
  xUnit?: string;
  yUnit?: string;
  points: LineHoverPoint[];
}
```

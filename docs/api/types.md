# Types

All types are re-exported from the main package entry point:

```ts
import type {
  ColormapName,
  CreateOptions,
  DataOptions,
  ScrolledDataOptions,
  StreamingDataOptions,
  ChartConfig,
  AxisConfig,
  StreamingAxisConfig,
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
  | 'grayscale'
```

Union of all available colormap names. Pass to [`CreateOptions.colormap`](#createoptions) or [`LeibnizFast.setColormap()`](/api/leibniz-fast#setcolormap).

| Value | Character |
|---|---|
| `'viridis'` | Blue → green → yellow. Default. Perceptually uniform. |
| `'inferno'` | Black → purple → orange → yellow. High contrast for dark data. |
| `'magma'` | Black → purple → pink → white. |
| `'plasma'` | Blue → purple → orange → yellow. Vivid, high contrast. |
| `'cividis'` | Blue-grey → yellow. Colorblind-friendly. |
| `'grayscale'` | Black → white. |

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

| Field | Type | Default | Description |
|---|---|---|---|
| `colormap` | [`ColormapName`](#colormapname) | `'viridis'` | Initial colormap applied to data values |
| `debug` | `boolean` | `false` | Log performance timing to the browser console |
| `chart` | [`ChartConfig`](#chartconfig) | `undefined` | Axes, title, and label configuration. Omit for a raw matrix view. |

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

| Field | Type | Required | Description |
|---|---|---|---|
| `rows` | `number` | **Yes** | Number of rows in the matrix |
| `cols` | `number` | **Yes** | Number of columns in the matrix |
| `xOffset` | `number` | No | Total columns received so far. Advances the streaming X axis label. Use with [`StreamingAxisConfig`](#streamingaxisconfig). |

---

## ScrolledDataOptions

```ts
interface ScrolledDataOptions extends DataOptions {
  newCols: number;
}
```

Options for [`LeibnizFast.setDataScrolled()`](/api/leibniz-fast#setdatascrolled). Extends [`DataOptions`](#dataoptions).

| Field | Type | Required | Description |
|---|---|---|---|
| `rows` | `number` | **Yes** | *(inherited)* Number of rows |
| `cols` | `number` | **Yes** | *(inherited)* Total display window width in columns |
| `xOffset` | `number` | No | *(inherited)* Total columns received for streaming axis |
| `newCols` | `number` | **Yes** | Number of new columns written this frame. GPU re-colorizes only these columns. |

---

## StreamingDataOptions

```ts
interface StreamingDataOptions {
  rows: number;
  cols: number;
}
```

Options for [`LeibnizFast.beginData()`](/api/leibniz-fast#begindata) and [`LeibnizFast.beginUpdate()`](/api/leibniz-fast#beginupdate).

| Field | Type | Required | Description |
|---|---|---|---|
| `rows` | `number` | **Yes** | Number of rows in the matrix |
| `cols` | `number` | **Yes** | Number of columns in the matrix |

---

## ChartConfig

```ts
interface ChartConfig {
  title?: string;
  xAxis?: AxisConfig | StreamingAxisConfig;
  yAxis?: AxisConfig;
  valueUnit?: string;
  font?: string;
  titleFont?: string;
  tickColor?: string;
  labelColor?: string;
  backgroundColor?: string;
}
```

Configuration for the 2D chart overlay rendered on top of the GPU canvas. All fields are optional; omit the entire object to disable the overlay.

| Field | Type | Default | Description |
|---|---|---|---|
| `title` | `string` | — | Text centered above the matrix |
| `xAxis` | [`AxisConfig`](#axisconfig) \| [`StreamingAxisConfig`](#streamingaxisconfig) | — | X axis configuration |
| `yAxis` | [`AxisConfig`](#axisconfig) | — | Y axis configuration |
| `valueUnit` | `string` | — | Unit string appended to hover tooltip values (e.g. `'dBFS'`, `'°C'`) |
| `font` | `string` (CSS font) | `'12px sans-serif'` | Font for tick labels |
| `titleFont` | `string` (CSS font) | `'bold 16px sans-serif'` | Font for the chart title |
| `tickColor` | `string` (CSS color) | `'#999'` | Color of tick marks and axis lines |
| `labelColor` | `string` (CSS color) | `'#ccc'` | Color of all text labels |
| `backgroundColor` | `string` (CSS color) | `'#1a1a1a'` | Background fill of the margin area |

See [Guide: Chart Customization](/guide/chart-customization) for full usage examples.

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

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | `string` | No | Human-readable axis name shown beside the axis |
| `unit` | `string` | No | Unit string displayed after the label and in the hover tooltip |
| `min` | `number` | **Yes** | Data-space value at the axis origin (bottom for Y, left for X) |
| `max` | `number` | **Yes** | Data-space value at the axis far end |

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

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | `string` | No | Human-readable axis name |
| `unit` | `string` | No | Unit string |
| `unitsPerCell` | `number` | **Yes** | Value increment per column |

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
  row: number;
  col: number;
  value: number;
  y?: number;
  x?: number;
  yUnit?: string;
  xUnit?: string;
  valueUnit?: string;
}
```

Passed to the [`HoverCallback`](#hovercallback) registered via [`LeibnizFast.onHover()`](/api/leibniz-fast#onhover).

| Field | Type | Description |
|---|---|---|
| `row` | `number` | Zero-based row index of the hovered cell |
| `col` | `number` | Zero-based column index of the hovered cell |
| `value` | `number` | Raw data value at `(row, col)` |
| `y` | `number?` | Interpolated Y axis value. Present only when `yAxis` is configured in [`ChartConfig`](#chartconfig). |
| `x` | `number?` | Interpolated X axis value. Present only when `xAxis` is configured. |
| `yUnit` | `string?` | Y axis `unit` string from `AxisConfig` |
| `xUnit` | `string?` | X axis `unit` string |
| `valueUnit` | `string?` | Value unit from `ChartConfig.valueUnit` |

---

## HoverCallback

```ts
type HoverCallback = (info: HoverInfo) => void
```

Function type for the hover event callback registered via [`LeibnizFast.onHover()`](/api/leibniz-fast#onhover).

```ts
const onHover: HoverCallback = (info) => {
  displayTooltip(`${info.value.toFixed(3)} at [${info.row}, ${info.col}]`);
};
viewer.onHover(onHover);
```

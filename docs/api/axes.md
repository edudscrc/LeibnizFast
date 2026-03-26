# Axes Utilities

::: warning Advanced / Internal API
These utilities are used internally by LeibnizFast to render the chart overlay. They are exported for advanced users who want to build custom overlay renderers or extend the chart system.

They are **not re-exported** from the main `leibniz-fast` entry point (`dist/index.js`). To use them, you must import directly from the source or build a secondary entry point from `js/axes.ts`.
:::

## Functions

### isStreamingAxis()

```ts
function isStreamingAxis(
  axis: AxisConfig | StreamingAxisConfig
): axis is StreamingAxisConfig
```

Type guard that returns `true` when `axis` is a [`StreamingAxisConfig`](/api/types#streamingaxisconfig) (i.e., has a `unitsPerCell` field).

| Parameter | Type | Description |
|---|---|---|
| `axis` | [`AxisConfig`](/api/types#axisconfig) \| [`StreamingAxisConfig`](/api/types#streamingaxisconfig) | Axis configuration to test |

**Returns:** `axis is StreamingAxisConfig`

```ts
if (isStreamingAxis(chart.xAxis)) {
  // chart.xAxis is StreamingAxisConfig
  console.log(chart.xAxis.unitsPerCell);
}
```

---

### generateTicks()

```ts
function generateTicks(
  min: number,
  max: number,
  availablePixels: number,
  ctx: CanvasRenderingContext2D,
  font: string
): Tick[]
```

Generates evenly-spaced tick marks for a visible data range, automatically removing ticks whose labels would overlap.

| Parameter | Type | Description |
|---|---|---|
| `min` | `number` | Data-space minimum of the visible range |
| `max` | `number` | Data-space maximum of the visible range |
| `availablePixels` | `number` | Pixel length of the axis |
| `ctx` | `CanvasRenderingContext2D` | 2D context used for label width measurement |
| `font` | `string` | CSS font string applied before measurement |

**Returns:** [`Tick[]`](#tick) — array of ticks with pixel positions relative to the axis origin.

---

### computeLayout()

```ts
function computeLayout(
  containerWidth: number,
  containerHeight: number,
  chart: ChartConfig,
  ctx: CanvasRenderingContext2D
): LayoutRect
```

Computes the rectangle occupied by the matrix within the container, accounting for axis labels, tick labels, and title margins.

| Parameter | Type | Description |
|---|---|---|
| `containerWidth` | `number` | Container width in CSS pixels |
| `containerHeight` | `number` | Container height in CSS pixels |
| `chart` | [`ChartConfig`](/api/types#chartconfig) | Chart configuration |
| `ctx` | `CanvasRenderingContext2D` | 2D context for text measurement |

**Returns:** [`LayoutRect`](#layoutrect)

---

### renderOverlay()

```ts
function renderOverlay(
  ctx: CanvasRenderingContext2D,
  layout: LayoutRect,
  chart: ChartConfig,
  visible: VisibleRange,
  containerWidth: number,
  containerHeight: number,
  dpr: number
): void
```

Renders the complete chart overlay — margin background, title, X axis, Y axis, tick marks, and tick labels — onto the provided 2D context.

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `CanvasRenderingContext2D` | Target 2D context (overlay canvas) |
| `layout` | [`LayoutRect`](#layoutrect) | Matrix area rectangle from `computeLayout()` |
| `chart` | [`ChartConfig`](/api/types#chartconfig) | Chart configuration |
| `visible` | [`VisibleRange`](#visiblerange) | Currently visible data range (from camera state) |
| `containerWidth` | `number` | Container width in CSS pixels |
| `containerHeight` | `number` | Container height in CSS pixels |
| `dpr` | `number` | Device pixel ratio (from `window.devicePixelRatio`) |

---

### uvToVisibleRange()

```ts
function uvToVisibleRange(
  uvOffset: [number, number],
  uvScale: [number, number],
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number
): VisibleRange
```

Maps camera UV coordinates (pan/zoom state) to the corresponding data-space axis ranges visible in the current view.

| Parameter | Type | Description |
|---|---|---|
| `uvOffset` | `[number, number]` | Camera UV offset `[x, y]` |
| `uvScale` | `[number, number]` | Camera UV scale `[x, y]` |
| `xMin` | `number` | Full X axis data minimum |
| `xMax` | `number` | Full X axis data maximum |
| `yMin` | `number` | Full Y axis data minimum |
| `yMax` | `number` | Full Y axis data maximum |

**Returns:** [`VisibleRange`](#visiblerange)

---

## Types

### Tick

```ts
interface Tick {
  value: number;
  label: string;
  position: number;
}
```

A single tick mark on an axis.

| Field | Type | Description |
|---|---|---|
| `value` | `number` | Data-space value at this tick |
| `label` | `string` | Formatted string for the tick label |
| `position` | `number` | Pixel position along the axis, relative to the matrix origin |

---

### LayoutRect

```ts
interface LayoutRect {
  x: number;
  y: number;
  width: number;
  height: number;
}
```

The rectangle occupied by the matrix within the container, in CSS pixels. Returned by [`computeLayout()`](#computelayout).

| Field | Type | Description |
|---|---|---|
| `x` | `number` | Left edge of the matrix area (left margin width) |
| `y` | `number` | Top edge of the matrix area (top margin height) |
| `width` | `number` | Width of the matrix area |
| `height` | `number` | Height of the matrix area |

---

### VisibleRange

```ts
interface VisibleRange {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}
```

The data-space range currently visible after applying the camera's pan and zoom. Returned by [`uvToVisibleRange()`](#uvtovisiblerange) and passed to [`renderOverlay()`](#renderoverlay).

| Field | Type | Description |
|---|---|---|
| `xMin` | `number` | Leftmost visible X axis value |
| `xMax` | `number` | Rightmost visible X axis value |
| `yMin` | `number` | Bottom-most visible Y axis value |
| `yMax` | `number` | Top-most visible Y axis value |

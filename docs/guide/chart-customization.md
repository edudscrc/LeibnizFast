# Chart Customization

The chart overlay renders axes, tick marks, a title, and a hover tooltip on top of the GPU canvas using a 2D Canvas element. It is entirely optional — omit `chart` from `CreateOptions` for a raw matrix view.

## Enabling the Overlay

Pass a [`ChartConfig`](/api/types#chartconfig) at creation time:

```ts
const viewer = await LeibnizFast.create(canvas, {
  colormap: 'viridis',
  chart: {
    title: 'Signal Amplitude',
    xAxis: { label: 'Time', unit: 's', min: 0, max: 10 },
    yAxis: { label: 'Frequency', unit: 'Hz', min: 0, max: 24000 },
    valueUnit: 'dBFS',
  },
});
```

Or add/update it after creation:

```ts
viewer.setChart({
  title: 'Updated Title',
  xAxis: { label: 'Depth', unit: 'm', min: 0, max: 500 },
});
```

Remove the overlay entirely by passing `null`:

```ts
viewer.setChart(null); // reverts to raw matrix mode
```

## `ChartConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `title` | `string` | — | Text displayed centered above the matrix |
| `xAxis` | [`AxisConfig`](/api/types#axisconfig) \| [`StreamingAxisConfig`](/api/types#streamingaxisconfig) | — | X axis configuration |
| `yAxis` | [`AxisConfig`](/api/types#axisconfig) | — | Y axis configuration |
| `valueUnit` | `string` | — | Unit appended to the value in the hover tooltip (e.g. `'dBFS'`, `'°C'`) |
| `font` | `string` (CSS) | `'12px sans-serif'` | Font for tick labels |
| `titleFont` | `string` (CSS) | `'bold 16px sans-serif'` | Font for the chart title |
| `tickColor` | `string` (CSS color) | `'#999'` | Color of tick marks and axis lines |
| `labelColor` | `string` (CSS color) | `'#ccc'` | Color of all text labels |
| `backgroundColor` | `string` (CSS color) | `'#1a1a1a'` | Background fill of the margin area around the matrix |

## Axis Types

### `AxisConfig` — Fixed Range

Use for static heatmaps and for the Y axis of streaming charts, where the physical range of the axis does not change over time.

```ts
const yAxis: AxisConfig = {
  label: 'Frequency',
  unit: 'kHz',
  min: 0,
  max: 24,
};
```

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | `string` | No | Human-readable axis name shown beside the axis |
| `unit` | `string` | No | Unit string displayed after the label |
| `min` | `number` | **Yes** | Data-space minimum value at the axis origin |
| `max` | `number` | **Yes** | Data-space maximum value at the axis far end |

### `StreamingAxisConfig` — Auto-Incrementing

Use for the X (time) axis of waterfall or streaming charts. The axis origin is always 0; each column increments the displayed value by `unitsPerCell`.

```ts
// 1 kHz stream — each column represents 1 ms
const xAxis: StreamingAxisConfig = {
  label: 'Time',
  unit: 's',
  unitsPerCell: 0.001,
};
```

| Field | Type | Required | Description |
|---|---|---|---|
| `label` | `string` | No | Human-readable axis name |
| `unit` | `string` | No | Unit string displayed after the label |
| `unitsPerCell` | `number` | **Yes** | Value increment per column |

## Colormaps

Six colormaps are built in. All are perceptually uniform and designed for scientific visualization.

| Name | Character |
|---|---|
| `'viridis'` | Blue → green → yellow. Default. Best general-purpose choice. |
| `'inferno'` | Black → purple → orange → yellow. High contrast for dark data. |
| `'magma'` | Black → purple → pink → white. |
| `'plasma'` | Blue → purple → orange → yellow. High contrast, vivid. |
| `'cividis'` | Blue-grey → yellow. Colorblind-friendly. |
| `'grayscale'` | Black → white. |

Change the colormap at any time without reloading data:

```ts
viewer.setColormap('inferno');
```

## Color Range

`setRange(min, max)` controls how data values map to colors. Values at or below `min` render as the first colormap color; values at or above `max` render as the last.

```ts
viewer.setRange(-1.0, 1.0);
```

`setData()` and `endData()` auto-detect the range from the data. `setRange()` overrides this. For waterfall streaming, you **must** call `setRange()` before the first `setDataScrolled()`.

## Hover Tooltip

Register a callback to receive per-cell hover information:

```ts
viewer.onHover((info) => {
  tooltip.textContent =
    `${info.yUnit ? info.y?.toFixed(1) + ' ' + info.yUnit : `row ${info.row}`}` +
    ` × ` +
    `${info.xUnit ? info.x?.toFixed(3) + ' ' + info.xUnit : `col ${info.col}`}` +
    ` = ${info.value.toFixed(4)}${info.valueUnit ? ' ' + info.valueUnit : ''}`;
});
```

The [`HoverInfo`](/api/types#hoverinfo) object contains:

| Field | Type | Description |
|---|---|---|
| `row` | `number` | Zero-based row index |
| `col` | `number` | Zero-based column index |
| `value` | `number` | Raw data value at `(row, col)` |
| `y` | `number?` | Interpolated Y axis value (present when `yAxis` is configured) |
| `x` | `number?` | Interpolated X axis value (present when `xAxis` is configured) |
| `yUnit` | `string?` | Y axis unit string |
| `xUnit` | `string?` | X axis unit string |
| `valueUnit` | `string?` | Value unit from `ChartConfig.valueUnit` |

## Updating the Title

Use `setTitle()` to update only the title without reconstructing the full config. If no chart overlay exists yet, it is created automatically.

```ts
viewer.setTitle('Frame 42 / 100');
```

## Complete Example

```ts
const viewer = await LeibnizFast.create(canvas, {
  colormap: 'plasma',
  chart: {
    title: 'Ocean Surface Temperature',
    xAxis: { label: 'Longitude', unit: '°E', min: -180, max: 180 },
    yAxis: { label: 'Latitude', unit: '°N', min: -90, max: 90 },
    valueUnit: '°C',
    backgroundColor: '#111',
    labelColor: '#ddd',
    tickColor: '#555',
  },
});

viewer.setData(temperatureGrid, { rows: 1800, cols: 3600 });
viewer.setRange(-2, 35);

viewer.onHover((info) => {
  console.log(
    `${info.y?.toFixed(2)}°N, ${info.x?.toFixed(2)}°E: ${info.value.toFixed(1)}°C`
  );
});
```

## See Also

- [API: setChart()](/api/leibniz-fast#setchart)
- [API: setColormap()](/api/leibniz-fast#setcolormap)
- [API: setRange()](/api/leibniz-fast#setrange)
- [API: onHover()](/api/leibniz-fast#onhover)
- [API: ChartConfig](/api/types#chartconfig)
- [API: HoverInfo](/api/types#hoverinfo)

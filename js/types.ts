/**
 * Available colormap names for matrix visualization.
 */
export type ColormapName =
  | 'viridis'
  | 'inferno'
  | 'magma'
  | 'plasma'
  | 'cividis'
  | 'grayscale';

/**
 * Result returned by {@link LeibnizFast.checkSupport}.
 */
export interface WebGpuSupport {
  /** True only when navigator.gpu, adapter creation, and device creation work. */
  supported: boolean;
  /** Human-readable failure reason or troubleshooting hint. */
  reason?: string;
  /** Best-effort adapter metadata when the browser exposes it. */
  adapterInfo?: Record<string, string | number | boolean>;
}

/**
 * Explicit value range for colormap mapping.
 */
export interface DataRange {
  /** Data value mapped to the first colormap color. */
  min: number;
  /** Data value mapped to the last colormap color. */
  max: number;
}

/**
 * Configuration for a fixed-range axis (used for static charts and the Y axis
 * of streaming charts).
 *
 * @example
 * ```ts
 * const yAxis: AxisConfig = { label: 'Frequency', unit: 'Hz', min: 0, max: 22050 };
 * ```
 */
export interface AxisConfig {
  /** Human-readable axis label (e.g. "Time", "Frequency"). */
  label?: string;
  /** Unit string displayed after the label (e.g. "s", "Hz"). */
  unit?: string;
  /** Minimum value of the axis range. */
  min: number;
  /** Maximum value of the axis range. */
  max: number;
}

/**
 * Configuration for a streaming X axis that auto-increments as columns
 * are appended. The first column starts at value 0; each subsequent column
 * adds `unitsPerCell`.
 *
 * @example
 * ```ts
 * const xAxis: StreamingAxisConfig = { label: 'Time', unit: 's', unitsPerCell: 0.001 };
 * ```
 */
export interface StreamingAxisConfig {
  /** Human-readable axis label (e.g. "Time"). */
  label?: string;
  /** Unit string displayed after the label (e.g. "s"). */
  unit?: string;
  /** Value increment per column. */
  unitsPerCell: number;
}

/** Shared styling for rendered chart overlays. */
export interface BaseChartConfig {
  /** Chart title displayed centered above the matrix. */
  title?: string;
  /** CSS font string for tick labels. Defaults to "12px sans-serif". */
  font?: string;
  /** CSS font string for the chart title. Defaults to "bold 16px sans-serif". */
  titleFont?: string;
  /** Color of tick marks and axis lines. Defaults to "#999". */
  tickColor?: string;
  /** Color of text labels (ticks, axis labels, title). Defaults to "#ccc". */
  labelColor?: string;
  /** Background color of the margin areas. Defaults to "#1a1a1a". */
  backgroundColor?: string;
}

/**
 * Heatmap chart configuration: axes, colorbar, labels, units, and title.
 *
 * Heatmaps are the default chart type when `type` is omitted.
 */
export interface HeatmapChartConfig extends BaseChartConfig {
  /** Explicit chart kind. Omit or set to `'heatmap'` for matrix colormaps. */
  type?: 'heatmap';
  /**
   * X axis configuration. Use {@link AxisConfig} for static charts or
   * {@link StreamingAxisConfig} for streaming charts where the X axis
   * auto-increments.
   */
  xAxis?: AxisConfig | StreamingAxisConfig;
  /** Y axis configuration. */
  yAxis?: AxisConfig;
  /**
   * Unit string for pixel values displayed in the tooltip (e.g. "rad", "dB").
   * The default colorbar uses this unit as its vertical label.
   */
  valueUnit?: string;
  /**
   * Whether to show the default right-side colorbar. Defaults to true when a
   * chart overlay is configured.
   */
  colorbar?: boolean;
}

/** RGBA line color: RGB in 0..255 and alpha in 0..1. */
export type RgbaColor = [
  red: number,
  green: number,
  blue: number,
  alpha: number,
];

/** Default line X axis: sample indices 0..N-1. */
export interface LineIndexXAxisConfig {
  /** Axis mode. Omit for the default index axis. */
  kind?: 'index';
  /** Human-readable axis label. */
  label?: string;
  /** Unit string displayed after the label. */
  unit?: string;
}

/** Line X axis generated with linspace(min, max, sampleCount). */
export interface LineLinearXAxisConfig {
  /** Axis mode for a generated linear coordinate array. */
  kind?: 'linear';
  /** Human-readable axis label. */
  label?: string;
  /** Unit string displayed after the label. */
  unit?: string;
  /** First X coordinate. */
  min: number;
  /** Last X coordinate. */
  max: number;
}

/** Line X axis supplied as one shared explicit coordinate array. */
export interface LineExplicitXAxisConfig {
  /** Axis mode for explicit shared X values. */
  kind: 'explicit';
  /** Human-readable axis label. */
  label?: string;
  /** Unit string displayed after the label. */
  unit?: string;
  /** Finite, strictly increasing X values shared by every series. */
  values: Float32Array | readonly number[];
}

/** Streaming line X axis that advances by a fixed amount per sample. */
export interface LineStreamingXAxisConfig {
  /** Axis mode for scrolling streaming line charts. */
  kind: 'streaming';
  /** Human-readable axis label. */
  label?: string;
  /** Unit string displayed after the label. */
  unit?: string;
  /** X coordinate of sample zero. Defaults to 0. */
  start?: number;
  /** X coordinate increment per sample. */
  unitsPerSample: number;
}

/** Shared line X axis configuration. */
export type LineXAxisConfig =
  | LineIndexXAxisConfig
  | LineLinearXAxisConfig
  | LineExplicitXAxisConfig
  | LineStreamingXAxisConfig;

/** Y axis configuration for line charts. */
export interface LineYAxisConfig {
  /** Human-readable axis label. */
  label?: string;
  /** Unit string displayed after the label. */
  unit?: string;
  /** Range mode. Defaults to sticky auto. */
  rangeMode?: 'stickyAuto' | 'fixed';
  /** Fixed minimum when `rangeMode` is `'fixed'`. */
  min?: number;
  /** Fixed maximum when `rangeMode` is `'fixed'`. */
  max?: number;
  /** Fractional padding added above/below auto ranges. Defaults to 0.05. */
  paddingRatio?: number;
}

/** Configuration for WebGPU-rendered line charts. */
export interface LineChartConfig extends BaseChartConfig {
  /** Explicit chart kind for line plots. */
  type: 'line';
  /** Shared X axis configuration. Defaults to sample indices. */
  xAxis?: LineXAxisConfig;
  /** Line Y axis configuration. Defaults to sticky auto range. */
  yAxis?: LineYAxisConfig;
  /** Draw background grid lines at tick positions. Defaults to false. */
  grid?: boolean;
  /** Show the clickable right-side legend. Defaults to true. */
  legend?: boolean;
}

/** Chart configuration for heatmaps or line plots. */
export type ChartConfig = HeatmapChartConfig | LineChartConfig;

/**
 * Options for creating a LeibnizFast instance.
 */
export interface CreateOptions {
  /** Initial colormap to use. Defaults to 'viridis'. */
  colormap?: ColormapName;
  /** Enable performance timing logs in the browser console. Defaults to false. */
  debug?: boolean;
  /** Chart configuration (axes, colorbar, title, labels). Omit for raw matrix view. */
  chart?: ChartConfig;
}

/** One named line series. */
export interface LineSeriesInput {
  /** Stable series id. Defaults to `name`. */
  id?: string;
  /** Display name used in the legend and hover info. */
  name: string;
  /** RGBA line color: RGB in 0..255, alpha in 0..1. */
  color: RgbaColor;
  /** Y values for this series. All series must have the same length. */
  data: Float32Array;
  /** Initial visibility. Defaults to true. */
  visible?: boolean;
}

/** New samples for one existing line series in a scrolling update. */
export interface LineSeriesUpdate {
  /** Existing series id. */
  id: string;
  /** New Y samples for this series. Length must equal `newSamples`. */
  data: Float32Array;
}

/** Options for full line data uploads and animated replacement frames. */
export interface LineDataOptions {
  /** Override the line chart X axis for this upload. */
  xAxis?: LineXAxisConfig;
  /** Override the line chart Y axis for this upload. */
  yAxis?: LineYAxisConfig;
  /** Total samples received so far for streaming X axes. */
  xOffset?: number;
  /** Recompute sticky auto Y range from the current visible series. */
  resetYRange?: boolean;
}

/** Options for scrolling line updates. */
export interface LineScrolledDataOptions {
  /** Number of new samples appended on the right side. */
  newSamples: number;
  /** Total samples received so far for streaming X axes. */
  xOffset?: number;
}

/**
 * Options for setting matrix data.
 */
export interface DataOptions {
  /** Number of rows in the matrix. */
  rows: number;
  /** Number of columns in the matrix. */
  cols: number;
  /**
   * Streaming X axis: total number of columns received so far, including
   * columns that have scrolled off the left edge. Used to compute the
   * current time window for the X axis. Only meaningful when the chart
   * uses a {@link StreamingAxisConfig} for the X axis.
   */
  xOffset?: number;
}

/**
 * A row-major chunk for {@link LeibnizFast.setDataChunks}.
 */
export interface DataChunk {
  /** Zero-based first row represented by this chunk. Chunks must be sequential. */
  startRow: number;
  /** Row-major f32 data. Length must be a positive multiple of `cols`. */
  data: Float32Array;
}

/**
 * Options for chunked matrix upload.
 */
export interface ChunkedDataOptions extends DataOptions {
  /**
   * Retain a CPU-side Float32Array cache for hover values and future full
   * recolorization. Defaults to false to minimize peak CPU memory.
   */
  retainData?: boolean;
  /**
   * Explicit data range. When omitted, chunks are scanned once while uploading.
   */
  range?: DataRange;
}

/** Heatmap hover information enriched with axis coordinates when configured. */
export interface HeatmapHoverInfo {
  /** Hover result kind. Omitted for backward compatibility with existing users. */
  kind?: 'heatmap';
  /** Zero-based row index in the matrix. */
  row: number;
  /** Zero-based column index in the matrix. */
  col: number;
  /** Raw data value at (row, col). */
  value: number;
  /**
   * Whether `value` is available. False when data was uploaded with
   * `retainData: false`; in that case `value` is `NaN`.
   */
  valueAvailable: boolean;
  /** Y axis value mapped from the row index (present when yAxis is configured). */
  y?: number;
  /** X axis value mapped from the column index (present when xAxis is configured). */
  x?: number;
  /** Y axis unit string (e.g. "m"), copied from the chart config. */
  yUnit?: string;
  /** X axis unit string (e.g. "s"), copied from the chart config. */
  xUnit?: string;
  /** Value unit string (e.g. "rad"), copied from the chart config. */
  valueUnit?: string;
  /** RGBA color currently used to render this heatmap value, when available. */
  color?: RgbaColor;
}

/** One finite line-series value resolved at the current hover X coordinate. */
export interface LineHoverPoint {
  /** Stable id of the series. */
  seriesId: string;
  /** Display name of the series. */
  seriesName: string;
  /** Zero-based series index in the current chart. */
  seriesIndex: number;
  /** Lower logical sample index of the interpolated segment. */
  sampleIndex: number;
  /** X axis coordinate at this hover point. */
  x: number;
  /** Interpolated Y value at this hover point. */
  value: number;
  /** Series color as supplied by the user. */
  color: RgbaColor;
}

/** Line hover information for all visible series at the current mouse X. */
export interface LineHoverInfo {
  /** Hover result kind. */
  kind: 'line';
  /** X axis coordinate at the mouse position. */
  x: number;
  /** Mouse X position in CSS pixels relative to the plot area. */
  mouseX: number;
  /** Mouse Y position in CSS pixels relative to the plot area. */
  mouseY: number;
  /** X axis unit string. */
  xUnit?: string;
  /** Y axis unit string. */
  yUnit?: string;
  /** Finite visible series values resolved at this X coordinate. */
  points: LineHoverPoint[];
}

/**
 * Information about a hovered chart value.
 */
export type HoverInfo = HeatmapHoverInfo | LineHoverInfo;

/**
 * Callback invoked when the user hovers over a chart value region.
 *
 * @param info - Heatmap cell information or line-series values at the current X coordinate
 */
export type HoverCallback = (info: HoverInfo) => void;

/**
 * Options for scrolled streaming update via setDataScrolled.
 *
 * Use when the buffer shifts left by `newCols` and writes new data
 * at the right edge (waterfall / scrolling time series pattern).
 */
export interface ScrolledDataOptions extends DataOptions {
  /**
   * Number of new columns added at the right edge since the last frame.
   * The GPU texture scrolls left by this amount and only the new columns
   * are colormapped, reducing per-frame work from O(rows × cols) to
   * O(rows × newCols).
   */
  newCols: number;
}

/**
 * Options for streaming data upload via beginData/appendChunk/endData.
 */
export interface StreamingDataOptions {
  /** Number of rows in the matrix. */
  rows: number;
  /** Number of columns in the matrix. */
  cols: number;
  /**
   * Explicit data range. When set, streaming uploads skip per-chunk min/max
   * tracking and use this fixed colormap range.
   */
  range?: DataRange;
  /**
   * Streaming uploads retain CPU-side data for hover values. Use
   * `setDataChunks(..., { retainData: false })` for a no-retention upload.
   */
  retainData?: boolean;
}

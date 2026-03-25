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

/**
 * Chart configuration: axes, labels, units, and title.
 *
 * @example
 * ```ts
 * const chart: ChartConfig = {
 *   title: 'Spectrogram',
 *   xAxis: { label: 'Time', unit: 's', min: 0, max: 10 },
 *   yAxis: { label: 'Frequency', unit: 'Hz', min: 0, max: 22050 },
 * };
 * ```
 */
export interface ChartConfig {
  /** Chart title displayed centered above the matrix. */
  title?: string;
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
   * When set, the tooltip value line shows "Value: 0.21 rad".
   */
  valueUnit?: string;
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
 * Options for creating a LeibnizFast instance.
 */
export interface CreateOptions {
  /** Initial colormap to use. Defaults to 'viridis'. */
  colormap?: ColormapName;
  /** Enable performance timing logs in the browser console. Defaults to false. */
  debug?: boolean;
  /** Chart configuration (axes, title, labels). Omit for raw matrix view. */
  chart?: ChartConfig;
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
 * Information about a hovered matrix cell, enriched with axis coordinates
 * when a chart configuration is present.
 */
export interface HoverInfo {
  /** Zero-based row index in the matrix. */
  row: number;
  /** Zero-based column index in the matrix. */
  col: number;
  /** Raw data value at (row, col). */
  value: number;
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
}

/**
 * Callback invoked when the user hovers over a matrix cell.
 *
 * @param info - Hover information including matrix indices, value, and axis coordinates
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
}

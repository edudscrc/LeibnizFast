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
 * Options for creating a LeibnizFast instance.
 */
export interface CreateOptions {
  /** Initial colormap to use. Defaults to 'viridis'. */
  colormap?: ColormapName;
}

/**
 * Options for setting matrix data.
 */
export interface DataOptions {
  /** Number of rows in the matrix. */
  rows: number;
  /** Number of columns in the matrix. */
  cols: number;
}

/**
 * Callback invoked when the user hovers over a matrix cell.
 *
 * @param row - Zero-based row index
 * @param col - Zero-based column index
 * @param value - The data value at (row, col)
 */
export type HoverCallback = (row: number, col: number, value: number) => void;

/**
 * Options for streaming data upload via beginData/appendChunk/endData.
 */
export interface StreamingDataOptions {
  /** Number of rows in the matrix. */
  rows: number;
  /** Number of columns in the matrix. */
  cols: number;
}

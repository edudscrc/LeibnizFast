/**
 * LeibnizFast — GPU-accelerated 2D matrix visualization.
 *
 * Thin TypeScript wrapper around the Rust/WASM core.
 * Handles WASM initialization, DOM event forwarding, and provides
 * a clean typed API.
 *
 * @example
 * ```ts
 * import { LeibnizFast } from 'leibniz-fast';
 *
 * const canvas = document.getElementById('canvas') as HTMLCanvasElement;
 * const viewer = await LeibnizFast.create(canvas, { colormap: 'viridis' });
 * viewer.setData(new Float32Array(data), { rows: 1000, cols: 2000 });
 * viewer.onHover((row, col, value) => console.log(row, col, value));
 * ```
 */

import type {
  ColormapName,
  CreateOptions,
  DataOptions,
  HoverCallback,
  StreamingDataOptions,
} from './types';

// Re-export types for consumers
export type {
  ColormapName,
  CreateOptions,
  DataOptions,
  HoverCallback,
  StreamingDataOptions,
};

/** Cached WASM module — initialized once on first `create()` call. */
let wasmModule: typeof import('../pkg/leibniz_fast') | null = null;

/**
 * Initialize the WASM module if not already loaded.
 * Caches the result so subsequent calls are instant.
 */
async function ensureWasmLoaded(): Promise<
  typeof import('../pkg/leibniz_fast')
> {
  if (!wasmModule) {
    // Dynamic import of the wasm-pack generated module
    wasmModule = await import('../pkg/leibniz_fast');
  }
  return wasmModule;
}

/**
 * GPU-accelerated 2D matrix visualization viewer.
 *
 * Use the static `create()` method to instantiate — do not call the
 * constructor directly.
 */
export class LeibnizFast {
  /** Internal WASM instance */
  private inner: any;
  /** Canvas element this viewer is attached to */
  private canvas: HTMLCanvasElement;
  /** Performance timing enabled */
  private debug: boolean;
  /** Bound event handlers for cleanup */
  private boundHandlers: {
    mousedown: (e: MouseEvent) => void;
    mousemove: (e: MouseEvent) => void;
    mouseup: (e: MouseEvent) => void;
    wheel: (e: WheelEvent) => void;
    resize: () => void;
  };

  private constructor(inner: any, canvas: HTMLCanvasElement, debug: boolean) {
    this.inner = inner;
    this.canvas = canvas;
    this.debug = debug;

    // Bind DOM event handlers
    this.boundHandlers = {
      mousedown: (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        this.inner.onMouseDown(e.clientX - rect.left, e.clientY - rect.top);
      },
      mousemove: (e: MouseEvent) => {
        const rect = canvas.getBoundingClientRect();
        this.inner.onMouseMove(e.clientX - rect.left, e.clientY - rect.top);
      },
      mouseup: () => {
        this.inner.onMouseUp();
      },
      wheel: (e: WheelEvent) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        this.inner.onWheel(
          e.clientX - rect.left,
          e.clientY - rect.top,
          -e.deltaY,
        );
      },
      resize: () => {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        this.inner.resize(canvas.width, canvas.height);
      },
    };

    // Register event listeners
    canvas.addEventListener('mousedown', this.boundHandlers.mousedown);
    canvas.addEventListener('mousemove', this.boundHandlers.mousemove);
    window.addEventListener('mouseup', this.boundHandlers.mouseup);
    canvas.addEventListener('wheel', this.boundHandlers.wheel, {
      passive: false,
    });
    window.addEventListener('resize', this.boundHandlers.resize);
  }

  /**
   * Create a new LeibnizFast viewer attached to the given canvas.
   *
   * Initializes WASM (if needed) and the GPU context.
   *
   * @param canvas - The HTML canvas element to render into
   * @param options - Optional configuration (colormap, etc.)
   * @returns A new LeibnizFast instance
   */
  static async create(
    canvas: HTMLCanvasElement,
    options?: CreateOptions,
  ): Promise<LeibnizFast> {
    const debug = options?.debug ?? false;
    const t0 = debug ? performance.now() : 0;
    const wasm = await ensureWasmLoaded();
    if (debug)
      console.log(
        `[perf] ensureWasmLoaded: ${(performance.now() - t0).toFixed(2)}ms`,
      );
    const t1 = debug ? performance.now() : 0;
    const inner = await wasm.LeibnizFast.create(
      canvas,
      options?.colormap ?? undefined,
      debug,
    );
    if (debug)
      console.log(
        `[perf] LeibnizFast.create (WASM): ${(performance.now() - t1).toFixed(2)}ms`,
      );
    if (debug)
      console.log(
        `[perf] LeibnizFast.create (total): ${(performance.now() - t0).toFixed(2)}ms`,
      );
    return new LeibnizFast(inner, canvas, debug);
  }

  /**
   * Set the matrix data to visualize.
   *
   * @param data - Flat Float32Array in row-major order
   * @param options - Matrix dimensions (rows, cols)
   */
  /** Time a synchronous call, logging duration when debug is enabled. */
  private timeSync<T>(label: string, fn: () => T): T {
    if (!this.debug) return fn();
    const t0 = performance.now();
    const result = fn();
    console.log(`[perf] ${label}: ${(performance.now() - t0).toFixed(2)}ms`);
    return result;
  }

  setData(data: Float32Array, options: DataOptions): void {
    this.timeSync('JS setData', () =>
      this.inner.setData(data, options.rows, options.cols),
    );
  }

  /**
   * Change the colormap used for visualization.
   *
   * @param name - One of the available colormap names
   */
  setColormap(name: ColormapName): void {
    this.timeSync('JS setColormap', () => this.inner.setColormap(name));
  }

  /**
   * Set the data range for colormap mapping.
   *
   * Values at or below `min` map to the first colormap color,
   * values at or above `max` map to the last.
   *
   * @param min - Minimum data value
   * @param max - Maximum data value
   */
  setRange(min: number, max: number): void {
    this.inner.setRange(min, max);
  }

  /**
   * Register a callback for hover events.
   *
   * @param callback - Called with (row, col, value) on cell hover
   */
  onHover(callback: HoverCallback): void {
    this.inner.onHover(callback);
  }

  /**
   * Begin a streaming data upload.
   *
   * Allocates buffers for the full matrix. Follow with `appendChunk()`
   * calls and finalize with `endData()`.
   *
   * @param options - Matrix dimensions (rows, cols)
   */
  beginData(options: StreamingDataOptions): void {
    this.timeSync('JS beginData', () =>
      this.inner.beginData(options.rows, options.cols),
    );
  }

  /**
   * Append a chunk of rows to the in-progress streaming upload.
   *
   * @param data - Float32Array containing a whole number of rows
   * @param startRow - Zero-based index of the first row in this chunk
   */
  appendChunk(data: Float32Array, startRow: number): void {
    this.timeSync('JS appendChunk', () =>
      this.inner.appendChunk(data, startRow),
    );
  }

  /**
   * Finalize a streaming upload. Computes data range, rebuilds
   * pipelines, and renders.
   */
  endData(): void {
    this.timeSync('JS endData', () => this.inner.endData());
  }

  /**
   * Get the maximum number of matrix elements supported by this device.
   *
   * @returns Maximum number of f32 elements that fit in a single GPU buffer
   */
  getMaxMatrixElements(): number {
    return this.inner.getMaxMatrixElements();
  }

  /**
   * Get the maximum matrix dimension (rows or cols) this device supports.
   *
   * Matrices with rows or cols exceeding this value will fail to render
   * because the output texture would exceed the GPU's texture size limit.
   *
   * @returns Maximum rows or cols value
   */
  getMaxTextureDimension(): number {
    return this.inner.getMaxTextureDimension();
  }

  /**
   * Clean up all resources (GPU, event listeners, WASM).
   * Must be called when the viewer is no longer needed.
   */
  destroy(): void {
    // Remove event listeners
    this.canvas.removeEventListener('mousedown', this.boundHandlers.mousedown);
    this.canvas.removeEventListener('mousemove', this.boundHandlers.mousemove);
    window.removeEventListener('mouseup', this.boundHandlers.mouseup);
    this.canvas.removeEventListener('wheel', this.boundHandlers.wheel);
    window.removeEventListener('resize', this.boundHandlers.resize);

    // Destroy WASM instance (frees GPU resources)
    this.inner.destroy();
  }
}

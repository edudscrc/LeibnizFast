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

import type { LeibnizFast as WasmLeibnizFast } from '../pkg/leibniz_fast';
import type {
  AxisConfig,
  ChartConfig,
  ColormapName,
  CreateOptions,
  DataOptions,
  HoverCallback,
  ScrolledDataOptions,
  StreamingAxisConfig,
  StreamingDataOptions,
} from './types';
import {
  computeLayout,
  isStreamingAxis,
  renderOverlay,
  uvToVisibleRange,
} from './axes';
import type { LayoutRect, VisibleRange } from './axes';

// Re-export types for consumers
export type {
  AxisConfig,
  ChartConfig,
  ColormapName,
  CreateOptions,
  DataOptions,
  HoverCallback,
  ScrolledDataOptions,
  StreamingAxisConfig,
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
    const mod = await import('../pkg/leibniz_fast');
    // The default export is the init function that loads the .wasm binary.
    // It must be called before any WASM class can be used.
    await mod.default();
    wasmModule = mod;
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
  private inner: WasmLeibnizFast;
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

  // --- Chart overlay state ---
  /** Chart configuration (axes, title, labels). Null when no chart mode. */
  private chartConfig: ChartConfig | null = null;
  /** Wrapper div that contains both canvases. */
  private wrapperDiv: HTMLDivElement | null = null;
  /** 2D overlay canvas for axes/title rendering. */
  private overlayCanvas: HTMLCanvasElement | null = null;
  /** 2D rendering context for the overlay canvas. */
  private overlayCtx: CanvasRenderingContext2D | null = null;
  /** Current layout (matrix area position within container). */
  private layout: LayoutRect = { x: 0, y: 0, width: 0, height: 0 };
  /** Streaming X axis: columns currently displayed in the matrix. */
  private streamingDisplayCols: number = 0;
  /** Streaming X axis: total columns received (including scrolled-off). */
  private streamingXOffset: number = 0;
  /** Cached canvas bounding rect, invalidated on resize. */
  private cachedCanvasRect: DOMRect | null = null;
  /** Whether an overlay rAF redraw is already scheduled. */
  private overlayDirty: boolean = false;

  private constructor(
    inner: WasmLeibnizFast,
    canvas: HTMLCanvasElement,
    debug: boolean,
    chartConfig: ChartConfig | null,
  ) {
    this.inner = inner;
    this.canvas = canvas;
    this.debug = debug;

    if (chartConfig) {
      this.chartConfig = chartConfig;
      this.setupChartOverlay();
    }

    // Bind DOM event handlers
    this.boundHandlers = {
      mousedown: (e: MouseEvent) => {
        const rect = this.getCanvasRect();
        this.inner.onMouseDown(e.clientX - rect.left, e.clientY - rect.top);
      },
      mousemove: (e: MouseEvent) => {
        const rect = this.getCanvasRect();
        this.inner.onMouseMove(e.clientX - rect.left, e.clientY - rect.top);
        this.scheduleOverlayUpdate();
      },
      mouseup: () => {
        this.inner.onMouseUp();
      },
      wheel: (e: WheelEvent) => {
        e.preventDefault();
        const rect = this.getCanvasRect();
        this.inner.onWheel(
          e.clientX - rect.left,
          e.clientY - rect.top,
          -e.deltaY,
        );
        this.scheduleOverlayUpdate();
      },
      resize: () => {
        this.handleResize();
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

    // Compute initial layout and render overlay (must happen after DOM setup)
    this.handleResize();
  }

  /**
   * Create a new LeibnizFast viewer attached to the given canvas.
   *
   * Initializes WASM (if needed) and the GPU context.
   *
   * @param canvas - The HTML canvas element to render into
   * @param options - Optional configuration (colormap, chart, etc.)
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
    return new LeibnizFast(inner, canvas, debug, options?.chart ?? null);
  }

  /** Time a synchronous call, logging duration when debug is enabled. */
  private timeSync<T>(label: string, fn: () => T): T {
    if (!this.debug) return fn();
    const t0 = performance.now();
    const result = fn();
    console.log(`[perf] ${label}: ${(performance.now() - t0).toFixed(2)}ms`);
    return result;
  }

  /**
   * Set the matrix data to visualize.
   *
   * @param data - Flat Float32Array in row-major order
   * @param options - Matrix dimensions (rows, cols)
   */
  setData(data: Float32Array, options: DataOptions): void {
    this.timeSync('JS setData', () =>
      this.inner.setData(data, options.rows, options.cols),
    );
    if (this.chartConfig?.xAxis && isStreamingAxis(this.chartConfig.xAxis)) {
      this.streamingDisplayCols = options.cols;
      if (options.xOffset !== undefined) {
        this.streamingXOffset = options.xOffset;
      }
    }
    this.updateOverlay();
  }

  /**
   * Scrolled streaming update: shift existing pixels left and only colormap
   * new columns at the right edge.
   *
   * Use this instead of `setData` for waterfall / scrolling time series where
   * the buffer shifts left by `newCols` each frame. Reduces per-frame GPU work
   * from O(rows × cols) to O(rows × newCols).
   *
   * **Requires `setRange()` to be called first.** Without a fixed range, this
   * falls back to a full `setData`.
   *
   * @param data - Full Float32Array in row-major order (used for hover lookups)
   * @param options - Matrix dimensions and number of new columns
   */
  setDataScrolled(data: Float32Array, options: ScrolledDataOptions): void {
    this.timeSync('JS setDataScrolled', () =>
      this.inner.setDataScrolled(
        data,
        options.rows,
        options.cols,
        options.newCols,
      ),
    );
    if (this.chartConfig?.xAxis && isStreamingAxis(this.chartConfig.xAxis)) {
      this.streamingDisplayCols = options.cols;
      if (options.xOffset !== undefined) {
        this.streamingXOffset = options.xOffset;
      }
    }
    this.scheduleOverlayUpdate();
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
    if (this.chartConfig?.xAxis && isStreamingAxis(this.chartConfig.xAxis)) {
      this.streamingDisplayCols = options.cols;
    }
  }

  /**
   * Begin a streaming update, reusing GPU resources when dimensions match.
   *
   * Fast path for real-time streaming: reuses the existing Float32Array
   * and GPU staging buffer when called with the same dimensions as the
   * previous frame, avoiding per-frame allocation and pipeline rebuild.
   * Falls back to `beginData()` on first use or when dimensions change.
   *
   * @param options - Matrix dimensions (rows, cols)
   */
  beginUpdate(options: StreamingDataOptions): void {
    this.timeSync('JS beginUpdate', () =>
      this.inner.beginUpdate(options.rows, options.cols),
    );
  }

  /**
   * Abort an in-progress streaming upload.
   *
   * Restores reusable resources for the next `beginUpdate()` call.
   * No-op if no upload is in progress.
   */
  abortData(): void {
    this.inner.abortData();
  }

  /**
   * Render a single frame without modifying data.
   *
   * Useful for decoupled rendering: ingest data at the source rate,
   * then call `render()` at display refresh rate via requestAnimationFrame.
   */
  render(): void {
    this.inner.render();
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
    this.updateOverlay();
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
   * Update the chart configuration (axes, title, labels).
   *
   * If no chart overlay exists yet, it will be created. If called with
   * `null`, the overlay is removed and the viewer reverts to raw matrix mode.
   *
   * @param config - Chart configuration, or null to remove
   */
  setChart(config: ChartConfig | null): void {
    if (config && !this.chartConfig) {
      this.chartConfig = config;
      this.setupChartOverlay();
    } else if (!config && this.chartConfig) {
      this.teardownChartOverlay();
      this.chartConfig = null;
    } else {
      this.chartConfig = config;
    }
    this.handleResize();
  }

  /**
   * Set the chart title.
   *
   * @param title - Title text displayed centered above the matrix
   */
  setTitle(title: string): void {
    if (!this.chartConfig) {
      this.chartConfig = { title };
      this.setupChartOverlay();
      this.handleResize();
    } else {
      this.chartConfig.title = title;
      this.updateOverlay();
    }
  }

  /**
   * Clean up all resources (GPU, event listeners, WASM, overlay DOM).
   * Must be called when the viewer is no longer needed.
   */
  destroy(): void {
    // Remove event listeners
    this.canvas.removeEventListener('mousedown', this.boundHandlers.mousedown);
    this.canvas.removeEventListener('mousemove', this.boundHandlers.mousemove);
    window.removeEventListener('mouseup', this.boundHandlers.mouseup);
    this.canvas.removeEventListener('wheel', this.boundHandlers.wheel);
    window.removeEventListener('resize', this.boundHandlers.resize);

    // Clean up overlay DOM
    this.teardownChartOverlay();

    // Destroy WASM instance (frees GPU resources)
    this.inner.destroy();
  }

  // ---------------------------------------------------------------------------
  // Private: chart overlay setup & rendering
  // ---------------------------------------------------------------------------

  /**
   * Create the wrapper div and overlay canvas for chart annotations.
   * Reparents the WebGPU canvas inside a container div.
   */
  private setupChartOverlay(): void {
    if (this.wrapperDiv) return; // already set up

    const parent = this.canvas.parentElement;
    if (!parent) return;

    // Create wrapper div matching the canvas's CSS size
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.width = this.canvas.style.width || '100%';
    wrapper.style.height = this.canvas.style.height || '100%';

    // Copy computed dimensions if inline styles aren't set
    const computedStyle = getComputedStyle(this.canvas);
    if (!this.canvas.style.width) {
      wrapper.style.width = computedStyle.width;
    }
    if (!this.canvas.style.height) {
      wrapper.style.height = computedStyle.height;
    }

    // Reparent: insert wrapper where canvas was, move canvas inside
    parent.insertBefore(wrapper, this.canvas);
    wrapper.appendChild(this.canvas);

    // Position the WebGPU canvas absolutely within the wrapper
    this.canvas.style.position = 'absolute';

    // Create overlay canvas
    const overlay = document.createElement('canvas');
    overlay.style.position = 'absolute';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.pointerEvents = 'none';
    wrapper.appendChild(overlay);

    this.wrapperDiv = wrapper;
    this.overlayCanvas = overlay;
    this.overlayCtx = overlay.getContext('2d');
  }

  /**
   * Remove the overlay canvas and wrapper div, restoring the original
   * canvas position in the DOM.
   */
  private teardownChartOverlay(): void {
    if (!this.wrapperDiv) return;

    const parent = this.wrapperDiv.parentElement;
    if (parent) {
      // Restore canvas to its original position
      this.canvas.style.position = '';
      this.canvas.style.left = '';
      this.canvas.style.top = '';
      this.canvas.style.width = '';
      this.canvas.style.height = '';
      parent.insertBefore(this.canvas, this.wrapperDiv);
      parent.removeChild(this.wrapperDiv);
    }

    if (this.overlayCanvas) {
      this.overlayCanvas.remove();
    }

    this.wrapperDiv = null;
    this.overlayCanvas = null;
    this.overlayCtx = null;
  }

  /**
   * Handle window resize: recalculate layout, resize both canvases,
   * update the WASM renderer, and redraw the overlay.
   */
  private handleResize(): void {
    this.cachedCanvasRect = null; // invalidate on resize
    const dpr = window.devicePixelRatio || 1;

    if (this.chartConfig && this.wrapperDiv && this.overlayCtx) {
      const wrapperRect = this.wrapperDiv.getBoundingClientRect();
      const containerW = wrapperRect.width;
      const containerH = wrapperRect.height;

      // Resize overlay canvas to cover the full container
      this.overlayCanvas!.width = containerW * dpr;
      this.overlayCanvas!.height = containerH * dpr;
      this.overlayCanvas!.style.width = `${containerW}px`;
      this.overlayCanvas!.style.height = `${containerH}px`;

      // Compute layout (margins for axes/title)
      this.layout = computeLayout(
        containerW,
        containerH,
        this.chartConfig,
        this.overlayCtx,
      );

      // Position and resize the WebGPU canvas to the matrix area
      this.canvas.style.left = `${this.layout.x}px`;
      this.canvas.style.top = `${this.layout.y}px`;
      this.canvas.style.width = `${this.layout.width}px`;
      this.canvas.style.height = `${this.layout.height}px`;
      this.canvas.width = this.layout.width * dpr;
      this.canvas.height = this.layout.height * dpr;

      // Update WASM renderer with matrix area size
      this.inner.resize(this.canvas.width, this.canvas.height);

      // Redraw overlay
      this.updateOverlay();
    } else {
      // No chart mode: standard resize
      const rect = this.canvas.getBoundingClientRect();
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.inner.resize(this.canvas.width, this.canvas.height);
    }
  }

  /**
   * Get the cached canvas bounding rect, computing it if invalidated.
   * Avoids forcing a layout reflow on every mouse event.
   */
  private getCanvasRect(): DOMRect {
    if (!this.cachedCanvasRect) {
      this.cachedCanvasRect = this.canvas.getBoundingClientRect();
    }
    return this.cachedCanvasRect;
  }

  /**
   * Schedule an overlay redraw on the next animation frame.
   * Multiple calls per frame are coalesced into a single redraw.
   */
  private scheduleOverlayUpdate(): void {
    if (this.overlayDirty) return;
    this.overlayDirty = true;
    requestAnimationFrame(() => {
      this.overlayDirty = false;
      this.updateOverlay();
    });
  }

  /**
   * Redraw the chart overlay (axes, ticks, labels, title).
   * Called after data changes, pan/zoom, and resize.
   */
  private updateOverlay(): void {
    if (!this.chartConfig || !this.overlayCtx || !this.wrapperDiv) return;

    const dpr = window.devicePixelRatio || 1;
    const wrapperRect = this.wrapperDiv.getBoundingClientRect();

    const visible = this.computeVisibleRange();

    renderOverlay(
      this.overlayCtx,
      this.layout,
      this.chartConfig,
      visible,
      wrapperRect.width,
      wrapperRect.height,
      dpr,
    );
  }

  /**
   * Compute the visible data range by mapping camera UV coordinates
   * to axis data coordinates.
   */
  private computeVisibleRange(): VisibleRange {
    const uv = this.inner.getVisibleRange();
    const uvOffset: [number, number] = [uv[0], uv[1]];
    const uvScale: [number, number] = [uv[2], uv[3]];

    // Determine full axis ranges
    const xRange = this.getFullXRange();
    const yRange = this.getFullYRange();

    return uvToVisibleRange(
      uvOffset,
      uvScale,
      xRange[0],
      xRange[1],
      yRange[0],
      yRange[1],
    );
  }

  /**
   * Get the full X axis range from the chart configuration.
   * For streaming axes, computes a sliding window:
   *   xMax = totalColsReceived * unitsPerCell
   *   xMin = (totalColsReceived - displayCols) * unitsPerCell
   */
  private getFullXRange(): [number, number] {
    if (!this.chartConfig?.xAxis) return [0, 1];
    const xAxis = this.chartConfig.xAxis;
    if (isStreamingAxis(xAxis)) {
      const xMax = this.streamingXOffset * xAxis.unitsPerCell;
      // Allow negative values before the buffer fills — the "0" tick
      // marks where streaming data begins, and the left portion shows
      // negative time (unfilled region).
      const xMin =
        (this.streamingXOffset - this.streamingDisplayCols) *
        xAxis.unitsPerCell;
      return [xMin, xMax];
    }
    return [xAxis.min, xAxis.max];
  }

  /**
   * Get the full Y axis range from the chart configuration.
   */
  private getFullYRange(): [number, number] {
    if (!this.chartConfig?.yAxis) return [0, 1];
    return [this.chartConfig.yAxis.min, this.chartConfig.yAxis.max];
  }
}

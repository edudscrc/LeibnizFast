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
  HoverInfo,
  ScrolledDataOptions,
  StreamingAxisConfig,
  StreamingDataOptions,
} from './types';
import {
  computeLayout,
  drawAxisHighlight,
  drawSelectionRect,
  isStreamingAxis,
  renderOverlay,
  uvToVisibleRange,
} from './axes';
import type { LayoutRect, VisibleRange } from './axes';

// ---------------------------------------------------------------------------
// Interaction types
// ---------------------------------------------------------------------------

/** Which region of the chart the mouse is over. */
type HitRegion = 'matrix' | 'x-axis' | 'y-axis' | 'none';

/** Current interaction mode state machine. */
type InteractionMode =
  | { type: 'idle' }
  | { type: 'matrix-pan' }
  | { type: 'axis-pan'; axis: 'x' | 'y'; lastPos: number }
  | {
      type: 'rect-select';
      startX: number;
      startY: number;
      currentX: number;
      currentY: number;
    }
  | {
      type: 'axis-select';
      axis: 'x' | 'y';
      startPos: number;
      currentPos: number;
    };

/** Minimum drag distance in CSS pixels to count as a selection. */
const MIN_SELECTION_PX = 5;

/** Duration of the zoom-reset animation in milliseconds. */
const ANIMATION_DURATION_MS = 300;

// Re-export types for consumers
export type {
  AxisConfig,
  ChartConfig,
  ColormapName,
  CreateOptions,
  DataOptions,
  HoverCallback,
  HoverInfo,
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
    mouseenter: () => void;
    mouseleave: () => void;
    wheel: (e: WheelEvent) => void;
    contextmenu: (e: MouseEvent) => void;
    dblclick: (e: MouseEvent) => void;
    resize: () => void;
  };

  // --- Hover / tooltip state ---
  /** User-registered hover callback. */
  private hoverCallback: HoverCallback | null = null;
  /** Last known mouse X in canvas-local pixels. */
  private lastMouseX: number = 0;
  /** Last known mouse Y in canvas-local pixels. */
  private lastMouseY: number = 0;
  /** Whether the mouse pointer is currently inside the canvas. */
  private mouseInside: boolean = false;

  // --- Interaction state ---
  /** Current interaction mode. */
  private interactionMode: InteractionMode = { type: 'idle' };
  /** Which axis region the mouse is currently hovering (for highlight). */
  private hoveredAxis: 'x' | 'y' | null = null;
  /** Active zoom-reset animation frame ID, or null if no animation is running. */
  private zoomAnimationId: number | null = null;

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
  /** Whether an overlay rAF redraw is already scheduled. */
  private overlayDirty: boolean = false;
  /** Current matrix row count (for hover coordinate mapping). */
  private matrixRows: number = 0;
  /** Current matrix column count (for hover coordinate mapping). */
  private matrixCols: number = 0;
  /** Reference to the last data array passed to setData/setDataScrolled. */
  private dataRef: Float32Array | null = null;
  /** Whether the stored data is column-major layout. */
  private dataColMajor: boolean = false;
  /** Ring cursor position for scrolled streaming data (0 when not streaming). */
  private ringCursor: number = 0;

  private constructor(
    inner: WasmLeibnizFast,
    canvas: HTMLCanvasElement,
    debug: boolean,
    chartConfig: ChartConfig | null,
  ) {
    this.inner = inner;
    this.canvas = canvas;
    this.debug = debug;

    // Bind DOM event handlers (must happen before setupChartOverlay,
    // which calls removeEventListeners/registerEventListeners)
    this.boundHandlers = {
      mousedown: (e: MouseEvent) => this.handleMouseDown(e),
      mousemove: (e: MouseEvent) => this.handleMouseMove(e),
      mouseenter: () => {
        this.mouseInside = true;
      },
      mouseleave: () => {
        this.mouseInside = false;
        this.hoveredAxis = null;
        this.updateCursor('default');
        this.scheduleOverlayUpdate();
      },
      mouseup: (e: MouseEvent) => this.handleMouseUp(e),
      wheel: (e: WheelEvent) => this.handleWheel(e),
      contextmenu: (e: MouseEvent) => e.preventDefault(),
      dblclick: (e: MouseEvent) => this.handleDblClick(e),
      resize: () => {
        this.handleResize();
      },
    };

    if (chartConfig) {
      this.chartConfig = chartConfig;
      this.setupChartOverlay();
    } else {
      this.registerEventListeners();
    }

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
    this.matrixRows = options.rows;
    this.matrixCols = options.cols;
    this.dataRef = data;
    this.dataColMajor = false;
    this.ringCursor = 0;
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
    this.refreshHoverIfNeeded();
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
    this.matrixRows = options.rows;
    this.matrixCols = options.cols;
    this.dataRef = data;
    this.dataColMajor = true; // scrolled path always uses column-major layout
    this.timeSync('JS setDataScrolled', () =>
      this.inner.setDataScrolled(
        data,
        options.rows,
        options.cols,
        options.newCols,
      ),
    );
    // Advance ring cursor to mirror the WASM renderer's ring_cursor
    this.ringCursor = (this.ringCursor + options.newCols) % options.cols;
    if (this.chartConfig?.xAxis && isStreamingAxis(this.chartConfig.xAxis)) {
      this.streamingDisplayCols = options.cols;
      if (options.xOffset !== undefined) {
        this.streamingXOffset = options.xOffset;
      }
    }
    this.scheduleOverlayUpdate();
    this.refreshHoverIfNeeded();
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
   * The callback receives a {@link HoverInfo} object with matrix indices,
   * the raw data value, and axis-mapped coordinates and units when a
   * chart configuration is present.
   *
   * @param callback - Called with enriched hover info on cell hover
   */
  onHover(callback: HoverCallback): void {
    this.hoverCallback = callback;
    // Register a thin WASM-side callback that enriches and forwards
    this.inner.onHover((row: number, col: number, value: number) => {
      if (!this.hoverCallback) return;
      this.hoverCallback(this.buildHoverInfo(row, col, value));
    });
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
    this.matrixRows = options.rows;
    this.matrixCols = options.cols;
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
    this.refreshHoverIfNeeded();
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
   * Reset the camera to show the full matrix (both axes).
   * Equivalent to double-clicking the matrix area.
   */
  resetZoom(): void {
    const uv = this.inner.getVisibleRange();
    this.animateToUvRect(
      { x: uv[0], y: uv[1], w: uv[2], h: uv[3] },
      { x: 0, y: 0, w: 1, h: 1 },
    );
  }

  /**
   * Clean up all resources (GPU, event listeners, WASM, overlay DOM).
   * Must be called when the viewer is no longer needed.
   */
  destroy(): void {
    if (this.zoomAnimationId !== null) {
      cancelAnimationFrame(this.zoomAnimationId);
      this.zoomAnimationId = null;
    }
    this.removeEventListeners();
    this.hoverCallback = null;
    this.dataRef = null;
    this.interactionMode = { type: 'idle' };
    this.hoveredAxis = null;

    // Clean up overlay DOM
    this.teardownChartOverlay();

    // Destroy WASM instance (frees GPU resources)
    this.inner.destroy();
  }

  // ---------------------------------------------------------------------------
  // Private: event listener registration
  // ---------------------------------------------------------------------------

  /** The element that receives mouse/wheel events (wrapper or canvas). */
  private get eventTarget(): HTMLElement {
    return this.wrapperDiv ?? this.canvas;
  }

  private registerEventListeners(): void {
    const target = this.eventTarget;
    target.addEventListener('mousedown', this.boundHandlers.mousedown);
    target.addEventListener('mousemove', this.boundHandlers.mousemove);
    target.addEventListener('mouseenter', this.boundHandlers.mouseenter);
    target.addEventListener('mouseleave', this.boundHandlers.mouseleave);
    target.addEventListener('wheel', this.boundHandlers.wheel, {
      passive: false,
    });
    target.addEventListener('contextmenu', this.boundHandlers.contextmenu);
    target.addEventListener('dblclick', this.boundHandlers.dblclick);
    window.addEventListener('mouseup', this.boundHandlers.mouseup);
    window.addEventListener('resize', this.boundHandlers.resize);
  }

  private removeEventListeners(): void {
    const target = this.eventTarget;
    target.removeEventListener('mousedown', this.boundHandlers.mousedown);
    target.removeEventListener('mousemove', this.boundHandlers.mousemove);
    target.removeEventListener('mouseenter', this.boundHandlers.mouseenter);
    target.removeEventListener('mouseleave', this.boundHandlers.mouseleave);
    target.removeEventListener('wheel', this.boundHandlers.wheel);
    target.removeEventListener('contextmenu', this.boundHandlers.contextmenu);
    target.removeEventListener('dblclick', this.boundHandlers.dblclick);
    window.removeEventListener('mouseup', this.boundHandlers.mouseup);
    window.removeEventListener('resize', this.boundHandlers.resize);
  }

  // ---------------------------------------------------------------------------
  // Private: hit region detection
  // ---------------------------------------------------------------------------

  /** Determine which chart region the mouse is over (wrapper-local coords). */
  private getHitRegion(wrapperX: number, wrapperY: number): HitRegion {
    if (!this.chartConfig) return 'matrix';
    const l = this.layout;

    // Matrix area
    if (
      wrapperX >= l.x &&
      wrapperX <= l.x + l.width &&
      wrapperY >= l.y &&
      wrapperY <= l.y + l.height
    ) {
      return 'matrix';
    }

    // X-axis region: below the matrix, horizontally within matrix bounds
    if (
      wrapperX >= l.x &&
      wrapperX <= l.x + l.width &&
      wrapperY > l.y + l.height
    ) {
      return 'x-axis';
    }

    // Y-axis region: left of the matrix, vertically within matrix bounds
    if (wrapperX < l.x && wrapperY >= l.y && wrapperY <= l.y + l.height) {
      return 'y-axis';
    }

    return 'none';
  }

  /** Convert a MouseEvent to wrapper-local coordinates. */
  private wrapperCoords(e: MouseEvent): { wx: number; wy: number } {
    const target = this.eventTarget;
    const rect = target.getBoundingClientRect();
    return { wx: e.clientX - rect.left, wy: e.clientY - rect.top };
  }

  /** Convert wrapper-local coords to canvas-local coords. */
  private toCanvasLocal(wx: number, wy: number): { cx: number; cy: number } {
    return { cx: wx - this.layout.x, cy: wy - this.layout.y };
  }

  /** Whether this is a streaming/waterfall chart (X axis is time). */
  private isStreamingChart(): boolean {
    return !!(
      this.chartConfig?.xAxis && isStreamingAxis(this.chartConfig.xAxis)
    );
  }

  // ---------------------------------------------------------------------------
  // Private: mouse event handlers
  // ---------------------------------------------------------------------------

  private handleMouseDown(e: MouseEvent): void {
    const { wx, wy } = this.wrapperCoords(e);
    const region = this.getHitRegion(wx, wy);
    const { cx, cy } = this.toCanvasLocal(wx, wy);
    const isRight = e.button === 2;
    const isLeft = e.button === 0;
    const streaming = this.isStreamingChart();

    if (isLeft && region === 'matrix') {
      // Standard left-drag pan via WASM
      this.interactionMode = { type: 'matrix-pan' };
      this.inner.onMouseDown(cx, cy);
      this.updateCursor('grabbing');
    } else if (isLeft && region === 'y-axis') {
      this.interactionMode = { type: 'axis-pan', axis: 'y', lastPos: wy };
      this.updateCursor('grabbing');
    } else if (isLeft && region === 'x-axis' && !streaming) {
      this.interactionMode = { type: 'axis-pan', axis: 'x', lastPos: wx };
      this.updateCursor('grabbing');
    } else if (isRight && region === 'matrix' && !streaming) {
      this.interactionMode = {
        type: 'rect-select',
        startX: wx,
        startY: wy,
        currentX: wx,
        currentY: wy,
      };
      this.updateCursor('crosshair');
    } else if (isRight && region === 'y-axis') {
      this.interactionMode = {
        type: 'axis-select',
        axis: 'y',
        startPos: wy,
        currentPos: wy,
      };
    } else if (isRight && region === 'x-axis' && !streaming) {
      this.interactionMode = {
        type: 'axis-select',
        axis: 'x',
        startPos: wx,
        currentPos: wx,
      };
    }
  }

  private handleMouseMove(e: MouseEvent): void {
    const { wx, wy } = this.wrapperCoords(e);
    const { cx, cy } = this.toCanvasLocal(wx, wy);
    const mode = this.interactionMode;

    switch (mode.type) {
      case 'matrix-pan': {
        this.lastMouseX = cx;
        this.lastMouseY = cy;
        this.inner.onMouseMove(cx, cy);
        this.scheduleOverlayUpdate();
        return;
      }

      case 'axis-pan': {
        if (mode.axis === 'x') {
          const dx = wx - mode.lastPos;
          mode.lastPos = wx;
          this.inner.panX(dx);
        } else {
          const dy = wy - mode.lastPos;
          mode.lastPos = wy;
          this.inner.panY(dy);
        }
        this.scheduleOverlayUpdate();
        return;
      }

      case 'rect-select': {
        mode.currentX = wx;
        mode.currentY = wy;
        this.scheduleOverlayUpdate();
        return;
      }

      case 'axis-select': {
        mode.currentPos = mode.axis === 'x' ? wx : wy;
        this.scheduleOverlayUpdate();
        return;
      }

      case 'idle': {
        // Update hover state
        this.lastMouseX = cx;
        this.lastMouseY = cy;
        const region = this.getHitRegion(wx, wy);

        if (region === 'matrix') {
          this.inner.onMouseMove(cx, cy);
          this.hoveredAxis = null;
          this.updateCursor('default');
        } else if (region === 'x-axis') {
          this.hoveredAxis = this.isStreamingChart() ? null : 'x';
          this.updateCursor(this.isStreamingChart() ? 'default' : 'col-resize');
        } else if (region === 'y-axis') {
          this.hoveredAxis = 'y';
          this.updateCursor('row-resize');
        } else {
          this.hoveredAxis = null;
          this.updateCursor('default');
        }
        this.scheduleOverlayUpdate();
        return;
      }
    }
  }

  private handleMouseUp(_e: MouseEvent): void {
    const mode = this.interactionMode;

    switch (mode.type) {
      case 'matrix-pan': {
        this.inner.onMouseUp();
        break;
      }

      case 'rect-select': {
        this.finishRectSelect(mode);
        break;
      }

      case 'axis-select': {
        this.finishAxisSelect(mode);
        break;
      }

      default:
        break;
    }

    this.interactionMode = { type: 'idle' };
    this.updateCursor('default');
    this.scheduleOverlayUpdate();
  }

  private handleWheel(e: WheelEvent): void {
    e.preventDefault();
    const { wx, wy } = this.wrapperCoords(e);
    const region = this.getHitRegion(wx, wy);
    const { cx, cy } = this.toCanvasLocal(wx, wy);
    const delta = -e.deltaY;
    const streaming = this.isStreamingChart();

    if (region === 'matrix') {
      this.inner.onWheel(cx, cy, delta);
    } else if (region === 'x-axis' && !streaming) {
      // Zoom X at the horizontal position mapped to canvas-local X
      this.inner.zoomAtX(cx, delta);
    } else if (region === 'y-axis') {
      // Zoom Y at the vertical position mapped to canvas-local Y
      this.inner.zoomAtY(cy, delta);
    }
    this.scheduleOverlayUpdate();
  }

  private handleDblClick(e: MouseEvent): void {
    const { wx, wy } = this.wrapperCoords(e);
    const region = this.getHitRegion(wx, wy);
    const streaming = this.isStreamingChart();

    const uv = this.inner.getVisibleRange();
    const from = { x: uv[0], y: uv[1], w: uv[2], h: uv[3] };

    if (region === 'matrix') {
      this.animateToUvRect(from, { x: 0, y: 0, w: 1, h: 1 });
    } else if (region === 'x-axis' && !streaming) {
      this.animateToUvRect(from, { x: 0, y: from.y, w: 1, h: from.h });
    } else if (region === 'y-axis') {
      this.animateToUvRect(from, { x: from.x, y: 0, w: from.w, h: 1 });
    }
  }

  /**
   * Smoothly animate from the current UV rect to a target UV rect
   * using ease-out cubic interpolation.
   */
  private animateToUvRect(
    from: { x: number; y: number; w: number; h: number },
    to: { x: number; y: number; w: number; h: number },
  ): void {
    // Cancel any in-progress animation
    if (this.zoomAnimationId !== null) {
      cancelAnimationFrame(this.zoomAnimationId);
      this.zoomAnimationId = null;
    }

    const startTime = performance.now();

    const step = (now: number) => {
      const elapsed = now - startTime;
      const rawT = Math.min(1, elapsed / ANIMATION_DURATION_MS);
      // Ease-out cubic: decelerates smoothly
      const t = 1 - (1 - rawT) ** 3;

      const x = from.x + (to.x - from.x) * t;
      const y = from.y + (to.y - from.y) * t;
      const w = from.w + (to.w - from.w) * t;
      const h = from.h + (to.h - from.h) * t;

      this.inner.zoomToUvRect(x, y, x + w, y + h);
      this.scheduleOverlayUpdate();

      if (rawT < 1) {
        this.zoomAnimationId = requestAnimationFrame(step);
      } else {
        this.zoomAnimationId = null;
      }
    };

    this.zoomAnimationId = requestAnimationFrame(step);
  }

  // ---------------------------------------------------------------------------
  // Private: selection finalization
  // ---------------------------------------------------------------------------

  /** Convert a rectangle selection to a UV rect and zoom into it. */
  private finishRectSelect(
    mode: Extract<InteractionMode, { type: 'rect-select' }>,
  ): void {
    const dx = Math.abs(mode.currentX - mode.startX);
    const dy = Math.abs(mode.currentY - mode.startY);
    if (dx < MIN_SELECTION_PX && dy < MIN_SELECTION_PX) return;

    const l = this.layout;
    // Clamp to matrix area and convert to canvas-local fractions
    const x0 = Math.max(
      0,
      (Math.min(mode.startX, mode.currentX) - l.x) / l.width,
    );
    const x1 = Math.min(
      1,
      (Math.max(mode.startX, mode.currentX) - l.x) / l.width,
    );
    const y0 = Math.max(
      0,
      (Math.min(mode.startY, mode.currentY) - l.y) / l.height,
    );
    const y1 = Math.min(
      1,
      (Math.max(mode.startY, mode.currentY) - l.y) / l.height,
    );

    // Convert canvas fractions to UV using current camera
    const uv = this.inner.getVisibleRange();
    const uMin = uv[0] + x0 * uv[2];
    const uMax = uv[0] + x1 * uv[2];
    const vMin = uv[1] + y0 * uv[3];
    const vMax = uv[1] + y1 * uv[3];

    this.inner.zoomToUvRect(uMin, vMin, uMax, vMax);
  }

  /** Convert an axis selection to a UV range and zoom that axis. */
  private finishAxisSelect(
    mode: Extract<InteractionMode, { type: 'axis-select' }>,
  ): void {
    const dist = Math.abs(mode.currentPos - mode.startPos);
    if (dist < MIN_SELECTION_PX) return;

    const l = this.layout;
    const uv = this.inner.getVisibleRange();

    if (mode.axis === 'x') {
      const x0 = Math.max(
        0,
        (Math.min(mode.startPos, mode.currentPos) - l.x) / l.width,
      );
      const x1 = Math.min(
        1,
        (Math.max(mode.startPos, mode.currentPos) - l.x) / l.width,
      );
      const uMin = uv[0] + x0 * uv[2];
      const uMax = uv[0] + x1 * uv[2];
      // Keep current Y range
      this.inner.zoomToUvRect(uMin, uv[1], uMax, uv[1] + uv[3]);
    } else {
      const y0 = Math.max(
        0,
        (Math.min(mode.startPos, mode.currentPos) - l.y) / l.height,
      );
      const y1 = Math.min(
        1,
        (Math.max(mode.startPos, mode.currentPos) - l.y) / l.height,
      );
      const vMin = uv[1] + y0 * uv[3];
      const vMax = uv[1] + y1 * uv[3];
      // Keep current X range
      this.inner.zoomToUvRect(uv[0], vMin, uv[0] + uv[2], vMax);
    }
  }

  // ---------------------------------------------------------------------------
  // Private: cursor management
  // ---------------------------------------------------------------------------

  private updateCursor(cursor: string): void {
    this.eventTarget.style.cursor = cursor;
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

    // Remove listeners from canvas before reparenting
    this.removeEventListeners();

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

    // Re-register listeners on wrapper (covers axis regions)
    this.registerEventListeners();
  }

  /**
   * Remove the overlay canvas and wrapper div, restoring the original
   * canvas position in the DOM.
   */
  private teardownChartOverlay(): void {
    if (!this.wrapperDiv) return;

    this.removeEventListeners();

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

    // Re-register on canvas directly
    this.registerEventListeners();
  }

  /**
   * Handle window resize: recalculate layout, resize both canvases,
   * update the WASM renderer, and redraw the overlay.
   */
  private handleResize(): void {
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
   * Redraw the chart overlay (axes, ticks, labels, title, selections, highlights).
   * Called after data changes, pan/zoom, and resize.
   */
  private updateOverlay(): void {
    if (!this.chartConfig || !this.overlayCtx || !this.wrapperDiv) return;

    const ctx = this.overlayCtx;
    const dpr = window.devicePixelRatio || 1;
    const wrapperRect = this.wrapperDiv.getBoundingClientRect();
    const containerW = wrapperRect.width;
    const containerH = wrapperRect.height;

    const visible = this.computeVisibleRange();

    renderOverlay(
      ctx,
      this.layout,
      this.chartConfig,
      visible,
      containerW,
      containerH,
      dpr,
    );

    // Draw interaction overlays on top of axes (inside the same DPR transform)
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Axis hover highlight
    if (this.hoveredAxis && this.interactionMode.type === 'idle') {
      drawAxisHighlight(
        ctx,
        this.layout,
        this.hoveredAxis,
        containerW,
        containerH,
      );
    }

    // Selection rectangle
    const mode = this.interactionMode;
    if (mode.type === 'rect-select') {
      const l = this.layout;
      const sx = Math.max(l.x, Math.min(mode.startX, mode.currentX));
      const sy = Math.max(l.y, Math.min(mode.startY, mode.currentY));
      const ex = Math.min(l.x + l.width, Math.max(mode.startX, mode.currentX));
      const ey = Math.min(l.y + l.height, Math.max(mode.startY, mode.currentY));
      drawSelectionRect(ctx, sx, sy, ex - sx, ey - sy);
    }

    // Axis selection band
    if (mode.type === 'axis-select') {
      const l = this.layout;
      if (mode.axis === 'x') {
        const sx = Math.max(l.x, Math.min(mode.startPos, mode.currentPos));
        const ex = Math.min(
          l.x + l.width,
          Math.max(mode.startPos, mode.currentPos),
        );
        drawSelectionRect(ctx, sx, l.y, ex - sx, l.height);
      } else {
        const sy = Math.max(l.y, Math.min(mode.startPos, mode.currentPos));
        const ey = Math.min(
          l.y + l.height,
          Math.max(mode.startPos, mode.currentPos),
        );
        drawSelectionRect(ctx, l.x, sy, l.width, ey - sy);
      }
    }

    ctx.restore();
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

  /**
   * Build a HoverInfo object from raw matrix indices and value.
   * Maps row/col to axis coordinates when a chart config is present.
   */
  private buildHoverInfo(row: number, col: number, value: number): HoverInfo {
    const info: HoverInfo = { row, col, value };
    const cfg = this.chartConfig;
    if (!cfg) return info;

    if (cfg.yAxis) {
      const [yMin, yMax] = this.getFullYRange();
      const rows = this.matrixRows;
      info.y = rows > 1 ? yMin + (row / (rows - 1)) * (yMax - yMin) : yMin;
      info.yUnit = cfg.yAxis.unit;
    }

    if (cfg.xAxis) {
      const [xMin, xMax] = this.getFullXRange();
      const cols = this.matrixCols;
      info.x = cols > 1 ? xMin + (col / (cols - 1)) * (xMax - xMin) : xMin;
      info.xUnit = cfg.xAxis.unit;
    }

    if (cfg.valueUnit) {
      info.valueUnit = cfg.valueUnit;
    }

    return info;
  }

  /**
   * Re-invoke the hover lookup at the last known mouse position.
   */
  private refreshHoverIfNeeded(): void {
    if (!this.mouseInside || !this.hoverCallback || !this.dataRef) return;

    const rows = this.matrixRows;
    const cols = this.matrixCols;
    if (rows === 0 || cols === 0) return;

    const uv = this.inner.getVisibleRange();
    const u = uv[0] + (this.lastMouseX / this.canvas.clientWidth) * uv[2];
    const v = uv[1] + (this.lastMouseY / this.canvas.clientHeight) * uv[3];

    const col = Math.floor(u * cols);
    const row = Math.floor(v * rows);
    if (row < 0 || row >= rows || col < 0 || col >= cols) return;

    const bufCol = (col + this.ringCursor) % cols;

    const data = this.dataRef;
    const idx = this.dataColMajor ? bufCol * rows + row : row * cols + bufCol;
    const value = data[idx];

    this.hoverCallback(this.buildHoverInfo(row, col, value));
  }
}

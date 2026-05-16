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
 * viewer.onHover((info) => console.log(info));
 * ```
 */

import type { LeibnizFast as WasmLeibnizFast } from '../pkg/leibniz_fast';
import type {
  AxisConfig,
  ChartConfig,
  ChunkedDataOptions,
  ColormapName,
  CreateOptions,
  DataChunk,
  DataOptions,
  HeatmapChartConfig,
  HeatmapHoverInfo,
  HoverCallback,
  HoverInfo,
  LineChartConfig,
  LineDataOptions,
  LineHoverInfo,
  LineHoverPoint,
  LineScrolledDataOptions,
  LineSeriesInput,
  LineSeriesUpdate,
  LineXAxisConfig,
  LineYAxisConfig,
  RgbaColor,
  ScrolledDataOptions,
  StreamingAxisConfig,
  StreamingDataOptions,
  WebGpuSupport,
} from './types';
import {
  computeLayout,
  drawAxisHighlight,
  drawLineHoverGuides,
  drawSelectionRect,
  isHeatmapChartConfig,
  isLineChartConfig,
  isStreamingAxis,
  renderOverlay,
  uvToVisibleRange,
} from './axes';
import type { ColorbarData, LayoutRect, VisibleRange } from './axes';

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
  ChunkedDataOptions,
  ColormapName,
  CreateOptions,
  DataChunk,
  DataOptions,
  HeatmapChartConfig,
  HeatmapHoverInfo,
  HoverCallback,
  HoverInfo,
  LineChartConfig,
  LineDataOptions,
  LineHoverInfo,
  LineHoverPoint,
  LineScrolledDataOptions,
  LineSeriesInput,
  LineSeriesUpdate,
  LineXAxisConfig,
  LineYAxisConfig,
  RgbaColor,
  ScrolledDataOptions,
  StreamingAxisConfig,
  StreamingDataOptions,
  WebGpuSupport,
};

/** Cached WASM module — initialized once on first `create()` call. */
let wasmModule: typeof import('../pkg/leibniz_fast') | null = null;

const WEBGPU_REQUIRED_MESSAGE =
  'LeibnizFast requires WebGPU. Use a WebGPU-capable browser, enable hardware acceleration, and serve the page from HTTPS or localhost. No CPU/WebGL fallback is available.';

type WebGpuDeviceLike = {
  destroy?: () => void;
};

type WebGpuAdapterLike = {
  requestDevice: () => Promise<WebGpuDeviceLike>;
  info?: unknown;
};

type WebGpuNavigatorLike = Navigator & {
  gpu?: {
    requestAdapter: (
      options?: Record<string, unknown>,
    ) => Promise<WebGpuAdapterLike | null>;
  };
};

type WasmLeibnizFastInner = WasmLeibnizFast & {
  getColorRange(): Float32Array;
  getColormapLut(): Uint8Array;
  setLineData(
    data: Float32Array,
    sampleCount: number,
    seriesCount: number,
    colors: Float32Array,
    visibility: Uint32Array,
    xValues: Float32Array | undefined,
    xMode: number,
    xStart: number,
    xStep: number,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
  ): void;
  setLineDataScrolled(
    data: Float32Array,
    newSamples: number,
    sampleCount: number,
    seriesCount: number,
    colors: Float32Array,
    visibility: Uint32Array,
    xStart: number,
    xStep: number,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
  ): void;
  updateLineConfig(
    colors: Float32Array,
    visibility: Uint32Array,
    xMode: number,
    xStart: number,
    xStep: number,
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
  ): void;
};

interface LineSeriesState {
  id: string;
  name: string;
  color: [number, number, number, number];
  visible: boolean;
  data: Float32Array;
}

interface ResolvedLineXAxis {
  config: LineXAxisConfig;
  xMode: number;
  xValues: Float32Array | undefined;
  xStart: number;
  xStep: number;
  xMin: number;
  xMax: number;
}

interface LineRange {
  min: number;
  max: number;
}

interface LineHoverGuidePoint {
  x: number;
  y: number;
  color: [number, number, number, number];
}

interface LineHoverGuideState {
  mouseX: number;
  mouseY: number;
  points: LineHoverGuidePoint[];
}

const LINE_X_MODE_LINEAR = 0;
const LINE_X_MODE_EXPLICIT = 1;
const DEFAULT_LINE_Y_PADDING_RATIO = 0.05;

function normalizeAdapterInfo(
  info: unknown,
): Record<string, string | number | boolean> | undefined {
  if (!info || typeof info !== 'object') return undefined;
  const result: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(info as Record<string, unknown>)) {
    if (
      typeof value === 'string' ||
      typeof value === 'number' ||
      typeof value === 'boolean'
    ) {
      result[key] = value;
    }
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function assertValidRange(range: { min: number; max: number }): void {
  if (
    !Number.isFinite(range.min) ||
    !Number.isFinite(range.max) ||
    range.max <= range.min
  ) {
    throw new Error(
      'Range must contain finite values with max greater than min.',
    );
  }
}

async function* toAsyncIterable<T>(
  values: Iterable<T> | AsyncIterable<T>,
): AsyncIterable<T> {
  if (Symbol.asyncIterator in Object(values)) {
    for await (const value of values as AsyncIterable<T>) {
      yield value;
    }
    return;
  }
  for (const value of values as Iterable<T>) {
    yield value;
  }
}

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
  /** Whether the last pointer position was inside the data plot region. */
  private mouseInPlot: boolean = false;

  // --- Interaction state ---
  /** Current interaction mode. */
  private interactionMode: InteractionMode = { type: 'idle' };
  /** Which axis region the mouse is currently hovering (for highlight). */
  private hoveredAxis: 'x' | 'y' | null = null;
  /** Active zoom-reset animation frame ID, or null if no animation is running. */
  private zoomAnimationId: number | null = null;
  /** True after destroy() has started; stale DOM events are ignored. */
  private disposed: boolean = false;

  // --- Chart overlay state ---
  /** Chart configuration (axes, colorbar, title, labels). Null when no chart mode. */
  private chartConfig: ChartConfig | null = null;
  /** Wrapper div that contains both canvases. */
  private wrapperDiv: HTMLDivElement | null = null;
  /** 2D overlay canvas for axes/title rendering. */
  private overlayCanvas: HTMLCanvasElement | null = null;
  /** 2D rendering context for the overlay canvas. */
  private overlayCtx: CanvasRenderingContext2D | null = null;
  /** DOM legend for line charts. */
  private legendDiv: HTMLDivElement | null = null;
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
  /** Active colormap lookup table for the overlay colorbar. */
  private colorbarLut: Uint8Array | null = null;
  /** Active heatmap value range cached outside Rust hover callbacks. */
  private colorRange: { min: number; max: number } | null = null;

  // --- Line chart state ---
  /** Current line series state, including library-owned data buffers. */
  private lineSeries: LineSeriesState[] = [];
  /** Current line sample count. */
  private lineSampleCount: number = 0;
  /** Resolved line X axis for the active data. */
  private lineXAxis: ResolvedLineXAxis | null = null;
  /** Current base Y range used by the WebGPU line renderer. */
  private lineYRange: LineRange | null = null;
  /** Whether sticky auto Y range is currently active. */
  private lineStickyYActive: boolean = true;
  /** Line ring cursor for JS hover lookup. */
  private lineRingCursor: number = 0;
  /** Total samples received for streaming line axes. */
  private lineTotalSamplesReceived: number = 0;
  /** Current line hover guide geometry in canvas-local CSS pixels. */
  private lineHoverGuide: LineHoverGuideState | null = null;
  /** Signature of the currently rendered legend structure. */
  private lineLegendSignature: string = '';

  private constructor(
    inner: WasmLeibnizFast,
    canvas: HTMLCanvasElement,
    debug: boolean,
    chartConfig: ChartConfig | null,
  ) {
    this.inner = inner;
    this.canvas = canvas;
    this.debug = debug;
    this.refreshColorbarLut();

    // Bind DOM event handlers (must happen before setupChartOverlay,
    // which calls removeEventListeners/registerEventListeners)
    this.boundHandlers = {
      mousedown: (e: MouseEvent) => this.handleMouseDown(e),
      mousemove: (e: MouseEvent) => this.handleMouseMove(e),
      mouseenter: () => {
        if (this.disposed) return;
        this.mouseInside = true;
      },
      mouseleave: () => {
        if (this.disposed) return;
        this.mouseInside = false;
        this.mouseInPlot = false;
        this.hoveredAxis = null;
        this.lineHoverGuide = null;
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
   * Check whether this browser can create a WebGPU adapter and device.
   *
   * A library cannot enable browser flags for the user. This method only
   * reports the current browser/page state so apps can show a friendly message.
   */
  static async checkSupport(): Promise<WebGpuSupport> {
    if (typeof window !== 'undefined' && window.isSecureContext === false) {
      return {
        supported: false,
        reason: `${WEBGPU_REQUIRED_MESSAGE} This page is not a secure context.`,
      };
    }

    if (typeof navigator === 'undefined') {
      return {
        supported: false,
        reason: `${WEBGPU_REQUIRED_MESSAGE} navigator is not available.`,
      };
    }

    const nav = navigator as WebGpuNavigatorLike;
    if (!nav.gpu) {
      return {
        supported: false,
        reason: `${WEBGPU_REQUIRED_MESSAGE} navigator.gpu is not available.`,
      };
    }

    let adapter: WebGpuAdapterLike | null = null;
    try {
      adapter = await nav.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
    } catch (error) {
      return {
        supported: false,
        reason: `${WEBGPU_REQUIRED_MESSAGE} Adapter request failed: ${String(error)}`,
      };
    }

    if (!adapter) {
      return {
        supported: false,
        reason: `${WEBGPU_REQUIRED_MESSAGE} No WebGPU adapter was returned.`,
      };
    }

    let device: WebGpuDeviceLike | null = null;
    try {
      device = await adapter.requestDevice();
    } catch (error) {
      return {
        supported: false,
        reason: `${WEBGPU_REQUIRED_MESSAGE} Device creation failed: ${String(error)}`,
        adapterInfo: normalizeAdapterInfo(adapter.info),
      };
    } finally {
      device?.destroy?.();
    }

    return {
      supported: true,
      adapterInfo: normalizeAdapterInfo(adapter.info),
    };
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
    const support = await LeibnizFast.checkSupport();
    if (!support.supported) {
      throw new Error(support.reason ?? WEBGPU_REQUIRED_MESSAGE);
    }
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

  /** Return the active heatmap chart config, or null in line/raw modes. */
  private getHeatmapChart(): HeatmapChartConfig | null {
    return this.chartConfig && isHeatmapChartConfig(this.chartConfig)
      ? this.chartConfig
      : null;
  }

  /** Return the active line chart config, or null otherwise. */
  private getLineChart(): LineChartConfig | null {
    return this.chartConfig && isLineChartConfig(this.chartConfig)
      ? this.chartConfig
      : null;
  }

  /** Whether the viewer is currently configured for line charts. */
  private isLineChart(): boolean {
    return this.getLineChart() !== null;
  }

  /** Fail fast when a heatmap API is called in line mode. */
  private assertHeatmapMode(method: string): HeatmapChartConfig | null {
    if (this.isLineChart()) {
      throw new Error(`${method}() is only available for heatmap charts.`);
    }
    return this.getHeatmapChart();
  }

  /** Fail fast when a line API is called outside line mode. */
  private assertLineMode(method: string): LineChartConfig {
    const chart = this.getLineChart();
    if (!chart) {
      throw new Error(`${method}() requires chart.type === 'line'.`);
    }
    return chart;
  }

  /** Refresh the cached colormap LUT used by the 2D overlay colorbar. */
  private refreshColorbarLut(): void {
    const lut = (this.inner as WasmLeibnizFastInner).getColormapLut();
    this.colorbarLut = lut.length > 0 ? lut : null;
  }

  /** Refresh the cached heatmap value range outside hover callbacks. */
  private refreshColorRange(): void {
    const range = (this.inner as WasmLeibnizFastInner).getColorRange();
    if (range.length < 2) {
      this.colorRange = null;
      return;
    }

    const min = range[0];
    const max = range[1];
    this.colorRange =
      Number.isFinite(min) && Number.isFinite(max) && max > min
        ? { min, max }
        : null;
  }

  /** Build colorbar render data from the cached colormap range. */
  private getColorbarData(): ColorbarData | null {
    const chart = this.getHeatmapChart();
    if (!chart || chart.colorbar === false) return null;
    if (!this.colorbarLut) return null;
    if (!this.colorRange) return null;

    const colorbar: ColorbarData = {
      min: this.colorRange.min,
      max: this.colorRange.max,
      colors: this.colorbarLut,
    };
    if (chart.valueUnit) {
      colorbar.label = chart.valueUnit;
    }
    return colorbar;
  }

  /** Resolve the current heatmap colormap color for a finite data value. */
  private getHeatmapValueColor(value: number): RgbaColor | undefined {
    if (!Number.isFinite(value) || !this.colorbarLut) return undefined;
    if (!this.colorRange) return undefined;

    const entries = Math.floor(this.colorbarLut.length / 3);
    if (entries <= 0) return undefined;

    const normalized = Math.max(
      0,
      Math.min(
        1,
        (value - this.colorRange.min) /
          (this.colorRange.max - this.colorRange.min),
      ),
    );
    const lutIndex = Math.max(
      0,
      Math.min(entries - 1, Math.round(normalized * (entries - 1))),
    );
    const offset = lutIndex * 3;
    return [
      this.colorbarLut[offset],
      this.colorbarLut[offset + 1],
      this.colorbarLut[offset + 2],
      1,
    ];
  }

  /**
   * Set the matrix data to visualize.
   *
   * @param data - Flat Float32Array in row-major order
   * @param options - Matrix dimensions (rows, cols)
   */
  setData(data: Float32Array, options: DataOptions): void {
    const chart = this.assertHeatmapMode('setData');
    this.matrixRows = options.rows;
    this.matrixCols = options.cols;
    this.dataRef = data;
    this.dataColMajor = false;
    this.ringCursor = 0;
    this.timeSync('JS setData', () =>
      this.inner.setData(data, options.rows, options.cols),
    );
    this.refreshColorRange();
    if (chart?.xAxis && isStreamingAxis(chart.xAxis)) {
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
   * throws instead of falling back to a full upload.
   *
   * @param data - Full Float32Array in row-major order (used for hover lookups)
   * @param options - Matrix dimensions and number of new columns
   */
  setDataScrolled(data: Float32Array, options: ScrolledDataOptions): void {
    const chart = this.assertHeatmapMode('setDataScrolled');
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
    this.refreshColorRange();
    // Advance ring cursor to mirror the WASM renderer's ring_cursor
    this.ringCursor = (this.ringCursor + options.newCols) % options.cols;
    if (chart?.xAxis && isStreamingAxis(chart.xAxis)) {
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
    this.assertHeatmapMode('setColormap');
    this.timeSync('JS setColormap', () => this.inner.setColormap(name));
    this.refreshColorbarLut();
    this.updateOverlay();
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
    this.assertHeatmapMode('setRange');
    assertValidRange({ min, max });
    this.inner.setRange(min, max);
    this.colorRange = { min, max };
    this.updateOverlay();
  }

  /**
   * Upload full line series data or replace the current animated line frame.
   *
   * @param series - One or more named series sharing the same X axis
   * @param options - Optional axis overrides for this upload
   */
  setLineData(series: LineSeriesInput[], options?: LineDataOptions): void {
    const chart = this.assertLineMode('setLineData');
    this.lineSeries = this.normalizeLineSeries(series);
    this.lineSampleCount = this.lineSeries[0]?.data.length ?? 0;
    this.lineRingCursor = 0;
    this.lineTotalSamplesReceived = options?.xOffset ?? this.lineSampleCount;
    const yAxis = options?.yAxis ?? chart.yAxis;
    if (yAxis?.rangeMode === 'fixed') {
      this.lineStickyYActive = false;
    } else if (options?.resetYRange === true || this.lineYRange === null) {
      this.lineStickyYActive = true;
    }
    this.lineXAxis = this.resolveLineXAxis(
      options?.xAxis ?? chart.xAxis,
      this.lineSampleCount,
      this.lineTotalSamplesReceived,
    );
    this.lineYRange = this.resolveLineYRange(
      yAxis,
      options?.resetYRange ?? this.lineYRange === null,
    );

    this.matrixRows = 1;
    this.matrixCols = this.lineSampleCount;

    this.timeSync('JS setLineData', () =>
      (this.inner as WasmLeibnizFastInner).setLineData(
        this.flattenLineData(),
        this.lineSampleCount,
        this.lineSeries.length,
        this.buildLineColorBuffer(),
        this.buildLineVisibilityBuffer(),
        this.lineXAxis?.xValues,
        this.lineXAxis?.xMode ?? LINE_X_MODE_LINEAR,
        this.lineXAxis?.xStart ?? 0,
        this.lineXAxis?.xStep ?? 1,
        this.lineXAxis?.xMin ?? 0,
        this.lineXAxis?.xMax ?? 1,
        this.lineYRange?.min ?? 0,
        this.lineYRange?.max ?? 1,
      ),
    );

    this.syncLineLegend();
    this.updateOverlay();
    this.refreshHoverIfNeeded();
  }

  /**
   * Append new samples to an existing scrolling line chart.
   *
   * New samples enter on the right. Old samples leave on the left without
   * shifting the internal buffers.
   */
  setLineDataScrolled(
    updates: LineSeriesUpdate[],
    options: LineScrolledDataOptions,
  ): void {
    const chart = this.assertLineMode('setLineDataScrolled');
    if (this.lineSeries.length === 0 || this.lineSampleCount === 0) {
      throw new Error(
        'setLineDataScrolled() requires an initial setLineData().',
      );
    }
    if (
      !Number.isInteger(options.newSamples) ||
      options.newSamples <= 0 ||
      options.newSamples > this.lineSampleCount
    ) {
      throw new Error(
        'newSamples must be a positive integer no greater than the active sample count.',
      );
    }

    const updateMap = new Map<string, Float32Array>();
    for (const update of updates) {
      if (updateMap.has(update.id)) {
        throw new Error(`Duplicate line update id: ${update.id}`);
      }
      if (update.data.length !== options.newSamples) {
        throw new Error(
          `Line update "${update.id}" length ${update.data.length} does not match newSamples ${options.newSamples}.`,
        );
      }
      updateMap.set(update.id, update.data);
    }
    if (updateMap.size !== this.lineSeries.length) {
      throw new Error(
        'setLineDataScrolled() requires one update per line series.',
      );
    }

    const flatUpdates = new Float32Array(
      this.lineSeries.length * options.newSamples,
    );
    const cursor = this.lineRingCursor;
    for (let s = 0; s < this.lineSeries.length; s++) {
      const state = this.lineSeries[s];
      const update = updateMap.get(state.id);
      if (!update)
        throw new Error(`Missing line update for series "${state.id}".`);
      flatUpdates.set(update, s * options.newSamples);

      const first = Math.min(options.newSamples, this.lineSampleCount - cursor);
      state.data.set(update.subarray(0, first), cursor);
      if (first < options.newSamples) {
        state.data.set(update.subarray(first), 0);
      }
    }

    this.lineTotalSamplesReceived =
      options.xOffset ?? this.lineTotalSamplesReceived + options.newSamples;
    this.lineXAxis = this.resolveLineXAxis(
      chart.xAxis,
      this.lineSampleCount,
      this.lineTotalSamplesReceived,
    );
    if (this.lineXAxis.xMode === LINE_X_MODE_EXPLICIT) {
      throw new Error(
        'setLineDataScrolled() does not support explicit X arrays.',
      );
    }
    this.lineYRange = this.resolveLineYRange(chart.yAxis, false);

    this.timeSync('JS setLineDataScrolled', () =>
      (this.inner as WasmLeibnizFastInner).setLineDataScrolled(
        flatUpdates,
        options.newSamples,
        this.lineSampleCount,
        this.lineSeries.length,
        this.buildLineColorBuffer(),
        this.buildLineVisibilityBuffer(),
        this.lineXAxis?.xStart ?? 0,
        this.lineXAxis?.xStep ?? 1,
        this.lineXAxis?.xMin ?? 0,
        this.lineXAxis?.xMax ?? 1,
        this.lineYRange?.min ?? 0,
        this.lineYRange?.max ?? 1,
      ),
    );

    this.lineRingCursor = (cursor + options.newSamples) % this.lineSampleCount;
    this.scheduleOverlayUpdate();
    this.refreshHoverIfNeeded();
  }

  /**
   * Toggle a line series by id. Hidden series are excluded from rendering,
   * hover, legend active state, and sticky-auto Y ranges.
   */
  setLineSeriesVisibility(id: string, visible: boolean): void {
    this.assertLineMode('setLineSeriesVisibility');
    const series = this.lineSeries.find((item) => item.id === id);
    if (!series) {
      throw new Error(`Unknown line series id: ${id}`);
    }
    series.visible = visible;
    this.lineYRange = this.resolveLineYRange(this.getLineChart()?.yAxis, true);
    this.updateLineGpuConfig();
    this.syncLineLegend();
    this.updateOverlay();
    this.refreshHoverIfNeeded();
  }

  /**
   * Register a callback for hover events.
   *
   * The callback receives a {@link HoverInfo} object. Heatmaps report one
   * matrix cell; line charts report all visible finite series values at the
   * current mouse X coordinate.
   *
   * @param callback - Called with enriched hover info while hovering the chart
   */
  onHover(callback: HoverCallback): void {
    this.hoverCallback = callback;
    // Register a thin WASM-side callback that enriches and forwards
    this.inner.onHover(
      (
        row: number,
        col: number,
        value: number,
        valueAvailable: boolean = true,
      ) => {
        if (!this.hoverCallback) return;
        this.hoverCallback(
          this.buildHoverInfo(row, col, value, valueAvailable),
        );
      },
    );
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
    const chart = this.assertHeatmapMode('beginData');
    if (options.retainData === false) {
      throw new Error(
        'beginData() retains CPU-side data. Use setDataChunks(..., { retainData: false }) for no-retention uploads.',
      );
    }
    if (options.range) {
      this.setRange(options.range.min, options.range.max);
    }
    this.matrixRows = options.rows;
    this.matrixCols = options.cols;
    this.timeSync('JS beginData', () =>
      this.inner.beginData(options.rows, options.cols),
    );
    if (chart?.xAxis && isStreamingAxis(chart.xAxis)) {
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
    this.assertHeatmapMode('beginUpdate');
    if (options.retainData === false) {
      throw new Error(
        'beginUpdate() retains CPU-side data. Use setDataChunks(..., { retainData: false }) for no-retention uploads.',
      );
    }
    if (options.range) {
      this.setRange(options.range.min, options.range.max);
    }
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
    this.assertHeatmapMode('appendChunk');
    this.timeSync('JS appendChunk', () =>
      this.inner.appendChunk(data, startRow),
    );
  }

  /**
   * Finalize a streaming upload. Computes data range, rebuilds
   * pipelines, and renders.
   */
  endData(): void {
    this.assertHeatmapMode('endData');
    this.timeSync('JS endData', () => this.inner.endData());
    this.refreshColorRange();
    this.updateOverlay();
    this.refreshHoverIfNeeded();
  }

  /**
   * Upload a matrix as sequential row-major chunks.
   *
   * This path avoids requiring one giant JavaScript Float32Array. By default,
   * chunks are uploaded directly to GPU textures and CPU-side values are not
   * retained, so hover callbacks report `valueAvailable: false`.
   */
  async setDataChunks(
    chunks: Iterable<DataChunk> | AsyncIterable<DataChunk>,
    options: ChunkedDataOptions,
  ): Promise<void> {
    const chart = this.assertHeatmapMode('setDataChunks');
    if (options.range) {
      assertValidRange(options.range);
    }

    const debug = this.debug;
    const t0 = debug ? performance.now() : 0;
    const retainData = options.retainData ?? false;
    const rangeMin = options.range?.min;
    const rangeMax = options.range?.max;

    this.matrixRows = options.rows;
    this.matrixCols = options.cols;
    this.dataRef = null;
    this.dataColMajor = false;
    this.ringCursor = 0;

    try {
      this.inner.beginDataChunks(
        options.rows,
        options.cols,
        retainData,
        rangeMin,
        rangeMax,
      );

      for await (const chunk of toAsyncIterable(chunks)) {
        this.inner.appendDataChunk(chunk.data, chunk.startRow);
      }

      this.inner.endDataChunks();
      this.refreshColorRange();
    } catch (error) {
      this.inner.abortDataChunks();
      throw error;
    }

    if (chart?.xAxis && isStreamingAxis(chart.xAxis)) {
      this.streamingDisplayCols = options.cols;
      if (options.xOffset !== undefined) {
        this.streamingXOffset = options.xOffset;
      }
    }

    if (debug) {
      console.log(
        `[perf] JS setDataChunks: ${(performance.now() - t0).toFixed(2)}ms`,
      );
    }
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
   * Update the chart configuration (axes, colorbar, title, labels).
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
    if (this.getLineChart() && this.lineSeries.length > 0) {
      const chart = this.getLineChart();
      if (chart) {
        this.lineXAxis = this.resolveLineXAxis(
          chart.xAxis,
          this.lineSampleCount,
          this.lineTotalSamplesReceived,
        );
        this.lineYRange = this.resolveLineYRange(chart.yAxis, true);
        this.updateLineGpuConfig();
      }
    }
    this.syncLineLegend();
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
    if (this.isLineChart()) this.resetLineStickyY();
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
    if (this.disposed) return;
    this.disposed = true;
    if (this.zoomAnimationId !== null) {
      cancelAnimationFrame(this.zoomAnimationId);
      this.zoomAnimationId = null;
    }
    this.removeEventListeners();
    this.hoverCallback = null;
    this.dataRef = null;
    this.colorRange = null;
    this.interactionMode = { type: 'idle' };
    this.mouseInside = false;
    this.mouseInPlot = false;
    this.hoveredAxis = null;
    this.lineHoverGuide = null;
    this.lineLegendSignature = '';

    // Clean up overlay DOM
    this.teardownChartOverlay(false);

    // Destroy WASM instance (frees GPU resources)
    this.inner.destroy();
  }

  // ---------------------------------------------------------------------------
  // Private: line chart data helpers
  // ---------------------------------------------------------------------------

  private normalizeLineSeries(series: LineSeriesInput[]): LineSeriesState[] {
    if (series.length === 0) {
      throw new Error('Line charts require at least one series.');
    }

    const previousVisibility = new Map(
      this.lineSeries.map((state) => [state.id, state.visible]),
    );
    const ids = new Set<string>();
    const sampleCount = series[0].data.length;
    if (sampleCount < 2) {
      throw new Error('Line charts require at least two samples per series.');
    }

    return series.map((input, index) => {
      const id = input.id ?? input.name;
      if (!id)
        throw new Error(`Line series at index ${index} needs an id or name.`);
      if (!input.name) throw new Error(`Line series "${id}" needs a name.`);
      if (ids.has(id)) throw new Error(`Duplicate line series id: ${id}`);
      ids.add(id);
      if (input.data.length !== sampleCount) {
        throw new Error('All line series must have the same sample count.');
      }
      this.assertValidLineColor(input.color, id);
      return {
        id,
        name: input.name,
        color: input.color,
        visible: input.visible ?? previousVisibility.get(id) ?? true,
        data: new Float32Array(input.data),
      };
    });
  }

  private assertValidLineColor(
    color: [number, number, number, number],
    id: string,
  ): void {
    const [r, g, b, a] = color;
    const valid =
      Number.isFinite(r) &&
      Number.isFinite(g) &&
      Number.isFinite(b) &&
      Number.isFinite(a) &&
      r >= 0 &&
      r <= 255 &&
      g >= 0 &&
      g <= 255 &&
      b >= 0 &&
      b <= 255 &&
      a >= 0 &&
      a <= 1;
    if (!valid) {
      throw new Error(
        `Line series "${id}" color must use RGB values in 0..255 and alpha in 0..1.`,
      );
    }
  }

  private resolveLineXAxis(
    config: LineXAxisConfig | undefined,
    sampleCount: number,
    xOffset?: number,
  ): ResolvedLineXAxis {
    const axis = config ?? { kind: 'index' as const };

    if ('values' in axis) {
      const values =
        axis.values instanceof Float32Array
          ? new Float32Array(axis.values)
          : new Float32Array(axis.values);
      if (values.length !== sampleCount) {
        throw new Error('Explicit line X values must match the series length.');
      }
      for (let i = 0; i < values.length; i++) {
        if (!Number.isFinite(values[i])) {
          throw new Error('Explicit line X values must be finite.');
        }
        if (i > 0 && values[i] <= values[i - 1]) {
          throw new Error(
            'Explicit line X values must be strictly increasing.',
          );
        }
      }
      return {
        config: axis,
        xMode: LINE_X_MODE_EXPLICIT,
        xValues: values,
        xStart: values[0],
        xStep: values.length > 1 ? values[1] - values[0] : 1,
        xMin: values[0],
        xMax: values[values.length - 1],
      };
    }

    if ('unitsPerSample' in axis) {
      const units = axis.unitsPerSample;
      if (!Number.isFinite(units) || units <= 0) {
        throw new Error(
          'Line streaming unitsPerSample must be finite and positive.',
        );
      }
      const start = axis.start ?? 0;
      if (!Number.isFinite(start)) {
        throw new Error('Line streaming start must be finite.');
      }
      const total = (xOffset ?? this.lineTotalSamplesReceived) || sampleCount;
      const xMin = start + Math.max(0, total - sampleCount) * units;
      const xMax = xMin + (sampleCount - 1) * units;
      return {
        config: axis,
        xMode: LINE_X_MODE_LINEAR,
        xValues: undefined,
        xStart: xMin,
        xStep: units,
        xMin,
        xMax,
      };
    }

    if ('min' in axis && 'max' in axis) {
      if (
        !Number.isFinite(axis.min) ||
        !Number.isFinite(axis.max) ||
        axis.max <= axis.min
      ) {
        throw new Error(
          'Line X axis min/max must be finite with max greater than min.',
        );
      }
      return {
        config: axis,
        xMode: LINE_X_MODE_LINEAR,
        xValues: undefined,
        xStart: axis.min,
        xStep: (axis.max - axis.min) / (sampleCount - 1),
        xMin: axis.min,
        xMax: axis.max,
      };
    }

    return {
      config: axis,
      xMode: LINE_X_MODE_LINEAR,
      xValues: undefined,
      xStart: 0,
      xStep: 1,
      xMin: 0,
      xMax: sampleCount - 1,
    };
  }

  private resolveLineYRange(
    config: LineYAxisConfig | undefined,
    reset: boolean,
  ): LineRange {
    if (config?.rangeMode === 'fixed') {
      if (
        config.min === undefined ||
        config.max === undefined ||
        !Number.isFinite(config.min) ||
        !Number.isFinite(config.max) ||
        config.max <= config.min
      ) {
        throw new Error('Fixed line Y ranges require finite min/max values.');
      }
      this.lineStickyYActive = false;
      return { min: config.min, max: config.max };
    }

    if (!this.lineStickyYActive && !reset && this.lineYRange) {
      return this.lineYRange;
    }

    const computed = this.computeLineAutoYRange(
      config?.paddingRatio ?? DEFAULT_LINE_Y_PADDING_RATIO,
    );
    if (reset || !this.lineYRange || !this.lineStickyYActive) {
      this.lineStickyYActive = true;
      return computed;
    }
    return {
      min: Math.min(this.lineYRange.min, computed.min),
      max: Math.max(this.lineYRange.max, computed.max),
    };
  }

  private computeLineAutoYRange(paddingRatio: number): LineRange {
    const padding = Number.isFinite(paddingRatio)
      ? Math.max(0, paddingRatio)
      : DEFAULT_LINE_Y_PADDING_RATIO;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    for (const series of this.lineSeries) {
      if (!series.visible) continue;
      for (const value of series.data) {
        if (!Number.isFinite(value)) continue;
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    }

    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return { min: -1, max: 1 };
    }
    if (min === max) {
      const delta = Math.max(1, Math.abs(min) * padding);
      return { min: min - delta, max: max + delta };
    }
    const delta = (max - min) * padding;
    return { min: min - delta, max: max + delta };
  }

  private flattenLineData(): Float32Array {
    const flat = new Float32Array(
      this.lineSeries.length * this.lineSampleCount,
    );
    for (let i = 0; i < this.lineSeries.length; i++) {
      flat.set(this.lineSeries[i].data, i * this.lineSampleCount);
    }
    return flat;
  }

  private buildLineColorBuffer(): Float32Array {
    const colors = new Float32Array(this.lineSeries.length * 4);
    for (let i = 0; i < this.lineSeries.length; i++) {
      const [r, g, b, a] = this.lineSeries[i].color;
      const offset = i * 4;
      colors[offset] = r / 255;
      colors[offset + 1] = g / 255;
      colors[offset + 2] = b / 255;
      colors[offset + 3] = a;
    }
    return colors;
  }

  private buildLineVisibilityBuffer(): Uint32Array {
    const visibility = new Uint32Array(this.lineSeries.length);
    for (let i = 0; i < this.lineSeries.length; i++) {
      visibility[i] = this.lineSeries[i].visible ? 1 : 0;
    }
    return visibility;
  }

  private updateLineGpuConfig(): void {
    if (!this.lineXAxis || !this.lineYRange || this.lineSeries.length === 0) {
      return;
    }
    if (this.lineXAxis.xMode === LINE_X_MODE_EXPLICIT) {
      (this.inner as WasmLeibnizFastInner).setLineData(
        this.flattenLineData(),
        this.lineSampleCount,
        this.lineSeries.length,
        this.buildLineColorBuffer(),
        this.buildLineVisibilityBuffer(),
        this.lineXAxis.xValues,
        this.lineXAxis.xMode,
        this.lineXAxis.xStart,
        this.lineXAxis.xStep,
        this.lineXAxis.xMin,
        this.lineXAxis.xMax,
        this.lineYRange.min,
        this.lineYRange.max,
      );
      return;
    }
    (this.inner as WasmLeibnizFastInner).updateLineConfig(
      this.buildLineColorBuffer(),
      this.buildLineVisibilityBuffer(),
      this.lineXAxis.xMode,
      this.lineXAxis.xStart,
      this.lineXAxis.xStep,
      this.lineXAxis.xMin,
      this.lineXAxis.xMax,
      this.lineYRange.min,
      this.lineYRange.max,
    );
  }

  private syncLineLegend(): void {
    const chart = this.getLineChart();
    if (!this.legendDiv || !chart || chart.legend === false) {
      if (this.legendDiv) this.legendDiv.style.display = 'none';
      this.lineLegendSignature = '';
      return;
    }

    this.legendDiv.style.display = 'block';

    const signature = this.getLineLegendSignature();
    if (
      signature !== this.lineLegendSignature ||
      this.legendDiv.childElementCount !== this.lineSeries.length
    ) {
      this.legendDiv.replaceChildren();
      for (const series of this.lineSeries) {
        this.legendDiv.appendChild(this.createLineLegendButton(series));
      }
      this.lineLegendSignature = signature;
    } else {
      const buttons = this.legendDiv.querySelectorAll<HTMLButtonElement>(
        '.lf-line-legend-item',
      );
      for (let i = 0; i < this.lineSeries.length; i++) {
        const button = buttons[i];
        if (button) this.updateLineLegendButton(button, this.lineSeries[i]);
      }
    }
    this.positionLineLegend();
  }

  private getLineLegendSignature(): string {
    return this.lineSeries
      .map(
        (series) =>
          `${series.id}\u0000${series.name}\u0000${series.color.join(',')}`,
      )
      .join('\u0001');
  }

  private createLineLegendButton(series: LineSeriesState): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.style.display = 'flex';
    button.style.alignItems = 'center';
    button.style.gap = '6px';
    button.style.width = '100%';
    button.style.margin = '0 0 4px 0';
    button.style.padding = '3px 4px';
    button.style.border = '0';
    button.style.borderRadius = '3px';
    button.style.background = 'transparent';
    button.style.color = 'inherit';
    button.style.cursor = 'pointer';
    button.style.textAlign = 'left';

    const swatch = document.createElement('span');
    swatch.className = 'lf-line-legend-swatch';
    swatch.style.width = '18px';
    swatch.style.height = '3px';
    swatch.style.flex = '0 0 auto';

    const label = document.createElement('span');
    label.className = 'lf-line-legend-label';
    label.style.overflow = 'hidden';
    label.style.textOverflow = 'ellipsis';
    label.style.whiteSpace = 'nowrap';

    button.append(swatch, label);
    button.addEventListener('mousedown', (event) => event.stopPropagation());
    button.addEventListener('mouseup', (event) => event.stopPropagation());
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      const id = button.dataset.seriesId;
      const current = this.lineSeries.find((item) => item.id === id);
      if (id && current) {
        this.setLineSeriesVisibility(id, !current.visible);
      }
    });

    this.updateLineLegendButton(button, series);
    return button;
  }

  private updateLineLegendButton(
    button: HTMLButtonElement,
    series: LineSeriesState,
  ): void {
    button.dataset.seriesId = series.id;
    button.className = series.visible
      ? 'lf-line-legend-item'
      : 'lf-line-legend-item is-hidden';
    button.title = series.name;
    button.style.opacity = series.visible ? '1' : '0.38';

    const swatch = button.querySelector<HTMLSpanElement>(
      '.lf-line-legend-swatch',
    );
    if (swatch) {
      const [r, g, b, a] = series.color;
      swatch.style.background = `rgba(${r}, ${g}, ${b}, ${a})`;
    }

    const label = button.querySelector<HTMLSpanElement>(
      '.lf-line-legend-label',
    );
    if (label) label.textContent = series.name;
  }

  private positionLineLegend(): void {
    if (!this.legendDiv || !this.wrapperDiv) return;
    const chart = this.getLineChart();
    if (!chart || chart.legend === false) return;

    const wrapperRect = this.wrapperDiv.getBoundingClientRect();
    const left = this.layout.x + this.layout.width + 16;
    const rightPadding = 16;
    const width = Math.max(80, wrapperRect.width - left - rightPadding);
    this.legendDiv.style.left = `${left}px`;
    this.legendDiv.style.top = `${this.layout.y}px`;
    this.legendDiv.style.width = `${width}px`;
    this.legendDiv.style.maxHeight = `${this.layout.height}px`;
  }

  private setLineYManual(): void {
    const chart = this.getLineChart();
    if (chart?.yAxis?.rangeMode !== 'fixed') {
      this.lineStickyYActive = false;
    }
  }

  private resetLineStickyY(): void {
    const chart = this.getLineChart();
    if (!chart || chart.yAxis?.rangeMode === 'fixed') return;
    this.lineStickyYActive = true;
    this.lineYRange = this.resolveLineYRange(chart.yAxis, true);
    this.updateLineGpuConfig();
  }

  private resolveLineSampleAtX(
    series: LineSeriesState,
    x: number,
  ): { sampleIndex: number; value: number } | null {
    if (
      !this.lineXAxis ||
      this.lineSampleCount < 2 ||
      x < this.lineXAxis.xMin ||
      x > this.lineXAxis.xMax
    ) {
      return null;
    }

    let sampleIndex = 0;
    let fraction = 0;

    const xValues = this.lineXAxis.xValues;
    if (xValues) {
      if (x === xValues[this.lineSampleCount - 1]) {
        sampleIndex = this.lineSampleCount - 2;
        fraction = 1;
      } else {
        let lo = 0;
        let hi = this.lineSampleCount - 1;
        while (hi - lo > 1) {
          const mid = lo + Math.floor((hi - lo) / 2);
          if (xValues[mid] <= x) lo = mid;
          else hi = mid;
        }
        sampleIndex = lo;
        const x0 = xValues[sampleIndex];
        const x1 = xValues[sampleIndex + 1];
        fraction = (x - x0) / (x1 - x0);
      }
    } else {
      const rawIndex = (x - this.lineXAxis.xStart) / this.lineXAxis.xStep;
      if (rawIndex >= this.lineSampleCount - 1) {
        sampleIndex = this.lineSampleCount - 2;
        fraction = 1;
      } else {
        sampleIndex = Math.max(0, Math.floor(rawIndex));
        fraction = rawIndex - sampleIndex;
      }
    }

    fraction = Math.max(0, Math.min(1, fraction));
    const firstIndex =
      (this.lineRingCursor + sampleIndex) % this.lineSampleCount;
    const secondIndex =
      (this.lineRingCursor + sampleIndex + 1) % this.lineSampleCount;
    const first = series.data[firstIndex];
    const second = series.data[secondIndex];
    if (!Number.isFinite(first) || !Number.isFinite(second)) return null;

    return {
      sampleIndex,
      value: first + (second - first) * fraction,
    };
  }

  private lineValueToCanvasPoint(
    x: number,
    value: number,
    uv: Float32Array,
  ): { x: number; y: number } | null {
    if (!this.lineXAxis || !this.lineYRange) return null;

    const xSpan = this.lineXAxis.xMax - this.lineXAxis.xMin;
    const ySpan = this.lineYRange.max - this.lineYRange.min;
    if (xSpan <= 0 || ySpan <= 0 || uv[2] <= 0 || uv[3] <= 0) return null;

    const fullU = (x - this.lineXAxis.xMin) / xSpan;
    const fullV = (this.lineYRange.max - value) / ySpan;
    return {
      x: ((fullU - uv[0]) / uv[2]) * this.canvas.clientWidth,
      y: ((fullV - uv[1]) / uv[3]) * this.canvas.clientHeight,
    };
  }

  private emitLineHover(canvasX: number, canvasY: number): void {
    if (!this.lineXAxis || !this.lineYRange || this.lineSampleCount < 2) {
      this.lineHoverGuide = null;
      return;
    }

    const canvasWidth = this.canvas.clientWidth;
    const canvasHeight = this.canvas.clientHeight;
    if (canvasWidth <= 0 || canvasHeight <= 0) {
      this.lineHoverGuide = null;
      return;
    }

    const clampedMouseX = Math.max(0, Math.min(canvasX, canvasWidth));
    const clampedMouseY = Math.max(0, Math.min(canvasY, canvasHeight));
    const uv = this.inner.getVisibleRange();
    const xSpan = this.lineXAxis.xMax - this.lineXAxis.xMin;
    const fullU = uv[0] + (clampedMouseX / canvasWidth) * uv[2];
    const x = this.lineXAxis.xMin + fullU * xSpan;
    const chart = this.getLineChart();
    const points: LineHoverPoint[] = [];
    const guidePoints: LineHoverGuidePoint[] = [];

    for (
      let seriesIndex = 0;
      seriesIndex < this.lineSeries.length;
      seriesIndex++
    ) {
      const series = this.lineSeries[seriesIndex];
      if (!series.visible) continue;

      const resolved = this.resolveLineSampleAtX(series, x);
      if (!resolved) continue;

      points.push({
        seriesId: series.id,
        seriesName: series.name,
        seriesIndex,
        sampleIndex: resolved.sampleIndex,
        x,
        value: resolved.value,
        color: [...series.color] as [number, number, number, number],
      });

      const point = this.lineValueToCanvasPoint(x, resolved.value, uv);
      if (
        point &&
        point.x >= 0 &&
        point.x <= canvasWidth &&
        point.y >= 0 &&
        point.y <= canvasHeight
      ) {
        guidePoints.push({
          x: point.x,
          y: point.y,
          color: series.color,
        });
      }
    }

    this.lineHoverGuide = {
      mouseX: clampedMouseX,
      mouseY: clampedMouseY,
      points: guidePoints,
    };

    this.hoverCallback?.({
      kind: 'line',
      x,
      mouseX: clampedMouseX,
      mouseY: clampedMouseY,
      xUnit: chart?.xAxis?.unit,
      yUnit: chart?.yAxis?.unit,
      points,
    });
    this.scheduleOverlayUpdate();
  }

  // ---------------------------------------------------------------------------
  // Private: event listener registration
  // ---------------------------------------------------------------------------

  /** The element that receives mouse/wheel events (wrapper or canvas). */
  private get eventTarget(): HTMLElement {
    return this.wrapperDiv ?? this.canvas;
  }

  private registerEventListeners(): void {
    if (this.disposed) return;
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
    const targets = new Set<HTMLElement>([this.canvas]);
    if (this.wrapperDiv) targets.add(this.wrapperDiv);

    for (const target of targets) {
      target.removeEventListener('mousedown', this.boundHandlers.mousedown);
      target.removeEventListener('mousemove', this.boundHandlers.mousemove);
      target.removeEventListener('mouseenter', this.boundHandlers.mouseenter);
      target.removeEventListener('mouseleave', this.boundHandlers.mouseleave);
      target.removeEventListener('wheel', this.boundHandlers.wheel);
      target.removeEventListener('contextmenu', this.boundHandlers.contextmenu);
      target.removeEventListener('dblclick', this.boundHandlers.dblclick);
    }
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
    const chart = this.getHeatmapChart();
    return !!(chart?.xAxis && isStreamingAxis(chart.xAxis));
  }

  // ---------------------------------------------------------------------------
  // Private: mouse event handlers
  // ---------------------------------------------------------------------------

  private handleMouseDown(e: MouseEvent): void {
    if (this.disposed) return;
    const { wx, wy } = this.wrapperCoords(e);
    const region = this.getHitRegion(wx, wy);
    const { cx, cy } = this.toCanvasLocal(wx, wy);
    const isRight = e.button === 2;
    const isLeft = e.button === 0;
    const streaming = this.isStreamingChart();

    if (isLeft && region === 'matrix') {
      // Standard left-drag pan via WASM
      if (this.isLineChart()) this.setLineYManual();
      this.mouseInPlot = true;
      this.lineHoverGuide = null;
      this.interactionMode = { type: 'matrix-pan' };
      this.inner.onMouseDown(cx, cy);
      this.updateCursor('grabbing');
    } else if (isLeft && region === 'y-axis') {
      this.mouseInPlot = false;
      if (this.isLineChart()) this.setLineYManual();
      this.interactionMode = { type: 'axis-pan', axis: 'y', lastPos: wy };
      this.updateCursor('grabbing');
    } else if (isLeft && region === 'x-axis') {
      this.mouseInPlot = false;
      this.interactionMode = { type: 'axis-pan', axis: 'x', lastPos: wx };
      this.updateCursor('grabbing');
    } else if (isRight && region === 'matrix' && !streaming) {
      this.mouseInPlot = true;
      if (this.isLineChart()) this.setLineYManual();
      this.interactionMode = {
        type: 'rect-select',
        startX: wx,
        startY: wy,
        currentX: wx,
        currentY: wy,
      };
      this.updateCursor('crosshair');
    } else if (isRight && region === 'y-axis') {
      this.mouseInPlot = false;
      if (this.isLineChart()) this.setLineYManual();
      this.interactionMode = {
        type: 'axis-select',
        axis: 'y',
        startPos: wy,
        currentPos: wy,
      };
    } else if (isRight && region === 'x-axis') {
      this.mouseInPlot = false;
      this.interactionMode = {
        type: 'axis-select',
        axis: 'x',
        startPos: wx,
        currentPos: wx,
      };
    }
  }

  private handleMouseMove(e: MouseEvent): void {
    if (this.disposed) return;
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
          this.mouseInPlot = true;
          if (this.isLineChart()) {
            this.emitLineHover(cx, cy);
          } else {
            this.lineHoverGuide = null;
            this.inner.onMouseMove(cx, cy);
          }
          this.hoveredAxis = null;
          this.updateCursor('default');
        } else if (region === 'x-axis') {
          this.mouseInPlot = false;
          this.lineHoverGuide = null;
          this.hoveredAxis = 'x';
          this.updateCursor('col-resize');
        } else if (region === 'y-axis') {
          this.mouseInPlot = false;
          this.lineHoverGuide = null;
          this.hoveredAxis = 'y';
          this.updateCursor('row-resize');
        } else {
          this.mouseInPlot = false;
          this.lineHoverGuide = null;
          this.hoveredAxis = null;
          this.updateCursor('default');
        }
        this.scheduleOverlayUpdate();
        return;
      }
    }
  }

  private handleMouseUp(_e: MouseEvent): void {
    if (this.disposed) return;
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
    if (this.disposed) return;
    e.preventDefault();
    const { wx, wy } = this.wrapperCoords(e);
    const region = this.getHitRegion(wx, wy);
    const { cx, cy } = this.toCanvasLocal(wx, wy);
    const delta = -e.deltaY;

    if (region === 'matrix') {
      if (this.isLineChart()) this.setLineYManual();
      this.inner.onWheel(cx, cy, delta);
    } else if (region === 'x-axis') {
      // Zoom X at the horizontal position mapped to canvas-local X
      this.inner.zoomAtX(cx, delta);
    } else if (region === 'y-axis') {
      if (this.isLineChart()) this.setLineYManual();
      // Zoom Y at the vertical position mapped to canvas-local Y
      this.inner.zoomAtY(cy, delta);
    }
    this.scheduleOverlayUpdate();
  }

  private handleDblClick(e: MouseEvent): void {
    if (this.disposed) return;
    const { wx, wy } = this.wrapperCoords(e);
    const region = this.getHitRegion(wx, wy);

    const uv = this.inner.getVisibleRange();
    const from = { x: uv[0], y: uv[1], w: uv[2], h: uv[3] };

    if (region === 'matrix') {
      if (this.isLineChart()) this.resetLineStickyY();
      this.animateToUvRect(from, { x: 0, y: 0, w: 1, h: 1 });
    } else if (region === 'x-axis') {
      this.animateToUvRect(from, { x: 0, y: from.y, w: 1, h: from.h });
    } else if (region === 'y-axis') {
      if (this.isLineChart()) this.resetLineStickyY();
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
    if (this.disposed) return;
    // Cancel any in-progress animation
    if (this.zoomAnimationId !== null) {
      cancelAnimationFrame(this.zoomAnimationId);
      this.zoomAnimationId = null;
    }

    const startTime = performance.now();

    const step = (now: number) => {
      if (this.disposed) return;
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

    if (this.isLineChart()) this.setLineYManual();
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
      if (this.isLineChart()) this.setLineYManual();
      this.inner.zoomToUvRect(uv[0], vMin, uv[0] + uv[2], vMax);
    }
  }

  // ---------------------------------------------------------------------------
  // Private: cursor management
  // ---------------------------------------------------------------------------

  private updateCursor(cursor: string): void {
    if (this.disposed) return;
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
    if (this.disposed) return;
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

    const legend = document.createElement('div');
    legend.className = 'lf-line-legend';
    legend.style.position = 'absolute';
    legend.style.display = 'none';
    legend.style.padding = '8px';
    legend.style.border = '1px solid rgba(255, 255, 255, 0.18)';
    legend.style.borderRadius = '4px';
    legend.style.background = 'rgba(12, 12, 16, 0.86)';
    legend.style.overflow = 'auto';
    legend.style.pointerEvents = 'auto';
    legend.style.font = '12px sans-serif';
    legend.style.color = '#ddd';
    wrapper.appendChild(legend);

    this.wrapperDiv = wrapper;
    this.overlayCanvas = overlay;
    this.overlayCtx = overlay.getContext('2d');
    this.legendDiv = legend;

    // Re-register listeners on wrapper (covers axis regions)
    this.registerEventListeners();
    this.syncLineLegend();
  }

  /**
   * Remove the overlay canvas and wrapper div, restoring the original
   * canvas position in the DOM.
   */
  private teardownChartOverlay(registerCanvasListeners: boolean = true): void {
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
    if (this.legendDiv) {
      this.legendDiv.remove();
    }

    this.wrapperDiv = null;
    this.overlayCanvas = null;
    this.overlayCtx = null;
    this.legendDiv = null;

    // Re-register on canvas directly when the viewer remains alive.
    if (registerCanvasListeners && !this.disposed) {
      this.registerEventListeners();
    }
  }

  /**
   * Handle window resize: recalculate layout, resize both canvases,
   * update the WASM renderer, and redraw the overlay.
   */
  private handleResize(): void {
    if (this.disposed) return;
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
      this.positionLineLegend();
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
    if (this.disposed) return;
    if (this.overlayDirty) return;
    this.overlayDirty = true;
    requestAnimationFrame(() => {
      if (this.disposed) return;
      this.overlayDirty = false;
      this.updateOverlay();
    });
  }

  /**
   * Redraw the chart overlay (axes, ticks, labels, title, selections, highlights).
   * Called after data changes, pan/zoom, and resize.
   */
  private updateOverlay(): void {
    if (this.disposed) return;
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
      this.getColorbarData(),
    );
    this.positionLineLegend();

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

    if (
      this.isLineChart() &&
      this.interactionMode.type === 'idle' &&
      this.lineHoverGuide
    ) {
      drawLineHoverGuides(
        ctx,
        this.layout,
        this.lineHoverGuide.mouseX,
        this.lineHoverGuide.mouseY,
        this.lineHoverGuide.points,
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

    if (this.isLineChart()) {
      const xSpan = xRange[1] - xRange[0];
      const ySpan = yRange[1] - yRange[0];
      return {
        xMin: xRange[0] + uvOffset[0] * xSpan,
        xMax: xRange[0] + (uvOffset[0] + uvScale[0]) * xSpan,
        yMin: yRange[1] - (uvOffset[1] + uvScale[1]) * ySpan,
        yMax: yRange[1] - uvOffset[1] * ySpan,
      };
    }

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
    if (this.isLineChart()) {
      return this.lineXAxis
        ? [this.lineXAxis.xMin, this.lineXAxis.xMax]
        : [0, Math.max(1, this.lineSampleCount - 1)];
    }

    const chart = this.getHeatmapChart();
    if (!chart?.xAxis) return [0, 1];
    const xAxis = chart.xAxis;
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
    if (this.isLineChart()) {
      return this.lineYRange
        ? [this.lineYRange.min, this.lineYRange.max]
        : [-1, 1];
    }

    const chart = this.getHeatmapChart();
    if (!chart?.yAxis) return [0, 1];
    return [chart.yAxis.min, chart.yAxis.max];
  }

  /**
   * Build a HoverInfo object from raw matrix indices and value.
   * Maps row/col to axis coordinates when a chart config is present.
   */
  private buildHoverInfo(
    row: number,
    col: number,
    value: number,
    valueAvailable: boolean = true,
  ): HoverInfo {
    const info: HeatmapHoverInfo = { row, col, value, valueAvailable };
    const cfg = this.getHeatmapChart();
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

    const color = this.getHeatmapValueColor(value);
    if (color) {
      info.color = color;
    }

    return info;
  }

  /**
   * Re-invoke the hover lookup at the last known mouse position.
   */
  private refreshHoverIfNeeded(): void {
    if (!this.mouseInside) return;

    if (this.isLineChart()) {
      if (!this.mouseInPlot) return;
      this.emitLineHover(this.lastMouseX, this.lastMouseY);
      return;
    }

    if (!this.hoverCallback) return;

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
    if (!data) {
      this.hoverCallback(this.buildHoverInfo(row, col, Number.NaN, false));
      return;
    }
    const idx = this.dataColMajor ? bufCol * rows + row : row * cols + bufCol;
    const value = data[idx];

    this.hoverCallback(this.buildHoverInfo(row, col, value, true));
  }
}

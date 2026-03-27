/**
 * Chart axes rendering — tick generation, layout computation, and
 * 2D Canvas overlay drawing.
 *
 * All text rendering uses a CanvasRenderingContext2D overlay positioned
 * on top of the WebGPU canvas. The WebGPU pipeline is not modified.
 */

import type { AxisConfig, ChartConfig, StreamingAxisConfig } from './types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_FONT = '12px sans-serif';
const DEFAULT_TITLE_FONT = 'bold 16px sans-serif';
const DEFAULT_TICK_COLOR = '#999';
const DEFAULT_LABEL_COLOR = '#ccc';
const DEFAULT_BG_COLOR = '#1a1a1a';

/** Length of each tick mark in CSS pixels. */
const TICK_LENGTH = 6;
/** Padding between elements in CSS pixels. */
const PADDING = 10;
/** Minimum spacing between tick labels in CSS pixels. */
const MIN_TICK_SPACING = 50;
/** Extra padding at container edges so labels don't clip. */
const EDGE_PADDING = 16;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** A single tick mark along an axis. */
export interface Tick {
  /** Data-space value at this tick. */
  value: number;
  /** Formatted string for the tick label. */
  label: string;
  /** Pixel position along the axis (relative to the matrix area origin). */
  position: number;
}

/** Rectangle describing the matrix area within the container. */
export interface LayoutRect {
  /** Left margin in CSS pixels. */
  x: number;
  /** Top margin in CSS pixels. */
  y: number;
  /** Matrix area width in CSS pixels. */
  width: number;
  /** Matrix area height in CSS pixels. */
  height: number;
}

/** Visible data range after camera transform. */
export interface VisibleRange {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

// ---------------------------------------------------------------------------
// Type guards
// ---------------------------------------------------------------------------

/**
 * Returns true if the axis config is a streaming config (has `unitsPerCell`).
 */
export function isStreamingAxis(
  axis: AxisConfig | StreamingAxisConfig,
): axis is StreamingAxisConfig {
  return 'unitsPerCell' in axis;
}

// ---------------------------------------------------------------------------
// Tick generation
// ---------------------------------------------------------------------------

/**
 * Round a number to a "nice" value — the nearest value in the sequence
 * 1, 2, 5, 10, 20, 50, ... that is >= `roughInterval`.
 */
function niceInterval(roughInterval: number): number {
  const exponent = Math.floor(Math.log10(roughInterval));
  const fraction = roughInterval / Math.pow(10, exponent);
  let niceFraction: number;
  if (fraction <= 1) niceFraction = 1;
  else if (fraction <= 2) niceFraction = 2;
  else if (fraction <= 5) niceFraction = 5;
  else niceFraction = 10;
  return niceFraction * Math.pow(10, exponent);
}

/**
 * Format a tick value for display. Uses fixed notation when sensible,
 * otherwise falls back to exponential.
 */
function formatTickValue(value: number, interval: number): string {
  if (value === 0) return '0';
  const absVal = Math.abs(value);
  if (absVal >= 1e6 || absVal < 1e-3) {
    return value.toExponential(1);
  }
  // Determine decimal places from the interval
  const decimals = Math.max(0, -Math.floor(Math.log10(interval)));
  return value.toFixed(decimals);
}

/**
 * Generate evenly-spaced tick marks for an axis.
 *
 * @param min - Data-space minimum of the visible range
 * @param max - Data-space maximum of the visible range
 * @param availablePixels - Pixel length of the axis
 * @param ctx - 2D context for text measurement
 * @param font - CSS font string
 * @returns Array of ticks with positions in [0, availablePixels]
 */
export function generateTicks(
  min: number,
  max: number,
  availablePixels: number,
  ctx: CanvasRenderingContext2D,
  font: string,
): Tick[] {
  if (availablePixels <= 0 || max <= min) return [];

  const range = max - min;
  const maxTicks = Math.max(2, Math.floor(availablePixels / MIN_TICK_SPACING));
  const roughInterval = range / maxTicks;
  const interval = niceInterval(roughInterval);

  const ticks: Tick[] = [];
  const firstTick = Math.ceil(min / interval) * interval;

  ctx.font = font;

  for (let value = firstTick; value <= max; value += interval) {
    // Snap near-zero values to exactly 0
    if (Math.abs(value) < interval * 1e-10) value = 0;

    const label = formatTickValue(value, interval);
    const fraction = (value - min) / range;
    const position = fraction * availablePixels;
    ticks.push({ value, label, position });
  }

  // Remove ticks whose labels overlap
  return removeOverlappingTicks(ticks, ctx, font);
}

/**
 * Filter out ticks whose labels would overlap, keeping the first and
 * removing subsequent overlapping ones.
 */
function removeOverlappingTicks(
  ticks: Tick[],
  ctx: CanvasRenderingContext2D,
  font: string,
): Tick[] {
  if (ticks.length <= 1) return ticks;
  ctx.font = font;

  const result: Tick[] = [ticks[0]];
  let lastRight =
    ticks[0].position + ctx.measureText(ticks[0].label).width / 2 + 4;

  for (let i = 1; i < ticks.length; i++) {
    const halfWidth = ctx.measureText(ticks[i].label).width / 2;
    const left = ticks[i].position - halfWidth;
    if (left >= lastRight) {
      result.push(ticks[i]);
      lastRight = ticks[i].position + halfWidth + 4;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Layout computation
// ---------------------------------------------------------------------------

/**
 * Compute the matrix area rectangle within the container, accounting for
 * margins needed by title, axis labels, and tick labels.
 *
 * @param containerWidth - Container width in CSS pixels
 * @param containerHeight - Container height in CSS pixels
 * @param chart - Chart configuration
 * @param ctx - 2D context for text measurement
 * @returns LayoutRect describing where the matrix should be rendered
 */
export function computeLayout(
  containerWidth: number,
  containerHeight: number,
  chart: ChartConfig,
  ctx: CanvasRenderingContext2D,
): LayoutRect {
  const font = chart.font ?? DEFAULT_FONT;
  const titleFont = chart.titleFont ?? DEFAULT_TITLE_FONT;

  // --- Top margin: title ---
  let topMargin = PADDING;
  if (chart.title) {
    ctx.font = titleFont;
    const titleMetrics = ctx.measureText(chart.title);
    const titleHeight =
      titleMetrics.actualBoundingBoxAscent +
      titleMetrics.actualBoundingBoxDescent;
    topMargin += titleHeight + PADDING;
  }

  // --- Bottom margin: X tick labels + X axis label + edge padding ---
  let bottomMargin = EDGE_PADDING;
  ctx.font = font;
  const sampleMetrics = ctx.measureText('0');
  const tickLabelHeight =
    sampleMetrics.actualBoundingBoxAscent +
    sampleMetrics.actualBoundingBoxDescent;

  if (chart.xAxis) {
    bottomMargin += TICK_LENGTH + 2 + tickLabelHeight + PADDING;
    if (chart.xAxis.label || chart.xAxis.unit) {
      const axisLabel = formatAxisLabel(chart.xAxis.label, chart.xAxis.unit);
      const labelMetrics = ctx.measureText(axisLabel);
      const labelHeight =
        labelMetrics.actualBoundingBoxAscent +
        labelMetrics.actualBoundingBoxDescent;
      bottomMargin += labelHeight + PADDING;
    }
  }

  // --- Left margin: Y axis label + Y tick labels + edge padding ---
  let leftMargin = EDGE_PADDING;
  if (chart.yAxis) {
    // Estimate max tick label width using extreme values
    const maxTickWidth = estimateMaxTickLabelWidth(
      chart.yAxis.min,
      chart.yAxis.max,
      ctx,
      font,
    );
    leftMargin += TICK_LENGTH + 2 + maxTickWidth + PADDING;

    if (chart.yAxis.label || chart.yAxis.unit) {
      leftMargin += tickLabelHeight + PADDING; // rotated text height = text ascent
    }
  }

  // --- Right margin ---
  const rightMargin = EDGE_PADDING;

  return {
    x: leftMargin,
    y: topMargin,
    width: Math.max(1, containerWidth - leftMargin - rightMargin),
    height: Math.max(1, containerHeight - topMargin - bottomMargin),
  };
}

/**
 * Estimate the maximum pixel width of tick labels for an axis range.
 */
function estimateMaxTickLabelWidth(
  min: number,
  max: number,
  ctx: CanvasRenderingContext2D,
  font: string,
): number {
  ctx.font = font;
  const range = max - min;
  const interval = niceInterval(range / 5);
  const candidates = [
    formatTickValue(min, interval),
    formatTickValue(max, interval),
    formatTickValue((min + max) / 2, interval),
  ];
  let maxWidth = 0;
  for (const label of candidates) {
    maxWidth = Math.max(maxWidth, ctx.measureText(label).width);
  }
  return maxWidth;
}

/**
 * Format an axis label with its unit: "Time (s)" or just "Time".
 */
function formatAxisLabel(label?: string, unit?: string): string {
  if (label && unit) return `${label} (${unit})`;
  if (label) return label;
  if (unit) return `(${unit})`;
  return '';
}

// ---------------------------------------------------------------------------
// Overlay rendering
// ---------------------------------------------------------------------------

/**
 * Render the chart overlay: title, axes, tick marks, tick labels, and
 * axis labels onto the 2D canvas.
 *
 * @param ctx - CanvasRenderingContext2D of the overlay canvas
 * @param layout - Matrix area rectangle
 * @param chart - Chart configuration
 * @param visible - Visible data range (after camera transform)
 * @param containerWidth - Container width in CSS pixels
 * @param containerHeight - Container height in CSS pixels
 * @param dpr - Device pixel ratio
 */
export function renderOverlay(
  ctx: CanvasRenderingContext2D,
  layout: LayoutRect,
  chart: ChartConfig,
  visible: VisibleRange,
  containerWidth: number,
  containerHeight: number,
  dpr: number,
): void {
  const font = chart.font ?? DEFAULT_FONT;
  const titleFont = chart.titleFont ?? DEFAULT_TITLE_FONT;
  const tickColor = chart.tickColor ?? DEFAULT_TICK_COLOR;
  const labelColor = chart.labelColor ?? DEFAULT_LABEL_COLOR;
  const bgColor = chart.backgroundColor ?? DEFAULT_BG_COLOR;

  // Scale for DPR — all drawing uses CSS pixel coordinates
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Clear entire overlay
  ctx.clearRect(0, 0, containerWidth, containerHeight);

  // Fill margin regions with background color
  drawMarginBackground(ctx, layout, containerWidth, containerHeight, bgColor);

  // Draw title
  if (chart.title) {
    drawTitle(ctx, chart.title, layout, containerWidth, titleFont, labelColor);
  }

  // Draw X axis
  if (chart.xAxis) {
    drawXAxis(ctx, layout, chart.xAxis, visible, font, tickColor, labelColor);
  }

  // Draw Y axis
  if (chart.yAxis) {
    drawYAxis(ctx, layout, chart.yAxis, visible, font, tickColor, labelColor);
  }

  ctx.restore();
}

/**
 * Fill margin areas around the matrix with the background color.
 */
function drawMarginBackground(
  ctx: CanvasRenderingContext2D,
  layout: LayoutRect,
  containerWidth: number,
  containerHeight: number,
  bgColor: string,
): void {
  ctx.fillStyle = bgColor;
  // Top
  ctx.fillRect(0, 0, containerWidth, layout.y);
  // Bottom
  ctx.fillRect(
    0,
    layout.y + layout.height,
    containerWidth,
    containerHeight - layout.y - layout.height,
  );
  // Left
  ctx.fillRect(0, layout.y, layout.x, layout.height);
  // Right
  ctx.fillRect(
    layout.x + layout.width,
    layout.y,
    containerWidth - layout.x - layout.width,
    layout.height,
  );
}

/**
 * Draw the chart title centered above the matrix area.
 */
function drawTitle(
  ctx: CanvasRenderingContext2D,
  title: string,
  layout: LayoutRect,
  containerWidth: number,
  titleFont: string,
  labelColor: string,
): void {
  ctx.font = titleFont;
  ctx.fillStyle = labelColor;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  const centerX = layout.x + layout.width / 2;
  ctx.fillText(title, centerX, layout.y - PADDING);
}

/**
 * Draw the X axis: line, ticks, tick labels, and axis label.
 */
function drawXAxis(
  ctx: CanvasRenderingContext2D,
  layout: LayoutRect,
  axisConfig: AxisConfig | StreamingAxisConfig,
  visible: VisibleRange,
  font: string,
  tickColor: string,
  labelColor: string,
): void {
  const axisY = layout.y + layout.height;

  // Axis line
  ctx.strokeStyle = tickColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(layout.x, axisY);
  ctx.lineTo(layout.x + layout.width, axisY);
  ctx.stroke();

  // Ticks
  const ticks = generateTicks(
    visible.xMin,
    visible.xMax,
    layout.width,
    ctx,
    font,
  );

  ctx.font = font;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';

  for (const tick of ticks) {
    const x = layout.x + tick.position;

    // Tick mark
    ctx.strokeStyle = tickColor;
    ctx.beginPath();
    ctx.moveTo(x, axisY);
    ctx.lineTo(x, axisY + TICK_LENGTH);
    ctx.stroke();

    // Tick label
    ctx.fillStyle = labelColor;
    ctx.fillText(tick.label, x, axisY + TICK_LENGTH + 2);
  }

  // Axis label + unit
  const axisLabel = formatAxisLabel(axisConfig.label, axisConfig.unit);
  if (axisLabel) {
    ctx.font = font;
    ctx.fillStyle = labelColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const sampleMetrics = ctx.measureText('0');
    const tickLabelHeight =
      sampleMetrics.actualBoundingBoxAscent +
      sampleMetrics.actualBoundingBoxDescent;
    const labelY = axisY + TICK_LENGTH + 2 + tickLabelHeight + PADDING;
    ctx.fillText(axisLabel, layout.x + layout.width / 2, labelY);
  }
}

/**
 * Draw the Y axis: line, ticks, tick labels, and axis label.
 */
function drawYAxis(
  ctx: CanvasRenderingContext2D,
  layout: LayoutRect,
  axisConfig: AxisConfig,
  visible: VisibleRange,
  font: string,
  tickColor: string,
  labelColor: string,
): void {
  const axisX = layout.x;

  // Axis line
  ctx.strokeStyle = tickColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(axisX, layout.y);
  ctx.lineTo(axisX, layout.y + layout.height);
  ctx.stroke();

  // Ticks — image convention: row 0 is at the top of the canvas, which
  // corresponds to yMin (the near/start of the physical range). generateTicks
  // produces position=0 for yMin, which maps to layout.y (top). No inversion.
  const ticks = generateTicks(
    visible.yMin,
    visible.yMax,
    layout.height,
    ctx,
    font,
  );

  ctx.font = font;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (const tick of ticks) {
    // No inversion: UV.y=0 is the top of the canvas, which corresponds to
    // yMin (row 0), so position=0 should map to the top of the matrix area.
    const y = layout.y + tick.position;

    // Tick mark
    ctx.strokeStyle = tickColor;
    ctx.beginPath();
    ctx.moveTo(axisX - TICK_LENGTH, y);
    ctx.lineTo(axisX, y);
    ctx.stroke();

    // Tick label
    ctx.fillStyle = labelColor;
    ctx.fillText(tick.label, axisX - TICK_LENGTH - 2, y);
  }

  // Axis label + unit (rotated -90 degrees)
  const axisLabel = formatAxisLabel(axisConfig.label, axisConfig.unit);
  if (axisLabel) {
    ctx.save();
    ctx.font = font;
    ctx.fillStyle = labelColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const centerY = layout.y + layout.height / 2;
    // After -90° rotation, "top" baseline points away from the left edge,
    // ensuring the label does not clip against the container border.
    ctx.translate(EDGE_PADDING, centerY);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(axisLabel, 0, 0);
    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// Selection & highlight rendering
// ---------------------------------------------------------------------------

/** Semi-transparent blue for selection rectangles. */
const SELECTION_FILL = 'rgba(59, 130, 246, 0.2)';
/** Dashed border color for selection rectangles. */
const SELECTION_STROKE = 'rgba(59, 130, 246, 0.8)';
/** Subtle highlight for axis hover feedback. */
const AXIS_HOVER_FILL = 'rgba(255, 255, 255, 0.05)';

/**
 * Draw a selection rectangle on the overlay canvas.
 *
 * @param ctx - 2D rendering context
 * @param x - Left edge in CSS pixels
 * @param y - Top edge in CSS pixels
 * @param w - Width in CSS pixels
 * @param h - Height in CSS pixels
 */
export function drawSelectionRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
): void {
  ctx.fillStyle = SELECTION_FILL;
  ctx.fillRect(x, y, w, h);

  ctx.strokeStyle = SELECTION_STROKE;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
}

/**
 * Draw a subtle highlight strip on an axis region to indicate interactivity.
 *
 * @param ctx - 2D rendering context
 * @param layout - Matrix area layout
 * @param axis - Which axis to highlight ('x' or 'y')
 * @param containerWidth - Full container width in CSS pixels
 * @param containerHeight - Full container height in CSS pixels
 */
export function drawAxisHighlight(
  ctx: CanvasRenderingContext2D,
  layout: LayoutRect,
  axis: 'x' | 'y',
  containerWidth: number,
  containerHeight: number,
): void {
  ctx.fillStyle = AXIS_HOVER_FILL;
  if (axis === 'x') {
    ctx.fillRect(
      layout.x,
      layout.y + layout.height,
      layout.width,
      containerHeight - layout.y - layout.height,
    );
  } else {
    ctx.fillRect(0, layout.y, layout.x, layout.height);
  }
}

/**
 * Map camera UV coordinates to data-space axis ranges.
 *
 * @param uvOffset - Camera UV offset [x, y]
 * @param uvScale - Camera UV scale [x, y]
 * @param xMin - Full X axis data minimum
 * @param xMax - Full X axis data maximum
 * @param yMin - Full Y axis data minimum
 * @param yMax - Full Y axis data maximum
 * @returns Visible data range
 */
export function uvToVisibleRange(
  uvOffset: [number, number],
  uvScale: [number, number],
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
): VisibleRange {
  const xRange = xMax - xMin;
  const yRange = yMax - yMin;
  return {
    xMin: xMin + uvOffset[0] * xRange,
    xMax: xMin + (uvOffset[0] + uvScale[0]) * xRange,
    yMin: yMin + uvOffset[1] * yRange,
    yMax: yMin + (uvOffset[1] + uvScale[1]) * yRange,
  };
}

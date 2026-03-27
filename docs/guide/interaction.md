# Mouse Interaction

LeibnizFast provides rich mouse interactions for exploring matrix data. The chart area is divided into three interactive regions: the **matrix area** (the heatmap itself), the **X-axis region** (below the matrix), and the **Y-axis region** (to the left of the matrix).

## Interaction Summary

| Action | Matrix Area | X-Axis Region | Y-Axis Region |
|--------|------------|---------------|---------------|
| **Scroll wheel** | Zoom both axes | Zoom X only | Zoom Y only |
| **Left-click drag** | Pan both axes | Pan X only | Pan Y only |
| **Right-click drag** | Rectangle zoom | Zoom X range | Zoom Y range |
| **Double-click** | Reset all zoom | Reset X zoom | Reset Y zoom |

## Zooming

### Scroll Wheel Zoom

- **On the matrix**: Zooms both axes simultaneously, centered on the cursor position.
- **On an axis region**: Zooms only that axis. Scrolling on the X-axis region zooms horizontally; scrolling on the Y-axis region zooms vertically.

### Rectangle Zoom (Right-Click Drag)

On the **matrix area**, hold the right mouse button and drag to draw a selection rectangle. When you release, the view zooms to fit the selected region. A semi-transparent blue overlay shows the selection as you drag.

A minimum drag distance of 5 pixels is required to trigger the zoom. Smaller drags are treated as a no-op, preventing accidental micro-zooms.

### Axis Range Zoom (Right-Click Drag on Axis)

On an **axis region**, hold the right mouse button and drag along the axis to select a data range. When you release, the view zooms to show only that range on the selected axis while keeping the other axis unchanged.

For example, if your X-axis shows 0--10 s and you right-click drag from the 4 s to 7 s tick mark, the view will zoom to show 4--7 s horizontally while keeping the full vertical range.

## Panning

### Matrix Pan (Left-Click Drag)

When zoomed in, left-click drag on the **matrix area** to pan the view in both directions.

### Axis Pan (Left-Click Drag on Axis)

When zoomed in, left-click drag on an **axis region** to pan only that axis. Dragging on the X-axis region pans horizontally; dragging on the Y-axis region pans vertically.

## Reset Zoom

**Double-click** to reset the zoom:

- **On the matrix**: Resets both axes to the full-matrix view.
- **On an axis region**: Resets only that axis, keeping the other axis at its current zoom level.

You can also call [`resetZoom()`](/api/leibniz-fast#resetzoom) programmatically.

## Visual Feedback

Interactive axis regions provide visual cues:

- **Cursor changes**: The cursor changes to `col-resize` when hovering over the X-axis region and `row-resize` over the Y-axis region, indicating that zoom/pan interactions are available.
- **Axis highlight**: A subtle highlight appears on the hovered axis region.
- **Selection overlay**: During right-click drag operations, a semi-transparent blue rectangle (or band) shows the selection in progress.
- **Drag cursors**: The cursor changes to `grabbing` during pan operations and `crosshair` during rectangle selection.

## Static vs. Waterfall Charts

The available interactions depend on the chart type:

### Static Charts

All interactions are available on both axes. Static charts use [`AxisConfig`](/api/types#axisconfig) with fixed `min`/`max` values for their X-axis.

### Waterfall (Streaming) Charts

Since the X-axis represents streaming time data, **X-axis interactions are disabled**:

- No scroll zoom on the X-axis region
- No left-click pan on the X-axis region
- No right-click range selection on the X-axis region
- No rectangle zoom on the matrix area
- No double-click reset on the X-axis region

**Y-axis interactions remain fully available**: scroll zoom, pan, right-click range selection, and double-click reset all work on the vertical axis.

## Context Menu

The browser's right-click context menu is automatically suppressed on the chart to enable right-click drag interactions.

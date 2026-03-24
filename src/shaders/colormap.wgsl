// Compute shader: applies colormap to raw matrix data and writes to a tile texture.
//
// Supports both single-texture (small matrices) and tiled-texture (large matrices)
// that exceed the device's maxTextureDimension2D limit.
//
// Each invocation handles one cell of the *tile*. The global_invocation_id.x/y
// are tile-relative coordinates. The absolute matrix position is obtained by
// adding col_offset/row_offset before reading from the flat data buffer.
//
// Workgroup size 16×16 is tunable but must match the dispatch calculation on
// the Rust side (ceiling division by 16).

// Binding 0: Raw matrix data as a flat array of f32 values (row-major order,
// full-matrix dimensions). The data is always indexed in full-matrix space.
@group(0) @binding(0) var<storage, read> matrix_data: array<f32>;

// Binding 1: Output RGBA texture — one pixel per tile cell.
// Size = tile_width × tile_height (≤ maxTextureDimension2D on both axes).
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

// Binding 2: Per-dispatch parameters
struct MatrixParams {
    // Rows of this tile/chunk (≤ maxTextureDimension2D)
    rows: u32,
    // Columns of this tile (≤ maxTextureDimension2D)
    cols: u32,
    // Data range for normalization
    min_val: f32,
    max_val: f32,
    // Absolute row in the full matrix where this tile/chunk starts
    row_offset: u32,
    // Absolute column in the full matrix where this tile starts
    col_offset: u32,
    // Total columns of the full matrix (used for flat buffer indexing)
    total_cols: u32,
    // Y offset when writing to tile texture (for multi-chunk-per-tile staging)
    texture_row_offset: u32,
    // X offset when writing to tile texture (for partial column updates / scrolling)
    texture_col_offset: u32,
    // When 1, the staging buffer is column-major: data[col * col_stride + row].
    // When 0 (default), the staging buffer is row-major: data[row * total_cols + col].
    col_major: u32,
    // Stride for column-major indexing (= total matrix rows). Unused when col_major=0.
    col_stride: u32,
    _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: MatrixParams;

// Binding 3: Colormap lookup table — 256×1 RGBA texture
@group(0) @binding(3) var colormap_lut: texture_2d<f32>;

// Binding 4: Sampler for the colormap LUT
@group(0) @binding(4) var colormap_sampler: sampler;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Tile-relative coordinates
    let tile_col = global_id.x;
    let tile_row = global_id.y;

    // Bounds check against this tile's dimensions
    if tile_col >= params.cols || tile_row >= params.rows {
        return;
    }

    // Absolute position in the full matrix
    let abs_row = tile_row + params.row_offset;
    let abs_col = tile_col + params.col_offset;

    // Read from the staging buffer. Row-major (default) uses the full-matrix row
    // stride; column-major (ring-buffer path) uses the column stride so the new
    // column strip can be uploaded without transposing.
    let idx = select(
        abs_row * params.total_cols + abs_col,     // row-major
        abs_col * params.col_stride + abs_row,      // column-major
        params.col_major != 0u,
    );
    let value = matrix_data[idx];

    // Normalize to [0, 1]
    let range = params.max_val - params.min_val;
    var normalized: f32;
    if range > 0.0 {
        normalized = clamp((value - params.min_val) / range, 0.0, 1.0);
    } else {
        normalized = 0.5;
    }

    // Sample the colormap LUT
    let color = textureSampleLevel(
        colormap_lut, colormap_sampler,
        vec2<f32>(normalized, 0.5), 0.0
    );

    // Write to tile-relative coordinates in the output texture.
    // texture_row_offset shifts Y when multiple chunks write to the same tile.
    // texture_col_offset shifts X for partial column updates (scrolled streaming).
    textureStore(output_texture, vec2<i32>(i32(tile_col + params.texture_col_offset), i32(tile_row + params.texture_row_offset)), color);
}

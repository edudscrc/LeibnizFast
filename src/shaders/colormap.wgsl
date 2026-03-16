// Compute shader: applies colormap to raw matrix data.
//
// Reads each matrix cell value, normalizes it to [0,1] using the data range,
// then samples the colormap LUT texture to produce an RGBA color.
// Output is written to a storage texture for the render pass to display.

// Binding 0: Raw matrix data as a flat array of f32 values (row-major order)
@group(0) @binding(0) var<storage, read> matrix_data: array<f32>;

// Binding 1: Output RGBA texture — each pixel corresponds to one matrix cell
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

// Binding 2: Matrix parameters — dimensions and data range for normalization
struct MatrixParams {
    rows: u32,
    cols: u32,
    min_val: f32,
    max_val: f32,
}
@group(0) @binding(2) var<uniform> params: MatrixParams;

// Binding 3: Colormap lookup table — 256x1 RGBA texture
@group(0) @binding(3) var colormap_lut: texture_2d<f32>;

// Binding 4: Sampler for the colormap LUT (linear interpolation between entries)
@group(0) @binding(4) var colormap_sampler: sampler;

// Workgroup size: 16x16 threads per workgroup.
// Each thread processes one matrix cell.
// Chosen as a good balance between occupancy and register usage on most GPUs.
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;

    // Bounds check: workgroup dispatch may overshoot matrix dimensions
    if col >= params.cols || row >= params.rows {
        return;
    }

    // Read the raw data value from the flat array (row-major indexing)
    let idx = row * params.cols + col;
    let value = matrix_data[idx];

    // Normalize value to [0, 1] range using the data min/max
    let range = params.max_val - params.min_val;
    var normalized: f32;
    if range > 0.0 {
        normalized = clamp((value - params.min_val) / range, 0.0, 1.0);
    } else {
        // Constant data: map everything to the middle of the colormap
        normalized = 0.5;
    }

    // Sample the colormap LUT texture at the normalized position.
    // UV coordinates: u = normalized value, v = 0.5 (center of 1-pixel-tall texture)
    let color = textureSampleLevel(colormap_lut, colormap_sampler, vec2<f32>(normalized, 0.5), 0.0);

    // Write the colored pixel to the output texture
    textureStore(output_texture, vec2<i32>(i32(col), i32(row)), color);
}

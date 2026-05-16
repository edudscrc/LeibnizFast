// Vertex + fragment shader for line charts.
//
// The vertex shader expands a logical line-list from storage buffers:
// vertex_index -> series, segment, endpoint. X values are either generated
// linearly or loaded from a shared explicit X buffer. Y values are stored
// series-major and can be ring-unwrapped by the cursor uniform.

@group(0) @binding(0) var<storage, read> y_values: array<f32>;
@group(0) @binding(1) var<storage, read> x_values: array<f32>;
@group(0) @binding(2) var<storage, read> colors: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> visibility: array<u32>;

struct LineParams {
  sample_count: u32,
  series_count: u32,
  ring_cursor: u32,
  x_mode: u32,
  x_min: f32,
  x_max: f32,
  y_min: f32,
  y_max: f32,
  uv_x_offset: f32,
  uv_y_offset: f32,
  uv_x_scale: f32,
  uv_y_scale: f32,
  x_start: f32,
  x_step: f32,
  _pad0: f32,
  _pad1: f32,
}

@group(0) @binding(4) var<uniform> params: LineParams;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
}

fn is_valid_value(value: f32) -> bool {
  return value == value && abs(value) < 3.402823e38;
}

fn physical_sample_index(logical_index: u32) -> u32 {
  return (params.ring_cursor + logical_index) % params.sample_count;
}

fn x_at(logical_index: u32) -> f32 {
  if params.x_mode == 1u {
    return x_values[logical_index];
  }
  return params.x_start + f32(logical_index) * params.x_step;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
  out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

  if params.sample_count < 2u || params.series_count == 0u {
    return out;
  }

  let vertices_per_series = (params.sample_count - 1u) * 2u;
  let series = vertex_index / vertices_per_series;
  if series >= params.series_count || visibility[series] == 0u {
    return out;
  }

  let local_vertex = vertex_index % vertices_per_series;
  let segment = local_vertex / 2u;
  let endpoint = local_vertex % 2u;
  let logical_index = segment + endpoint;

  let y0 =
    y_values[series * params.sample_count + physical_sample_index(segment)];
  let y1 =
    y_values[series * params.sample_count + physical_sample_index(
      segment + 1u,
    )];
  if !is_valid_value(y0) || !is_valid_value(y1) {
    return out;
  }

  let y = select(y0, y1, endpoint == 1u);
  let x = x_at(logical_index);
  if
    !is_valid_value(x)
      || params.x_max <= params.x_min
      || params.y_max <= params.y_min
  {
    return out;
  }

  let full_u = (x - params.x_min) / (params.x_max - params.x_min);
  let full_v = (params.y_max - y) / (params.y_max - params.y_min);
  let screen_u = (full_u - params.uv_x_offset) / params.uv_x_scale;
  let screen_v = (full_v - params.uv_y_offset) / params.uv_y_scale;

  out.position =
    vec4<f32>(screen_u * 2.0 - 1.0, 1.0 - screen_v * 2.0, 0.0, 1.0);
  out.color = colors[series];
  return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return input.color;
}

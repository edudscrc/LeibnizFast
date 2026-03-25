//! # Renderer
//!
//! Manages the wgpu device, surface, and rendering pipeline. Handles
//! initialization of the GPU context and orchestrates compute and render passes.
//!
//! Supports two rendering paths:
//! - **WebGPU (compute)**: compute shader applies colormap on GPU
//! - **WebGL2 (CPU fallback)**: colormap applied on CPU, uploaded as RGBA texture
//!
//! ## Tiling
//! When matrix dimensions exceed `maxTextureDimension2D`, the output texture is
//! split into a grid of tiles (each ≤ `max_texture_dimension` per side). The
//! compute pass is dispatched once per tile; the render pass draws one quad per tile
//! with UV coordinates mapped to that tile's region of the full matrix.

use crate::camera::Camera;
use crate::chunked_upload::WORKGROUP_ALIGNMENT;
use crate::colormap::ColormapTexture;
use crate::matrix::{MatrixParams, MatrixView};
use crate::perf::PerfTimer;
use crate::pipeline::PipelineFactory;
use crate::tile_grid::TileGrid;

// ---------------------------------------------------------------------------
// Per-tile camera buffer
// ---------------------------------------------------------------------------

// A camera uniform buffer that maps a tile's region of the matrix.
//
// The render shader already has a `CameraUniforms` struct with `uv_offset` and
// `uv_scale`. For tiled rendering we create one buffer per tile so that each
// tile's quad maps its texture [0,1] to the correct sub-region of the full
// matrix, then the main camera transform is applied on top.
//
// The data layout matches `CameraUniforms` in render.wgsl:
//   offset: vec2<f32>  (bytes 0-7)
//   scale:  vec2<f32>  (bytes 8-15)
//
// For tiled rendering we embed the tile UV region into the buffer and set the
// camera uniform to identity — the tile quad is drawn with clip-space coords
// that already map to the tile's screen region.
//
// Simpler approach used here: reuse the existing camera buffer but override
// it before each tile's draw call using a per-tile buffer pre-computed during
// `rebuild_pipelines` that encodes the **tile's UV region within the full matrix**.
// The render shader then further applies the user's camera transform on top.
//
// Actually: the cleanest solution is to render each tile as a sub-quad covering
// only the fraction of clip-space that corresponds to that tile. The render shader
// receives the main camera and samples the tile texture using coordinates remapped
// to [0,1] within the tile.  We achieve this by passing a *tile camera* that maps
// screen UV -> tile UV, computed as:
//
//   tile_uv = (screen_uv * camera_scale + camera_offset - tile_uv_offset) / tile_uv_size
//
// But that would require modifying the shader. Instead we use a simpler two-step
// approach:
//
// **Step 1 (compute)**: Each tile texture contains the correctly colored pixels for
// its sub-region. No UV trickery needed.
//
// **Step 2 (render)**: We draw the tile covering the fraction of clip space
// [left, right] x [bottom, top] that the tile occupies in the full matrix, then
// sample the tile texture at UV [0,1] adjusted by the camera transform *relative
// to the tile's position*.
//
// The shader already handles this: we pass a `CameraUniforms` that has been
// pre-composed with the tile position. Specifically, for tile (tx, ty):
//
//   tile_cam_offset = camera.uv_offset - vec2(col_uv_off, row_uv_off)
//   tile_cam_offset /= vec2(col_uv_size, row_uv_size)
//   tile_cam_scale  = camera.uv_scale  / vec2(col_uv_size, row_uv_size)
//
// Then the quad covers full clip-space (unchanged) and the shader uses the
// composed transform to decide which part of the tile texture to sample,
// discarding fragments outside [0,1].

/// Core renderer that owns the wgpu device, queue, and surface.
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    /// Whether the device supports compute shaders
    pub has_compute: bool,
    /// Maximum 2D texture dimension reported by the device
    pub max_texture_dimension: u32,

    // --- Shared pipelines (rebuilt on colormap/data change) ---
    compute_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for compute pipeline (kept to create per-tile groups)
    compute_bind_group_layout: Option<wgpu::BindGroupLayout>,
    render_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group layout for render pipeline (group 0: per-tile)
    render_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Bind group layout for colormap (group 1: shared)
    colormap_bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// Shared colormap bind group (group 1) for render pipeline
    colormap_bind_group: Option<wgpu::BindGroup>,
    /// Range uniform buffer (min_val, max_val) for fragment shader
    range_buffer: Option<wgpu::Buffer>,

    // --- Per-tile resources ---
    /// Colored tile textures, indexed by tile_grid.tile_index(tx, ty)
    tile_textures: Vec<wgpu::Texture>,
    tile_texture_views: Vec<wgpu::TextureView>,
    /// Per-tile compute bind groups (one per tile)
    tile_compute_bind_groups: Vec<wgpu::BindGroup>,
    /// Per-tile params buffers (row_offset, col_offset per tile)
    tile_params_buffers: Vec<wgpu::Buffer>,
    /// Per-tile camera uniform buffers (encode tile UV region)
    tile_camera_buffers: Vec<wgpu::Buffer>,
    /// Per-tile render bind groups (one per tile)
    tile_render_bind_groups: Vec<wgpu::BindGroup>,

    // --- Tile layout ---
    tile_grid: Option<TileGrid>,

    /// Matrix dimensions
    matrix_rows: u32,
    matrix_cols: u32,

    /// Whether the colormap has already been applied to the textures (staged path).
    colormap_applied: bool,
    /// Enable performance timing logs.
    debug: bool,

    /// Reusable buffer for reading matrix chunks, avoiding per-chunk allocations.
    /// Sized to `staging_capacity_rows * cols` during pipeline rebuild.
    chunk_buffer: Vec<f32>,

    /// Ring buffer write cursor: the next texture column to write to.
    /// Advances by `new_cols` on each scrolled update. Wraps at `matrix_cols`.
    ring_cursor: u32,
}

impl Renderer {
    /// Borrow the reusable chunk buffer, growing it if needed.
    ///
    /// Returns a mutable slice of exactly `len` elements, resizing the
    /// internal Vec only when the current capacity is too small.
    pub fn chunk_buf_mut(&mut self, len: usize) -> &mut [f32] {
        if self.chunk_buffer.len() < len {
            self.chunk_buffer.resize(len, 0.0f32);
        }
        &mut self.chunk_buffer[..len]
    }
    /// Create a new Renderer from an HTML canvas element.
    pub async fn new(canvas: &web_sys::HtmlCanvasElement, debug: bool) -> Result<Self, String> {
        let _timer = PerfTimer::new("Renderer::new", debug);
        let width = canvas.width();
        let height = canvas.height();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("Failed to create surface: {e}"))?;

        let t_adapter = PerfTimer::new("request_adapter", debug);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find a suitable GPU adapter")?;
        t_adapter.finish();

        log::info!("Adapter: {:?}", adapter.get_info().name);

        // Request device with the adapter's own limits.
        let t_device = PerfTimer::new("request_device", debug);
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("LeibnizFast Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter_limits.clone(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {e}"))?;
        t_device.finish();

        let device_limits = device.limits();
        let has_compute = device_limits.max_storage_buffers_per_shader_stage > 0;
        let max_texture_dimension = device_limits.max_texture_dimension_2d;
        log::info!(
            "Compute shaders: {} (storage buffers: {})",
            if has_compute {
                "available"
            } else {
                "unavailable (WebGL2 fallback)"
            },
            device_limits.max_storage_buffers_per_shader_stage
        );
        log::info!("Max texture dimension: {max_texture_dimension}");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            has_compute,
            max_texture_dimension,
            compute_pipeline: None,
            compute_bind_group_layout: None,
            render_pipeline: None,
            render_bind_group_layout: None,
            colormap_bind_group_layout: None,
            colormap_bind_group: None,
            range_buffer: None,
            tile_textures: Vec::new(),
            tile_texture_views: Vec::new(),
            tile_compute_bind_groups: Vec::new(),
            tile_params_buffers: Vec::new(),
            tile_camera_buffers: Vec::new(),
            tile_render_bind_groups: Vec::new(),
            tile_grid: None,
            matrix_rows: 0,
            matrix_cols: 0,
            colormap_applied: false,
            debug,
            chunk_buffer: Vec::new(),
            ring_cursor: 0,
        })
    }

    /// Resize the rendering surface to new dimensions.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface.configure(&self.device, &self.surface_config);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pipeline rebuild
    // -----------------------------------------------------------------------

    /// Rebuild pipelines after data or colormap changes.
    ///
    /// Computes a `TileGrid` from the matrix dimensions and the device's
    /// `maxTextureDimension2D`. Creates one texture per tile, one compute bind
    /// group per tile, and one render bind group per tile.
    ///
    /// **No dimension limit is enforced** — matrices of any size are supported
    /// as long as memory is available.  The only limit that matters is that each
    /// individual tile fits within `max_texture_dimension`, which is guaranteed
    /// by `TileGrid`.
    pub fn rebuild_pipelines(
        &mut self,
        matrix: &Option<MatrixView>,
        colormap: &Option<ColormapTexture>,
        camera: &Camera,
        rows: u32,
        cols: u32,
    ) -> Result<(), String> {
        let _timer = PerfTimer::new("rebuild_pipelines", self.debug);
        let colormap = colormap.as_ref().ok_or("No colormap set")?;

        self.matrix_rows = rows;
        self.matrix_cols = cols;
        self.colormap_applied = false;

        let grid = TileGrid::new(rows, cols, self.max_texture_dimension);
        log::info!(
            "Tiling: {}×{} matrix → {}×{} tiles ({} total)",
            cols,
            rows,
            grid.tiles_x,
            grid.tiles_y,
            grid.tile_count()
        );

        let factory = PipelineFactory::new(&self.device, self.debug);

        // --- Shared pipelines (recreate on each rebuild) ---
        let (compute_pipeline, compute_layout) = factory.create_compute_pipeline();
        let (render_pipeline, render_layout, colormap_layout) =
            factory.create_render_pipeline(self.surface_config.format);

        // Range uniform buffer for fragment shader (min_val, max_val)
        use wgpu::util::DeviceExt;
        let range_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Range Params Buffer"),
                contents: bytemuck::cast_slice(&[crate::matrix::RangeParams {
                    min_val: 0.0,
                    max_val: 1.0,
                    _pad0: 0.0,
                    _pad1: 0.0,
                }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Shared colormap bind group (group 1)
        let colormap_bind_group =
            factory.create_colormap_bind_group(&colormap_layout, colormap, &range_buffer);

        // --- Per-tile resources ---
        let needs_storage = self.has_compute;
        let tile_tex_pairs = factory.create_tiled_textures(&grid, needs_storage);

        let (tile_textures, tile_texture_views): (Vec<_>, Vec<_>) =
            tile_tex_pairs.into_iter().unzip();

        // Per-tile params buffers and compute bind groups (WebGPU only)
        let (tile_params_buffers, tile_compute_bind_groups) = if self.has_compute {
            use crate::matrix::MatrixParams;
            use wgpu::util::DeviceExt;

            let matrix = matrix
                .as_ref()
                .ok_or("No matrix data set (WebGPU path requires MatrixView)")?;

            let mut params_bufs = Vec::with_capacity(grid.tile_count());
            let mut bind_groups = Vec::with_capacity(grid.tile_count());

            for (tx, ty) in grid.iter_tiles() {
                let (row_start, _) = grid.tile_row_range(ty);
                let (col_start, _) = grid.tile_col_range(tx);
                let tile_w = grid.tile_width(tx);
                let tile_h = grid.tile_height(ty);

                let params = MatrixParams {
                    rows: tile_h,
                    cols: tile_w,
                    min_val: 0.0, // updated when colormap is applied
                    max_val: 1.0,
                    row_offset: row_start,
                    col_offset: col_start,
                    total_cols: cols,
                    texture_row_offset: 0,
                    texture_col_offset: 0,
                    col_major: 0,
                    col_stride: 0,
                    _pad2: 0,
                };
                let params_buf =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Tile Params Buffer"),
                            contents: bytemuck::cast_slice(&[params]),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        });

                let idx = grid.tile_index(tx, ty);
                let bg = factory.create_compute_bind_group(
                    &compute_layout,
                    &matrix.data_buffer,
                    &params_buf,
                    &tile_texture_views[idx],
                );
                params_bufs.push(params_buf);
                bind_groups.push(bg);
            }
            (params_bufs, bind_groups)
        } else {
            (Vec::new(), Vec::new())
        };

        // Per-tile camera buffers and render bind groups
        let (tile_camera_buffers, tile_render_bind_groups) =
            self.build_tile_render_data(&grid, camera, &render_layout, &tile_texture_views);

        // Commit — reset ring cursor (tile dimensions may have changed)
        self.ring_cursor = 0;
        self.tile_textures = tile_textures;
        self.tile_texture_views = tile_texture_views;
        self.tile_params_buffers = tile_params_buffers;
        self.tile_compute_bind_groups = tile_compute_bind_groups;
        self.tile_camera_buffers = tile_camera_buffers;
        self.tile_render_bind_groups = tile_render_bind_groups;
        self.compute_pipeline = Some(compute_pipeline);
        self.compute_bind_group_layout = Some(compute_layout);
        self.render_pipeline = Some(render_pipeline);
        self.render_bind_group_layout = Some(render_layout);
        self.colormap_bind_group_layout = Some(colormap_layout);
        self.colormap_bind_group = Some(colormap_bind_group);
        self.range_buffer = Some(range_buffer);
        self.tile_grid = Some(grid);

        Ok(())
    }

    /// Rebuild the shared colormap bind group (render group 1) after a colormap change.
    ///
    /// Only creates one bind group (shared across all tiles) — very cheap.
    pub fn rebuild_colormap_bind_group(
        &mut self,
        colormap: &ColormapTexture,
    ) -> Result<(), String> {
        let _timer = PerfTimer::new("rebuild_colormap_bind_group", self.debug);
        let layout = self
            .colormap_bind_group_layout
            .as_ref()
            .ok_or("Pipelines not initialized")?;
        let range_buffer = self
            .range_buffer
            .as_ref()
            .ok_or("Range buffer not initialized")?;

        let factory = PipelineFactory::new(&self.device, self.debug);
        self.colormap_bind_group =
            Some(factory.create_colormap_bind_group(layout, colormap, range_buffer));
        Ok(())
    }

    /// Update the range uniform buffer for the fragment shader.
    ///
    /// Cost: one `queue.write_buffer()` call (16 bytes). No bind group rebuild.
    pub fn update_range_buffer(&self, min_val: f32, max_val: f32) {
        if let Some(ref buf) = self.range_buffer {
            let data = crate::matrix::RangeParams {
                min_val,
                max_val,
                _pad0: 0.0,
                _pad1: 0.0,
            };
            self.queue
                .write_buffer(buf, 0, bytemuck::cast_slice(&[data]));
        }
    }

    /// Build per-tile camera buffers and render bind groups.
    ///
    /// For each tile (tx, ty) we compute the camera uniform that maps
    /// the main camera transform to the tile's sub-region. The render
    /// shader samples the tile texture using coordinates remapped from
    /// full-matrix UV to tile UV.
    fn build_tile_render_data(
        &self,
        grid: &TileGrid,
        camera: &Camera,
        render_layout: &wgpu::BindGroupLayout,
        tile_views: &[wgpu::TextureView],
    ) -> (Vec<wgpu::Buffer>, Vec<wgpu::BindGroup>) {
        use wgpu::util::DeviceExt;
        let factory = PipelineFactory::new(&self.device, self.debug);

        grid.iter_tiles()
            .map(|(tx, ty)| {
                // UV region of this tile in the full matrix [0,1]
                let tile_col_off = grid.col_uv_offset(tx);
                let tile_row_off = grid.row_uv_offset(ty);
                let tile_col_sz = grid.col_uv_size(tx);
                let tile_row_sz = grid.row_uv_size(ty);

                // Compose camera + tile: map screen UV → tile-local UV
                //   full_matrix_uv = camera.offset + screen_uv * camera.scale
                //   tile_uv = (full_matrix_uv - tile_offset) / tile_size
                // So in the tile shader:
                //   tile_uv = (camera.offset - tile_offset) / tile_size
                //           + screen_uv * (camera.scale / tile_size)
                let cam = camera.state.get_uniforms();
                let composed_offset = [
                    (cam.uv_offset[0] - tile_col_off) / tile_col_sz,
                    (cam.uv_offset[1] - tile_row_off) / tile_row_sz,
                ];
                let composed_scale = [cam.uv_scale[0] / tile_col_sz, cam.uv_scale[1] / tile_row_sz];

                // Pack into 32-byte buffer matching WGSL CameraUniforms:
                // X stays in full-matrix UV space so the ring offset is applied
                // once at the correct scale before per-tile mapping.
                // Y is pre-composed to tile-local UV (no ring on Y).
                let data: [f32; 8] = [
                    cam.uv_offset[0],            // uv_x_offset (full-matrix)
                    composed_offset[1],          // uv_y_offset (tile-local)
                    cam.uv_scale[0],             // uv_x_scale  (full-matrix)
                    composed_scale[1],           // uv_y_scale  (tile-local)
                    self.ring_offset_fraction(), // ring_offset
                    0.0,                         // _pad0
                    tile_col_off,                // tile_x_offset
                    tile_col_sz,                 // tile_x_size
                ];

                let tile_camera_buf =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Tile Camera Buffer"),
                            contents: bytemuck::cast_slice(&data),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        });

                let idx = grid.tile_index(tx, ty);
                let bind_group = factory.create_render_bind_group(
                    render_layout,
                    &tile_views[idx],
                    &tile_camera_buf,
                );

                (tile_camera_buf, bind_group)
            })
            .unzip()
    }

    /// Refresh only the per-tile camera buffers (after pan/zoom), without
    /// rebuilding textures or pipelines.
    pub fn update_tile_camera_buffers(&self, grid: &TileGrid, camera: &Camera) {
        let cam = camera.state.get_uniforms();
        let ring_off = self.ring_offset_fraction();
        for (tx, ty) in grid.iter_tiles() {
            let idx = grid.tile_index(tx, ty);
            let tile_col_off = grid.col_uv_offset(tx);
            let tile_row_off = grid.row_uv_offset(ty);
            let tile_col_sz = grid.col_uv_size(tx);
            let tile_row_sz = grid.row_uv_size(ty);

            let data: [f32; 8] = [
                cam.uv_offset[0],                                // uv_x_offset (full-matrix)
                (cam.uv_offset[1] - tile_row_off) / tile_row_sz, // uv_y_offset (tile-local)
                cam.uv_scale[0],                                 // uv_x_scale  (full-matrix)
                cam.uv_scale[1] / tile_row_sz,                   // uv_y_scale  (tile-local)
                ring_off,                                        // ring_offset
                0.0,                                             // _pad0
                tile_col_off,                                    // tile_x_offset
                tile_col_sz,                                     // tile_x_size
            ];
            self.queue.write_buffer(
                &self.tile_camera_buffers[idx],
                0,
                bytemuck::cast_slice(&data),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Texture uploads (WebGL2 fallback)
    // -----------------------------------------------------------------------

    /// Upload a region of CPU-colormapped RGBA data (WebGL2, tiled or chunked).
    ///
    /// For non-tiled cases `tile_index = 0` and the row_offset is relative to
    /// the texture top. For tiled cases we write to the correct tile texture.
    pub fn upload_colored_texture_region(
        &self,
        rgba_data: &[u8],
        cols: u32,
        chunk_rows: u32,
        row_offset: u32,
    ) {
        if let Some(texture) = self.tile_textures.first() {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: row_offset,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                rgba_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(cols * 4),
                    rows_per_image: Some(chunk_rows),
                },
                wgpu::Extent3d {
                    width: cols,
                    height: chunk_rows,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    // -----------------------------------------------------------------------
    // Compute (colormap) dispatch
    // -----------------------------------------------------------------------

    /// Write MatrixParams to a specific tile's params buffer.
    fn write_tile_params(&self, tile_idx: usize, params: &MatrixParams) {
        self.queue.write_buffer(
            &self.tile_params_buffers[tile_idx],
            0,
            bytemuck::cast_slice(&[*params]),
        );
    }

    /// Copy raw float data to R32Float tile textures via compute shader.
    ///
    /// Iterates over all tiles and chunks, reading data via `read_fn`, writing
    /// it to the staging buffer, and dispatching the compute shader for each chunk.
    /// The compute shader writes raw float values; colormap application happens
    /// in the fragment shader.
    ///
    /// `read_fn(start_index, buffer)` fills `buffer` with f32 data starting
    /// at the given flat index.
    pub fn apply_colormap_tiled(
        &mut self,
        matrix: &MatrixView,
        read_fn: &dyn Fn(usize, &mut [f32]),
        cols: u32,
    ) {
        let _timer = PerfTimer::new("apply_colormap_tiled", self.debug);
        let grid = match &self.tile_grid {
            Some(g) => g.clone(),
            None => return,
        };
        let staging_rows = matrix.staging_capacity_rows();
        let compute_pipeline = match &self.compute_pipeline {
            Some(p) => p,
            None => return,
        };

        // Iterate by row-band first, then dispatch all X-tiles sharing those rows
        // in a single encoder+submit. This avoids redundant data reads and reduces
        // GPU submission overhead from O(tiles_x × chunks) to O(chunks).
        for ty in 0..grid.tiles_y {
            let (row_start, row_end) = grid.tile_row_range(ty);
            let tile_h = row_end - row_start;
            let mut current_tile_row: u32 = 0;

            while current_tile_row < tile_h {
                let chunk_rows = staging_rows.min(tile_h - current_tile_row);
                let abs_row_start = row_start + current_tile_row;

                // Read full rows once for all X-tiles in this band
                let chunk_len = (chunk_rows as usize) * (cols as usize);
                if self.chunk_buffer.len() < chunk_len {
                    self.chunk_buffer.resize(chunk_len, 0.0f32);
                }
                let start_idx = (abs_row_start as usize) * (cols as usize);
                read_fn(start_idx, &mut self.chunk_buffer[..chunk_len]);
                matrix.write_staging_chunk(&self.queue, &self.chunk_buffer[..chunk_len]);

                // Single encoder for all X-tiles in this chunk
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Tiled Compute Encoder"),
                        });

                for tx in 0..grid.tiles_x {
                    let tile_idx = grid.tile_index(tx, ty);
                    let (col_start, _) = grid.tile_col_range(tx);
                    let tile_w = grid.tile_width(tx);

                    self.write_tile_params(
                        tile_idx,
                        &MatrixParams {
                            rows: chunk_rows,
                            cols: tile_w,
                            min_val: 0.0,
                            max_val: 1.0,
                            row_offset: 0,
                            col_offset: col_start,
                            total_cols: cols,
                            texture_row_offset: current_tile_row,
                            texture_col_offset: 0,
                            col_major: 0,
                            col_stride: 0,
                            _pad2: 0,
                        },
                    );

                    let bind_group = &self.tile_compute_bind_groups[tile_idx];
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Tiled Data Copy Pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(compute_pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.dispatch_workgroups(
                            tile_w.div_ceil(WORKGROUP_ALIGNMENT),
                            chunk_rows.div_ceil(WORKGROUP_ALIGNMENT),
                            1,
                        );
                    }
                }

                self.queue.submit(std::iter::once(encoder.finish()));
                current_tile_row += chunk_rows;
            }
        }

        self.colormap_applied = true;
    }

    /// Ring-buffer scrolled update: write new columns at `ring_cursor` and advance.
    ///
    /// The texture acts as a circular buffer. New columns are written at the
    /// current cursor position (wrapping at texture width). The render shader
    /// applies `ring_offset` to unwrap the ring so the oldest column appears at
    /// the left edge and the newest at the right.
    ///
    /// Requires the JS-side buffer to be **column-major**: column `c` occupies
    /// `data[c * rows .. (c+1) * rows]`. This means the JS ring cursor and the
    /// texture ring cursor are both in sync: same position, both advance by
    /// `new_cols` each frame.
    ///
    /// Cost: O(rows × new_cols) per frame — independent of total column count.
    /// For the common case (rows ≤ staging capacity), a single `read_fn` call
    /// per x-tile per segment transfers all new column data in one JS/WASM copy.
    pub fn apply_colormap_ring(
        &mut self,
        matrix: &MatrixView,
        read_fn: &dyn Fn(usize, &mut [f32]),
        cols: u32,
        new_cols: u32,
    ) {
        let _timer = PerfTimer::new("apply_colormap_ring", self.debug);
        let grid = match &self.tile_grid {
            Some(g) => g.clone(),
            None => return,
        };
        let compute_pipeline = match &self.compute_pipeline {
            Some(p) => p,
            None => return,
        };

        if new_cols == 0 {
            return;
        }

        // Clamp: if all columns are replaced it is equivalent to a full refresh
        // starting from the current cursor position (ring stays in-place).
        let new_cols = new_cols.min(cols);

        let rows = self.matrix_rows;
        let rows_usize = rows as usize;
        let staging_rows = matrix.staging_capacity_rows();

        // Split at the wrap boundary: [cursor..cols) then [0..remainder)
        let cursor = self.ring_cursor;
        let first_segment = new_cols.min(cols - cursor);
        let second_segment = new_cols - first_segment;

        // Each segment is a contiguous ring-position range starting at tex_col_start.
        // Because the JS buffer is column-major with the same ring layout, the JS
        // positions are identical to the texture positions within [0, cols).
        let segments: [(u32, u32); 2] = [(cursor, first_segment), (0, second_segment)];

        for &(seg_start, seg_cols) in &segments {
            if seg_cols == 0 {
                continue;
            }
            let seg_end = seg_start + seg_cols;

            for tx in 0..grid.tiles_x {
                let (tile_col_start, tile_col_end) = grid.tile_col_range(tx);

                // Clip segment to this tile's column range
                let clip_start = seg_start.max(tile_col_start);
                let clip_end = seg_end.min(tile_col_end);
                if clip_start >= clip_end {
                    continue;
                }
                let tile_new_cols = clip_end - clip_start;
                let tile_new_cols_usize = tile_new_cols as usize;

                // Texture-local write offset within this tile
                let tex_col_offset = clip_start - tile_col_start;

                for ty in 0..grid.tiles_y {
                    let tile_idx = grid.tile_index(tx, ty);
                    let (row_start, row_end) = grid.tile_row_range(ty);
                    let tile_h = row_end - row_start;
                    let mut current_tile_row: u32 = 0;

                    while current_tile_row < tile_h {
                        let chunk_rows = staging_rows.min(tile_h - current_tile_row);
                        let abs_row_start = row_start + current_tile_row;
                        let chunk_rows_usize = chunk_rows as usize;

                        let staging_len = tile_new_cols_usize * chunk_rows_usize;
                        if self.chunk_buffer.len() < staging_len {
                            self.chunk_buffer.resize(staging_len, 0.0f32);
                        }

                        // Read new columns from column-major ring buffer.
                        //
                        // The JS buffer stores column c at data[c*rows .. (c+1)*rows].
                        // For the common case where all rows fit in one chunk
                        // (chunk_rows == tile_h), columns clip_start..clip_end are
                        // contiguous in memory — one read_fn call copies everything.
                        //
                        // For multi-chunk (large row counts), each column's portion
                        // is non-contiguous, so we fall back to per-column reads.
                        if chunk_rows == tile_h {
                            // Fast path: contiguous block covering all tile rows
                            let js_start = clip_start as usize * rows_usize;
                            read_fn(js_start, &mut self.chunk_buffer[..staging_len]);
                        } else {
                            // Slow path: multiple row chunks, read per column
                            for c in 0..tile_new_cols_usize {
                                let js_col = clip_start as usize + c;
                                let js_off = js_col * rows_usize + abs_row_start as usize;
                                let stage_start = c * chunk_rows_usize;
                                read_fn(
                                    js_off,
                                    &mut self.chunk_buffer
                                        [stage_start..stage_start + chunk_rows_usize],
                                );
                            }
                        }

                        matrix.write_staging_chunk(&self.queue, &self.chunk_buffer[..staging_len]);

                        // Dispatch compute: column-major staging, no row/col offsets
                        // needed since the staging buffer is tile-local.
                        let mut encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Ring Colormap Encoder"),
                                });
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Ring Colormap Pass"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(compute_pipeline);
                            pass.set_bind_group(0, &self.tile_compute_bind_groups[tile_idx], &[]);
                            self.write_tile_params(
                                tile_idx,
                                &MatrixParams {
                                    rows: chunk_rows,
                                    cols: tile_new_cols,
                                    min_val: 0.0,
                                    max_val: 1.0,
                                    row_offset: 0,
                                    col_offset: 0,
                                    total_cols: 0,
                                    texture_row_offset: current_tile_row,
                                    texture_col_offset: tex_col_offset,
                                    col_major: 1,
                                    col_stride: chunk_rows,
                                    _pad2: 0,
                                },
                            );
                            pass.dispatch_workgroups(
                                tile_new_cols.div_ceil(WORKGROUP_ALIGNMENT),
                                chunk_rows.div_ceil(WORKGROUP_ALIGNMENT),
                                1,
                            );
                        }
                        self.queue.submit(std::iter::once(encoder.finish()));

                        current_tile_row += chunk_rows;
                    }
                }
            }
        }

        self.ring_cursor = (cursor + new_cols) % cols;
        self.colormap_applied = true;
    }

    /// Get the current ring buffer offset as a fraction of texture width [0, 1).
    /// Returns 0.0 when the ring buffer is not active (cursor at 0).
    pub fn ring_offset_fraction(&self) -> f32 {
        if self.matrix_cols == 0 {
            return 0.0;
        }
        self.ring_cursor as f32 / self.matrix_cols as f32
    }

    /// Reset the ring cursor to 0 (e.g. after a full colormap or dimension change).
    pub fn reset_ring_cursor(&mut self) {
        self.ring_cursor = 0;
    }

    // -----------------------------------------------------------------------
    // Frame rendering
    // -----------------------------------------------------------------------

    /// Render a single frame.
    ///
    /// Raw float data is in R32Float tile textures. The fragment shader reads
    /// the float, normalizes using the range uniform, and applies the colormap
    /// LUT. Two bind groups are set per tile draw:
    /// - Group 0 (per-tile): data texture + camera uniform
    /// - Group 1 (shared):   colormap LUT + sampler + range uniform
    pub fn render_frame(
        &self,
        _colormap: &Option<ColormapTexture>,
        camera: &Camera,
    ) -> Result<(), String> {
        let _timer = PerfTimer::new("render_frame", self.debug);
        // Update tile camera buffers for current camera state
        if let Some(ref grid) = self.tile_grid {
            self.update_tile_camera_buffers(grid, camera);
        }

        let render_pipeline = match self.render_pipeline.as_ref() {
            Some(p) => p,
            None => return self.render_clear(),
        };

        if self.tile_render_bind_groups.is_empty() {
            return self.render_clear();
        }

        let colormap_bg = match self.colormap_bind_group.as_ref() {
            Some(bg) => bg,
            None => return self.render_clear(),
        };

        let output = self
            .surface
            .get_current_texture()
            .map_err(|e| format!("Failed to get surface texture: {e}"))?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LeibnizFast Encoder"),
            });

        // Render pass: one quad per tile
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(render_pipeline);
            // Group 1 (shared colormap) is the same for all tiles
            render_pass.set_bind_group(1, colormap_bg, &[]);
            for bind_group in &self.tile_render_bind_groups {
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Render a clear frame (no data loaded yet).
    fn render_clear(&self) -> Result<(), String> {
        let output = self
            .surface
            .get_current_texture()
            .map_err(|e| format!("Failed to get surface texture: {e}"))?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Clear Encoder"),
            });

        let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Clear Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.2,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        drop(_render_pass);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Maximum GPU buffer size.
    pub fn max_buffer_size(&self) -> u64 {
        self.device.limits().max_buffer_size
    }

    /// Maximum 2D texture dimension.
    pub fn max_texture_dimension(&self) -> u32 {
        self.max_texture_dimension
    }
}

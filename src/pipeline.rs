//! # Pipeline Factory
//!
//! Centralizes creation of wgpu compute and render pipelines, bind group layouts,
//! and bind groups. Keeps the Renderer clean and allows future swapping of
//! pipeline configurations (e.g., WebGL2 fallback).
//!
//! Tiling support: for matrices larger than `maxTextureDimension2D`, we create
//! multiple smaller textures (tiles) and render each as a separate quad.

use crate::colormap::ColormapTexture;
use crate::perf::PerfTimer;
use crate::tile_grid::TileGrid;

/// Factory for creating wgpu pipelines and their associated resources.
pub struct PipelineFactory<'a> {
    device: &'a wgpu::Device,
    /// Enable performance timing logs.
    debug: bool,
}

impl<'a> PipelineFactory<'a> {
    /// Create a new pipeline factory for the given device.
    pub fn new(device: &'a wgpu::Device, debug: bool) -> Self {
        Self { device, debug }
    }

    /// Create a single tile texture with the given dimensions.
    fn create_tile_texture(
        &self,
        width: u32,
        height: u32,
        needs_storage: bool,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let usage = if needs_storage {
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING
        } else {
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Colored Matrix Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Create all tile textures for a tiled matrix.
    ///
    /// Returns a flat `Vec` indexed by `tile_grid.tile_index(tx, ty)`.
    pub fn create_tiled_textures(
        &self,
        tile_grid: &TileGrid,
        needs_storage: bool,
    ) -> Vec<(wgpu::Texture, wgpu::TextureView)> {
        let _timer = PerfTimer::new("create_tiled_textures", self.debug);
        tile_grid
            .iter_tiles()
            .map(|(tx, ty)| {
                let w = tile_grid.tile_width(tx);
                let h = tile_grid.tile_height(ty);
                self.create_tile_texture(w, h, needs_storage)
            })
            .collect()
    }

    /// Create the compute pipeline (shared across all tiles).
    ///
    /// Returns the pipeline and the bind group **layout** so per-tile bind groups
    /// can be created separately via `create_compute_bind_group`.
    pub fn create_compute_pipeline(&self) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let _timer = PerfTimer::new("create_compute_pipeline", self.debug);
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Compute Bind Group Layout"),
                    entries: &[
                        // Binding 0: Matrix data buffer (read-only storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Binding 1: Output tile texture (write-only storage texture)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        // Binding 2: Matrix params uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Binding 3: Colormap LUT texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Binding 4: Colormap sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Colormap Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/colormap.wgsl").into()),
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Colormap Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        (pipeline, bind_group_layout)
    }

    /// Create a compute bind group for one tile.
    ///
    /// Each tile gets its own `params_buffer` with correct `row_offset` and
    /// `col_offset` for that tile's region. The `data_buffer` is shared across
    /// all tiles (it holds the full matrix or staging data).
    pub fn create_compute_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        data_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        colormap: &ColormapTexture,
        tile_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(tile_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&colormap.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&colormap.sampler),
                },
            ],
        })
    }

    /// Create the render pipeline (shared across tiles).
    ///
    /// Returns the pipeline and its bind group layout.
    pub fn create_render_pipeline(
        &self,
        surface_format: wgpu::TextureFormat,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        let _timer = PerfTimer::new("create_render_pipeline", self.debug);
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Render Bind Group Layout"),
                    entries: &[
                        // Binding 0: Colored tile texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Binding 1: Nearest-neighbor sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // Binding 2: Camera uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Render Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/render.wgsl").into()),
            });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        (pipeline, bind_group_layout)
    }

    /// Create a render bind group for one tile, using the tile's own texture view
    /// and a per-tile camera buffer.
    pub fn create_render_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        tile_view: &wgpu::TextureView,
        camera_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nearest Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(tile_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: camera_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }
}

//! # Renderer
//!
//! Manages the wgpu device, surface, and rendering pipeline. Handles
//! initialization of the GPU context and orchestrates compute and render passes.
//!
//! Supports two rendering paths:
//! - **WebGPU (compute)**: Uses a compute shader to apply colormaps on the GPU
//! - **WebGL2 (CPU fallback)**: Applies colormaps on the CPU and uploads RGBA texture

use crate::camera::Camera;
use crate::colormap::ColormapTexture;
use crate::matrix::MatrixView;
use crate::pipeline::PipelineFactory;

/// Core renderer that owns the wgpu device, queue, and surface.
///
/// Detects at init whether compute shaders are available (WebGPU) or not
/// (WebGL2), and selects the appropriate rendering path.
pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    /// Whether the device supports compute shaders (WebGPU = true, WebGL2 = false)
    pub has_compute: bool,
    /// Compute pipeline for colormap application (WebGPU path only)
    compute_pipeline: Option<wgpu::ComputePipeline>,
    compute_bind_group: Option<wgpu::BindGroup>,
    /// Render pipeline for displaying the textured quad
    render_pipeline: Option<wgpu::RenderPipeline>,
    render_bind_group: Option<wgpu::BindGroup>,
    /// The colored output texture (from compute shader or CPU upload)
    colored_texture: Option<wgpu::Texture>,
    colored_texture_view: Option<wgpu::TextureView>,
    /// Matrix dimensions needed for compute dispatch
    matrix_rows: u32,
    matrix_cols: u32,
}

impl Renderer {
    /// Create a new Renderer from an HTML canvas element.
    ///
    /// Performs async wgpu initialization: Instance → Surface → Adapter → Device.
    /// Detects compute shader support to choose between WebGPU and WebGL2 paths.
    pub async fn new(canvas: &web_sys::HtmlCanvasElement) -> Result<Self, String> {
        let width = canvas.width();
        let height = canvas.height();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("Failed to create surface: {e}"))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find a suitable GPU adapter")?;

        log::info!("Adapter: {:?}", adapter.get_info().name);

        // Request device with the adapter's own limits.
        // We avoid downlevel_webgl2_defaults() because it forces storage buffer
        // limits to 0, which prevents compute shaders even when the adapter supports them.
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

        // Check device limits (not adapter limits) for compute shader support.
        // On WebGL2 backends the adapter may advertise storage buffer support
        // but the device may be created with lower limits.
        let device_limits = device.limits();
        let has_compute = device_limits.max_storage_buffers_per_shader_stage > 0;
        log::info!(
            "Compute shaders: {} (storage buffers: {})",
            if has_compute {
                "available"
            } else {
                "unavailable (WebGL2 fallback)"
            },
            device_limits.max_storage_buffers_per_shader_stage
        );

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
            compute_pipeline: None,
            compute_bind_group: None,
            render_pipeline: None,
            render_bind_group: None,
            colored_texture: None,
            colored_texture_view: None,
            matrix_rows: 0,
            matrix_cols: 0,
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

    /// Rebuild pipelines after data or colormap changes.
    ///
    /// On WebGPU: creates compute + render pipelines using the GPU `MatrixView`.
    /// On WebGL2: creates render-only pipeline (colormap applied on CPU), using
    /// only the matrix dimensions — no GPU storage buffer needed.
    pub fn rebuild_pipelines(
        &mut self,
        matrix: &Option<MatrixView>,
        colormap: &Option<ColormapTexture>,
        camera: &Camera,
        rows: u32,
        cols: u32,
    ) -> Result<(), String> {
        let colormap = colormap.as_ref().ok_or("No colormap set")?;

        self.matrix_rows = rows;
        self.matrix_cols = cols;

        let factory = PipelineFactory::new(&self.device);

        if self.has_compute {
            // WebGPU path: compute shader applies colormap on GPU
            let matrix = matrix.as_ref().ok_or("No matrix data set (WebGPU path requires MatrixView)")?;

            let (colored_texture, colored_view) =
                factory.create_colored_texture(cols, rows, true);
            self.colored_texture_view = Some(colored_view);

            let (compute_pipeline, compute_bind_group) = factory
                .create_compute_pipeline_and_bindings(
                    matrix,
                    colormap,
                    self.colored_texture_view.as_ref().unwrap(),
                );
            self.compute_pipeline = Some(compute_pipeline);
            self.compute_bind_group = Some(compute_bind_group);

            self.colored_texture = Some(colored_texture);
        } else {
            // WebGL2 path: create texture for CPU upload (no STORAGE_BINDING)
            let (colored_texture, colored_view) =
                factory.create_colored_texture(cols, rows, false);
            self.colored_texture_view = Some(colored_view);
            self.colored_texture = Some(colored_texture);
        }

        // Render pipeline is the same for both paths
        let (render_pipeline, render_bind_group) = factory.create_render_pipeline_and_bindings(
            self.colored_texture_view.as_ref().unwrap(),
            camera,
            self.surface_config.format,
        );
        self.render_pipeline = Some(render_pipeline);
        self.render_bind_group = Some(render_bind_group);

        Ok(())
    }

    /// Upload CPU-colormapped RGBA data to the colored texture (WebGL2 fallback).
    ///
    /// Called when compute shaders are unavailable. The colormap is applied on the
    /// CPU and the resulting RGBA pixels are uploaded to the texture.
    pub fn upload_colored_texture(&self, rgba_data: &[u8], cols: u32, rows: u32) {
        if let Some(ref texture) = self.colored_texture {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                rgba_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(cols * 4),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d {
                    width: cols,
                    height: rows,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Render a single frame.
    ///
    /// On WebGPU: compute pass (colormap) → render pass (textured quad).
    /// On WebGL2: render pass only (texture already has colors from CPU upload).
    pub fn render_frame(
        &self,
        _colormap: &Option<ColormapTexture>,
        _camera: &Camera,
    ) -> Result<(), String> {
        // If pipelines haven't been built yet (no data set), just clear
        let render_pipeline = match self.render_pipeline.as_ref() {
            Some(p) => p,
            None => return self.render_clear(),
        };
        let render_bind_group = match self.render_bind_group.as_ref() {
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

        // Compute pass: only on WebGPU path
        if self.has_compute {
            if let (Some(compute_pipeline), Some(compute_bind_group)) =
                (&self.compute_pipeline, &self.compute_bind_group)
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Colormap Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(compute_pipeline);
                compute_pass.set_bind_group(0, compute_bind_group, &[]);

                let workgroups_x = (self.matrix_cols + 15) / 16;
                let workgroups_y = (self.matrix_rows + 15) / 16;
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }
        }

        // Render pass: draw textured quad with camera transform
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
            render_pass.set_bind_group(0, render_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
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
}

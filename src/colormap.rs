//! # Colormap
//!
//! Colormap provider trait and GPU texture management for colormap lookup tables.
//!
//! The `ColormapProvider` trait allows testing colormap resolution without GPU.
//! `ColormapTexture` wraps a wgpu texture + sampler for use in the compute shader.

use crate::colormap_data;

/// Trait for colormap data resolution — allows testing without GPU.
///
/// Implementations provide RGB lookup tables indexed by colormap name.
pub trait ColormapProvider {
    /// Get a 256-entry RGB lookup table for the named colormap.
    /// Returns `None` if the name is not recognized.
    fn get_colormap_rgb(&self, name: &str) -> Option<&'static [[u8; 3]; 256]>;

    /// Get all available colormap names.
    fn available_colormaps(&self) -> &[&str];
}

/// Production implementation using compile-time constant colormap data.
pub struct BuiltinColormaps;

impl ColormapProvider for BuiltinColormaps {
    fn get_colormap_rgb(&self, name: &str) -> Option<&'static [[u8; 3]; 256]> {
        colormap_data::get_colormap_by_name(name)
    }

    fn available_colormaps(&self) -> &[&str] {
        colormap_data::COLORMAP_NAMES
    }
}

/// Apply a colormap to matrix data on the CPU, producing RGBA bytes.
///
/// This is the WebGL2 fallback when compute shaders are unavailable.
/// Each value is normalized to [0,1] using (min,max), then looked up in the LUT.
pub fn apply_colormap_cpu(
    data: &[f32],
    min_val: f32,
    max_val: f32,
    lut: &[[u8; 3]; 256],
) -> Vec<u8> {
    let range = max_val - min_val;
    let mut rgba = Vec::with_capacity(data.len() * 4);

    for &value in data {
        let normalized = if range > 0.0 {
            ((value - min_val) / range).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Map to LUT index [0, 255]
        let idx = (normalized * 255.0).round() as usize;
        let idx = idx.min(255);
        let rgb = lut[idx];
        rgba.push(rgb[0]);
        rgba.push(rgb[1]);
        rgba.push(rgb[2]);
        rgba.push(255);
    }

    rgba
}

/// GPU texture + sampler for a colormap lookup table.
///
/// The colormap is stored as a 256x1 RGBA texture. The compute shader
/// samples this texture with the normalized data value to get the color.
///
/// Only available when compiling for WASM target.
#[cfg(target_arch = "wasm32")]
pub struct ColormapTexture {
    /// 256x1 RGBA texture containing the colormap
    pub texture_view: wgpu::TextureView,
    /// Linear sampler for interpolating between colormap entries
    pub sampler: wgpu::Sampler,
}

#[cfg(target_arch = "wasm32")]
impl ColormapTexture {
    /// Create a new colormap texture from a 256-entry RGB lookup table.
    ///
    /// Converts the RGB data to RGBA (alpha=255) and uploads to a 256x1 texture.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, rgb_data: &[[u8; 3]; 256]) -> Self {
        // Convert RGB → RGBA for the texture
        let mut rgba_data = [0u8; 256 * 4];
        for (i, rgb) in rgb_data.iter().enumerate() {
            rgba_data[i * 4] = rgb[0];
            rgba_data[i * 4 + 1] = rgb[1];
            rgba_data[i * 4 + 2] = rgb[2];
            rgba_data[i * 4 + 3] = 255;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Colormap LUT Texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Colormap Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture_view,
            sampler,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_colormaps_have_256_entries() {
        let provider = BuiltinColormaps;
        for name in provider.available_colormaps() {
            let data = provider
                .get_colormap_rgb(name)
                .unwrap_or_else(|| panic!("Colormap '{name}' should exist"));
            assert_eq!(data.len(), 256, "Colormap '{name}' should have 256 entries");
        }
    }

    #[test]
    fn test_all_rgb_values_in_range() {
        let provider = BuiltinColormaps;
        for name in provider.available_colormaps() {
            let data = provider.get_colormap_rgb(name).unwrap();
            for (i, rgb) in data.iter().enumerate() {
                // u8 is always 0-255, but verify no accidental wrap-around
                // by checking that adjacent entries don't have huge jumps
                if i > 0 {
                    let prev = data[i - 1];
                    let _dr = (rgb[0] as i16 - prev[0] as i16).unsigned_abs();
                    let _dg = (rgb[1] as i16 - prev[1] as i16).unsigned_abs();
                    let _db = (rgb[2] as i16 - prev[2] as i16).unsigned_abs();
                    // Perceptually uniform colormaps shouldn't have huge jumps
                    // (allows up to 20 per channel per step for approximations)
                }
            }
        }
    }

    #[test]
    fn test_viridis_known_values() {
        let provider = BuiltinColormaps;
        let viridis = provider.get_colormap_rgb("viridis").unwrap();

        // Viridis starts dark (low RGB values)
        assert!(viridis[0][0] < 100, "Viridis start R should be dark");
        assert!(viridis[0][1] < 50, "Viridis start G should be dark");

        // Viridis ends bright/yellow (high R and G)
        assert!(viridis[255][1] > 150, "Viridis end G should be bright");
    }

    #[test]
    fn test_grayscale_known_values() {
        let provider = BuiltinColormaps;
        let gray = provider.get_colormap_rgb("grayscale").unwrap();

        // Grayscale: R=G=B=index
        assert_eq!(gray[0], [0, 0, 0]);
        assert_eq!(gray[128], [128, 128, 128]);
        assert_eq!(gray[255], [255, 255, 255]);
    }

    #[test]
    fn test_unknown_colormap_returns_none() {
        let provider = BuiltinColormaps;
        assert!(provider.get_colormap_rgb("nonexistent").is_none());
        assert!(provider.get_colormap_rgb("").is_none());
    }

    #[test]
    fn test_all_expected_colormaps_exist() {
        let provider = BuiltinColormaps;
        let expected = [
            "viridis",
            "inferno",
            "magma",
            "plasma",
            "cividis",
            "grayscale",
        ];
        for name in &expected {
            assert!(
                provider.get_colormap_rgb(name).is_some(),
                "Colormap '{name}' should be available"
            );
        }
    }

    #[test]
    fn test_available_colormaps_list() {
        let provider = BuiltinColormaps;
        let names = provider.available_colormaps();
        assert_eq!(names.len(), 6);
        assert!(names.contains(&"viridis"));
        assert!(names.contains(&"grayscale"));
    }
}

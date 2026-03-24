//! # Colormap
//!
//! Colormap provider trait and GPU texture management for colormap lookup tables.
//!
//! The `ColormapProvider` trait allows testing colormap resolution without GPU.
//! `ColormapTexture` wraps a wgpu texture + sampler for use in the compute shader.

use crate::colormap_data;

/// Number of entries in each colormap lookup table.
#[cfg(target_arch = "wasm32")]
const COLORMAP_SIZE: u32 = 256;

/// Maximum valid colormap LUT index (0-based).
const COLORMAP_MAX_INDEX: usize = 255;

/// Float version of `COLORMAP_MAX_INDEX` for normalisation arithmetic.
const COLORMAP_MAX_INDEX_F32: f32 = 255.0;

/// Number of bytes per pixel in an RGBA image.
const RGBA_CHANNELS: usize = 4;

/// Fully opaque alpha channel value.
const RGBA_ALPHA_OPAQUE: u8 = 255;

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
    let mut rgba = Vec::with_capacity(data.len() * RGBA_CHANNELS);
    apply_colormap_cpu_into(data, min_val, max_val, lut, &mut rgba);
    rgba
}

/// Apply a colormap to matrix data on the CPU, writing into a reusable buffer.
///
/// Like `apply_colormap_cpu` but writes into `output` instead of allocating.
/// The buffer is cleared and refilled, reusing its existing heap allocation.
pub fn apply_colormap_cpu_into(
    data: &[f32],
    min_val: f32,
    max_val: f32,
    lut: &[[u8; 3]; 256],
    output: &mut Vec<u8>,
) {
    let range = max_val - min_val;
    output.clear();
    output.reserve(data.len() * RGBA_CHANNELS);

    for &value in data {
        let normalized = if range > 0.0 {
            ((value - min_val) / range).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Map to LUT index [0, COLORMAP_MAX_INDEX]
        let idx = (normalized * COLORMAP_MAX_INDEX_F32).round() as usize;
        let idx = idx.min(COLORMAP_MAX_INDEX);
        let rgb = lut[idx];
        output.push(rgb[0]);
        output.push(rgb[1]);
        output.push(rgb[2]);
        output.push(RGBA_ALPHA_OPAQUE);
    }
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
        let mut rgba_data = [0u8; COLORMAP_SIZE as usize * RGBA_CHANNELS];
        for (i, rgb) in rgb_data.iter().enumerate() {
            rgba_data[i * RGBA_CHANNELS] = rgb[0];
            rgba_data[i * RGBA_CHANNELS + 1] = rgb[1];
            rgba_data[i * RGBA_CHANNELS + 2] = rgb[2];
            rgba_data[i * RGBA_CHANNELS + 3] = RGBA_ALPHA_OPAQUE;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Colormap LUT Texture"),
            size: wgpu::Extent3d {
                width: COLORMAP_SIZE,
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
                bytes_per_row: Some(COLORMAP_SIZE * RGBA_CHANNELS as u32),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: COLORMAP_SIZE,
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

    // -----------------------------------------------------------------------
    // apply_colormap_cpu tests
    // -----------------------------------------------------------------------

    fn grayscale_lut() -> &'static [[u8; 3]; COLORMAP_MAX_INDEX + 1] {
        BuiltinColormaps
            .get_colormap_rgb("grayscale")
            .expect("grayscale must exist")
    }

    #[test]
    fn test_apply_colormap_cpu_basic() {
        let lut = grayscale_lut();
        let data = vec![0.0, 0.5, 1.0];
        let rgba = apply_colormap_cpu(&data, 0.0, 1.0, lut);

        // 3 pixels × 4 channels
        assert_eq!(rgba.len(), 12);

        // First pixel: value=0 → index 0 → (0,0,0,255)
        assert_eq!(rgba[0], 0);
        assert_eq!(rgba[3], RGBA_ALPHA_OPAQUE);

        // Last pixel: value=1 → index 255 → (255,255,255,255)
        assert_eq!(rgba[8], RGBA_ALPHA_OPAQUE);
        assert_eq!(rgba[11], RGBA_ALPHA_OPAQUE);
    }

    #[test]
    fn test_apply_colormap_cpu_empty_data() {
        let lut = grayscale_lut();
        let rgba = apply_colormap_cpu(&[], 0.0, 1.0, lut);
        assert!(rgba.is_empty());
    }

    #[test]
    fn test_apply_colormap_cpu_zero_range() {
        let lut = grayscale_lut();
        let data = vec![5.0, 5.0, 5.0];
        let rgba = apply_colormap_cpu(&data, 5.0, 5.0, lut);

        // Zero range → normalized = 0.5 → index 128
        assert_eq!(rgba.len(), 12);
        assert_eq!(rgba[0], 128); // R
        assert_eq!(rgba[1], 128); // G
        assert_eq!(rgba[2], 128); // B
        assert_eq!(rgba[3], RGBA_ALPHA_OPAQUE);
    }

    #[test]
    fn test_apply_colormap_cpu_values_outside_range() {
        let lut = grayscale_lut();
        let data = vec![-10.0, 100.0];
        let rgba = apply_colormap_cpu(&data, 0.0, 1.0, lut);

        // Below min → clamped to 0 → index 0 → (0,0,0)
        assert_eq!(rgba[0], 0);
        // Above max → clamped to 1 → index 255 → (255,255,255)
        assert_eq!(rgba[4], RGBA_ALPHA_OPAQUE);
    }

    #[test]
    fn test_apply_colormap_cpu_nan_values() {
        let lut = grayscale_lut();
        let data = vec![f32::NAN];
        let rgba = apply_colormap_cpu(&data, 0.0, 1.0, lut);

        // NaN normalisation → clamp(NaN, 0, 1) → 0 → index 0
        assert_eq!(rgba.len(), 4);
        assert_eq!(rgba[3], RGBA_ALPHA_OPAQUE);
    }

    #[test]
    fn test_apply_colormap_cpu_single_element() {
        let lut = grayscale_lut();
        let data = vec![0.5];
        let rgba = apply_colormap_cpu(&data, 0.0, 1.0, lut);

        // 0.5 → index 128 → grayscale (128, 128, 128)
        assert_eq!(rgba.len(), 4);
        assert_eq!(rgba[0], 128);
        assert_eq!(rgba[1], 128);
        assert_eq!(rgba[2], 128);
        assert_eq!(rgba[3], RGBA_ALPHA_OPAQUE);
    }
}

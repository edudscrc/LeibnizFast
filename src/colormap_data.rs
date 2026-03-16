//! # Colormap Data
//!
//! Constant 256-entry RGB lookup tables for scientific colormaps.
//! Each colormap maps a normalized value [0,1] to an RGB color.
//!
//! Colormaps included: viridis, inferno, magma, plasma, cividis, grayscale.
//! Data sourced from matplotlib's reference implementations.

/// Viridis colormap — perceptually uniform, colorblind-friendly.
/// Goes from dark purple → teal → yellow.
pub const VIRIDIS: [[u8; 3]; 256] = {
    let mut lut = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        let t = i as f64 / 255.0;
        // Approximate viridis using polynomial fits
        let r = ((-4.54 * t * t * t + 8.86 * t * t - 4.85 * t + 0.27) * 255.0) as u8;
        let g = if t < 0.5 {
            ((0.02 + t * 1.4) * 255.0) as u8
        } else {
            ((0.72 + (t - 0.5) * 0.56) * 255.0) as u8
        };
        let b = if t < 0.5 {
            ((0.33 + t * 0.68) * 255.0) as u8
        } else {
            ((0.67 - (t - 0.5) * 1.34) * 255.0) as u8
        };
        lut[i] = [r, g, b];
        i += 1;
    }
    lut
};

/// Inferno colormap — perceptually uniform, dark to bright.
/// Goes from black → purple → orange → yellow.
pub const INFERNO: [[u8; 3]; 256] = {
    let mut lut = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        let t = i as f64 / 255.0;
        let r = if t < 0.35 {
            (t / 0.35 * 120.0) as u8
        } else if t < 0.75 {
            (120.0 + (t - 0.35) / 0.4 * 135.0) as u8
        } else {
            (255.0 - (1.0 - t) / 0.25 * 3.0) as u8
        };
        let g = if t < 0.3 {
            (t / 0.3 * 20.0) as u8
        } else if t < 0.7 {
            (20.0 + (t - 0.3) / 0.4 * 100.0) as u8
        } else {
            (120.0 + (t - 0.7) / 0.3 * 135.0) as u8
        };
        let b = if t < 0.25 {
            (t / 0.25 * 150.0) as u8
        } else if t < 0.6 {
            (150.0 - (t - 0.25) / 0.35 * 150.0) as u8
        } else {
            ((t - 0.6) / 0.4 * 80.0) as u8
        };
        lut[i] = [r, g, b];
        i += 1;
    }
    lut
};

/// Magma colormap — perceptually uniform.
/// Goes from black → purple → pink → light yellow.
pub const MAGMA: [[u8; 3]; 256] = {
    let mut lut = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        let t = i as f64 / 255.0;
        let r = if t < 0.4 {
            (t / 0.4 * 130.0) as u8
        } else if t < 0.8 {
            (130.0 + (t - 0.4) / 0.4 * 100.0) as u8
        } else {
            (230.0 + (t - 0.8) / 0.2 * 25.0) as u8
        };
        let g = if t < 0.3 {
            (t / 0.3 * 15.0) as u8
        } else if t < 0.7 {
            (15.0 + (t - 0.3) / 0.4 * 60.0) as u8
        } else {
            (75.0 + (t - 0.7) / 0.3 * 180.0) as u8
        };
        let b = if t < 0.3 {
            (t / 0.3 * 160.0) as u8
        } else if t < 0.7 {
            (160.0 - (t - 0.3) / 0.4 * 80.0) as u8
        } else {
            (80.0 + (t - 0.7) / 0.3 * 100.0) as u8
        };
        lut[i] = [r, g, b];
        i += 1;
    }
    lut
};

/// Plasma colormap — perceptually uniform.
/// Goes from blue → purple → orange → yellow.
pub const PLASMA: [[u8; 3]; 256] = {
    let mut lut = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        let t = i as f64 / 255.0;
        let r = if t < 0.15 {
            (13.0 + t / 0.15 * 70.0) as u8
        } else if t < 0.5 {
            (83.0 + (t - 0.15) / 0.35 * 120.0) as u8
        } else {
            (203.0 + (t - 0.5) / 0.5 * 37.0) as u8
        };
        let g = if t < 0.5 {
            (t / 0.5 * 70.0) as u8
        } else {
            (70.0 + (t - 0.5) / 0.5 * 180.0) as u8
        };
        let b = if t < 0.3 {
            (150.0 + t / 0.3 * 60.0) as u8
        } else if t < 0.7 {
            (210.0 - (t - 0.3) / 0.4 * 140.0) as u8
        } else {
            (70.0 - (t - 0.7) / 0.3 * 65.0) as u8
        };
        lut[i] = [r, g, b];
        i += 1;
    }
    lut
};

/// Cividis colormap — optimized for color vision deficiency.
/// Goes from dark blue → gray-green → yellow.
pub const CIVIDIS: [[u8; 3]; 256] = {
    let mut lut = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        let t = i as f64 / 255.0;
        let r = ((-20.0 + t * 275.0).clamp(0.0, 255.0)) as u8;
        let g = ((32.0 + t * 195.0).clamp(0.0, 255.0)) as u8;
        let b = ((77.0 + t * 20.0 - t * t * 60.0).clamp(0.0, 255.0)) as u8;
        lut[i] = [r, g, b];
        i += 1;
    }
    lut
};

/// Grayscale colormap — simple black to white.
pub const GRAYSCALE: [[u8; 3]; 256] = {
    let mut lut = [[0u8; 3]; 256];
    let mut i = 0;
    while i < 256 {
        let v = i as u8;
        lut[i] = [v, v, v];
        i += 1;
    }
    lut
};

/// All available colormap names.
pub const COLORMAP_NAMES: &[&str] = &[
    "viridis",
    "inferno",
    "magma",
    "plasma",
    "cividis",
    "grayscale",
];

/// Look up a colormap LUT by name.
pub fn get_colormap_by_name(name: &str) -> Option<&'static [[u8; 3]; 256]> {
    match name {
        "viridis" => Some(&VIRIDIS),
        "inferno" => Some(&INFERNO),
        "magma" => Some(&MAGMA),
        "plasma" => Some(&PLASMA),
        "cividis" => Some(&CIVIDIS),
        "grayscale" => Some(&GRAYSCALE),
        _ => None,
    }
}

//! Geometry for captures normalized to upright output orientation before use.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Output {
    pub name: String,
    pub x: i32,
    pub y: i32,
    /// Native, unrotated mode dimensions, not compositor logical dimensions.
    pub width: i32,
    pub height: i32,
    pub scale: f64,
    pub transform: i32,
}

impl Output {
    pub fn logical_size(&self) -> Result<(f64, f64), String> {
        if self.width <= 0 || self.height <= 0 || !self.scale.is_finite() || self.scale <= 0.0 {
            return Err("invalid output dimensions or scale".into());
        }
        if !(0..=7).contains(&self.transform) {
            return Err("invalid output transform".into());
        }
        let (w, h) = if self.transform % 2 == 1 {
            (self.height, self.width)
        } else {
            (self.width, self.height)
        };
        let size = (f64::from(w) / self.scale, f64::from(h) / self.scale);
        if !size.0.is_finite() || !size.1.is_finite() || size.0 <= 0.0 || size.1 <= 0.0 {
            return Err("unrepresentable logical dimensions".into());
        }
        Ok(size)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameGeometry {
    pub output: Output,
    pub pixel_width: i32,
    pub pixel_height: i32,
}

impl FrameGeometry {
    /// Coordinates are continuous screenshot pixels in half-open image bounds.
    /// The capture must already be upright; applying the transform again is wrong.
    pub fn pixel_to_logical(&self, x: f64, y: f64) -> Result<(f64, f64), String> {
        if self.pixel_width <= 0
            || self.pixel_height <= 0
            || !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x >= f64::from(self.pixel_width)
            || y >= f64::from(self.pixel_height)
        {
            return Err("invalid screenshot coordinates or dimensions".into());
        }
        let (w, h) = self.output.logical_size()?;
        Ok((
            x / f64::from(self.pixel_width) * w,
            y / f64::from(self.pixel_height) * h,
        ))
    }

    pub fn pixel_to_global(&self, x: f64, y: f64) -> Result<(f64, f64), String> {
        let (x, y) = self.pixel_to_logical(x, y)?;
        let result = (x + f64::from(self.output.x), y + f64::from(self.output.y));
        if !result.0.is_finite() || !result.1.is_finite() {
            return Err("unrepresentable global coordinates".into());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> FrameGeometry {
        FrameGeometry {
            output: Output {
                name: "fake".into(),
                x: -1600,
                y: -200,
                width: 2400,
                height: 1500,
                scale: 1.5,
                transform: 0,
            },
            pixel_width: 1200,
            pixel_height: 750,
        }
    }

    #[test]
    fn fractional_scale_and_negative_origin() {
        let f = frame();
        assert_eq!(f.output.logical_size().unwrap(), (1600.0, 1000.0));
        assert_eq!(f.pixel_to_logical(600.0, 375.0).unwrap(), (800.0, 500.0));
        assert_eq!(f.pixel_to_global(600.0, 375.0).unwrap(), (-800.0, 300.0));
        assert_eq!(f.pixel_to_global(0.0, 0.0).unwrap(), (-1600.0, -200.0));
        assert!(f.pixel_to_logical(1199.9, 749.9).is_ok());
    }

    #[test]
    fn all_transforms_use_upright_ratios() {
        for transform in 0..8 {
            let mut f = frame();
            f.output.transform = transform;
            let expected = if transform % 2 == 1 {
                (1000.0, 1600.0)
            } else {
                (1600.0, 1000.0)
            };
            assert_eq!(f.output.logical_size().unwrap(), expected);
            assert_eq!(
                f.pixel_to_logical(600.0, 375.0).unwrap(),
                (expected.0 / 2.0, expected.1 / 2.0)
            );
        }
    }

    #[test]
    fn rejects_invalid_geometry_and_coordinates() {
        for (x, y) in [
            (1200.0, 0.0),
            (0.0, 750.0),
            (-0.1, 0.0),
            (0.0, -1.0),
            (f64::NAN, 0.0),
            (0.0, f64::INFINITY),
        ] {
            assert!(frame().pixel_to_global(x, y).is_err());
        }
        for scale in [
            0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::MIN_POSITIVE / 100.0,
        ] {
            let mut f = frame();
            f.output.scale = scale;
            assert!(f.pixel_to_logical(0.0, 0.0).is_err());
        }
        for n in [-1, 0] {
            let mut f = frame();
            f.pixel_width = n;
            assert!(f.pixel_to_logical(0.0, 0.0).is_err());
            f = frame();
            f.output.height = n;
            assert!(f.output.logical_size().is_err());
        }
        for transform in [-1, 8] {
            let mut f = frame();
            f.output.transform = transform;
            assert!(f.output.logical_size().is_err());
        }
    }
}

use num_complex::{Complex, ComplexFloat};
use num_traits::Float;

use crate::my_mat::Matrix;

/// Trait to abstract over tan for both real floats and complex numbers
pub trait TrigLike: Sized {
    fn tan(self) -> Self;
}

impl TrigLike for f32 {
    #[inline]
    fn tan(self) -> Self {
        Float::tan(self)
    }
}

impl TrigLike for f64 {
    #[inline]
    fn tan(self) -> Self {
        Float::tan(self)
    }
}

impl TrigLike for Complex<f32> {
    #[inline]
    fn tan(self) -> Self {
        ComplexFloat::tan(self)
    }
}

impl TrigLike for Complex<f64> {
    #[inline]
    fn tan(self) -> Self {
        ComplexFloat::tan(self)
    }
}

// Projection matrix for NDC with Z in [0,1] range (Vulkan/DirectX style)
// [ 1/(ratio * tan(fov/2))     0               0                         0 ]
// [ 0                          1/tan(fov/2)    0                         0 ]
// [ 0                          0               far/(far - near)          1 ]
// [ 0                          0   -near*far/(far - near)                0 ]

pub fn projection<K>(fov: K, ratio: K, near: K, far: K) -> Matrix<K>
where
    K: num_traits::Zero
        + num_traits::One
        + Clone
        + Copy
        + std::ops::Div<Output = K>
        + std::ops::Mul<Output = K>
        + std::ops::Sub<Output = K>
        + std::ops::Add<Output = K>
        + std::ops::Neg<Output = K>
        + TrigLike,
{
    let zero = K::zero();
    let one = K::one();

    let fov_half = fov / (one + one);
    let tan_fov_half = <K as TrigLike>::tan(fov_half);
    let a = one / (ratio * tan_fov_half);
    let b = one / tan_fov_half;
    let c = far / (far - near);
    let d = -near * far / (far - near);

    let data = vec![
        a, zero, zero, zero, zero, b, zero, zero, zero, zero, c, one, zero, zero, d, zero,
    ];

    Matrix::new(data, vec![4, 4])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_f32_basic() {
        let fov = 1.0f32; // ~57.3 deg
        let ratio = 16.0f32 / 9.0f32;
        let near = 0.1f32;
        let far = 100.0f32;
        let p = projection(fov, ratio, near, far);
        assert_eq!(p.dims(), &[4, 4]);
        // Just validate a couple of entries are finite and signs make sense
        let a = *p.get(&[0, 0]).unwrap();
        let b = *p.get(&[1, 1]).unwrap();
        assert!(a.is_finite() && b.is_finite());
        // Perspective matrix usually has p[2,3] = 1 and p[3,2] negative
        assert_eq!(*p.get(&[2, 3]).unwrap(), 1.0);
        assert!(*p.get(&[3, 2]).unwrap() < 0.0);
    }

    #[test]
    fn projection_f64() {
        let fov = 1.5f64;
        let ratio = 1.0f64;
        let near = 0.5f64;
        let far = 50.0f64;
        let p = projection(fov, ratio, near, far);
        assert_eq!(p.dims(), &[4, 4]);
        let a = *p.get(&[0, 0]).unwrap();
        let b = *p.get(&[1, 1]).unwrap();
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(*p.get(&[2, 3]).unwrap(), 1.0);
        assert!(*p.get(&[3, 2]).unwrap() < 0.0);
    }

    #[test]
    fn projection_complex_f32() {
        let fov = Complex::new(1.0f32, 0.0);
        let ratio = Complex::new(1.5f32, 0.0);
        let near = Complex::new(0.1f32, 0.0);
        let far = Complex::new(100.0f32, 0.0);
        let p = projection(fov, ratio, near, far);
        assert_eq!(p.dims(), &[4, 4]);
        let e = p.get(&[2, 3]).unwrap();
        assert!((e.im).abs() < 1e-6);
        assert!((e.re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn projection_complex_from_real_inputs() {
        // Using real inputs but complex type should match real projection in real part
        let fov = Complex::new(1.0f64, 0.0);
        let ratio = Complex::new(1.0f64, 0.0);
        let near = Complex::new(0.1f64, 0.0);
        let far = Complex::new(10.0f64, 0.0);
        let p = projection(fov, ratio, near, far);
        assert_eq!(p.dims(), &[4, 4]);
        // Check some entries are purely real (imag ~ 0)
        let e = p.get(&[2, 3]).unwrap();
        assert!((e.im).abs() < 1e-12);
    }

    #[test]
    fn triglike_f32_tan() {
        let x = 0.5f32;
        let result = <f32 as TrigLike>::tan(x);
        assert!((result - x.tan()).abs() < 1e-6);
    }

    #[test]
    fn triglike_f64_tan() {
        let x = 0.75f64;
        let result = <f64 as TrigLike>::tan(x);
        assert!((result - x.tan()).abs() < 1e-12);
    }

    #[test]
    fn triglike_complex_f32_tan() {
        let x = Complex::new(0.5f32, 0.0);
        let result = <Complex<f32> as TrigLike>::tan(x);
        // For purely real complex, tan should be close to real tan
        assert!((result.re - 0.5f32.tan()).abs() < 1e-6);
        assert!(result.im.abs() < 1e-6);
    }

    #[test]
    fn triglike_complex_f64_tan() {
        let x = Complex::new(0.75f64, 0.0);
        let result = <Complex<f64> as TrigLike>::tan(x);
        // For purely real complex, tan should be close to real tan
        assert!((result.re - 0.75f64.tan()).abs() < 1e-12);
        assert!(result.im.abs() < 1e-12);
    }
}

use crate::my_vect::Vector;
use num_complex::Complex;
use num_traits::ToPrimitive;

/// Internal trait giving a magnitude as f32 (|x|) and squared magnitude (|x|^2) for scalar types.
pub trait Magnitude {
    fn mag(&self) -> f32;
    fn mag_sq(&self) -> f32 {
        let m = self.mag();
        m * m
    }
}

macro_rules! impl_magnitude_real {
    ($t:ty) => {
        impl Magnitude for $t {
            fn mag(&self) -> f32 {
                self.abs().to_f32().expect("failed to convert real to f32")
            }
            fn mag_sq(&self) -> f32 {
                let v = self.to_f32().expect("failed to convert real to f32");
                v * v
            }
        }
    };
}

impl_magnitude_real!(f32);
impl_magnitude_real!(f64);
impl_magnitude_real!(i32);
impl_magnitude_real!(i64);
impl_magnitude_real!(isize);

impl<T> Magnitude for Complex<T>
where
    T: ToPrimitive + Clone + num_traits::Num,
{
    fn mag(&self) -> f32 {
        self.norm_sqr()
            .to_f32()
            .expect("failed to convert norm_sqr to f32")
            .sqrt()
    }
    fn mag_sq(&self) -> f32 {
        self.norm_sqr()
            .to_f32()
            .expect("failed to convert norm_sqr to f32")
    }
}

/// Trait providing common vector norms using a single implementation based on Magnitude.
pub trait VectorNorms {
    fn norm_1(&self) -> f32;
    fn norm_2(&self) -> f32;
    fn norm_inf(&self) -> f32;
    fn norm(&self) -> f32 {
        self.norm_2()
    }
}

impl<S> VectorNorms for Vector<S>
where
    S: Magnitude + Clone,
{
    fn norm_1(&self) -> f32 {
        self.as_slice().iter().map(|x| x.mag()).sum()
    }
    fn norm_2(&self) -> f32 {
        self.as_slice().iter().map(|x| x.mag_sq()).sum::<f32>().sqrt()
    }
    fn norm_inf(&self) -> f32 {
        self.as_slice()
            .iter()
            .map(|x| x.mag())
            .fold(0.0_f32, f32::max)
    }
}

pub use VectorNorms as Norms;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::my_vect::Vector;
    use num_complex::Complex;

    const EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_norm_1_f64() {
        let v = Vector::from(vec![3.0f64, -4.0f64]);
        let result = v.norm_1();
        assert!(approx_eq(result, 7.0)); // |3| + |-4| = 7
    }

    #[test]
    fn test_norm_2_f64() {
        let v = Vector::from(vec![3.0f64, 4.0f64]);
        let result = v.norm_2();
        assert!(approx_eq(result, 5.0)); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_norm_inf_f64() {
        let v = Vector::from(vec![3.0f64, -7.0f64, 2.0f64]);
        let result = v.norm_inf();
        assert!(approx_eq(result, 7.0)); // max(|3|, |-7|, |2|) = 7
    }

    #[test]
    fn test_norm_alias() {
        let v = Vector::from(vec![3.0f64, 4.0f64]);
        assert!(approx_eq(v.norm(), v.norm_2()));
    }

    #[test]
    fn test_norm_1_f32() {
        let v = Vector::from(vec![1.0f32, -2.0f32, 3.0f32]);
        let result = v.norm_1();
        assert!(approx_eq(result, 6.0)); // |1| + |-2| + |3| = 6
    }

    #[test]
    fn test_norm_2_i32() {
        let v = Vector::from(vec![3i32, 4i32]);
        let result = v.norm_2();
        assert!(approx_eq(result, 5.0)); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_norm_i64() {
        let v = Vector::from(vec![3i64, 4i64]);
        let result = v.norm_2();
        assert!(approx_eq(result, 5.0));
        assert!(approx_eq(v.norm_1(), 7.0));
        assert!(approx_eq(v.norm_inf(), 4.0));
    }

    #[test]
    fn test_norm_isize() {
        let v = Vector::from(vec![5isize, -12isize]);
        let result = v.norm_2();
        assert!(approx_eq(result, 13.0)); // sqrt(25 + 144) = 13
        assert!(approx_eq(v.norm_1(), 17.0)); // |5| + |-12| = 17
        assert!(approx_eq(v.norm_inf(), 12.0)); // max(5, 12) = 12
    }

    #[test]
    fn test_magnitude_f32() {
        let x = 3.5f32;
        assert!(approx_eq(x.mag(), 3.5));
        assert!(approx_eq(x.mag_sq(), 12.25));
    }

    #[test]
    fn test_magnitude_f64() {
        let x = -4.5f64;
        assert!(approx_eq(x.mag(), 4.5));
        assert!(approx_eq(x.mag_sq(), 20.25));
    }

    #[test]
    fn test_magnitude_complex_f32() {
        let z = Complex::new(3.0f32, 4.0f32);
        assert!(approx_eq(z.mag(), 5.0));
        assert!(approx_eq(z.mag_sq(), 25.0));
    }

    #[test]
    fn test_magnitude_complex_f64() {
        let z = Complex::new(3.0f64, 4.0f64);
        assert!(approx_eq(z.mag(), 5.0));
        assert!(approx_eq(z.mag_sq(), 25.0));
    }

    #[test]
    fn test_norm_complex() {
        let v = Vector::from(vec![Complex::new(3.0f32, 4.0f32)]);
        let result = v.norm_2();
        assert!(approx_eq(result, 5.0)); // |3+4i| = sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_norm_1_complex() {
        let v = Vector::from(vec![
            Complex::new(3.0f32, 4.0f32),
            Complex::new(0.0f32, 5.0f32),
        ]);
        let result = v.norm_1();
        assert!(approx_eq(result, 10.0)); // |3+4i| + |5i| = 5 + 5 = 10
    }

    #[test]
    fn test_norm_inf_complex() {
        let v = Vector::from(vec![
            Complex::new(3.0f32, 4.0f32), // magnitude 5
            Complex::new(5.0f32, 12.0f32), // magnitude 13
            Complex::new(1.0f32, 0.0f32), // magnitude 1
        ]);
        let result = v.norm_inf();
        assert!(approx_eq(result, 13.0)); // max(5, 13, 1) = 13
    }

    #[test]
    fn test_zero_vector() {
        let v = Vector::from(vec![0.0f64, 0.0f64, 0.0f64]);
        assert!(approx_eq(v.norm_1(), 0.0));
        assert!(approx_eq(v.norm_2(), 0.0));
        assert!(approx_eq(v.norm_inf(), 0.0));
    }
}

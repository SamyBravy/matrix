use crate::my_vect::Vector;
use num_complex::Complex;
use num_traits::ToPrimitive;

/// Trait providing common vector norms.
///
/// Implemented for `Vector<V>` where `V` is a real arithmetic type and for
/// `Vector<Complex<T>>` for complex-valued vectors. Use `use crate::norm::VectorNorms;`
/// (or fully qualify) so the methods are in scope at call sites.
pub trait VectorNorms {
    /// L1 norm: sum of absolute values, returned as f32.
    fn norm_1(&self) -> f32;

    /// L2 norm (Euclidean). Alias `norm` provided.
    fn norm_2(&self) -> f32;

    /// Infinity norm: max absolute value.
    fn norm_inf(&self) -> f32;

    /// Convenience alias for L2 norm to match previous API.
    fn norm(&self) -> f32 {
        self.norm_2()
    }
}

// Implementations for common real element types to avoid overlap with Complex<T>.
macro_rules! impl_real_norms {
    ($t:ty) => {
        impl VectorNorms for Vector<$t> {
            fn norm_1(&self) -> f32 {
                self.iter()
                    .map(|x| x.abs().to_f32().expect("failed to convert element to f32"))
                    .sum()
            }

            fn norm_2(&self) -> f32 {
                let sum_sq: f32 = self
                    .iter()
                    .map(|x| {
                        let v = x.to_f32().expect("failed to convert element to f32");
                        v * v
                    })
                    .sum();
                sum_sq.sqrt()
            }

            fn norm_inf(&self) -> f32 {
                self.iter()
                    .map(|x| x.abs().to_f32().expect("failed to convert element to f32"))
                    .fold(0.0_f32, f32::max)
            }
        }
    };
}

impl_real_norms!(f32);
impl_real_norms!(f64);
impl_real_norms!(i32);
impl_real_norms!(i64);
impl_real_norms!(isize);

// Specialized implementation for complex-valued vectors.
impl<T> VectorNorms for Vector<Complex<T>>
where
    T: ToPrimitive + Clone + num_traits::Num,
{
    fn norm_1(&self) -> f32 {
        self.iter()
            .map(|z| {
                z.norm_sqr()
                    .to_f32()
                    .expect("failed to convert norm_sqr to f32")
                    .sqrt()
            })
            .sum()
    }

    fn norm_2(&self) -> f32 {
        let ret: f32 = self
            .iter()
            .map(|z| {
                z.norm_sqr()
                    .to_f32()
                    .expect("failed to convert norm_sqr to f32")
            })
            .sum();
        ret.sqrt()
    }

    fn norm_inf(&self) -> f32 {
        self.iter()
            .map(|z| {
                z.norm_sqr()
                    .to_f32()
                    .expect("failed to convert norm_sqr to f32")
                    .sqrt()
            })
            .fold(0.0_f32, f32::max)
    }
}

// Re-export trait for easy use in other modules
pub use VectorNorms as Norms;

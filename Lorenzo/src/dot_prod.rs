use crate::my_vect::Vector;
use num_complex::Complex;
use num_traits::{ToPrimitive, Zero};

macro_rules! impl_dot_for_real {
    ($t:ty) => {
        impl Vector<$t> {
            pub fn dot(&self, other: &Vector<$t>) -> $t
            where
                $t: Clone + std::ops::Add<Output = $t> + std::ops::Mul<Output = $t> + Zero,
            {
                if self.len() != other.len() {
                    panic!("Vectors must have the same length for dot product");
                }
                let mut acc: $t = Zero::zero();
                for i in 0..self.len() {
                    let term = self[i].clone() * other[i].clone();
                    acc = acc + term;
                }
                acc
            }
        }
    };
}

impl_dot_for_real!(f32);
impl_dot_for_real!(f64);
impl_dot_for_real!(i32);
impl_dot_for_real!(i64);
impl_dot_for_real!(isize);

// Hermitian dot product for complex vectors: sum conj(self_i) * other_i
impl<T> Vector<Complex<T>>
where
    T: Clone + Zero + num_traits::Num + std::ops::Neg<Output = T>,
{
    pub fn dot(&self, other: &Vector<Complex<T>>) -> Complex<T> {
        if self.len() != other.len() {
            panic!("Vectors must have the same length for dot product");
        }
        let mut acc: Complex<T> = Complex::new(Zero::zero(), Zero::zero());
        for i in 0..self.len() {
            let term = self[i].clone().conj() * other[i].clone();
            acc = acc + term;
        }
        acc
    }
}

pub trait DotF32 {
    fn dot_f32(&self, other: &Self) -> f32;
}

macro_rules! impl_dot_f32_for_real {
    ($t:ty) => {
        impl DotF32 for Vector<$t> {
            fn dot_f32(&self, other: &Self) -> f32 {
                let acc = self.dot(other);
                acc.to_f32().expect("failed to convert dot product to f32")
            }
        }
    };
}

impl_dot_f32_for_real!(f32);
impl_dot_f32_for_real!(f64);
impl_dot_f32_for_real!(i32);
impl_dot_f32_for_real!(i64);
impl_dot_f32_for_real!(isize);

impl<T> DotF32 for Vector<Complex<T>>
where
    T: ToPrimitive + Clone + num_traits::Num + std::ops::Neg<Output = T>,
{
    fn dot_f32(&self, other: &Self) -> f32 {
        let c = self.dot(other);
        c.re.to_f32()
            .expect("failed to convert complex real part to f32")
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::my_vect::Vector;
    use num_complex::Complex;

    #[test]
    fn test_dot_product_f64() {
        let u = Vector::from(vec![1.0f64, 2.0f64, 3.0f64]);
        let v = Vector::from(vec![4.0f64, 5.0f64, 6.0f64]);
        let result = u.dot(&v);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_dot_product_f32() {
        let u = Vector::from(vec![1.0f32, 2.0f32]);
        let v = Vector::from(vec![3.0f32, 4.0f32]);
        let result = u.dot(&v);
        assert_eq!(result, 11.0); // 1*3 + 2*4 = 3 + 8 = 11
    }

    #[test]
    fn test_dot_product_i32() {
        let u = Vector::from(vec![1i32, 2i32, 3i32]);
        let v = Vector::from(vec![4i32, 5i32, 6i32]);
        let result = u.dot(&v);
        assert_eq!(result, 32); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_product_complex_hermitian() {
        // For complex vectors, dot product is Hermitian: conj(u) * v
        let u = Vector::from(vec![Complex::new(1.0f32, 2.0f32)]);
        let v = Vector::from(vec![Complex::new(3.0f32, 4.0f32)]);
        let result = u.dot(&v);
        // conj(1+2i) * (3+4i) = (1-2i) * (3+4i) = 3 + 4i - 6i - 8i^2 = 3 - 2i + 8 = 11 - 2i
        assert_eq!(result, Complex::new(11.0, -2.0));
    }

    #[test]
    fn test_dot_f32_real() {
        let u = Vector::from(vec![1.0f64, 2.0f64]);
        let v = Vector::from(vec![3.0f64, 4.0f64]);
        let result = u.dot_f32(&v);
        assert_eq!(result, 11.0);
    }

    #[test]
    fn test_dot_f32_complex() {
        let u = Vector::from(vec![Complex::new(1.0f32, 2.0f32)]);
        let v = Vector::from(vec![Complex::new(3.0f32, 4.0f32)]);
        let result = u.dot_f32(&v);
        // Real part of (11 - 2i) is 11
        assert_eq!(result, 11.0);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_dot_product_different_lengths() {
        let u = Vector::from(vec![1.0f64, 2.0f64]);
        let v = Vector::from(vec![3.0f64, 4.0f64, 5.0f64]);
        u.dot(&v);
    }
}

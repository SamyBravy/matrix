use crate::my_mat::{Matrix, Scalar};
use crate::my_vect::Vector;
use std::ops::{Add, Mul, Sub};

pub trait Scale<K> {
    type Output;
    fn scale(&self, scalar: K) -> Self::Output;
}

impl<K> Add for Vector<K>
where
    K: Add<Output = K> + Clone + Default,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let len = if self.len() > rhs.len() {
            self.len()
        } else {
            rhs.len()
        };
        let mut result = Vec::with_capacity(len);
        for n in 0..len {
            let a = if n < self.len() {
                self[n].clone()
            } else {
                K::default()
            };
            let b = if n < rhs.len() {
                rhs[n].clone()
            } else {
                K::default()
            };
            result.push(a + b);
        }
        Vector::from(result)
    }
}

impl<K> Sub for Vector<K>
where
    K: Sub<Output = K> + Clone + Default,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let len = if self.len() > rhs.len() {
            self.len()
        } else {
            rhs.len()
        };
        let mut result = Vec::with_capacity(len);
        for n in 0..len {
            let a = if n < self.len() {
                self[n].clone()
            } else {
                K::default()
            };
            let b = if n < rhs.len() {
                rhs[n].clone()
            } else {
                K::default()
            };
            result.push(a - b);
        }
        Vector::from(result)
    }
}

impl<K> Scale<K> for Vector<K>
where
    K: Scalar + Clone + Mul<Output = K>,
{
    type Output = Self;

    fn scale(&self, scalar: K) -> Self::Output {
        let len = self.len();
        let mut result = Vec::with_capacity(len);
        for n in 0..len {
            result.push(self[n].clone() * scalar.clone());
        }
        Vector::from(result)
    }
}

impl<K> Add for Matrix<K>
where
    K: Add<Output = K> + Clone + Default,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // Fast path: same shape -> preserve shape
        if self.dims() == rhs.dims() {
            let a_flat: Vec<K> = self.linear_iter().cloned().collect();
            let b_flat: Vec<K> = rhs.linear_iter().cloned().collect();
            let mut data = Vec::with_capacity(a_flat.len());
            for i in 0..a_flat.len() {
                data.push(a_flat[i].clone() + b_flat[i].clone());
            }
            return Matrix::new(data, self.dims().to_vec());
        }

        // Lenient fallback: flat elementwise addition, pad with Default
        let a_flat: Vec<K> = self.linear_iter().cloned().collect();
        let b_flat: Vec<K> = rhs.linear_iter().cloned().collect();
        let len = usize::max(a_flat.len(), b_flat.len());
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let a = a_flat.get(i).cloned().unwrap_or_default();
            let b = b_flat.get(i).cloned().unwrap_or_default();
            data.push(a + b);
        }

        Matrix::new(data, vec![len])
    }
}


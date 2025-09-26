use crate::my_mat::{Matrix};
use crate::my_vect::Vector;
use std::ops::{Add, Mul, Sub, AddAssign, SubAssign, MulAssign};

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
    K: Clone + Mul<Output = K>,
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

impl<K> AddAssign for Vector<K>
where
    K: Add<Output = K> + Clone + Default,
{
    fn add_assign(&mut self, rhs: Self) {
        let len = self.len().max(rhs.len());
        let mut result = Vec::with_capacity(len);
        for n in 0..len {
            let a = if n < self.len() { self[n].clone() } else { K::default() };
            let b = if n < rhs.len() { rhs[n].clone() } else { K::default() };
            result.push(a + b);
        }
        self.data = result;
    }
}

impl<K> SubAssign for Vector<K>
where
    K: Sub<Output = K> + Clone + Default,
{
    fn sub_assign(&mut self, rhs: Self) {
        let len = self.len().max(rhs.len());
        let mut result = Vec::with_capacity(len);
        for n in 0..len {
            let a = if n < self.len() { self[n].clone() } else { K::default() };
            let b = if n < rhs.len() { rhs[n].clone() } else { K::default() };
            result.push(a - b);
        }
        self.data = result;
    }
}

impl<K> Mul<K> for Vector<K>
where
    K: Mul<Output = K> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: K) -> Self::Output {
        let mut result = Vec::with_capacity(self.len());
        for item in self.data {
            result.push(item * rhs.clone());
        }
        Vector::from(result)
    }
}

impl <K> MulAssign<K> for Vector<K>
where
		K: Mul<Output = K> + Clone,
{
		fn mul_assign(&mut self, rhs: K) {
				for item in &mut self.data {
						*item = item.clone() * rhs.clone();
				}
		}
		
}

impl<K> Add for Matrix<K>
where
    K: Add<Output = K> + Clone + Default,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.dims() == rhs.dims() {
            let a_flat: Vec<K> = self.linear_iter().cloned().collect();
            let b_flat: Vec<K> = rhs.linear_iter().cloned().collect();
            let mut data = Vec::with_capacity(a_flat.len());
            for i in 0..a_flat.len() {
                data.push(a_flat[i].clone() + b_flat[i].clone());
            }
            return Matrix::new(data, self.dims().to_vec());
        }
        panic!("Matrix addition only supports same shape matrices");
    }
}

impl<K> AddAssign for Matrix<K>
where
    K: Add<Output = K> + Clone + Default,
{
    fn add_assign(&mut self, rhs: Self) {
        if self.dims() != rhs.dims() {
            panic!("Matrix addition only supports same shape matrices");
        }
        let a_flat: Vec<K> = self.linear_iter().cloned().collect();
        let b_flat: Vec<K> = rhs.linear_iter().cloned().collect();
        let mut data = Vec::with_capacity(a_flat.len());
        for i in 0..a_flat.len() {
            data.push(a_flat[i].clone() + b_flat[i].clone());
        }
        self.data = data;
        // shape remains the same
    }
}

impl<K> Sub for Matrix<K>
where
    K: Sub<Output = K> + Clone + Default,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.dims() == rhs.dims() {
            let a_flat: Vec<K> = self.linear_iter().cloned().collect();
            let b_flat: Vec<K> = rhs.linear_iter().cloned().collect();
            let mut data = Vec::with_capacity(a_flat.len());
            for i in 0..a_flat.len() {
                data.push(a_flat[i].clone() - b_flat[i].clone());
            }
            return Matrix::new(data, self.dims().to_vec());
        }
        panic!("Matrix subtraction only supports same shape matrices");
    }
}

impl<K> SubAssign for Matrix<K>
where
    K: Sub<Output = K> + Clone + Default,
{
    fn sub_assign(&mut self, rhs: Self) {
        if self.dims() != rhs.dims() {
            panic!("Matrix subtraction only supports same shape matrices");
        }
        let a_flat: Vec<K> = self.linear_iter().cloned().collect();
        let b_flat: Vec<K> = rhs.linear_iter().cloned().collect();
        let mut data = Vec::with_capacity(a_flat.len());
        for i in 0..a_flat.len() {
            data.push(a_flat[i].clone() - b_flat[i].clone());
        }
        self.data = data;
    }
}

impl<K> Scale<K> for Matrix<K>
where
    K: Clone + Mul<Output = K>,
{
    type Output = Self;

    fn scale(&self, scalar: K) -> Self::Output {
        let a_flat: Vec<K> = self.linear_iter().cloned().collect();
        let mut data = Vec::with_capacity(a_flat.len());
        for i in 0..a_flat.len() {
            data.push(a_flat[i].clone() * scalar.clone());
        }
        Matrix::new(data, self.dims().to_vec())
    }
}

impl<K> Mul<K> for Matrix<K>
where
    K: Mul<Output = K> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: K) -> Self::Output {
        let a_flat: Vec<K> = self.linear_iter().cloned().collect();
        let mut data = Vec::with_capacity(a_flat.len());
        for item in a_flat {
            data.push(item * rhs.clone());
        }
        Matrix::new(data, self.dims().to_vec())
    }
}

impl<K> MulAssign<K> for Matrix<K>
where
    K: Mul<Output = K> + Clone,
{
    fn mul_assign(&mut self, rhs: K) {
        for item in &mut self.data {
            *item = item.clone() * rhs.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add() {
        let u = Vector::from(vec![2.0, 3.0]);
        let v = Vector::from(vec![5.0, 7.0]);
        let result = u.add(v);
        assert_eq!(result, Vector::from(vec![7.0, 10.0]));
    }

    #[test]
    fn vector_add_assign() {
        let mut u = Vector::from(vec![2.0, 3.0]);
        let v = Vector::from(vec![5.0, 7.0]);
        u += v;
        assert_eq!(u, Vector::from(vec![7.0, 10.0]));
    }

    #[test]
    fn vector_sub() {
        let u = Vector::from(vec![2.0, 3.0]);
        let v = Vector::from(vec![5.0, 7.0]);
        let result = u.sub(v);
        assert_eq!(result, Vector::from(vec![-3.0, -4.0]));
    }

    #[test]
    fn vector_sub_assign() {
        let mut u = Vector::from(vec![2.0, 3.0]);
        let v = Vector::from(vec![5.0, 7.0]);
        u -= v;
        assert_eq!(u, Vector::from(vec![-3.0, -4.0]));
    }

    #[test]
    fn vector_scale() {
        let u = Vector::from(vec![2.0, 3.0]);
        let result = u.scale(2.0);
        assert_eq!(result, Vector::from(vec![4.0, 6.0]));
    }

    #[test]
    fn vector_mul() {
        let u = Vector::from(vec![2.0, 3.0]);
        let result = u * 2.0;
        assert_eq!(result, Vector::from(vec![4.0, 6.0]));
    }

    #[test]
    fn vector_mul_assign() {
        let mut u = Vector::from(vec![2.0, 3.0]);
        u *= 2.0;
        assert_eq!(u, Vector::from(vec![4.0, 6.0]));
    }

    #[test]
    fn vector_iter() {
        let u = Vector::from(vec![1.0, 2.0, 3.0]);
        let collected: Vec<f64> = (&u).into_iter().cloned().collect();
        assert_eq!(collected, vec![1.0, 2.0, 3.0]);
        // Test owned iteration
        let owned: Vec<f64> = u.into_iter().collect();
        assert_eq!(owned, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn matrix_add() {
        let u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let v = Matrix::try_from_nested(vec![vec![7.0, 4.0], vec![-2.0, 2.0]]).unwrap();
        let result = u.add(v);
        let expected = Matrix::try_from_nested(vec![vec![8.0, 6.0], vec![1.0, 6.0]]).unwrap();
        assert_eq!(result.linear_iter().cloned().collect::<Vec<_>>(), expected.linear_iter().cloned().collect::<Vec<_>>());
    }

    #[test]
    fn matrix_add_assign() {
        let mut u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let v = Matrix::try_from_nested(vec![vec![7.0, 4.0], vec![-2.0, 2.0]]).unwrap();
        u += v;
        let expected = Matrix::try_from_nested(vec![vec![8.0, 6.0], vec![1.0, 6.0]]).unwrap();
        assert_eq!(u.linear_iter().cloned().collect::<Vec<_>>(), expected.linear_iter().cloned().collect::<Vec<_>>());
    }

    #[test]
    fn matrix_sub() {
        let u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let v = Matrix::try_from_nested(vec![vec![7.0, 4.0], vec![-2.0, 2.0]]).unwrap();
        let result = u.sub(v);
        let expected = Matrix::try_from_nested(vec![vec![-6.0, -2.0], vec![5.0, 2.0]]).unwrap();
        assert_eq!(result.linear_iter().cloned().collect::<Vec<_>>(), expected.linear_iter().cloned().collect::<Vec<_>>());
    }

    #[test]
    fn matrix_sub_assign() {
        let mut u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let v = Matrix::try_from_nested(vec![vec![7.0, 4.0], vec![-2.0, 2.0]]).unwrap();
        u -= v;
        let expected = Matrix::try_from_nested(vec![vec![-6.0, -2.0], vec![5.0, 2.0]]).unwrap();
        assert_eq!(u.linear_iter().cloned().collect::<Vec<_>>(), expected.linear_iter().cloned().collect::<Vec<_>>());
    }

    #[test]
    fn matrix_scale() {
        let u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let result = u.scale(2.0);
        let expected = Matrix::try_from_nested(vec![vec![2.0, 4.0], vec![6.0, 8.0]]).unwrap();
        assert_eq!(result.linear_iter().cloned().collect::<Vec<_>>(), expected.linear_iter().cloned().collect::<Vec<_>>());
    }

    #[test]
    fn matrix_mul() {
        let u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let result = u * 2.0;
        let expected = Matrix::try_from_nested(vec![vec![2.0, 4.0], vec![6.0, 8.0]]).unwrap();
        assert_eq!(result.linear_iter().cloned().collect::<Vec<_>>(), expected.linear_iter().cloned().collect::<Vec<_>>());
    }

    #[test]
    fn matrix_mul_assign() {
        let mut u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        u *= 2.0;
        let expected = Matrix::try_from_nested(vec![vec![2.0, 4.0], vec![6.0, 8.0]]).unwrap();
        assert_eq!(u.linear_iter().cloned().collect::<Vec<_>>(), expected.linear_iter().cloned().collect::<Vec<_>>());
    }
}

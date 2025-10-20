use crate::my_mat::Matrix;
use num_traits::{One, Zero};

impl<K> Matrix<K>
where
    K: Zero
        + One
        + Clone
        + Copy
        + PartialEq
        + std::ops::Mul<Output = K>
        + std::ops::Add<Output = K>
        + std::ops::Sub<Output = K>
        + std::ops::Div<Output = K>
        + std::ops::Neg<Output = K>,
{
    /// Compute the inverse of a square matrix using Gauss-Jordan elimination.
    /// Returns `None` if the matrix is singular (non-invertible).
    ///
    /// # Panics
    /// Panics if the matrix is not 2-dimensional or not square.
    pub fn inverse(&self) -> Option<Matrix<K>> {
        if self.dims().len() != 2 {
            panic!("Matrix must be 2 dimensional for inverse");
        }
        if !self.is_square() {
            panic!("Matrix must be square for inverse");
        }

        let n = self.dims()[0];
        
        // Create augmented matrix [A | I]
        let mut aug_data = Vec::with_capacity(n * n * 2);
        for i in 0..n {
            for j in 0..n {
                aug_data.push(self.get(&[i, j]).unwrap().clone());
            }
            // Add identity matrix columns
            for j in 0..n {
                if i == j {
                    aug_data.push(K::one());
                } else {
                    aug_data.push(K::zero());
                }
            }
        }
        let mut aug = Matrix::new(aug_data, vec![n, 2 * n]);

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot (non-zero element in column i at or below row i)
            let mut pivot_row = i;
            let mut found_pivot = false;
            for k in i..n {
                if aug.get(&[k, i]).unwrap().clone() != K::zero() {
                    pivot_row = k;
                    found_pivot = true;
                    break;
                }
            }
            
            // If no pivot found, matrix is singular
            if !found_pivot {
                return None;
            }

            // Swap rows if needed
            if pivot_row != i {
                for j in 0..(2 * n) {
                    let idx_i = vec![i, j];
                    let idx_pivot = vec![pivot_row, j];
                    let temp = aug.get(&idx_i[..]).unwrap().clone();
                    *aug.get_mut(&idx_i[..]).unwrap() = aug.get(&idx_pivot[..]).unwrap().clone();
                    *aug.get_mut(&idx_pivot[..]).unwrap() = temp;
                }
            }

            // Scale pivot row to make pivot element = 1
            let pivot = aug.get(&[i, i]).unwrap().clone();
            for j in 0..(2 * n) {
                let idx = vec![i, j];
                let val = aug.get(&idx[..]).unwrap().clone();
                *aug.get_mut(&idx[..]).unwrap() = val / pivot;
            }

            // Eliminate column i in all other rows
            for k in 0..n {
                if k != i {
                    let factor = aug.get(&[k, i]).unwrap().clone();
                    for j in 0..(2 * n) {
                        let idx_k = vec![k, j];
                        let idx_i = vec![i, j];
                        let val_k = aug.get(&idx_k[..]).unwrap().clone();
                        let val_i = aug.get(&idx_i[..]).unwrap().clone();
                        *aug.get_mut(&idx_k[..]).unwrap() = val_k - factor * val_i;
                    }
                }
            }
        }

        // Extract the right half (the inverse matrix)
        let mut inv_data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in n..(2 * n) {
                inv_data.push(aug.get(&[i, j]).unwrap().clone());
            }
        }

        Some(Matrix::new(inv_data, vec![n, n]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn inverse_2x2_float() {
        let m = Matrix::try_from_nested(vec![vec![4.0, 7.0], vec![2.0, 6.0]]).unwrap();
        let inv = m.inverse().unwrap();
        
        // Expected inverse: (1/10) * [[6, -7], [-2, 4]]
        let expected = Matrix::try_from_nested(vec![
            vec![0.6, -0.7],
            vec![-0.2, 0.4],
        ])
        .unwrap();
        
        // Check each element
        for i in 0..2 {
            for j in 0..2 {
                let actual: f64 = *inv.get(&[i, j]).unwrap();
                let exp: f64 = *expected.get(&[i, j]).unwrap();
                let diff = (actual - exp).abs();
                assert!(diff < 1e-10, "Element [{}, {}] mismatch", i, j);
            }
        }
    }

    #[test]
    fn inverse_3x3_float() {
        let m = Matrix::try_from_nested(vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![5.0, 6.0, 0.0],
        ])
        .unwrap();
        let inv = m.inverse().unwrap();
        
        // Verify by multiplication: M * M^(-1) = I
        let product = &m * &inv;
        
        // Check it's close to identity
        for i in 0..3 {
            for j in 0..3 {
                let expected_val = if i == j { 1.0 } else { 0.0 };
                let actual_val: f64 = *product.get(&[i, j]).unwrap();
                let diff = (actual_val - expected_val).abs();
                assert!(diff < 1e-10, "M * M^(-1) [{}, {}] = {} (expected {})", i, j, actual_val, expected_val);
            }
        }
    }

    #[test]
    fn inverse_identity_is_identity() {
        let id = Matrix::try_from_nested(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
        .unwrap();
        let inv = id.inverse().unwrap();
        
        // Identity inverse should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected_val = if i == j { 1.0 } else { 0.0 };
                let actual_val: f64 = *inv.get(&[i, j]).unwrap();
                let diff = (actual_val - expected_val).abs();
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    fn inverse_singular_returns_none() {
        // Singular matrix (det = 0)
        let m = Matrix::try_from_nested(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0], // Second row is 2x first row
        ])
        .unwrap();
        assert!(m.inverse().is_none());
    }

    #[test]
    fn inverse_2x2_complex() {
        let m = Matrix::try_from_nested(vec![
            vec![Complex::new(1.0, 1.0), Complex::new(0.0, 1.0)],
            vec![Complex::new(1.0, 0.0), Complex::new(1.0, 1.0)],
        ])
        .unwrap();
        let inv = m.inverse().unwrap();
        
        // Verify by multiplication
        let product = &m * &inv;
        
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { Complex::new(1.0, 0.0) } else { Complex::new(0.0, 0.0) };
                let actual = *product.get(&[i, j]).unwrap();
                let diff = (actual - expected).norm();
                assert!(diff < 1e-10, "Complex M * M^(-1) [{}, {}] error = {}", i, j, diff);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Matrix must be square")]
    fn inverse_non_square_panics() {
        let m = Matrix::try_from_nested(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ])
        .unwrap();
        let _ = m.inverse();
    }
}
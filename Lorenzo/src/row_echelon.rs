use crate::my_mat::Matrix;

impl<K> Matrix<K>
where
    K: num_traits::Zero
        + Clone
        + Copy
        + std::ops::Mul<Output = K>
        + std::ops::Add<Output = K>
        + PartialEq,
{
    fn swap_rows(&mut self, row1: usize, row2: usize) {
        if self.dims().len() != 2 {
            panic!("Matrix must be 2-dimensional");
        }
        let cols = self.dims()[1];
        for col in 0..cols {
            let idx1 = vec![row1, col];
            let idx2 = vec![row2, col];
            let temp = self[&idx1[..]].clone();
            *self.get_mut(&idx1[..]).unwrap() = self[&idx2[..]].clone();
            *self.get_mut(&idx2[..]).unwrap() = temp;
        }
    }
    // #[warn(dead_code)]
    // fn scale_row(&mut self, row: usize, factor: K) {
    //     if self.dims().len() != 2 {
    //         panic!("Matrix must be 2-dimensional");
    //     }
    //     let cols = self.dims()[1];
    //     for col in 0..cols {
    //         let idx = vec![row, col];
    //         let val = self[&idx[..]].clone();
    //         *self.get_mut(&idx[..]).unwrap() = val * factor;
    //     }
    // }

    fn add_rows(&mut self, row1: usize, row2: usize, factor: K) {
        if self.dims().len() != 2 {
            panic!("Matrix must be 2-dimensional");
        }
        let cols = self.dims()[1];
        for col in 0..cols {
            let idx1 = vec![row1, col];
            let idx2 = vec![row2, col];
            let val1 = self[&idx1[..]].clone();
            let val2 = self[&idx2[..]].clone();
            *self.get_mut(&idx1[..]).unwrap() = val1 + val2 * factor;
        }
    }

    pub fn row_echelon(&self) -> (Matrix<K>, u32)
    where
        K: std::ops::Div<Output = K> + std::ops::Neg<Output = K>,
    {
        let mut matrix = self.clone();
        if matrix.dims().len() != 2 {
            panic!("Matrix must be 2-dimensional for row echelon form");
        }
        let rows = matrix.dims()[0];
        let mut swaps: u32 = 0;
        for row in 0..rows {
            let pivot_idx = vec![row, row];
            let pivot = matrix[&pivot_idx[..]].clone();

            if pivot == K::zero() {
                for i in row + 1..rows {
                    let idx = vec![i, row];
                    if matrix[&idx[..]] != K::zero() {
                        matrix.swap_rows(row, i);
                        swaps += 1;
                        break;
                    }
                }
            }

            let pivot_idx = vec![row, row];
            if matrix[&pivot_idx[..]] == K::zero() {
                continue;
            }

            let pivot = matrix[&pivot_idx[..]].clone();
            for i in row + 1..rows {
                let idx = vec![i, row];
                let factor = matrix[&idx[..]].clone() / pivot.clone();
                matrix.add_rows(i, row, -factor);
            }
        }

        (matrix, swaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_echelon_identity_matrix() {
        let m = Matrix::try_from_nested(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
        .unwrap();
        let result = m.row_echelon().0;

        // Identity matrix should remain unchanged
        assert_eq!(result.get(&[0, 0]), Some(&1.0));
        assert_eq!(result.get(&[1, 1]), Some(&1.0));
        assert_eq!(result.get(&[2, 2]), Some(&1.0));
    }

    #[test]
    fn row_echelon_simple_3x3() {
        let m = Matrix::try_from_nested(vec![
            vec![2.0, 1.0, -1.0],
            vec![-3.0, -1.0, 2.0],
            vec![-2.0, 1.0, 2.0],
        ])
        .unwrap();
        let result = m.row_echelon().0;

        // First pivot should be non-zero
        assert_ne!(result.get(&[0, 0]), Some(&0.0));

        // Elements below first pivot should be zero
        let val: f64 = *result.get(&[1, 0]).unwrap();
        assert!(val.abs() < 1e-10);
        let val: f64 = *result.get(&[2, 0]).unwrap();
        assert!(val.abs() < 1e-10);
    }

    #[test]
    fn row_echelon_2x2() {
        let m = Matrix::try_from_nested(vec![vec![3.0, 2.0], vec![1.0, 4.0]]).unwrap();
        let result = m.row_echelon().0;

        // Check upper triangular form
        assert_ne!(result.get(&[0, 0]), Some(&0.0));
        let below: f64 = *result.get(&[1, 0]).unwrap();
        assert!(below.abs() < 1e-10);
    }

    #[test]
    fn row_echelon_with_zero_pivot() {
        let m = Matrix::try_from_nested(vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            vec![2.0, 1.0, 4.0],
        ])
        .unwrap();
        let result = m.row_echelon().0;

        // Should swap rows to avoid zero pivot
        assert_ne!(result.get(&[0, 0]), Some(&0.0));
    }

    #[test]
    fn row_echelon_rectangular_3x4() {
        let m = Matrix::try_from_nested(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 7.0, 8.0],
            vec![3.0, 6.0, 10.0, 13.0],
        ])
        .unwrap();
        let result = m.row_echelon().0;

        // Check dimensions preserved
        assert_eq!(result.dims(), &[3, 4]);

        // First element should be non-zero
        assert_ne!(result.get(&[0, 0]), Some(&0.0));
    }

    #[test]
    #[should_panic(expected = "Matrix must be 2-dimensional")]
    fn row_echelon_panics_on_1d() {
        let m = Matrix::try_from_nested(vec![1.0, 2.0, 3.0]).unwrap();
        let _ = m.row_echelon().0;
    }

    #[test]
    fn row_echelon_integers() {
        let m =
            Matrix::try_from_nested(vec![vec![2, 1, -1], vec![-3, -1, 2], vec![-2, 1, 2]]).unwrap();
        let result = m.row_echelon().0;

        // Should complete without panic
        assert_eq!(result.dims(), &[3, 3]);
    }
}

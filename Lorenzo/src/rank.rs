use crate::my_mat::Matrix;

impl<K> Matrix<K>
where
    K: num_traits::Zero
        + Clone
        + Copy
        + std::ops::Mul<Output = K>
        + std::ops::Add<Output = K>
        + std::ops::Div<Output = K>
        + std::ops::Neg<Output = K>
        + PartialEq,
{
    pub fn rank(&self) -> usize {
        if self.dims().len() != 2 {
            panic!("Matrix must be 2-dimensional");
        }
        if self.dims()[0] != self.dims()[1] {
            panic!("Matrix must be square");
        }
        let (echelon_form, _swaps) = self.row_echelon();
        let rows = echelon_form.dims()[0];
        let cols = echelon_form.dims()[1];
        let mut rank = 0;

        for row in 0..rows {
            let mut is_zero_row = true;
            for col in 0..cols {
                let idx = vec![row, col];
                if echelon_form[&idx[..]] != K::zero() {
                    is_zero_row = false;
                    break;
                }
            }
            if !is_zero_row {
                rank += 1;
            }
        }
        rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_identity_matrix() {
        // Identity matrix has full rank
        let m = Matrix::try_from_nested(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
        .unwrap();
        assert_eq!(m.rank(), 3);
    }

    #[test]
    fn rank_full_rank_matrix() {
        // Full rank 3x3 matrix
        let m = Matrix::try_from_nested(vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![5.0, 6.0, 0.0],
        ])
        .unwrap();
        assert_eq!(m.rank(), 3);
    }

    #[test]
    fn rank_singular_matrix() {
        // Singular matrix (row 2 = 2 * row 1)
        let m = Matrix::try_from_nested(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![4.0, 5.0, 6.0],
        ])
        .unwrap();
        assert_eq!(m.rank(), 2);
    }

    #[test]
    fn rank_zero_matrix() {
        // All zeros - rank should be 0
        let m = Matrix::try_from_nested(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ])
        .unwrap();
        assert_eq!(m.rank(), 0);
    }

    #[test]
    fn rank_2x2_full_rank() {
        let m = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(m.rank(), 2);
    }

    #[test]
    fn rank_2x2_rank_1() {
        // Second row is multiple of first
        let m = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![2.0, 4.0]]).unwrap();
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn rank_single_element() {
        let m = Matrix::try_from_nested(vec![vec![5.0]]).unwrap();
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn rank_single_element_zero() {
        let m = Matrix::try_from_nested(vec![vec![0.0]]).unwrap();
        assert_eq!(m.rank(), 0);
    }

    #[test]
    #[should_panic(expected = "Matrix must be 2-dimensional")]
    fn rank_panics_on_1d() {
        let v = Matrix::try_from_nested(vec![1.0, 2.0, 3.0]).unwrap();
        let _ = v.rank();
    }

    #[test]
    #[should_panic(expected = "Matrix must be square")]
    fn rank_panics_on_non_square() {
        let m = Matrix::try_from_nested(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();
        let _ = m.rank();
    }

    #[test]
    fn rank_with_integers() {
        let m = Matrix::try_from_nested(vec![vec![1, 2], vec![3, 4]]).unwrap();
        assert_eq!(m.rank(), 2);
    }
}

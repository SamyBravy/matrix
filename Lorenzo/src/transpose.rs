use crate::my_mat::Matrix;

impl<K> Matrix<K>
where
    K: Clone,
{
    /// Transposes a 2D matrix. For N-D tensors (N > 2), reverses all dimensions.
    /// For 2D matrices, performs proper matrix transposition (rows ↔ columns).
    pub fn transpose(&self) -> Matrix<K> {
        // For 2D matrices, perform proper transposition
        if self.dims().len() == 2 {
            let rows = self.dims()[0];
            let cols = self.dims()[1];
            let mut new_data = Vec::with_capacity(self.len());

            // Transpose: read column-by-column from original, which becomes row-by-row in result
            for j in 0..cols {
                for i in 0..rows {
                    new_data.push(self.get(&[i, j]).unwrap().clone());
                }
            }

            Matrix::new(new_data, vec![cols, rows])
        } else {
            // For other dimensions, just reverse the shape (tensor dimension swap)
            let reversed_shape: Vec<usize> = self.dims().iter().rev().cloned().collect();
            Matrix::new(self.data_clone(), reversed_shape)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn transpose_2x3_matrix() {
        // Create a 2x3 matrix:
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let m = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let t = m.transpose();

        // After transpose should be 3x2:
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        assert_eq!(t.dims(), &[3, 2]);
        assert_eq!(t.get(&[0, 0]), Some(&1));
        assert_eq!(t.get(&[0, 1]), Some(&4));
        assert_eq!(t.get(&[1, 0]), Some(&2));
        assert_eq!(t.get(&[1, 1]), Some(&5));
        assert_eq!(t.get(&[2, 0]), Some(&3));
        assert_eq!(t.get(&[2, 1]), Some(&6));
    }

    #[test]
    fn transpose_square_matrix() {
        // Create a 3x3 matrix
        let m = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]).unwrap();
        let t = m.transpose();

        assert_eq!(t.dims(), &[3, 3]);
        assert_eq!(t.get(&[0, 0]), Some(&1));
        assert_eq!(t.get(&[0, 1]), Some(&4));
        assert_eq!(t.get(&[0, 2]), Some(&7));
        assert_eq!(t.get(&[1, 0]), Some(&2));
        assert_eq!(t.get(&[1, 1]), Some(&5));
        assert_eq!(t.get(&[1, 2]), Some(&8));
        assert_eq!(t.get(&[2, 0]), Some(&3));
        assert_eq!(t.get(&[2, 1]), Some(&6));
        assert_eq!(t.get(&[2, 2]), Some(&9));
    }

    #[test]
    fn transpose_1d_vector() {
        // Transpose of a 1D vector [4] becomes [4] (reversed is still [4])
        let v = Matrix::try_from_nested(vec![1, 2, 3, 4]).unwrap();
        let t = v.transpose();
        assert_eq!(t.dims(), &[4]);
        assert_eq!(t.get(&[0]), Some(&1));
        assert_eq!(t.get(&[3]), Some(&4));
    }

    #[test]
    fn transpose_3d_matrix() {
        // Create a 2x3x4 matrix
        let m = Matrix::new((0..24).collect(), vec![2, 3, 4]);
        let t = m.transpose();

        // Shape should be reversed: [4, 3, 2]
        assert_eq!(t.dims(), &[4, 3, 2]);

        // For N-D tensors, transpose just reverses dimensions (data unchanged)
        // So accessing the same linear index gives the same value
        assert_eq!(t.get(&[0, 0, 0]), Some(&0));
        // Linear index 1 in original is at [0,0,1], but in reversed shape:
        // we need to compute what [0,0,1] maps to in shape [4,3,2]
        // Actually the data is unchanged, so linear index 1 = value 1
        // In shape [4,3,2], index [0,0,1] computes to: 0*3*2 + 0*2 + 1 = 1
        assert_eq!(t.get(&[0, 0, 1]), Some(&1));
    }

    #[test]
    fn transpose_complex_matrix() {
        let m = Matrix::try_from_nested(vec![
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)],
        ])
        .unwrap();
        let t = m.transpose();

        assert_eq!(t.dims(), &[2, 2]);
        assert_eq!(t.get(&[0, 0]), Some(&Complex::new(1.0, 2.0)));
        assert_eq!(t.get(&[0, 1]), Some(&Complex::new(5.0, 6.0)));
        assert_eq!(t.get(&[1, 0]), Some(&Complex::new(3.0, 4.0)));
        assert_eq!(t.get(&[1, 1]), Some(&Complex::new(7.0, 8.0)));
    }

    #[test]
    fn transpose_twice_is_original_shape() {
        let m = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let t = m.transpose();
        let tt = t.transpose();

        // Double transpose should restore original shape
        assert_eq!(tt.dims(), m.dims());
        assert_eq!(tt.dims(), &[2, 3]);
    }
}

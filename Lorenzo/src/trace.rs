use crate::my_mat::Matrix;

impl<K> Matrix<K> {
    pub fn trace(&self) -> K
    where
        K: std::ops::Add<Output = K> + Clone + Default,
    {
        let dims = self.dims();

        if !dims.iter().all(|d| *d == dims[0]) {
            panic!("Matrix is not square");
        }

        let mut sum = K::default();
        for i in 0..dims[0] {
            let idx: Vec<usize> = vec![i; dims.len()];
            sum = sum + self[&idx[..]].clone(); //self [[2,2,2]]
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_of_square_matrix_sums_diagonal() {
        let matrix =
            Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]).unwrap();

        assert_eq!(matrix.trace(), 1 + 5 + 9);
    }

    #[test]
    #[should_panic(expected = "Matrix is not square")]
    fn trace_panics_for_rectangle_matrix() {
        let matrix = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();

        let _ = matrix.trace();
    }

    #[test]
    fn trace_of_one_dimensional_matrix() {
        // 1D matrix [3] passes the square check (all dims equal)
        // and returns sum of diagonal [0], [1], [2]
        let matrix = Matrix::try_from_nested(vec![1]).unwrap();
        assert_eq!(matrix.trace(), 1);
    }

    #[test]
    fn trace_of_3d_cubic_matrix() {
        // 3x3x3 cubic matrix
        let matrix = Matrix::new(
            vec![
                // First 3x3 layer (i=0)
                1, 2, 3, 4, 5, 6, 7, 8, 9, // Second 3x3 layer (i=1)
                10, 11, 12, 13, 14, 15, 16, 17, 18, // Third 3x3 layer (i=2)
                19, 20, 21, 22, 23, 24, 25, 26, 27,
            ],
            vec![3, 3, 3],
        );

        // Trace: [0,0,0]=1 + [1,1,1]=14 + [2,2,2]=27
        assert_eq!(matrix.trace(), 1 + 14 + 27);
    }

    #[test]
    fn trace_of_4d_hypercube_matrix() {
        // 2x2x2x2 hypercube
        let matrix = Matrix::new((0..16).collect::<Vec<i32>>(), vec![2, 2, 2, 2]);

        // Trace: [0,0,0,0]=0 + [1,1,1,1]=15
        assert_eq!(matrix.trace(), 0 + 15);
    }

    #[test]
    #[should_panic(expected = "Matrix is not square")]
    fn trace_panics_for_non_square_3d_matrix() {
        // 3x3x2 is not square
        let matrix = Matrix::new(vec![1; 18], vec![3, 3, 2]);

        let _ = matrix.trace();
    }

    #[test]
    fn trace_of_complex_square_matrix() {
        use num_complex::Complex;

        let matrix = Matrix::try_from_nested(vec![
            vec![Complex::new(1.0, 0.0), Complex::new(2.0, 1.0)],
            vec![Complex::new(3.0, -1.0), Complex::new(4.0, 2.0)],
        ])
        .unwrap();

        assert_eq!(matrix.trace(), Complex::new(5.0, 2.0));
    }
}

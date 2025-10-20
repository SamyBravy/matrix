use crate::my_mat::Matrix;
use num_traits::{One, Zero};

impl<K> Matrix<K>
where
    K: Zero
        + One
        + Clone
        + PartialEq
        + std::ops::Mul<Output = K>
        + std::ops::Add<Output = K>
        + std::ops::Sub<Output = K>
        + std::ops::Neg<Output = K>,
{
    /// Compute determinant using cofactor expansion (Laplace expansion).
    /// Works correctly for all types including integers (no division needed).
    pub fn determinant(&self) -> K {
        if self.dims().len() != 2 {
            panic!("Matrix must be 2 dimensional for determinant");
        }
        if !self.is_square() {
            panic!("Matrix must be square for determinant");
        }

        let n = self.dims()[0];

        // Base cases
        if n == 0 {
            return K::one();
        }
        if n == 1 {
            return self.get(&[0]).unwrap().clone();
        }
        if n == 2 {
            let a = self.get(&[0, 0]).unwrap().clone();
            let b = self.get(&[0, 1]).unwrap().clone();
            let c = self.get(&[1, 0]).unwrap().clone();
            let d = self.get(&[1, 1]).unwrap().clone();
            return a * d - b * c;
        }
        let mut det = K::zero();
        for j in 0..n {
            let element = self.get(&[0, j]).unwrap().clone();
            if element == K::zero() {
                continue;
            }
            let mut submatrix_data = Vec::new();
            for i in 1..n {
                for k in 0..n {
                    if k != j {
                        submatrix_data.push(self.get(&[i, k]).unwrap().clone());
                    }
                }
            }
            let submatrix = Matrix::new(submatrix_data, vec![n - 1, n - 1]);

            let cofactor = submatrix.determinant();
            let sign = if j % 2 == 0 { K::one() } else { -K::one() };
            det = det + sign * element * cofactor;
        }

        det
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn det_2x2_int() {
        let m = Matrix::try_from_nested(vec![vec![1, 2], vec![3, 4]]).unwrap();
        // det = 1*4 - 2*3 = -2
        assert_eq!(m.determinant(), -2);
    }

    #[test]
    fn det_3x3_int() {
        let m =
            Matrix::try_from_nested(vec![vec![6, 1, 1], vec![4, -2, 5], vec![2, 8, 7]]).unwrap();
        // known determinant = -306
        assert_eq!(m.determinant(), -306);
    }

    #[test]
    fn det_2x2_complex() {
        let m = Matrix::try_from_nested(vec![
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)],
        ])
        .unwrap();
        // determinant = (1+2i)*(7+8i) - (3+4i)*(5+6i)
        // = (7-16 + i(8+14)) - (15-24 + i(18+20))
        // = (-9 + 22i) - (-9 + 38i) = -16i
        let det = m.determinant();
        let expected = Complex::new(0.0, -16.0);
        // Allow small floating point error
        let re_diff: f64 = det.re - expected.re;
        let im_diff: f64 = det.im - expected.im;
        assert!(re_diff.abs() < 1e-10 && im_diff.abs() < 1e-10);
    }
}

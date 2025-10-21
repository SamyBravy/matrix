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

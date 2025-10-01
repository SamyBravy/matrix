use crate::my_mat::Matrix;

impl<K> Matrix<K>
where
    K: Clone,
{
    pub fn transpose(&self) -> Matrix<K> {
        let reversed_shape: Vec<usize> = self.dims().iter().rev().cloned().collect();
        Matrix::new(self.data_clone(), reversed_shape)
    }
}
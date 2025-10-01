// use crate::my_mat::Matrix;

// fn is_row_echelon<K>(matrix: &Matrix<K>) -> bool 
// where 
// 		K: num_traits::Zero + PartialEq + Clone,
// {
// 	for i in 0..matrix.dims()[0] {
// 		let idx = vec![i; matrix.dims()[0]];
// 		if matrix[&idx[..]].iter().any(|&x| x != K::zero()) {
// 			return false;
// 		}
// 	}
// 	true
// }
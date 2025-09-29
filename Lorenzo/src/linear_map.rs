use crate::my_mat::Matrix;
use crate::my_vect::Vector;
use num_traits::Zero;
use std::ops::{Add, Mul, MulAssign};

/// Element-wise scalar multiplication (works for any N-D shape).
impl<K> Mul<K> for Matrix<K>
where
    K: Mul<Output = K> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: K) -> Self::Output {
        let data: Vec<K> = self
            .linear_iter()
            .cloned()
            .map(|x| x * rhs.clone())
            .collect();
        Matrix::new(data, self.dims().to_vec())
    }
}

/// Borrowed element-wise scalar multiplication so inputs aren't consumed.
impl<'a, K> Mul<K> for &'a Matrix<K>
where
    K: Mul<Output = K> + Clone,
{
    type Output = Matrix<K>;

    fn mul(self, rhs: K) -> Self::Output {
        let data: Vec<K> = self
            .linear_iter()
            .cloned()
            .map(|x| x * rhs.clone())
            .collect();
        Matrix::new(data, self.dims().to_vec())
    }
}

impl<K> MulAssign<K> for Matrix<K>
where
    K: Mul<Output = K> + Clone,
{
    fn mul_assign(&mut self, rhs: K) {
        for x in &mut self.into_iter() {
            *x = x.clone() * rhs.clone();
        }
    }
}

impl<K> Matrix<K>
where
    K: Clone + Add<Output = K> + Mul<Output = K> + Zero,
{
    /// Contract a rank-1 matrix (shape [N]) with a vector of length N returning a scalar (dot product).
    /// This is not exposed via `Mul` because that implementation always returns a vector; using an
    /// inherent method avoids an API that changes return type based on runtime shape.
    pub fn dot_vector(&self, v: &Vector<K>) -> K {
        assert!(self.dims().len() == 1, "dot_vector requires a rank-1 matrix (shape [N])");
        let n = self.dims()[0];
        assert!(v.len() == n, "Length mismatch: matrix has {} elements, vector has {}", n, v.len());
        let mut acc = K::zero();
        for i in 0..n {
            let val = self.get(&[i]).expect("in bounds").clone();
            let prod = val * v[i].clone();
            acc = acc + prod;
        }
        acc
    }
}

/// Generalized matrix X matrix multiplication for N-D arrays.
///
/// Semantics (batched / tensor-style):
/// Let A have shape [a0, a1, ..., a_{p-2}, K]
/// Let B have shape [K, b1, b2, ..., b_{q-1}]
/// Result has shape [a0, a1, ..., a_{p-2}, b1, b2, ..., b_{q-1}]
///
/// Constraints:
/// - A and B must each have at least 1 dimension (last of A matches first of B).
/// - The contracted dimension K = A.shape[last] = B.shape[0].
/// - No additional broadcasting; all "batch" dims (a0..a_{p-2}) stay intact.
///
/// This reduces to classic 2-D matmul when A and B are both rank-2.
/// Complexity: O(prod(result_shape) * K)
impl<K> Mul<Matrix<K>> for Matrix<K>
where
    K: Clone + Add<Output = K> + Mul<Output = K> + Zero,
{
    type Output = Matrix<K>;

    fn mul(self, rhs: Matrix<K>) -> Self::Output {
        let a_shape = self.dims();
        let b_shape = rhs.dims();
        assert!(
            !a_shape.is_empty() && !b_shape.is_empty(),
            "Both operands must have rank >= 1"
        );

        let k_left = *a_shape.last().unwrap();
        let k_right = b_shape[0];
        assert!(
            k_left == k_right,
            "Contracted dimensions differ: {} vs {}",
            k_left,
            k_right
        );

        // Result shape: A without last + B without first
        let mut out_shape = Vec::with_capacity(a_shape.len() + b_shape.len() - 2);
        out_shape.extend_from_slice(&a_shape[..a_shape.len() - 1]);
        out_shape.extend_from_slice(&b_shape[1..]);

        if out_shape.is_empty() {
            panic!("Rank-1 X Rank-1 would produce a scalar; not supported");
        }

        let out_len: usize = out_shape.iter().product();
        let mut out_data: Vec<K> = vec![K::zero(); out_len];

        let a_rank = a_shape.len();
        let b_rank = b_shape.len();

        // Helper closures
    let _multi_to_flat = |idx: &[usize], shape: &[usize]| -> usize {
            let mut flat = 0;
            let mut stride = 1;
            for (dim, &sz) in shape.iter().rev().enumerate() {
                let coord = idx[shape.len() - 1 - dim];
                flat += coord * stride;
                stride *= sz;
            }
            flat
        };

        let index_to_multi = |mut lin: usize, shape: &[usize]| -> Vec<usize> {
            if shape.is_empty() {
                return vec![];
            }
            let mut coords = vec![0; shape.len()];
            for i in (0..shape.len()).rev() {
                let sz = shape[i];
                coords[i] = lin % sz;
                lin /= sz;
            }
            coords
        };

        for out_lin in 0..out_len {
            let out_multi = index_to_multi(out_lin, &out_shape);

            // Split output indices into A-batch and B-tail
            let a_batch_len = a_rank.saturating_sub(1);
            let b_tail_len = b_rank.saturating_sub(1);
            let a_batch_indices = &out_multi[..a_batch_len];
            let b_tail_indices = &out_multi[a_batch_len..a_batch_len + b_tail_len];

            let mut acc = K::zero();

            for k in 0..k_left {
                // Build A index
                let mut a_idx = Vec::with_capacity(a_rank);
                a_idx.extend_from_slice(a_batch_indices);
                a_idx.push(k);

                // Build B index
                let mut b_idx = Vec::with_capacity(b_rank);
                b_idx.push(k);
                b_idx.extend_from_slice(b_tail_indices);

                let a_val = self.get(&a_idx).expect("A index in bounds").clone();
                let b_val = rhs.get(&b_idx).expect("B index in bounds").clone();
                acc = acc + a_val * b_val;
            }
            out_data[out_lin] = acc;
        }

        Matrix::new(out_data, out_shape)
    }
}

/// Helper performing generalized matmul on borrowed matrices.
fn matmul_ref<K>(a: &Matrix<K>, b: &Matrix<K>) -> Matrix<K>
where
    K: Clone + Add<Output = K> + Mul<Output = K> + Zero,
{
    let a_shape = a.dims();
    let b_shape = b.dims();
    assert!(
        !a_shape.is_empty() && !b_shape.is_empty(),
        "Both operands must have rank >= 1"
    );

    let k_left = *a_shape.last().unwrap();
    let k_right = b_shape[0];
    assert!(
        k_left == k_right,
        "Contracted dimensions differ: {} vs {}",
        k_left,
        k_right
    );

    let mut out_shape = Vec::with_capacity(a_shape.len() + b_shape.len() - 2);
    out_shape.extend_from_slice(&a_shape[..a_shape.len() - 1]);
    out_shape.extend_from_slice(&b_shape[1..]);
    if out_shape.is_empty() {
        panic!("Rank-1 X Rank-1 would produce a scalar; not supported");
    }
    let out_len: usize = out_shape.iter().product();
    let mut out_data: Vec<K> = vec![K::zero(); out_len];

    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    let index_to_multi = |mut lin: usize, shape: &[usize]| -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }
        let mut coords = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            let sz = shape[i];
            coords[i] = lin % sz;
            lin /= sz;
        }
        coords
    };

    for out_lin in 0..out_len {
        let out_multi = index_to_multi(out_lin, &out_shape);
        let a_batch_len = a_rank.saturating_sub(1);
        let b_tail_len = b_rank.saturating_sub(1);
        let a_batch_indices = &out_multi[..a_batch_len];
        let b_tail_indices = &out_multi[a_batch_len..a_batch_len + b_tail_len];
        let mut acc = K::zero();
        for k in 0..k_left {
            let mut a_idx = Vec::with_capacity(a_rank);
            a_idx.extend_from_slice(a_batch_indices);
            a_idx.push(k);
            let mut b_idx = Vec::with_capacity(b_rank);
            b_idx.push(k);
            b_idx.extend_from_slice(b_tail_indices);
            let a_val = a.get(&a_idx).expect("A index in bounds").clone();
            let b_val = b.get(&b_idx).expect("B index in bounds").clone();
            acc = acc + a_val * b_val;
        }
        out_data[out_lin] = acc;
    }
    Matrix::new(out_data, out_shape)
}

/// Borrowed matrix × matrix multiplication (non-consuming).
impl<'a, K> Mul<&'a Matrix<K>> for &'a Matrix<K>
where
    K: Clone + Add<Output = K> + Mul<Output = K> + Zero,
{
    type Output = Matrix<K>;
    fn mul(self, rhs: &'a Matrix<K>) -> Self::Output {
        matmul_ref(self, rhs)
    }
}

/// Matrix X Vector contraction over the last axis of the matrix.
/// Let A have shape [d0,..., d_{m-2}, K], vector has length K.
/// Result shape: [d0,..., d_{m-2}]
impl<K> Mul<Vector<K>> for Matrix<K>
where
    K: Clone + Add<Output = K> + Mul<Output = K> + Zero,
{
    type Output = Vector<K>;

    fn mul(self, rhs: Vector<K>) -> Self::Output {
        let a_shape = self.dims();
        assert!(
            !a_shape.is_empty(),
            "Matrix must have rank >= 1 for matXvec"
        );
        let k = *a_shape.last().unwrap();
        assert!(
            rhs.len() == k,
            "Last dimension {} of matrix must match vector length {}",
            k,
            rhs.len()
        );

        // Output shape = matrix shape without last; if rank==1 result is scalar (not representable
        // as Vector<K> of length 0), so we forbid rank==1 here.
        if a_shape.len() == 1 {
            panic!("Rank-1 (vector) X vector would yield a scalar; not supported");
        }
        let out_len: usize = a_shape[..a_shape.len() - 1].iter().product();
        let mut out = vec![K::zero(); out_len];

        // Helpers
        let index_to_multi = |mut lin: usize, shape: &[usize]| -> Vec<usize> {
            if shape.is_empty() {
                return vec![];
            }
            let mut coords = vec![0; shape.len()];
            for i in (0..shape.len()).rev() {
                let sz = shape[i];
                coords[i] = lin % sz;
                lin /= sz;
            }
            coords
        };

        for out_lin in 0..out_len {
            let batch_multi = index_to_multi(out_lin, &a_shape[..a_shape.len() - 1]);
            let mut acc = K::zero();
            for k_i in 0..k {
                let mut a_idx = batch_multi.clone();
                a_idx.push(k_i);
                let a_val = self.get(&a_idx).expect("in bounds").clone();
                let b_val = rhs.get(k_i).expect("vector index").clone();
                acc = acc + a_val * b_val;
            }
            out[out_lin] = acc;
        }

        Vector::from(out)
    }
}

/// Borrowed matrix × borrowed vector contraction (non-consuming).
impl<'a, K> Mul<&'a Vector<K>> for &'a Matrix<K>
where
    K: Clone + Add<Output = K> + Mul<Output = K> + Zero,
{
    type Output = Vector<K>;
    fn mul(self, rhs: &'a Vector<K>) -> Self::Output {
        let a_shape = self.dims();
        assert!(
            !a_shape.is_empty(),
            "Matrix must have rank >= 1 for matXvec"
        );
        let k = *a_shape.last().unwrap();
        assert!(
            rhs.len() == k,
            "Last dimension {} of matrix must match vector length {}",
            k,
            rhs.len()
        );
        if a_shape.len() == 1 {
            panic!("Rank-1 (vector) X vector would yield a scalar; not supported");
        }
        let out_len: usize = a_shape[..a_shape.len() - 1].iter().product();
        let mut out = vec![K::zero(); out_len];
        let index_to_multi = |mut lin: usize, shape: &[usize]| -> Vec<usize> {
            if shape.is_empty() {
                return vec![];
            }
            let mut coords = vec![0; shape.len()];
            for i in (0..shape.len()).rev() {
                let sz = shape[i];
                coords[i] = lin % sz;
                lin /= sz;
            }
            coords
        };
        for out_lin in 0..out_len {
            let batch_multi = index_to_multi(out_lin, &a_shape[..a_shape.len() - 1]);
            let mut acc = K::zero();
            for k_i in 0..k {
                let mut a_idx = batch_multi.clone();
                a_idx.push(k_i);
                let a_val = self.get(&a_idx).expect("in bounds").clone();
                let b_val = rhs.get(k_i).expect("vector index").clone();
                acc = acc + a_val * b_val;
            }
            out[out_lin] = acc;
        }
        Vector::from(out)
    }
}

// ------------------------- Tests -------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::my_mat::Matrix;

    #[test]
    fn matmul_2d() {
        let a = Matrix::from_nested_unchecked(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let b = Matrix::from_nested_unchecked(vec![vec![7, 8], vec![9, 10], vec![11, 12]]);
        let c = a * b;
        assert_eq!(c.dims(), &[2, 2]);
        assert_eq!(
            c.linear_iter().cloned().collect::<Vec<_>>(),
            vec![58, 64, 139, 154]
        );
    }

    #[test]
    fn matmul_nd() {
        // A shape [2,3,4], B shape [4,5] => result [2,3,5]
        let a = Matrix::from_nested_unchecked(vec![
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]],
            vec![vec![2, 3, 4, 5], vec![6, 7, 8, 9], vec![10, 11, 12, 13]],
        ]);
        let b = Matrix::from_nested_unchecked(vec![
            vec![1, 0, 2, 0, 1],
            vec![0, 1, 0, 2, 0],
            vec![1, 0, 1, 0, 1],
            vec![0, 1, 0, 1, 0],
        ]); // shape [4,5]

        let c = a * b; // shape [2,3,5]
        assert_eq!(c.dims(), &[2, 3, 5]);
    }

    #[test]
    fn mat_vec_nd() {
        // A shape [2,3,4], v length 4 -> result shape [2,3]
        let a = Matrix::from_nested_unchecked(vec![
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]],
            vec![vec![2, 3, 4, 5], vec![6, 7, 8, 9], vec![10, 11, 12, 13]],
        ]);
        let v = Vector::from(vec![1, 0, 1, 0]); // length 4
        let r = a * v;
        assert_eq!(r.len(), 2 * 3);
    }

    #[test]
    #[should_panic]
    fn matmul_bad_contract_dim() {
        let a = Matrix::from_nested_unchecked(vec![vec![1, 2]]);
        let b = Matrix::from_nested_unchecked(vec![vec![1, 2]]);
        let _ = a * b; // 2 (last of A) != 1 (first of B)
    }

    #[test]
    fn borrowed_matmul_2d() {
        let a = Matrix::from_nested_unchecked(vec![vec![1, 2], vec![3, 4]]);
        let b = Matrix::from_nested_unchecked(vec![vec![5, 6], vec![7, 8]]);
        let c = &a * &b; // borrowed multiply
        // a and b still usable
        assert_eq!(a.dims(), &[2, 2]);
        assert_eq!(b.dims(), &[2, 2]);
        assert_eq!(c.dims(), &[2, 2]);
        assert_eq!(c.linear_iter().cloned().collect::<Vec<_>>(), vec![19, 22, 43, 50]);
    }

    #[test]
    fn borrowed_mat_vec() {
        // A shape [2,3,4], v length 4 -> result shape [2,3]
        let a = Matrix::from_nested_unchecked(vec![
            vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]],
            vec![vec![2, 3, 4, 5], vec![6, 7, 8, 9], vec![10, 11, 12, 13]],
        ]);
        let v = Vector::from(vec![1, 0, 1, 0]);
        let r = &a * &v;
        assert_eq!(r.len(), 2 * 3);
        // ensure a, v still usable
        assert_eq!(a.dims(), &[2, 3, 4]);
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn borrowed_scalar_mul() {
        let a = Matrix::from_nested_unchecked(vec![vec![1, 2], vec![3, 4]]);
        let b = &a * 10;
        assert_eq!(b.linear_iter().cloned().collect::<Vec<_>>(), vec![10, 20, 30, 40]);
        // Original unchanged
        assert_eq!(a.linear_iter().cloned().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn rank1_matrix_vector_scalar_dot() {
        let m = Matrix::from_nested_unchecked(vec![1, 2, 3]); // shape [3]
        let v = Vector::from(vec![4, 5, 6]);
        let s = m.dot_vector(&v);
        assert_eq!(s, 32); // 1*4 + 2*5 + 3*6
    }
}

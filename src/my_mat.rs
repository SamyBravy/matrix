use std::fmt;

pub struct Matrix<K> {
    data: Vec<K>,
    shape: Vec<usize>,
    strides: Vec<usize>, // strides for C-order indexing
}

impl<K> Matrix<K> {
    pub fn new(data: Vec<K>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert!(
            expected == data.len(),
            "data length {} does not match shape product {}",
            data.len(),
            expected
        );
        let strides = compute_strides(&shape);
        Matrix {
            data,
            shape,
            strides,
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.shape
    }

    // Map multi-dimensional indices to flat index (C-order).
    fn index_flat(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        let mut idx = 0usize;
        for (i, &ind) in indices.iter().enumerate() {
            if ind >= self.shape[i] {
                return None;
            }
            idx += ind * self.strides[i];
        }
        Some(idx)
    }

    pub fn get(&self, indices: &[usize]) -> Option<&K> {
        self.index_flat(indices).map(|i| &self.data[i])
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut K> {
        self.index_flat(indices).map(move |i| &mut self.data[i])
    }

    pub fn linear_iter(&self) -> impl Iterator<Item = &K> {
        self.data.iter()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![0; n];
    let mut acc = 1usize;
    for i in (0..n).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i]);
    }
    strides
}

pub trait Scalar {}
impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for usize {}
impl Scalar for num_complex::Complex<f64> {}
impl Scalar for num_complex::Complex<f32> {}
pub trait Nested<T> {
    fn shape_checked(&self) -> Result<Vec<usize>, String>;
    fn flatten_into(&self, out: &mut Vec<T>);
}

impl<T> Nested<T> for T
where
    T: Scalar + Clone,
{
    fn shape_checked(&self) -> Result<Vec<usize>, String> {
        Ok(vec![])
    }
    fn flatten_into(&self, out: &mut Vec<T>) {
        out.push(self.clone());
    }
}

impl<U, T> Nested<T> for Vec<U>
where
    U: Nested<T>,
{
    fn shape_checked(&self) -> Result<Vec<usize>, String> {
        if self.is_empty() {
            // Treat empty vector as a dimension of size 0
            return Ok(vec![0]);
        }
        let first_shape = self[0].shape_checked()?;
        for (i, item) in self.iter().enumerate().skip(1) {
            let s = item.shape_checked()?;
            if s != first_shape {
                return Err(format!(
                    "inconsistent inner shapes at index {}: {:?} != {:?}",
                    i, s, first_shape
                ));
            }
        }
        let mut shape = Vec::with_capacity(1 + first_shape.len());
        shape.push(self.len());
        shape.extend(first_shape);
        Ok(shape)
    }

    fn flatten_into(&self, out: &mut Vec<T>) {
        for item in self {
            item.flatten_into(out);
        }
    }
}

impl<K> Matrix<K>
where
    K: Scalar + Clone,
{
    pub fn try_from_nested<N>(nested: N) -> Result<Self, String>
    where
        N: Nested<K>,
    {
        let shape = nested.shape_checked()?;
        let mut data = Vec::new();
        nested.flatten_into(&mut data);
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(format!(
                "flattened data length {} does not match shape product {}",
                data.len(),
                expected
            ));
        }
        let strides = compute_strides(&shape);
        Ok(Matrix {
            data,
            shape,
            strides,
        })
    }

    pub fn from_nested_unchecked<N>(nested: N) -> Self
    where
        N: Nested<K>,
    {
        match Self::try_from_nested(nested) {
            Ok(m) => m,
            Err(e) => panic!("from_nested_unchecked: {}", e),
        }
    }
}

use crate::my_vect::Vector;

//from vector
impl<K> Matrix<K>
where
    K: Clone,
{
    /// Construct a 1xN matrix (row) from a Vector.
    pub fn from_vector_row(vec: Vector<K>) -> Self {
        let len = vec.len();
        let data = vec.into_vec();
        let shape = vec![1, len];
        let strides = compute_strides(&shape);
        Matrix {
            data,
            shape,
            strides,
        }
    }

    /// Construct an N-length 1D matrix (shape = [N]) from a Vector.
    pub fn from_vector_1d(vec: Vector<K>) -> Self {
        let len = vec.len();
        let data = vec.into_vec();
        let shape = vec![len];
        let strides = compute_strides(&shape);
        Matrix {
            data,
            shape,
            strides,
        }
    }

    /// Try to build a Matrix from a Vector with an explicit shape, validating size.
    pub fn try_from_vector(vec: Vector<K>, shape: Vec<usize>) -> Result<Self, String> {
        let len = vec.len();
        let data = vec.into_vec();
        let expected: usize = shape.iter().product();
        if expected != len {
            return Err(format!(
                "data length {} does not match shape product {}",
                len, expected
            ));
        }
        let strides = compute_strides(&shape);
        Ok(Matrix {
            data,
            shape,
            strides,
        })
    }

    /// Convenience panicking constructor when you know the shape is correct.
    pub fn from_vector_unchecked(vec: Vector<K>, shape: Vec<usize>) -> Self {
        match Self::try_from_vector(vec, shape) {
            Ok(m) => m,
            Err(e) => panic!("from_vector_unchecked: {}", e),
        }
    }
}

impl<K> From<Vector<K>> for Matrix<K>
where
    K: Clone,
{
    fn from(vec: Vector<K>) -> Self {
        Self::from_vector_1d(vec)
    }
}

//println overload

impl<K> fmt::Display for Matrix<K>
where
    K: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nd = self.shape.len();
        match nd {
            0 => writeln!(f, "").map(|_| ()).or(Ok(())), // empty
            1 => {
                for i in 0..self.shape[0] {
                    if let Some(v) = self.get(&[i]) {
                        write!(f, "{} ", v)?;
                    }
                }
                writeln!(f)
            }
            2 => {
                let (r, c) = (self.shape[0], self.shape[1]);
                for i in 0..r {
                    for j in 0..c {
                        if let Some(v) = self.get(&[i, j]) {
                            write!(f, "{} ", v)?;
                        }
                    }
                    writeln!(f)?;
                }
                Ok(())
            }
            _ => {
                let lead_dims = nd - 2;
                let mut prefix = vec![0usize; lead_dims];
                let mut first_block = true;
                loop {
                    if !first_block {
                        writeln!(f)?;
                    }
                    first_block = false;

                    write!(f, "slice ")?;
                    write!(f, "[")?;
                    for (pi, p) in prefix.iter().enumerate() {
                        if pi > 0 {
                            write!(f, ",")?;
                        }
                        write!(f, "{}", p)?;
                    }
                    writeln!(f, "]:")?;

                    let rows = self.shape[nd - 2];
                    let cols = self.shape[nd - 1];
                    let mut idx = prefix.clone();
                    for i in 0..rows {
                        idx.push(i);
                        for j in 0..cols {
                            idx.push(j);
                            if let Some(v) = self.get(&idx) {
                                write!(f, "{} ", v)?;
                            }
                            idx.pop();
                        }
                        idx.pop();
                        writeln!(f)?;
                    }
                    let mut carry = true;
                    for d in (0..lead_dims).rev() {
                        if carry {
                            prefix[d] += 1;
                            if prefix[d] >= self.shape[d] {
                                prefix[d] = 0;
                                carry = true;
                            } else {
                                carry = false;
                            }
                        }
                    }
                    if carry {
                        break;
                    }
                }
                Ok(())
            }
        }
    }
}

// TESTS
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_1d_matrix() {
        let m = Matrix::<i32>::try_from_nested(vec![1, 2, 3]).unwrap();
        assert_eq!(m.dims(), &[3]);
        assert_eq!(m.get(&[0]), Some(&1));
        assert_eq!(m.get(&[1]), Some(&2));
        assert_eq!(m.get(&[2]), Some(&3));
        assert_eq!(m.get(&[3]), None); // out of bounds
    }

    #[test]
    fn creates_2d_matrix_and_order_is_row_major() {
        // 3x2 matrix
        let nested = vec![vec![1, 2], vec![3, 4], vec![5, 6]];
        let m = Matrix::<i32>::try_from_nested(nested).unwrap();
        assert_eq!(m.dims(), &[3, 2]);

        // Check a few indices
        assert_eq!(m.get(&[0, 0]), Some(&1));
        assert_eq!(m.get(&[0, 1]), Some(&2));
        assert_eq!(m.get(&[1, 0]), Some(&3));
        assert_eq!(m.get(&[2, 1]), Some(&6));

        // Verify row-major flattening order by reading rows
        let mut flat = Vec::new();
        for i in 0..3 {
            for j in 0..2 {
                flat.push(*m.get(&[i, j]).unwrap());
            }
        }
        assert_eq!(flat, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn creates_3d_matrix() {
        let d0 = 2usize;
        let d1 = 3usize;
        let d2 = 4usize;
        // Build nested Vec with value formula matching row-major flattening
        let mut nested = Vec::with_capacity(d0);
        for i in 0..d0 {
            let mut v1 = Vec::with_capacity(d1);
            for j in 0..d1 {
                let mut v2 = Vec::with_capacity(d2);
                for k in 0..d2 {
                    v2.push(((i * d1 + j) * d2 + k) as i32);
                }
                v1.push(v2);
            }
            nested.push(v1);
        }
        let m = Matrix::<i32>::try_from_nested(nested).unwrap();
        assert_eq!(m.dims(), &[d0, d1, d2]);
        // Spot-check a few indices
        assert_eq!(m.get(&[0, 0, 0]), Some(&0));
        let exp1 = ((0 * d1 + 1) * d2 + 2) as i32;
        assert_eq!(m.get(&[0, 1, 2]).copied(), Some(exp1));
        let exp2 = ((1 * d1 + 2) * d2 + 3) as i32;
        assert_eq!(m.get(&[1, 2, 3]).copied(), Some(exp2));
        assert_eq!(m.get(&[2, 0, 0]), None); // out of bounds on first dim

        // Display should include a slice header and a 3x4 block for slice [0]
        let s = format!("{}", m);
        assert!(s.contains("slice [0]:"));
        assert!(s.contains("0 1 2 3 \n4 5 6 7 \n8 9 10 11 \n"));
    }

    #[test]
    fn empty_outer_dimension() {
        let m = Matrix::<i32>::try_from_nested(Vec::<Vec<i32>>::new()).unwrap();
        assert_eq!(m.dims(), &[0]);
        assert_eq!(m.get(&[0]), None);
    }

    #[test]
    fn empty_inner_dimension() {
        let nested: Vec<Vec<i32>> = vec![vec![], vec![], vec![]];
        let m = Matrix::<i32>::try_from_nested(nested).unwrap();
        assert_eq!(m.dims(), &[3, 0]);
        assert_eq!(m.get(&[0, 0]), None);
    }

    #[test]
    fn irregular_siblings_return_err() {
        let nested = vec![vec![1, 2], vec![3]]; // ragged
        let res = Matrix::<i32>::try_from_nested(nested);
        assert!(res.is_err());
    }

    #[test]
    #[should_panic]
    fn unchecked_panics_on_irregular() {
        let nested = vec![vec![1, 2], vec![3]]; // ragged
        let _ = Matrix::<i32>::from_nested_unchecked(nested);
    }

    #[test]
    fn supports_complex_scalars() {
        use num_complex::Complex;
        let nested = vec![Complex::new(1.0f64, 2.0), Complex::new(3.0, 4.0)];
        let m = Matrix::<Complex<f64>>::try_from_nested(nested).unwrap();
        assert_eq!(m.dims(), &[2]);
        assert_eq!(m.get(&[1]).unwrap().re, 3.0);
        assert_eq!(m.get(&[1]).unwrap().im, 4.0);
    }
}

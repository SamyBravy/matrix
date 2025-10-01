use std::fmt;

#[derive(Clone, Debug)]
/// A multi-dimensional array (tensor) with arbitrary shape, stored in a contiguous `Vec<K>`.
/// Uses row-major (C-order) indexing for efficient access.
/// Supports element-wise operations, indexing, and construction from nested vectors or vectors.
pub struct Matrix<K> {
    data: Vec<K>,
    shape: Vec<usize>,
    strides: Vec<usize>, // strides for C-order indexing
}

impl<K> Matrix<K> {
    /// Creates a new `Matrix` from a flat data vector and a shape vector.
    ///
    /// # Panics
    /// Panics if the product of the shape dimensions does not equal the length of the data vector.
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

    /// Returns the shape of the matrix as a slice of dimensions.
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
						println!("ind: {}, stride: {}", ind, self.strides[i]);
            idx += ind * self.strides[i];
        }
        Some(idx)
    }

    /// Returns a reference to the element at the given multi-dimensional indices, or `None` if out of bounds.
    pub fn get(&self, indices: &[usize]) -> Option<&K> {
        self.index_flat(indices).map(|i| &self.data[i])
    }

    /// Returns a mutable reference to the element at the given multi-dimensional indices, or `None` if out of bounds.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut K> {
        self.index_flat(indices).map(move |i| &mut self.data[i])
    }

    /// Returns an iterator over all elements in row-major order.
    pub fn linear_iter(&self) -> impl Iterator<Item = &K> {
        self.data.iter()
    }

    /// Returns the total number of elements in the matrix.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_square(&self) -> bool {
        self.shape.len() == 2 && self.shape[0] == self.shape[1]
    }

    pub fn data_clone(&self) -> Vec<K>
    where
        K: Clone,
    {
        self.data.clone()
    }

    /// Helper method for formatting n-dimensional matrices recursively
    fn format_nd_recursive(
        &self,
        f: &mut fmt::Formatter<'_>,
        prefix: &[usize],
        depth: usize,
    ) -> fmt::Result
    where
        K: fmt::Display,
    {
        let nd = self.shape.len();
        if depth == nd - 2 {
            // At the second-to-last dimension, format as 2D slice
            write!(f, "[")?;
            let rows = self.shape[nd - 2];
            let cols = self.shape[nd - 1];
            for i in 0..rows {
                if i > 0 {
                    write!(f, ",\n{}", " ".repeat((depth + 1) * 2))?;
                }
                write!(f, "[")?;
                for j in 0..cols {
                    if j > 0 {
                        write!(f, ", ")?;
                    }
                    let mut idx = prefix.to_vec();
                    idx.push(i);
                    idx.push(j);
                    if let Some(v) = self.get(&idx) {
                        write!(f, "{}", v)?;
                    }
                }
                write!(f, "]")?;
            }
            write!(f, "]")
        } else {
            // Recursively format higher dimensions
            write!(f, "[")?;
            for i in 0..self.shape[depth] {
                if i > 0 {
                    write!(f, ",\n{}", " ".repeat((depth + 1) * 2))?;
                }
                let mut new_prefix = prefix.to_vec();
                new_prefix.push(i);
                self.format_nd_recursive(f, &new_prefix, depth + 1)?;
            }
            write!(f, "]")
        }
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
		println!("strides: {:?}", strides);
    strides
}
use std::ops::{Index, IndexMut};

impl<K> Index<&[usize]> for Matrix<K> {
    type Output = K;

    fn index(&self, index: &[usize]) -> &Self::Output {
        match self.index_flat(index) {
            Some(i) => &self.data[i],
            None => panic!(
                "matrix index out of bounds or wrong dimensionality: {:?}",
                index
            ),
        }
    }
}

impl<K> IndexMut<&[usize]> for Matrix<K> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        match self.index_flat(index) {
            Some(i) => &mut self.data[i],
            None => panic!(
                "matrix index out of bounds or wrong dimensionality: {:?}",
                index
            ),
        }
    }
}

impl<K> Index<usize> for Matrix<K> {
    type Output = K;

    fn index(&self, index: usize) -> &Self::Output {
        if self.shape.len() != 1 {
            panic!(
                "Index<usize> only supported for 1-D matrices, found {} dims",
                self.shape.len()
            );
        }
        &self.data[index]
    }
}

impl<K> IndexMut<usize> for Matrix<K> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if self.shape.len() != 1 {
            panic!(
                "IndexMut<usize> only supported for 1-D matrices, found {} dims",
                self.shape.len()
            );
        }
        &mut self.data[index]
    }
}

/// Marker trait for types that can be used as scalar elements in matrices.
pub trait Scalar {}
impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for usize {}
impl Scalar for num_complex::Complex<f64> {}
impl Scalar for num_complex::Complex<f32> {}

/// Trait for types that can be recursively nested to build matrices.
pub trait Nested<T> {
    /// Returns the shape of this nested structure, or an error if siblings differ.
    fn shape_checked(&self) -> Result<Vec<usize>, String>;
    /// Appends all leaf elements into out in row-major (C-order) traversal.
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
            0 => write!(f, "[]"), // empty matrix
            1 => {
                write!(f, "[")?;
                for i in 0..self.shape[0] {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if let Some(v) = self.get(&[i]) {
                        write!(f, "{}", v)?;
                    }
                }
                write!(f, "]")
            }
            2 => {
                let (r, c) = (self.shape[0], self.shape[1]);
                write!(f, "[")?;
                for i in 0..r {
                    if i > 0 {
                        write!(f, ",\n ")?;
                    }
                    write!(f, "[")?;
                    for j in 0..c {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        if let Some(v) = self.get(&[i, j]) {
                            write!(f, "{}", v)?;
                        }
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")
            }
            _ => {
                // For 3D and higher, use nested bracket notation
                self.format_nd_recursive(f, &[], 0)
            }
        }
    }
}

impl<K> IntoIterator for Matrix<K>
where
    K: Clone,
{
    type Item = K;
    type IntoIter = std::vec::IntoIter<K>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, K> IntoIterator for &'a Matrix<K>
where
    K: Clone,
{
    type Item = &'a K;
    type IntoIter = std::slice::Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}
impl<'a, K> IntoIterator for &'a mut Matrix<K>
where
    K: Clone,
{
    type Item = &'a mut K;
    type IntoIter = std::slice::IterMut<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

// impl<'a,K> IntoIterator for &'a mut Matrix<K>
// where
// 		K: Clone,
// {
// 		type Item = &'a mut K;
// 		type IntoIter = std::slice::IterMut<'a, K>;

// 		fn into_iter(self) -> Self::IntoIter {
// 				self.data.into_iter()
// 		}

// }

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

        // Display should use nested bracket notation
        let s = format!("{}", m);
        // Should contain nested brackets for 3D structure
        assert!(s.starts_with("["));
        assert!(s.contains("[0, 1, 2, 3]"));
        assert!(s.contains("[4, 5, 6, 7]"));
        assert!(s.contains("[8, 9, 10, 11]"));
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

    #[test]
    fn creates_4d_matrix() {
        // Create a 2x2x2x2 4D matrix
        let d0 = 2usize;
        let d1 = 2usize;
        let d2 = 2usize;
        let d3 = 2usize;

        let mut nested = Vec::with_capacity(d0);
        for i in 0..d0 {
            let mut v1 = Vec::with_capacity(d1);
            for j in 0..d1 {
                let mut v2 = Vec::with_capacity(d2);
                for k in 0..d2 {
                    let mut v3 = Vec::with_capacity(d3);
                    for l in 0..d3 {
                        v3.push((((i * d1 + j) * d2 + k) * d3 + l) as i32);
                    }
                    v2.push(v3);
                }
                v1.push(v2);
            }
            nested.push(v1);
        }

        let m = Matrix::<i32>::try_from_nested(nested).unwrap();
        assert_eq!(m.dims(), &[d0, d1, d2, d3]);

        // Test some specific indices
        assert_eq!(m.get(&[0, 0, 0, 0]), Some(&0));
        assert_eq!(m.get(&[0, 0, 0, 1]), Some(&1));
        assert_eq!(m.get(&[1, 1, 1, 1]), Some(&15)); // Last element
        assert_eq!(m.get(&[2, 0, 0, 0]), None); // Out of bounds

        // Test display format contains nested brackets
        let s = format!("{}", m);
        assert!(s.starts_with("["));
        assert!(s.contains("[0, 1]"));
        assert!(s.contains("[14, 15]"));
    }

    #[test]
    fn creates_5d_matrix() {
        // Create a small 2x2x2x2x2 5D matrix
        let dims = [2, 2, 2, 2, 2];
        let mut flat_data = Vec::new();

        // Generate data in row-major order
        for i0 in 0..dims[0] {
            for i1 in 0..dims[1] {
                for i2 in 0..dims[2] {
                    for i3 in 0..dims[3] {
                        for i4 in 0..dims[4] {
                            let val = ((((i0 * dims[1] + i1) * dims[2] + i2) * dims[3] + i3)
                                * dims[4]
                                + i4) as i32;
                            flat_data.push(val);
                        }
                    }
                }
            }
        }

        let m = Matrix::new(flat_data, dims.to_vec());
        assert_eq!(m.dims(), &dims);

        // Test corner cases
        assert_eq!(m.get(&[0, 0, 0, 0, 0]), Some(&0));
        assert_eq!(m.get(&[1, 1, 1, 1, 1]), Some(&31)); // Last element
        assert_eq!(m.get(&[0, 1, 0, 1, 0]), Some(&10));

        // Verify it displays without crashing
        let s = format!("{}", m);
        assert!(s.len() > 0);
        assert!(s.starts_with("["));
    }

    #[test]
    fn test_1d_matrix_display() {
        let m = Matrix::<i32>::try_from_nested(vec![1, 2, 3, 4]).unwrap();
        let s = format!("{}", m);
        assert_eq!(s, "[1, 2, 3, 4]");
    }

    #[test]
    fn test_2d_matrix_display() {
        let m = Matrix::<i32>::try_from_nested(vec![vec![1, 2], vec![3, 4]]).unwrap();
        let s = format!("{}", m);
        assert_eq!(s, "[[1, 2],\n [3, 4]]");
    }

    #[test]
    fn test_empty_matrix_display() {
        // Create a truly empty matrix with 0-length dimension
        let m = Matrix::<i32>::try_from_nested(Vec::<i32>::new()).unwrap();
        let s = format!("{}", m);
        assert_eq!(s, "[]");
    }

    #[test]
    fn test_matrix_from_vector_examples() {
        // Test different ways to create matrices from vectors
        let v = Vector::from(vec![1, 2, 3, 4, 5, 6]);

        // Create 1D matrix
        let m1d = Matrix::from_vector_1d(v.clone());
        assert_eq!(m1d.dims(), &[6]);
        assert_eq!(format!("{}", m1d), "[1, 2, 3, 4, 5, 6]");

        // Create row matrix (1x6)
        let m_row = Matrix::from_vector_row(v.clone());
        assert_eq!(m_row.dims(), &[1, 6]);
        assert_eq!(format!("{}", m_row), "[[1, 2, 3, 4, 5, 6]]");

        // Create 2x3 matrix
        let m2x3 = Matrix::try_from_vector(v.clone(), vec![2, 3]).unwrap();
        assert_eq!(m2x3.dims(), &[2, 3]);
        assert_eq!(format!("{}", m2x3), "[[1, 2, 3],\n [4, 5, 6]]");

        // Create 3x2 matrix
        let m3x2 = Matrix::try_from_vector(v, vec![3, 2]).unwrap();
        assert_eq!(m3x2.dims(), &[3, 2]);
        assert_eq!(format!("{}", m3x2), "[[1, 2],\n [3, 4],\n [5, 6]]");
    }
}

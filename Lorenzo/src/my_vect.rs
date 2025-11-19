use crate::my_mat::Matrix;
use std::{
    fmt,
    ops::{Index, IndexMut},
};

#[derive(Clone)]
pub struct Vector<K> {
    data: Vec<K>,
}

impl<K> Vector<K> {
    fn new(data: Vec<K>) -> Self {
        Vector { data }
    }

    pub fn into_vec(self) -> Vec<K> {
        self.data
    }
}

impl<K> Default for Vector<K>
where
    K: Default + Clone,
{
    fn default() -> Self {
        Vector { data: vec![] }
    }
}

impl<K> From<Vec<K>> for Vector<K> {
    fn from(data: Vec<K>) -> Self {
        Vector { data }
    }
}

impl<K, const N: usize> From<[K; N]> for Vector<K> {
    fn from(arr: [K; N]) -> Self {
        Vector { data: arr.into() }
    }
}

impl<K> From<Matrix<K>> for Vector<K>
where
    K: Clone,
{
    fn from(mat: Matrix<K>) -> Self {
        let mut data = Vec::with_capacity(mat.len());
        for item in mat.linear_iter() {
            data.push(item.clone());
        }
        Vector::new(data)
    }
}

impl<K> Vector<K> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Borrow the underlying slice.
    pub fn as_slice(&self) -> &[K] {
        &self.data
    }
}

impl<K> Vector<K>
where
    K: Clone,
{
    pub fn iter(&self) -> impl Iterator<Item = &K> {
        self.data.iter()
    }
}

// Display and Debug
impl<K> fmt::Display for Vector<K>
where
    K: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", item)?;
        }
        write!(f, "]")
    }
}

impl<K> fmt::Debug for Vector<K>
where
    K: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector {:?}", self.data)
    }
}

// operators
impl<K> PartialEq for Vector<K>
where
    K: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<K> IndexMut<usize> for Vector<K> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

// Iterator implementations
impl<K> IntoIterator for Vector<K>
where
    K: Clone,
{
    type Item = K;
    type IntoIter = std::vec::IntoIter<K>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, K> IntoIterator for &'a Vector<K>
where
    K: Clone,
{
    type Item = &'a K;
    type IntoIter = std::slice::Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, K> IntoIterator for &'a mut Vector<K>
where
    K: Clone,
{
    type Item = &'a mut K;
    type IntoIter = std::slice::IterMut<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<K> Index<usize> for Vector<K> {
    type Output = K;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<K> Vector<K> {
    pub fn insert_at(&mut self, index: usize, element: K) {
        self.data.insert(index, element);
    }
    pub fn remove_at(&mut self, index: usize) -> K {
        self.data.remove(index)
    }
}

impl<K> Vector<K> {
    pub fn get(&self, index: usize) -> Option<&K> {
        self.data.get(index)
    }
    pub fn resize(&mut self, new_len: usize, value: K)
    where
        K: Clone,
    {
        self.data.resize(new_len, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn vector_new_and_into_vec() {
        let v = Vector::from(vec![1, 2, 3]);
        let data = v.into_vec();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn vector_default() {
        let v: Vector<i32> = Vector::default();
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn vector_from_array() {
        let v = Vector::from([1, 2, 3, 4]);
        assert_eq!(v.len(), 4);
        assert_eq!(v[0], 1);
        assert_eq!(v[3], 4);
    }

    #[test]
    fn vector_from_matrix() {
        let m = Matrix::try_from_nested(vec![1, 2, 3]).unwrap();
        let v = Vector::from(m);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn vector_len_and_as_slice() {
        let v = Vector::from(vec![1, 2, 3]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn vector_iter() {
        let v = Vector::from(vec![1, 2, 3]);
        let sum: i32 = v.iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn vector_display() {
        let v = Vector::from(vec![1, 2, 3]);
        assert_eq!(format!("{}", v), "[1, 2, 3]");
    }

    #[test]
    fn vector_debug() {
        let v = Vector::from(vec![1, 2, 3]);
        let debug_str = format!("{:?}", v);
        assert!(debug_str.contains("Vector"));
        assert!(debug_str.contains("1"));
    }

    #[test]
    fn vector_partial_eq() {
        let v1 = Vector::from(vec![1, 2, 3]);
        let v2 = Vector::from(vec![1, 2, 3]);
        let v3 = Vector::from(vec![1, 2, 4]);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn vector_index() {
        let v = Vector::from(vec![10, 20, 30]);
        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
        assert_eq!(v[2], 30);
    }

    #[test]
    fn vector_index_mut() {
        let mut v = Vector::from(vec![10, 20, 30]);
        v[1] = 99;
        assert_eq!(v[1], 99);
    }

    #[test]
    fn vector_into_iterator() {
        let v = Vector::from(vec![1, 2, 3]);
        let sum: i32 = v.into_iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn vector_into_iterator_ref() {
        let v = Vector::from(vec![1, 2, 3]);
        let sum: i32 = (&v).into_iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn vector_into_iterator_mut() {
        let mut v = Vector::from(vec![1, 2, 3]);
        for x in &mut v {
            *x *= 2;
        }
        assert_eq!(v[0], 2);
        assert_eq!(v[1], 4);
        assert_eq!(v[2], 6);
    }

    #[test]
    fn vector_insert_at() {
        let mut v = Vector::from(vec![1, 3, 4]);
        v.insert_at(1, 2);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 3);
        assert_eq!(v[3], 4);
    }

    #[test]
    fn vector_remove_at() {
        let mut v = Vector::from(vec![1, 2, 3, 4]);
        let removed = v.remove_at(2);
        assert_eq!(removed, 3);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1);
        assert_eq!(v[1], 2);
        assert_eq!(v[2], 4);
    }

    #[test]
    fn vector_get() {
        let v = Vector::from(vec![10, 20, 30]);
        assert_eq!(v.get(0), Some(&10));
        assert_eq!(v.get(2), Some(&30));
        assert_eq!(v.get(3), None);
    }

    #[test]
    fn vector_resize() {
        let mut v = Vector::from(vec![1, 2, 3]);
        v.resize(5, 0);
        assert_eq!(v.len(), 5);
        assert_eq!(v[3], 0);
        assert_eq!(v[4], 0);

        v.resize(2, 0);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn vector_complex() {
        let v = Vector::from(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], Complex::new(1.0, 2.0));
    }
}

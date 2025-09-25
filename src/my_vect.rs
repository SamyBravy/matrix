use crate::my_mat::Matrix;
use std::{fmt, ops::Index, ops::IndexMut};

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

impl<K> From<Vec<K>> for Vector<K> {
    fn from(data: Vec<K>) -> Self {
        Vector { data }
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
}

impl<K> Vector<K>
where
    K: Clone,
{
    fn iter(&self) -> impl Iterator<Item = &K> {
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

impl<K> Index<usize> for Vector<K> {
    type Output = K;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

use crate::matrix::Matrix;
use std::ops::Index;
#[derive(Clone)]
pub struct Vector<K> {
    data: Vec<K>,
}

impl<K> Vector<K> {
    pub fn new(data: Vec<K>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, K> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, K> {
        self.data.iter_mut()
    }
}

impl<K> IntoIterator for Vector<K> {
    type Item = K;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, K> IntoIterator for &'a Vector<K> {
    type Item = &'a K;
    type IntoIter = std::slice::Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, K> IntoIterator for &'a mut Vector<K> {
    type Item = &'a mut K;
    type IntoIter = std::slice::IterMut<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<K> Index<usize> for Vector<K> {
    type Output = K;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.data.len() {
            panic!(
                "Index out of bounds: the len is {} but the index is {}",
                self.data.len(),
                index
            );
        }
        &self.data[index]
    }
}

impl<K> std::fmt::Display for Vector<K>
where
    K: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.data.iter().enumerate() {
            write!(f, "{}", item)?;
            if i < self.data.len() - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

// impl<K> From<Matrix<K>> for Vector<K>
// where
//     K: Clone,
// {
//     fn from(mat: Matrix<K>) -> Self {
// 		//
// 	}
// }

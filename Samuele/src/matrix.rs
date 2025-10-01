use crate::vector::Vector;
use std::ops::Index;

#[derive(Clone)]
pub struct Matrix<K> {
    data: Vector<K>,
    shape: Vector<usize>,
}

impl<K> Matrix<K> {
    pub fn new(data: Vector<K>, shape: Vector<usize>) -> Self {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            panic!(
                "Data length does not match shape: expected {} but got {}",
                expected_size,
                data.len()
            );
        }
        
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> Vector<usize> {
        self.shape.clone()
    }

    pub fn is_square(&self) -> bool {
        self.shape.iter().all(|n| *n == self.shape[0])
    }

    pub fn data(&self) -> &Vector<K> {
        &self.data
    }

    pub fn reshape(&mut self, new_shape: Vector<usize>)
    where
        K: Clone,
    {
        let expected_size: usize = new_shape.iter().product();
        if self.data.len() != expected_size {
            panic!(
                "Data length does not match new shape: expected {} but got {}",
                expected_size,
                self.data.len()
            );
        }

        self.shape = new_shape;
    }

 //   fn print(&self, f: &mut std::fmt::Formatter<'_>, depth: usize) -> std::fmt::Result {
 //       if depth == self.shape.len() - 1 {
  //          write!("{}", Vector::from(self.data()))
//        }
  //  }
}

impl<K> Index<&[usize]> for Matrix<K> {
    type Output = K;

    fn index(&self, index: &[usize]) -> &Self::Output {
        if index.len() != self.shape.len() {
            panic!(
                "Invalid index: expected a {}-dimensional index but got a {}-dimensional index",
                self.shape.len(),
                index.len()
            );
        }
        if !index
            .iter()
            .enumerate()
            .all(|(dim, i)| *i < self.shape[dim])
        {
            panic!(
                "Index out of bounds: the shape is {} but the index is {:?}",
                self.shape, index
            );
        }

        let idx: usize = 
        &self.data[idx]
    }
}

//impl<K> std::fmt::Display for Matrix<K>
//where
 //   K: std::fmt::Display,
//{
//    fn fmt(&self, f: &mut // std::fmt::Formatter<'_>) -> std::fmt::Result {
 //       self.print(f, 0)
//    }
//}

impl<K> From<Vector<K>> for Matrix<K> {
    fn from(vect: Vector<K>) -> Self {
        let len = vect.len();
        Matrix::new(vect, Vector::new(vec![len]))
    }
}

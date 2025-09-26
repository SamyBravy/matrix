use crate::my_vect::Vector;
impl<K> Vector::<K> {

pub fn dot(&self, v: Vector<K>) -> K
 where
	 K: Clone + std::ops::AddAssign + std::ops::Mul<Output = K>,
{
		if self.len() != v.len() {
				panic!("Vectors must have the same length for dot product");
		}
		let mut result = self.data[0].clone() * v.data[0].clone();
		for i in 1..self.len() {
			result += self.data[i].clone() * v.data[i].clone();
		}
		result
	}
}

impl<T> Vector<num_complex::Complex<T>>
where
    T: Clone + num_traits::Num + std::ops::Neg<Output = T> + std::ops::Add<Output = T> + Default,
    num_complex::Complex<T>: std::ops::AddAssign,
{
    /// Hermitian dot product for complex vectors: sum_i conj(self_i) * v_i.
    /// Panics if lengths differ.
    pub fn dot_conj(&self, v: &Vector<num_complex::Complex<T>>) -> num_complex::Complex<T> {
        if self.len() != v.len() {
            panic!("Vectors must have the same length for dot product");
        }
        let mut result = self.data[0].clone().conj() * v.data[0].clone();
        for i in 1..self.len() {
            result += self.data[i].clone().conj() * v.data[i].clone();
        }
        result
    }
}
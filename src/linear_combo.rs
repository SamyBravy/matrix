use crate::my_vect::Vector;

pub fn linear_combination<K>(u: &[Vector<K>], coefs: &[K]) -> Vector<K>
where
    K: Clone + std::ops::Mul<Output = K> + std::ops::Add<Output = K> + Default,
{
    if u.len() != coefs.len() {
        panic!("vectors and coefficients must have the same length");
    }
    if u.is_empty() {
        panic!("input vector slice must not be empty");
    }
    let dim = u[0].len();
    let mut result: Vector<K> = Vector::from(vec![K::default(); dim]);
    for (vec, coef) in u.iter().zip(coefs.iter()) {
        if vec.len() != dim {
            panic!("all vectors must have the same dimension");
        }
        for i in 0..dim {
            result.data[i] = result.data[i].clone() + vec[i].clone() * coef.clone();
        }
    }

    result
}

use crate::my_vect::Vector;

pub fn linear_combination<K>(u: &[Vector<K>], coefs: &[K]) -> Vector<K>
where
    K: Clone + std::ops::Mul<Output = K> + std::ops::Add<Output = K> + Default,
{
    if u.len() != coefs.len() {
        panic!("Wrong input!")
    }
    let mut result: Vector<K> = Vector::default();
    for (v, mult) in u.iter().zip(coefs.iter()) {
        for i in 0..v.len() {
            result.data[i] = result.data[i].clone() + v[i].clone() * mult.clone();
        }
    }
    result
}

use crate::my_vect::Vector;

pub fn cross_product<K>(u: &Vector<K>, v: &Vector<K>) -> Vector<K>
where
    K: Clone + std::ops::Sub<Output = K> + std::ops::Mul<Output = K>,
{
    if u.len() != 3 || v.len() != 3 {
        panic!("Cross product is only defined for 3-dimensional vectors");
    }
    Vector::from(vec![
        u[1].clone() * v[2].clone() - u[2].clone() * v[1].clone(),
        u[2].clone() * v[0].clone() - u[0].clone() * v[2].clone(),
        u[0].clone() * v[1].clone() - u[1].clone() * v[0].clone(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_product_basic() {
        let u = Vector::from(vec![1, 2, 3]);
        let v = Vector::from(vec![4, 5, 6]);
        let w = cross_product(&u, &v);
        // (1,2,3) x (4,5,6) = (-3, 6, -3)
        assert_eq!(w, Vector::from(vec![-3, 6, -3]));
    }

    #[test]
    fn cross_product_anti_commutative() {
        let u = Vector::from(vec![1, 2, 3]);
        let v = Vector::from(vec![4, 5, 6]);
        let w1 = cross_product(&u, &v);
        let w2 = cross_product(&v, &u);
        // w2 should be -w1
        assert_eq!(w2, Vector::from(vec![w1[0] * -1, w1[1] * -1, w1[2] * -1]));
    }

    #[test]
    #[should_panic(expected = "Cross product is only defined for 3-dimensional vectors")]
    fn cross_product_bad_dim() {
        let u = Vector::from(vec![1, 2]); // not length 3
        let v = Vector::from(vec![3, 4]);
        let _ = cross_product(&u, &v);
    }

    #[test]
    #[should_panic(expected = "Cross product is only defined for 3-dimensional vectors")]
    fn cross_product_bad_second_operand_dim() {
        let u = Vector::from(vec![1, 2, 3]);
        let v = Vector::from(vec![4, 5]);
        let _ = cross_product(&u, &v);
    }
}

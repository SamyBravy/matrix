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
            result[i] = result[i].clone() + vec[i].clone() * coef.clone();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::my_vect::Vector;

    #[test]
    fn test_linear_combination_basic() {
        let u = vec![
            Vector::from(vec![1.0, 2.0]),
            Vector::from(vec![3.0, 4.0]),
        ];
        let coefs = vec![2.0, -1.0];
        let result = linear_combination(&u, &coefs);
        let expected = Vector::from(vec![-1.0, 0.0]); // 2*(1,2) + (-1)*(3,4) = (2,4) + (-3,-4) = (-1,0)
        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_linear_combination_three_vectors() {
        let u = vec![
            Vector::from(vec![1.0, 0.0, 0.0]),
            Vector::from(vec![0.0, 1.0, 0.0]),
            Vector::from(vec![0.0, 0.0, 1.0]),
        ];
        let coefs = vec![2.0, 3.0, 4.0];
        let result = linear_combination(&u, &coefs);
        let expected = Vector::from(vec![2.0, 3.0, 4.0]);
        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_linear_combination_zero_coefficient() {
        let u = vec![
            Vector::from(vec![1.0, 2.0]),
            Vector::from(vec![3.0, 4.0]),
        ];
        let coefs = vec![1.0, 0.0];
        let result = linear_combination(&u, &coefs);
        let expected = Vector::from(vec![1.0, 2.0]);
        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    #[should_panic(expected = "vectors and coefficients must have the same length")]
    fn test_linear_combination_mismatched_lengths() {
        let u = vec![Vector::from(vec![1.0, 2.0])];
        let coefs = vec![1.0, 2.0]; // Different length
        linear_combination(&u, &coefs);
    }

    #[test]
    #[should_panic(expected = "input vector slice must not be empty")]
    fn test_linear_combination_empty_vectors() {
        let u: Vec<Vector<f64>> = vec![];
        let coefs: Vec<f64> = vec![];
        linear_combination(&u, &coefs);
    }

    #[test]
    #[should_panic(expected = "all vectors must have the same dimension")]
    fn test_linear_combination_different_dimensions() {
        let u = vec![
            Vector::from(vec![1.0, 2.0]),
            Vector::from(vec![3.0, 4.0, 5.0]), // Different dimension
        ];
        let coefs = vec![1.0, 1.0];
        linear_combination(&u, &coefs);
    }
}

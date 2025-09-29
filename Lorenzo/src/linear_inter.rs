pub fn lerp<V>(u: V, v: V, t: f32) -> V
where
    V: std::ops::Mul<f32, Output = V> + std::ops::Add<Output = V>,
{
    if t < 0. || t > 1. {
        panic!("t must be in [0, 1]");
    }
    u * (1. - t) + v * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::my_vect::Vector;

    #[test]
    fn test_lerp_midpoint() {
        let u = Vector::from(vec![0.0, 0.0]);
        let v = Vector::from(vec![2.0, 4.0]);
        let result = lerp(u, v, 0.5);
        let expected = Vector::from(vec![1.0, 2.0]);
        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    fn test_lerp_at_start() {
        let u = Vector::from(vec![1.0, 2.0, 3.0]);
        let v = Vector::from(vec![4.0, 5.0, 6.0]);
        let result = lerp(u.clone(), v, 0.0);
        assert_eq!(result.len(), u.len());
        for i in 0..result.len() {
            assert_eq!(result[i], u[i]);
        }
    }

    #[test]
    fn test_lerp_at_end() {
        let u = Vector::from(vec![1.0, 2.0, 3.0]);
        let v = Vector::from(vec![4.0, 5.0, 6.0]);
        let result = lerp(u, v.clone(), 1.0);
        assert_eq!(result.len(), v.len());
        for i in 0..result.len() {
            assert_eq!(result[i], v[i]);
        }
    }

    #[test]
    fn test_lerp_quarter() {
        let u = Vector::from(vec![0.0, 0.0]);
        let v = Vector::from(vec![4.0, 8.0]);
        let result = lerp(u, v, 0.25);
        let expected = Vector::from(vec![1.0, 2.0]); // 0.75*0 + 0.25*4 = 1, 0.75*0 + 0.25*8 = 2
        assert_eq!(result.len(), expected.len());
        for i in 0..result.len() {
            assert_eq!(result[i], expected[i]);
        }
    }

    #[test]
    #[should_panic(expected = "t must be in [0, 1]")]
    fn test_lerp_negative_t() {
        let u = Vector::from(vec![1.0, 2.0]);
        let v = Vector::from(vec![3.0, 4.0]);
        lerp(u, v, -0.1);
    }

    #[test]
    #[should_panic(expected = "t must be in [0, 1]")]
    fn test_lerp_t_greater_than_one() {
        let u = Vector::from(vec![1.0, 2.0]);
        let v = Vector::from(vec![3.0, 4.0]);
        lerp(u, v, 1.1);
    }
}

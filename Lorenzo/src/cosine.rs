use crate::my_vect::Vector;
use crate::norm::Norms;
use crate::dot_prod::DotF32;

pub fn angle_cos<K>(u: &Vector<K>, v: &Vector<K>) -> f32
where
    Vector<K>: DotF32 + Norms,
{
    let dot = u.dot_f32(v);
    let denom = u.norm_2() * v.norm_2();
    if denom == 0.0 {
        return 0.0;
    }
    let val = dot / denom;
    val.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::my_vect::Vector;
    use num_complex::Complex;

    const EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_angle_cos_perpendicular() {
        let u = Vector::from(vec![1.0f64, 0.0f64]);
        let v = Vector::from(vec![0.0f64, 1.0f64]);
        let cos_angle = angle_cos(&u, &v);
        assert!(approx_eq(cos_angle, 0.0)); // 90 degrees
    }

    #[test]
    fn test_angle_cos_parallel() {
        let u = Vector::from(vec![1.0f64, 1.0f64]);
        let v = Vector::from(vec![2.0f64, 2.0f64]);
        let cos_angle = angle_cos(&u, &v);
        assert!(approx_eq(cos_angle, 1.0)); // 0 degrees
    }

    #[test]
    fn test_angle_cos_opposite() {
        let u = Vector::from(vec![1.0f64, 0.0f64]);
        let v = Vector::from(vec![-1.0f64, 0.0f64]);
        let cos_angle = angle_cos(&u, &v);
        assert!(approx_eq(cos_angle, -1.0)); // 180 degrees
    }

    #[test]
    fn test_angle_cos_complex() {
        let u = Vector::from(vec![Complex::new(1.0f32, 0.0f32)]);
        let v = Vector::from(vec![Complex::new(0.0f32, 1.0f32)]);
        let cos_angle = angle_cos(&u, &v);
        assert!(approx_eq(cos_angle, 0.0)); // perpendicular complex vectors
    }

    #[test]
    fn test_angle_cos_zero_vector_returns_zero() {
        let u = Vector::from(vec![0.0f64, 0.0f64]);
        let v = Vector::from(vec![5.0f64, -7.5f64]);
        let cos_angle = angle_cos(&u, &v);
        assert!(approx_eq(cos_angle, 0.0));
    }
}

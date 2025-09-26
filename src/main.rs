mod add_sub_scale;
mod dot_prod;
mod linear_combo;
mod linear_inter;
mod my_mat;
mod my_vect;
mod norm;
use crate::my_mat::Matrix;
use crate::my_vect::Vector;
use crate::norm::Norms;

fn main() {
    // Vector addition
    let mut u = Vector::from(vec![2., 3.]);
    let v = Vector::from(vec![5., 7.]);
    u += v;
    println!("{}", u);
    // [7.0, 10.0]

    // Vector subtraction
    let mut u = Vector::from(vec![2., 3.]);
    let v = Vector::from(vec![5., 7.]);
    u -= v;
    println!("{}", u);
    // [-3.0, -4.0]

    // Vector scaling
    let mut u = Vector::from(vec![2., 3.]);
    u *= 2.;
    println!("{}", u);
    // [4.0, 6.0]

    // Matrix addition
    let mut u = Matrix::try_from_nested(vec![vec![1., 2.], vec![3., 4.]]).unwrap();
    let v = Matrix::try_from_nested(vec![vec![7., 4.], vec![-2., 2.]]).unwrap();
    u += v;
    println!("{}", u);
    // 8.0 6.0
    // 1.0 6.0

    // Matrix subtraction
    let mut u = Matrix::try_from_nested(vec![vec![1., 2.], vec![3., 4.]]).unwrap();
    let v = Matrix::try_from_nested(vec![vec![7., 4.], vec![-2., 2.]]).unwrap();
    u -= v;
    println!("{}", u);
    // -6.0 -2.0
    // 5.0 2.0

    // Matrix scaling
    let mut u = Matrix::try_from_nested(vec![vec![1., 2.], vec![3., 4.]]).unwrap();
    u *= 2.;
    println!("{}", u);
    // 2.0 4.0
    // 6.0 8.0

    // Linear combination of vectors
    let u = vec![
        Vector::from(vec![1., 2.]),
        Vector::from(vec![3., 4.]),
        Vector::from(vec![5., 6.]),
    ];
    let coefs = vec![2., 0., -1.];
    let result = linear_combo::linear_combination(&u, &coefs);
    println!("{}", result);
    // [ -3.0, -4.0 ]
    let e1 = Vector::from([1., 0., 0.]);
    let e2 = Vector::from([0., 1., 0.]);
    let e3 = Vector::from([0., 0., 1.]);
    let v1 = Vector::from([1., 2., 3.]);
    let v2 = Vector::from([0., 10., -100.]);
    println!(
        "{}",
        linear_combo::linear_combination(&[e1, e2, e3], &[10., -2., 0.5])
    );
    // [10.]
    // [-2.]
    // [0.5]
    println!(
        "{}",
        linear_combo::linear_combination(&[v1, v2], &[10., -2.])
    );
    // [10.]
    // [0.]
    // [230.]

    // Linear interpolation between vectors
    let u = Vector::from([1., 2., 3.]);
    let v = Vector::from([4., 5., 6.]);
    let t = 0.5;
    let result = linear_inter::lerp(u, v, t);
    println!("{}", result);
    // [2.5, 3.5, 4.5]

    // Norm examples (real)
    let v = Vector::from(vec![3.0f64, 4.0f64]);
    println!("L1 norm (real): {}", v.norm_1());
    println!("L2 norm (real): {}", v.norm_2());
    println!("Linf norm (real): {}", v.norm_inf());

    // Norm examples (complex)
    let vc = Vector::from(vec![num_complex::Complex::new(3.0f32, 4.0f32)]);
    println!("L1 norm (complex): {}", vc.norm_1());
    println!("L2 norm (complex): {}", vc.norm_2());
    println!("Linf norm (complex): {}", vc.norm_inf());
}

// Unit tests and small examples exercised via `cargo test`.
#[cfg(test)]
mod tests {
    use crate::my_mat::Matrix;
    use crate::my_vect::Vector;
    use crate::norm::Norms;
    use num_complex::Complex;
    use num_traits::ToPrimitive;

    const EPS_F32: f32 = 1e-6;

    fn approx_eq_f32(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS_F32
    }

    fn to_f64_vec<T: ToPrimitive + Clone>(it: impl IntoIterator<Item = T>) -> Vec<f64> {
        it.into_iter()
            .map(|x| x.to_f64().expect("failed to convert to f64"))
            .collect()
    }

    #[test]
    fn vector_add_sub_scale() {
        let mut u = Vector::from(vec![2.0, 3.0]);
        let v = Vector::from(vec![5.0, 7.0]);
        u += v;
        let got = to_f64_vec(u.iter().cloned());
        assert_eq!(got, vec![7.0, 10.0]);

        let mut u = Vector::from(vec![2.0, 3.0]);
        let v = Vector::from(vec![5.0, 7.0]);
        u -= v;
        let got = to_f64_vec(u.iter().cloned());
        assert_eq!(got, vec![-3.0, -4.0]);

        let mut u = Vector::from(vec![2.0, 3.0]);
        u *= 2.0;
        let got = to_f64_vec(u.iter().cloned());
        assert_eq!(got, vec![4.0, 6.0]);
    }

    #[test]
    fn matrix_add_sub_scale_and_index() {
        let mut u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let v = Matrix::try_from_nested(vec![vec![7.0, 4.0], vec![-2.0, 2.0]]).unwrap();
        u += v;
        // check a few entries using index by slice
        assert_eq!(u.get(&[0usize, 0usize]), Some(&8.0));
        assert_eq!(u.get(&[0usize, 1usize]), Some(&6.0));
        assert_eq!(u.get(&[1usize, 0usize]), Some(&1.0));
        assert_eq!(u.get(&[1usize, 1usize]), Some(&6.0));

        let mut u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let v = Matrix::try_from_nested(vec![vec![7.0, 4.0], vec![-2.0, 2.0]]).unwrap();
        u -= v;
        assert_eq!(u.get(&[0usize, 0usize]), Some(&-6.0));

        let mut u = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        u *= 2.0;
        assert_eq!(u.get(&[1usize, 1usize]), Some(&8.0));
    }

    #[test]
    fn linear_combination_and_lerp_examples() {
        let u = vec![
            Vector::from(vec![1.0, 2.0]),
            Vector::from(vec![3.0, 4.0]),
            Vector::from(vec![5.0, 6.0]),
        ];
        let coefs = vec![2.0, 0.0, -1.0];
        let result = crate::linear_combo::linear_combination(&u, &coefs);
        let got = to_f64_vec(result.iter().cloned());
        assert_eq!(got, vec![-3.0, -2.0]);

        let e1 = Vector::from([1.0, 0.0, 0.0]);
        let e2 = Vector::from([0.0, 1.0, 0.0]);
        let e3 = Vector::from([0.0, 0.0, 1.0]);
        let v = crate::linear_combo::linear_combination(&[e1, e2, e3], &[10.0, -2.0, 0.5]);
        let got = to_f64_vec(v.iter().cloned());
        assert_eq!(got, vec![10.0, -2.0, 0.5]);

        let u = Vector::from([1.0, 2.0, 3.0]);
        let v = Vector::from([4.0, 5.0, 6.0]);
        let t = 0.5;
        let r = crate::linear_inter::lerp(u, v, t);
        let got = to_f64_vec(r.iter().cloned());
        assert_eq!(got, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn norms_real_and_complex() {
        // real vector (f64) -> norms return f32
        let v = Vector::from(vec![3.0f64, 4.0f64]);
        let n2 = v.norm_2();
        assert!(approx_eq_f32(n2, 5.0));

        // complex vector (Complex<f32>)
        let v = Vector::from(vec![Complex::new(3.0f32, 4.0f32)]);
        let n2 = v.norm_2();
        assert!(approx_eq_f32(n2, 5.0));
    }
}

mod add_sub_scale;
mod cosine;
mod cross_prod;
mod dot_prod;
mod linear_combo;
mod linear_inter;
mod linear_map;
mod my_mat;
mod my_vect;
mod norm;
use crate::my_mat::Matrix;
use crate::my_vect::Vector;
use crate::norm::Norms;
use crate::add_sub_scale::Scale;
use num_complex::Complex;

// ANSI colors
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

fn header(title: &str) {
    println!("{CYAN}===== {title} ====={RESET}");
}

fn sub(label: &str) {
    println!("{MAGENTA}-- {label} --{RESET}");
}

fn main() {
    // -------------- Addition / Subtraction / Scaling --------------
    header("Vector Arithmetic (Add / Sub / Scale)");
    sub("2D vectors");
    let a2 = Vector::from(vec![2.0, 3.0]);
    let b2 = Vector::from(vec![5.0, 7.0]);
    println!("{BLUE}a = {a2}{RESET}");
    println!("{BLUE}b = {b2}{RESET}");
    let sum2 = a2.clone() + b2.clone();
    let diff2 = a2.clone() - b2.clone();
    let scaled2 = a2.clone() * 2.0;
    println!("{GREEN}a + b = {sum2}{RESET}");
    println!("{RED}a - b = {diff2}{RESET}");
    println!("{YELLOW}2 * a = {scaled2}{RESET}");
    println!("{YELLOW}3 * a (via scale trait) = {}{RESET}", a2.scale(3.0));

    sub("4D vectors");
    let a4 = Vector::from(vec![1.0, 2.0, 3.0, 4.0]);
    let b4 = Vector::from(vec![4.0, 3.0, 2.0, 1.0]);
    println!("{BLUE}a = {a4}{RESET}");
    println!("{BLUE}b = {b4}{RESET}");
    println!("{GREEN}a + b = {}{RESET}", a4.clone() + b4.clone());
    println!("{RED}a - b = {}{RESET}", a4.clone() - b4.clone());
    println!("{YELLOW}0.5 * a = {}{RESET}", a4.clone() * 0.5);

    sub("Complex vectors");
    let ac = Vector::from(vec![Complex::new(1.0f32, 2.0), Complex::new(3.0, -1.0)]);
    let bc = Vector::from(vec![Complex::new(-2.0f32, 0.5), Complex::new(1.0, 4.0)]);
    println!("{BLUE}a = {ac}{RESET}");
    println!("{BLUE}b = {bc}{RESET}");
    println!("{GREEN}a + b = {}{RESET}", ac.clone() + bc.clone());
    println!("{RED}a - b = {}{RESET}", ac.clone() - bc.clone());
    println!("{YELLOW}i * a (scale by Complex) = {}{RESET}", {
        let i = Complex::new(0.0f32, 1.0f32);
        let mut tmp = Vec::new();
        for z in ac.iter() {
            tmp.push(z.clone() * i);
        }
        Vector::from(tmp)
    });

    // -------------- Linear Combination --------------
    header("Linear Combination");
    sub("2D");
    let lc2 = linear_combo::linear_combination(
        &[Vector::from(vec![1.0, 0.0]), Vector::from(vec![0.0, 1.0])],
        &[3.0, -2.0],
    );
    println!("{GREEN}3*e1 -2*e2 = {lc2}{RESET}");
    sub("4D");
    let basis4 = [
        Vector::from(vec![1.0, 0.0, 0.0, 0.0]),
        Vector::from(vec![0.0, 1.0, 0.0, 0.0]),
        Vector::from(vec![0.0, 0.0, 1.0, 0.0]),
        Vector::from(vec![0.0, 0.0, 0.0, 1.0]),
    ];
    let lc4 = linear_combo::linear_combination(&basis4, &[1.0, -1.0, 2.0, 0.5]);
    println!("{GREEN}Combination = {lc4}{RESET}");
    sub("Complex");
    let cc1 = Vector::from(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
    let cc2 = Vector::from(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]);
    // Coefficients must match the vector scalar type (Complex<f32>)
    let complex_lc = linear_combo::linear_combination(
        &[cc1, cc2],
        &[Complex::new(2.0f32, 0.0f32), Complex::new(3.0f32, 0.0f32)],
    );
    println!("{GREEN}2*e1 + 3*e2 = {complex_lc}{RESET}");

    // -------------- Linear Interpolation (Lerp) --------------
    header("Linear Interpolation (lerp)");
    sub("2D");
    let l2 = linear_inter::lerp(
        Vector::from(vec![0.0, 0.0]),
        Vector::from(vec![2.0, 4.0]),
        0.25,
    );
    println!("{GREEN}lerp((0,0),(2,4),0.25) = {l2}{RESET}");
    sub("4D");
    let l4 = linear_inter::lerp(
        Vector::from(vec![0.0, 0.0, 0.0, 0.0]),
        Vector::from(vec![4.0, 3.0, 2.0, 1.0]),
        0.5,
    );
    println!("{GREEN}midpoint = {l4}{RESET}");
    sub("Complex (manual lerp)");
    println!("{YELLOW}Note:{RESET} Vector<Complex> scaling by f32 not implemented, so we lerp element-wise.");
    let c_u = Vector::from(vec![Complex::new(1.0, 2.0), Complex::new(-1.0, 0.5)]);
    let c_v = Vector::from(vec![Complex::new(3.0, -2.0), Complex::new(0.0, 1.0)]);
    let t = 0.3f32;
    let mut lerp_complex_data = Vec::new();
    for (zu, zv) in c_u.iter().zip(c_v.iter()) {
        let diff = zv - zu;
        // diff * t (Complex * f32 is supported)
        lerp_complex_data.push(zu + diff * t);
    }
    println!(
        "{GREEN}lerp_complex(t=0.3) = {}{RESET}",
        Vector::from(lerp_complex_data)
    );

    // -------------- Dot Product --------------
    header("Dot Product");
    sub("2D");
    let d2a = Vector::from(vec![1.0, 2.0]);
    let d2b = Vector::from(vec![3.0, 4.0]);
    // Disambiguate dot method (multiple impls for different scalar types)
    println!(
        "{GREEN}(1,2)·(3,4) = {}{RESET}",
        Vector::<f64>::dot(&d2a, &d2b)
    );
    sub("4D");
    let d4a = Vector::from(vec![1.0, 0.0, -1.0, 2.0]);
    let d4b = Vector::from(vec![2.0, -1.0, 4.0, 0.5]);
    println!("{GREEN}4D dot = {}{RESET}", Vector::<f64>::dot(&d4a, &d4b));
    sub("Complex (Hermitian)");
    let dc1 = Vector::from(vec![Complex::new(1.0, 2.0), Complex::new(0.0, -1.0)]);
    let dc2 = Vector::from(vec![Complex::new(3.0, -4.0), Complex::new(0.5, 2.0)]);
    println!("{GREEN}Hermitian dot = {}{RESET}", dc1.dot(&dc2));

    // -------------- Norms --------------
    header("Norms (L1, L2, Linf)");
    sub("2D");
    let n2 = Vector::from(vec![3.0f64, 4.0]);
    println!(
        "{GREEN}L1={} L2={} Linf={} (norm alias={}){RESET}",
        n2.norm_1(),
        n2.norm_2(),
        n2.norm_inf(),
        n2.norm()
    );
    sub("4D");
    let n4 = Vector::from(vec![1.0, -2.0, 3.0, -4.0]);
    println!(
        "{GREEN}L1={} L2={} Linf={}{RESET}",
        n4.norm_1(),
        n4.norm_2(),
        n4.norm_inf()
    );
    sub("Complex");
    let nc = Vector::from(vec![Complex::new(3.0f32, 4.0f32), Complex::new(1.0, -1.0)]);
    println!(
        "{GREEN}L1={} L2={} Linf={}{RESET}",
        nc.norm_1(),
        nc.norm_2(),
        nc.norm_inf()
    );

    // -------------- Cosine --------------
    header("Cosine Similarity");
    sub("2D perpendicular");
    let c2a = Vector::from(vec![1.0, 0.0]);
    let c2b = Vector::from(vec![0.0, 1.0]);
    println!("{GREEN}cos = {}{RESET}", cosine::angle_cos(&c2a, &c2b));
    sub("4D");
    let c4a = Vector::from(vec![1.0, 1.0, 0.0, 0.0]);
    let c4b = Vector::from(vec![2.0, 2.0, 0.0, 0.0]);
    println!(
        "{GREEN}parallel cos = {}{RESET}",
        cosine::angle_cos(&c4a, &c4b)
    );
    sub("Complex");
    let cc_a = Vector::from(vec![Complex::new(1.0, 0.0)]);
    let cc_b = Vector::from(vec![Complex::new(0.0, 1.0)]);
    println!("{GREEN}cos = {}{RESET}", cosine::angle_cos(&cc_a, &cc_b));

    // -------------- Matrix Operations --------------
    header("Matrix Operations (Add/Sub/Scale)");
    sub("2x2 real");
    let m_a = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
    let m_b = Matrix::try_from_nested(vec![vec![5.0, -1.0], vec![0.5, 2.0]]).unwrap();
    println!("{BLUE}A = {m_a}{RESET}");
    println!("{BLUE}B = {m_b}{RESET}");
    println!("{GREEN}A + B = {}{RESET}", (m_a.clone() + m_b.clone()));
    println!("{RED}A - B = {}{RESET}", (m_a.clone() - m_b.clone()));
    println!("{YELLOW}2 * A = {}{RESET}", (m_a.clone() * 2.0));
    sub("4x4 real");
    let m4 = Matrix::try_from_nested(vec![
        vec![1.0, 0.0, 2.0, -1.0],
        vec![0.0, 3.0, 0.0, 4.0],
        vec![5.0, 0.0, 6.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
    ])
    .unwrap();
    println!("{BLUE}M = {m4}{RESET}");
    println!("{YELLOW}0.5 * M = {}{RESET}", (&m4 * 0.5));
    sub("Complex 2x2");
    let mc = Matrix::try_from_nested(vec![
        vec![Complex::new(1.0, 1.0), Complex::new(0.0, -1.0)],
        vec![Complex::new(2.0, 0.5), Complex::new(-1.0, 2.0)],
    ])
    .unwrap();
    println!("{BLUE}C = {mc}{RESET}");
    println!("{YELLOW}i * C = {}{RESET}", {
        let i = Complex::new(0.0, 1.0);
        &mc * i
    });

    // -------------- Matrix Multiplication --------------
    header("Matrix Multiplication & Contraction");
    sub("2D: (2x3)*(3x2)");
    let mm_a = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    let mm_b = Matrix::try_from_nested(vec![vec![7, 8], vec![9, 10], vec![11, 12]]).unwrap();
    println!("{GREEN}Result = {}{RESET}", mm_a * mm_b);
    sub("4x4 * 4x4 (identity-ish)");
    let id_like = Matrix::try_from_nested(vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ])
    .unwrap();
    println!("{GREEN}M * I = {}{RESET}", &m4 * &id_like);
    sub("Complex 2x2");
    let mmc_a = Matrix::try_from_nested(vec![
        vec![Complex::new(1.0, 1.0), Complex::new(2.0, -1.0)],
        vec![Complex::new(0.0, 2.0), Complex::new(3.0, 0.0)],
    ])
    .unwrap();
    let mmc_b = Matrix::try_from_nested(vec![
        vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
        vec![Complex::new(-1.0, 1.0), Complex::new(2.0, -2.0)],
    ])
    .unwrap();
    println!("{GREEN}Complex product = {}{RESET}", &mmc_a * &mmc_b);

    // -------------- Matrix * Vector --------------
    header("Matrix * Vector");
    sub("2x3 * 3");
    let mv_a = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
    let mv_v = Vector::from(vec![1, 0, -1]);
    println!("{GREEN}Result = {}{RESET}", mv_a * mv_v);
    sub("4x4 * 4");
    let mv_v4 = Vector::from(vec![1.0, -1.0, 0.5, 2.0]);
    println!("{GREEN}Result = {}{RESET}", &m4 * &mv_v4);
    sub("Complex 2x2 * 2");
    let mv_c_v = Vector::from(vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)]);
    println!("{GREEN}Result = {}{RESET}", &mc * &mv_c_v);

    // -------------- Cross Product --------------
    header("Cross Product (3D only)");
    sub("3D real");
    let x_u = Vector::from(vec![1.0, 2.0, 3.0]);
    let x_v = Vector::from(vec![4.0, 5.0, 6.0]);
    println!(
        "{GREEN}u x v = {}{RESET}",
        cross_prod::cross_product(&x_u, &x_v)
    );
    sub("3D complex");
    let x_uc = Vector::from(vec![
        Complex::new(1.0, 1.0),
        Complex::new(0.0, 2.0),
        Complex::new(-1.0, 0.5),
    ]);
    let x_vc = Vector::from(vec![
        Complex::new(2.0, -1.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
    ]);
    println!(
        "{GREEN}u x v = {}{RESET}",
        cross_prod::cross_product(&x_uc, &x_vc)
    );
    println!("{YELLOW}Note:{RESET} 4D cross product isn't defined in standard 3D sense.");

    // -------------- Rank-1 Matrix · Vector --------------
    header("Rank-1 Matrix · Vector (Scalar Dot)");
    let r1 = Matrix::from_nested_unchecked(vec![1, 2, 3, 4]);
    let rv = Vector::from(vec![5, 6, 7, 8]);
    println!(
        "{GREEN}[1,2,3,4] · [5,6,7,8] = {}{RESET}",
        r1.dot_vector(&rv)
    );
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

mod add_sub_scale;
mod cosine;
mod cross_prod;
mod determinant;
mod dot_prod;
mod inverse;
mod linear_combo;
mod linear_inter;
mod linear_map;
mod my_mat;
mod my_vect;
mod norm;
mod rank;
mod row_echelon;
mod trace;
mod transpose;
use crate::add_sub_scale::Scale;
use crate::my_mat::Matrix;
use crate::my_vect::Vector;
use crate::norm::Norms;
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
    println!(
        "{YELLOW}Note:{RESET} Vector<Complex> scaling by f32 not implemented, so we lerp element-wise."
    );
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

    // -------------- Trace (Diagonal Sum) --------------
    header("Trace (Diagonal Sum)");
    sub("2x2 real");
    let t2 = Matrix::try_from_nested(vec![vec![1, 2], vec![3, 4]]).unwrap();
    println!("{BLUE}M = {t2}{RESET}");
    println!("{GREEN}trace = {}{RESET}", t2.trace());

    sub("3x3 real");
    let t3 = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]).unwrap();
    println!("{BLUE}M = {t3}{RESET}");
    println!("{GREEN}trace = {}{RESET}", t3.trace());

    sub("3x3x3 cubic (3D)");
    let t3d = Matrix::new(
        vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27,
        ],
        vec![3, 3, 3],
    );
    println!("{BLUE}3D cubic matrix [3,3,3]{RESET}");
    println!(
        "{GREEN}trace = {} (sum of [0,0,0], [1,1,1], [2,2,2]){RESET}",
        t3d.trace()
    );

    sub("2x2x2x2 hypercube (4D)");
    let t4d = Matrix::new((0..16).collect(), vec![2, 2, 2, 2]);
    println!("{BLUE}4D hypercube [2,2,2,2]{RESET}");
    println!(
        "{GREEN}trace = {} (sum of [0,0,0,0], [1,1,1,1]){RESET}",
        t4d.trace()
    );

    sub("Complex 2x2");
    let tc = Matrix::try_from_nested(vec![
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)],
    ])
    .unwrap();
    println!("{BLUE}C = {tc}{RESET}");
    println!("{GREEN}trace = {}{RESET}", tc.trace());

    // -------------- Row Echelon Form --------------
    header("Row Echelon Form (Gaussian Elimination)");
    sub("3x3 real");
    let re3 = Matrix::try_from_nested(vec![
        vec![2.0, 1.0, -1.0],
        vec![-3.0, -1.0, 2.0],
        vec![-2.0, 1.0, 2.0],
    ])
    .unwrap();
    println!("{BLUE}Original = {re3}{RESET}");
    println!("{GREEN}Row Echelon = {}{RESET}", re3.row_echelon().0);

    sub("2x2 real");
    let re2 = Matrix::try_from_nested(vec![vec![3.0, 2.0], vec![1.0, 4.0]]).unwrap();
    println!("{BLUE}Original = {re2}{RESET}");
    println!("{GREEN}Row Echelon = {}{RESET}", re2.row_echelon().0);

    sub("3x4 rectangular");
    let re_rect = Matrix::try_from_nested(vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 4.0, 7.0, 8.0],
        vec![3.0, 6.0, 10.0, 13.0],
    ])
    .unwrap();
    println!("{BLUE}Original = {re_rect}{RESET}");
    println!("{GREEN}Row Echelon = {}{RESET}", re_rect.row_echelon().0);

    sub("Identity matrix (should remain unchanged)");
    let re_id = Matrix::try_from_nested(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ])
    .unwrap();
    println!("{BLUE}Original = {re_id}{RESET}");
    println!("{GREEN}Row Echelon = {}{RESET}", re_id.row_echelon().0);

    // -------------- Determinant examples --------------
    header("Determinant");
    sub("2x2 integer");
    let d2 = Matrix::try_from_nested(vec![vec![1, 2], vec![3, 4]]).unwrap();
    println!("{BLUE}M = {d2}{RESET}");
    println!("{GREEN}det = {}{RESET}", d2.determinant());

    sub("3x3 integer");
    let d3 = Matrix::try_from_nested(vec![vec![6, 1, 1], vec![4, -2, 5], vec![2, 8, 7]]).unwrap();
    println!("{BLUE}M = {d3}{RESET}");
    println!("{GREEN}det = {}{RESET}", d3.determinant());

    sub("2x2 complex");
    let dc = Matrix::try_from_nested(vec![
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)],
    ])
    .unwrap();
    println!("{BLUE}C = {dc}{RESET}");
    println!("{GREEN}det = {}{RESET}", dc.determinant());

    // -------------- Matrix Inverse --------------
    header("Matrix Inverse");
    sub("2x2 float");
    let m2 = Matrix::try_from_nested(vec![vec![4.0, 7.0], vec![2.0, 6.0]]).unwrap();
    println!("{BLUE}M = {m2}{RESET}");
    if let Some(inv2) = m2.inverse() {
        // Format inverse nicely
        println!("{GREEN}M^(-1) = [{RESET}");
        for i in 0..2 {
            print!("{GREEN} [{RESET}");
            for j in 0..2 {
                let val: f64 = *inv2.get(&[i, j]).unwrap();
                if j > 0 {
                    print!(", ");
                }
                print!("{:.4}", val);
            }
            println!("{GREEN}]{RESET}");
        }
        println!("{GREEN}]{RESET}");

        // Verify: M * M^(-1) = I
        let product = &m2 * &inv2;
        println!("{YELLOW}M * M^(-1) ≈ [{RESET}");
        for i in 0..2 {
            print!("{YELLOW} [{RESET}");
            for j in 0..2 {
                let val: f64 = *product.get(&[i, j]).unwrap();
                if j > 0 {
                    print!(", ");
                }
                print!("{:.4}", val);
            }
            println!("{YELLOW}]{RESET}");
        }
        println!("{YELLOW}]{RESET}");
    }

    sub("3x3 float");
    let m3 = Matrix::try_from_nested(vec![
        vec![1.0, 2.0, 3.0],
        vec![0.0, 1.0, 4.0],
        vec![5.0, 6.0, 0.0],
    ])
    .unwrap();
    println!("{BLUE}M = {m3}{RESET}");
    if let Some(inv3) = m3.inverse() {
        println!("{GREEN}M^(-1) = [{RESET}");
        for i in 0..3 {
            print!("{GREEN} [{RESET}");
            for j in 0..3 {
                let val: f64 = *inv3.get(&[i, j]).unwrap();
                if j > 0 {
                    print!(", ");
                }
                print!("{:6.2}", val);
            }
            println!("{GREEN}]{RESET}");
        }
        println!("{GREEN}]{RESET}");
    }

    sub("Singular matrix (no inverse)");
    let singular = Matrix::try_from_nested(vec![vec![1.0, 2.0], vec![2.0, 4.0]]).unwrap();
    println!("{BLUE}M (singular) = {singular}{RESET}");
    match singular.inverse() {
        Some(_) => println!("{GREEN}M^(-1) exists{RESET}"),
        None => println!("{RED}M^(-1) does not exist (det = 0){RESET}"),
    }

    sub("2x2 complex");
    let mc = Matrix::try_from_nested(vec![
        vec![Complex::new(1.0, 1.0), Complex::new(0.0, 1.0)],
        vec![Complex::new(1.0, 0.0), Complex::new(1.0, 1.0)],
    ])
    .unwrap();
    println!("{BLUE}M = {mc}{RESET}");
    if let Some(invc) = mc.inverse() {
        println!("{GREEN}M^(-1) = {invc}{RESET}");
    }
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

    #[test]
    fn dot_product_examples_match_main() {
        let d2a = Vector::from(vec![1.0, 2.0]);
        let d2b = Vector::from(vec![3.0, 4.0]);
        assert_eq!(Vector::<f64>::dot(&d2a, &d2b), 11.0);

        let d4a = Vector::from(vec![1.0, 0.0, -1.0, 2.0]);
        let d4b = Vector::from(vec![2.0, -1.0, 4.0, 0.5]);
        assert_eq!(Vector::<f64>::dot(&d4a, &d4b), -1.0);

        let dc1 = Vector::from(vec![Complex::new(1.0f32, 2.0), Complex::new(0.0, -1.0)]);
        let dc2 = Vector::from(vec![Complex::new(3.0f32, -4.0), Complex::new(0.5, 2.0)]);
        let expected = Complex::new(-7.0f32, -9.5);
        assert_eq!(dc1.dot(&dc2), expected);
    }

    #[test]
    fn matrix_multiplication_example_matches_main() {
        let mm_a = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let mm_b = Matrix::try_from_nested(vec![vec![7, 8], vec![9, 10], vec![11, 12]]).unwrap();
        let product = mm_a * mm_b;
        assert_eq!(product.dims(), &[2, 2]);
        let values: Vec<_> = product.linear_iter().cloned().collect();
        assert_eq!(values, vec![58, 64, 139, 154]);
    }

    #[test]
    fn matrix_vector_example_matches_main() {
        let mv_a = Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        let mv_v = Vector::from(vec![1, 0, -1]);
        let result = mv_a * mv_v;
        let values: Vec<_> = result.into_vec();
        assert_eq!(values, vec![-2, -2]);
    }

    #[test]
    fn cross_product_example_matches_main() {
        let x_u = Vector::from(vec![1.0, 2.0, 3.0]);
        let x_v = Vector::from(vec![4.0, 5.0, 6.0]);
        let cross = crate::cross_prod::cross_product(&x_u, &x_v);
        let values: Vec<_> = cross.into_vec();
        assert_eq!(values, vec![-3.0, 6.0, -3.0]);
    }

    #[test]
    fn rank1_matrix_dot_example_matches_main() {
        let r1 = Matrix::from_nested_unchecked(vec![1, 2, 3, 4]);
        let rv = Vector::from(vec![5, 6, 7, 8]);
        assert_eq!(r1.dot_vector(&rv), 70);
    }

    #[test]
    fn trace_examples_match_main() {
        // 2x2 trace
        let t2 = Matrix::try_from_nested(vec![vec![1, 2], vec![3, 4]]).unwrap();
        assert_eq!(t2.trace(), 1 + 4);

        // 3x3 trace
        let t3 =
            Matrix::try_from_nested(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]).unwrap();
        assert_eq!(t3.trace(), 1 + 5 + 9);

        // 3x3x3 cubic trace
        let t3d = Matrix::new(
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27,
            ],
            vec![3, 3, 3],
        );
        assert_eq!(t3d.trace(), 1 + 14 + 27);

        // 2x2x2x2 hypercube trace
        let t4d = Matrix::new((0..16).collect(), vec![2, 2, 2, 2]);
        assert_eq!(t4d.trace(), 0 + 15);

        // Complex trace
        let tc = Matrix::try_from_nested(vec![
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)],
        ])
        .unwrap();
        assert_eq!(tc.trace(), Complex::new(8.0, 10.0));
    }

    #[test]
    fn row_echelon_examples_match_main() {
        // 3x3 matrix
        let re3 = Matrix::try_from_nested(vec![
            vec![2.0, 1.0, -1.0],
            vec![-3.0, -1.0, 2.0],
            vec![-2.0, 1.0, 2.0],
        ])
        .unwrap();
        let result = re3.row_echelon().0;

        // Should produce upper triangular form
        assert_eq!(result.dims(), &[3, 3]);
        assert_ne!(result.get(&[0, 0]), Some(&0.0));

        // Below diagonal should be zero (with floating point tolerance)
        let val: f64 = *result.get(&[1, 0]).unwrap();
        assert!(val.abs() < 1e-10);

        // 2x2 matrix
        let re2 = Matrix::try_from_nested(vec![vec![3.0, 2.0], vec![1.0, 4.0]]).unwrap();
        let result2 = re2.row_echelon().0;
        assert_eq!(result2.dims(), &[2, 2]);

        // Identity should remain unchanged
        let id = Matrix::try_from_nested(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
        .unwrap();
        let result_id = id.row_echelon().0;
        assert_eq!(result_id.get(&[0, 0]), Some(&1.0));
        assert_eq!(result_id.get(&[1, 1]), Some(&1.0));
        assert_eq!(result_id.get(&[2, 2]), Some(&1.0));
    }
}

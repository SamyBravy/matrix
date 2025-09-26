mod my_mat;
mod my_vect;
mod add_sub_scale;
mod linear_combo;

use crate::my_mat::Matrix;
use crate::my_vect::Vector;

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
}

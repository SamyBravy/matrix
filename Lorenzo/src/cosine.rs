use crate::my_vect::Vector;
use crate::norm::Norms;
use crate::dot_prod::DotF32;

fn angle_cos<K>(u: &Vector<K>, v: &Vector<K>) -> f32
where
    Vector<K>: DotF32 + Norms,
{
    let dot = u.dot_f32(v);
    dot / (u.norm_2() * v.norm_2())
}

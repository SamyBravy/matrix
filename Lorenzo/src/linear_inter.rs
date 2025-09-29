pub fn lerp<V>(u: V, v: V, t: f32) -> V
where
    V: std::ops::Mul<f32, Output = V> + std::ops::Add<Output = V>,
{
    if t < 0. || t > 1. {
        panic!("t must be in [0, 1]");
    }
    u * (1. - t) + v * t
}

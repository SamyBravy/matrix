struct Vector<K> {
	data: Vec<K>,
}

impl<K> Vector<K> {
	pub fn len(&self) -> usize {
		self.data.len()
	}
}

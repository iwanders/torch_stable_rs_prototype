//! An owning Size.
//!
//! Why do we need this? Well because .sizes() -> &[usize], which is not owned.
//! This also means that calling sizes() borrows the tensor, this makes it impossible to call a mutable method.
//! In short;
//! ```
//! let v = t.view_mut(t.sizes());
//! ```
//! is a compilation error.
//!
//! This is the equivalen to `torch.Size`.

#[derive(Default, Clone, Debug)]
pub struct Size(tinyvec::TinyVec<[usize; 8]>);

impl Size {
    pub fn from(v: &[usize]) -> Size {
        let mut z = Size::default();
        z.0 = v.iter().copied().collect();
        z
    }
}

impl std::ops::Deref for Size {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        &self.0.as_slice()
    }
}

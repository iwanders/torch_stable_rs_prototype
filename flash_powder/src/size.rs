//! An owning Size.
//!
//! Why do we need this? Well because .sizes() -> &[usize], which is not owned.
//! This also means that calling sizes() borrows the tensor, this makes it impossible to call a mutable method.
//! In short;
//! ```ignore
//! # use flash_powder::prelude::*;
//! let mut t = Tensor::randn(&[3,3], &Default::default()).unwrap();
//! let v = t.view_mut(t.sizes());
//! ```
//! is a compilation error.
//!
//! We avoid this with;
//! ```rust
//! # use flash_powder::prelude::*;
//! let mut t = Tensor::randn(&[3,3], &Default::default()).unwrap();
//! let shape = t.shape();
//! let v = t.view_mut(&shape);
//! ```
//!
//! This is the rough equivalent to `torch.Size`.
//!
//! It can deref into `&[usize]`!.

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
impl<'a> PartialEq<&[usize]> for &Size {
    fn eq(&self, other: &&[usize]) -> bool {
        self.0.as_slice() == *other
    }
}

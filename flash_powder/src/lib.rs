//!
//! Main building blocks:
//!
//! - [`Tensor`]; Owning tensor, this owns the data, created with [`TensorFactory`][factory::TensorFactory]. (think `Vec<u8>`)
//! - [`Ten<'_>`]; Const borrow of Tensor, this has a parent, its lifetime cannot exceed the parent. (think `&[u8]`)
//! - [`TenMut<'_>`]; Mutable borrow of Tensor, this has a mutable parent, its lifetime cannot exceed the parent. (think `&mut [u8]`)
//!
//! All of these provide [`TensorAccess`] and all functions and methods are implemented on that trait.
//!
//! - [`properties::TensorProperties`]: Methods to retrieve tensor properties like dimension and size.
//! - [`data`]`::{`[`DataRef`][`data::DataRef`], [`DataMut`][`data::DataMut`]`}`: Traits to access the tensor's data as bytes or other types.
//! - [`core_methods`]`::`[`CoreMethods`][`core_methods::CoreMethods`]: Methods / Functions on [`TensorAccess`] that require const access.
//! - [`core_methods`]`::`[`CoreMethodsMut`][`core_methods::CoreMethodsMut`]: Methods / Functions on [`TensorAccess`] that require mutable access.
//! - [`functional`]: Holds free functions line [`conv2d`][`functional::conv2d`] and [`relu`][`functional::relu`], just like PyTorch's Functional.
//!
//!
//! Other principles;
//! - No unsafe in the public interface, safe behaviour as you'd expect.
//! - No interior mutability, all methods are const correct.
//! - Modifying one tensor will not modify another, unless through an mutable borrow.
//! - Rust style lifetimes on tensors, either tied together with an explicit lifetime, or completely separate.

pub mod conversion;
pub mod core_methods;
pub mod data;
pub mod factory;
pub mod functional;
pub mod properties;
pub mod tensor;
use tensor::{Ten, TenMut, Tensor, TensorAccess};
pub use torch_stable::{StableTorchResult, headeronly::core::ScalarType};

/// The prelude that contains all the important types and traits.
pub mod prelude {
    use super::*;
    #[doc(inline)]
    pub use core_methods::{CoreMethods, CoreMethodsMut};
    #[doc(inline)]
    pub use data::{DataMut, DataRef};
    #[doc(inline)]
    pub use factory::TensorFactory;
    #[doc(inline)]
    pub use properties::TensorProperties;
    #[doc(inline)]
    pub use tensor::{Ten, TenMut, Tensor, TensorAccess};

    // #[doc(inline)]
    // pub use crate::functional;
}
#[cfg(test)]
mod test {
    use super::*;
    use crate::factory::TensorFactory;
    use tensor::Tensor;
    pub use torch_stable::StableTorchResult;

    #[test]
    fn test_flash_powder_create_error() -> StableTorchResult<()> {
        let a = Tensor::zeros(&[usize::MAX, 5], &Default::default());

        assert!(a.is_err());
        let failure = a.err().unwrap();
        let v = failure.to_string();
        println!("v: {v}");
        assert!(v.contains("(zeros: Dimension size must be non-negative.)"));

        Ok(())
    }
}

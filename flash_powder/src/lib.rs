//!
//! Main building blocks:
//!
//! - [`Tensor`]; Owning tensor, this owns the data, created with [`NativeFunctionsOwned`][native_functions::NativeFunctionsOwned]. (think `Vec<u8>`)
//! - [`Ten<'_>`]; Const borrow of Tensor, this has a parent, its lifetime cannot exceed the parent. (think `&[u8]`)
//! - [`TenMut<'_>`]; Mutable borrow of Tensor, this has a mutable parent, its lifetime cannot exceed the parent. (think `&mut [u8]`)
//!
//! All of these provide [`TensorAccess`] and all functions and methods are implemented on that trait.
//!
//! - [`methods::TensorMethods`]: Methods to retrieve tensor properties like dimension and size.
//! - [`data`]`::{`[`DataRef`][`data::DataRef`], [`DataMut`][`data::DataMut`]`}`: Traits to access the tensor's data as bytes or other types.
//! - [`native_functions`]`::`[`NativeFunctions`][`native_functions::NativeFunctions`]: Methods / Functions on [`TensorAccess`] that require const access.
//! - [`native_functions`]`::`[`NativeFunctionsMut`][`native_functions::NativeFunctionsMut`]: Methods / Functions on [`TensorAccess`] that require mutable access.
//!
//!
//! Other principles;
//! - No unsafe in the public interface, safe behaviour as you'd expect.
//! - No interior mutability, all methods are const correct.
//! - Modifying one tensor will not modify another, unless it has a mutable borrow.
//! - Rust style lifetimes on tensors, either tied together with an explicit lifetime, or completely separate.

pub mod conversion;
pub mod data;
pub mod factory;
pub mod functional;
pub mod methods;
pub mod native_functions;
pub mod tensor;
use tensor::{Ten, TenMut, Tensor, TensorAccess};
pub use torch_stable::StableTorchResult;
pub mod prelude {
    use super::*;
    #[doc(inline)]
    pub use data::{DataMut, DataRef};
    #[doc(inline)]
    pub use factory::NativeFunctionsOwned;
    #[doc(inline)]
    pub use methods::TensorMethods;
    #[doc(inline)]
    pub use native_functions::{NativeFunctions, NativeFunctionsMut};
    #[doc(inline)]
    pub use tensor::{Ten, TenMut, Tensor, TensorAccess};

    #[doc(inline)]
    pub use crate::functional;
}
#[cfg(test)]
mod test {
    use super::*;
    use crate::factory::NativeFunctionsOwned;
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

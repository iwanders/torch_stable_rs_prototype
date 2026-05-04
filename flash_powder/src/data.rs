//! Accessors to the data in the Tensor.
//!
//! These only work if the Tensor is on the cpu.
use anyhow::bail;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

use crate::tensor::{Ten, TenMut, Tensor, TensorAccess};
use torch_stable::StableTorchResult;

pub trait DataRef: TensorAccess {
    fn u8_ref(&self) -> StableTorchResult<&[u8]> {
        let z = self.get_tensor();
        let element_size = z.element_size();
        let elements = z.numel();
        let data_ptr = z.const_data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }
    fn f32_ref(&self) -> StableTorchResult<&[f32]> {
        let byte_ref = self.u8_ref()?;
        use zerocopy::TryFromBytes;
        match <[f32]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn d_ref<T: IntoBytes + TryFromBytes + Immutable>(&self) -> StableTorchResult<&[T]> {
        let byte_ref = self.u8_ref()?;
        match <[T]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
}
impl DataRef for Tensor {}
impl<'a> DataRef for Ten<'a> {}
impl<'a> DataRef for TenMut<'a> {}

pub trait DataMut: TensorAccess {
    fn u8_mut(&mut self) -> StableTorchResult<&mut [u8]> {
        let z = self.get_tensor_mut();
        let element_size = z.element_size();
        let elements = z.numel();
        let data_ptr = z.mutable_data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts_mut(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    fn f32_mut(&mut self) -> StableTorchResult<&mut [f32]> {
        let byte_ref = self.u8_mut()?;
        match <[f32]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn d_mut<T: IntoBytes + TryFromBytes + Immutable>(&mut self) -> StableTorchResult<&mut [T]> {
        let byte_ref = self.u8_mut()?;
        match <[T]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
}
impl DataMut for Tensor {}
impl<'a> DataMut for TenMut<'a> {}

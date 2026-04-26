use anyhow::bail;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

use crate::tensor::{Ten, TenMut, Tensor, TensorAccess};
use torch_stable::StableTorchResult;

pub trait DataAccess: TensorAccess {
    fn data_ref(&self) -> StableTorchResult<&[u8]> {
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
        let byte_ref = self.data_ref()?;
        use zerocopy::TryFromBytes;
        match <[f32]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn t_ref<T: IntoBytes + TryFromBytes + Immutable>(&self) -> StableTorchResult<&[T]> {
        let byte_ref = self.data_ref()?;
        match <[T]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
}
impl DataAccess for Tensor {}
impl<'a> DataAccess for Ten<'a> {}
impl<'a> DataAccess for TenMut<'a> {}

pub trait DataManipulationMut: TensorAccess {
    fn data_mut(&self) -> StableTorchResult<&mut [u8]> {
        let z = self.get_tensor();
        let element_size = z.element_size();
        let elements = z.numel();
        let data_ptr = z.mutable_data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts_mut(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    fn f32_mut(&self) -> StableTorchResult<&mut [f32]> {
        let byte_ref = self.data_mut()?;
        match <[f32]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn t_mut<T: IntoBytes + TryFromBytes + Immutable>(&mut self) -> StableTorchResult<&mut [T]> {
        let byte_ref = self.data_mut()?;
        match <[T]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
}
impl DataManipulationMut for Tensor {}
//impl<'a> DataManipulationMut for Ten<'a> {}
impl<'a> DataManipulationMut for TenMut<'a> {}

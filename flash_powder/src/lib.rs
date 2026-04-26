// What is closest to a torch that works through oxidization? It's magnesium flash powder.

// Strictly follow Rusts' semantics.
// Since it is; String / str
// It will be: Tensor / Ten
//
// We don't put the scalartype in the type, because we can multiply one type to another and have torch handle that.
// We don't put the device type in the type, because that can also be toggled with a global switch easily.

use anyhow::bail;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

pub mod native_functions;
pub mod tensor;
use torch_stable::aoti_torch::*;
use torch_stable::headeronly::core::ScalarType;
use torch_stable::stable::device::{Device, DeviceIndex};
use torch_stable::stable::ops::{EmtpyOptions, ToOptions};
use torch_stable::unsafe_call_dispatch_panic;
use torch_stable::{
    StableTorchResult,
    aoti_torch::{AtenTensorHandle, StableIValue, aoti_torch_zero_},
    stable::tensor::Tensor as StableTensor,
    unsafe_call_bail, unsafe_call_dispatch_bail,
};

use tensor::{Ten, TenMut, Tensor, TensorAccess};

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

    fn fill_tensor<T: TensorAccess>(&mut self, value: &T) -> StableTorchResult<()> {
        let mut stack: [StableIValue; 2] =
            [(self.get_tensor()).into(), (value.get_tensor()).into()];
        unsafe_call_dispatch_bail!("aten::fill", "Tensor", stack.as_mut_slice());
        Ok(())
    }
    fn fill_f64(&mut self, value: f64) -> StableTorchResult<()> {
        unsafe_call_bail!(aoti_torch_aten_fill__Scalar(self.get_tensor().get(), value));
        Ok(())
    }
}
impl DataManipulationMut for Tensor {}
impl<'a> DataManipulationMut for Ten<'a> {}
impl<'a> DataManipulationMut for TenMut<'a> {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_flash_powder_to() -> StableTorchResult<()> {
        let t = Tensor::zeros(
            &[5, 5],
            &EmtpyOptions {
                ..Default::default()
            },
        )?;
        assert_eq!(t.scalar_type(), ScalarType::Float);
        let orig = t.const_data_ptr();

        let z = t.to(&ToOptions {
            ..Default::default()
        })?;
        assert_eq!(z.storage_offset(), 0);
        assert_ne!(orig, z.const_data_ptr());

        Ok(())
    }

    #[test]
    fn test_flash_powder_create_error() -> StableTorchResult<()> {
        unsafe {
            torch_stable::stable::c::torch_exception_set_exception_printing(false);
        }
        let a = Tensor::zeros(
            &[usize::MAX, 5],
            &EmtpyOptions {
                ..Default::default()
            },
        );

        let error_msg_ptr =
            unsafe { torch_stable::stable::c::torch_exception_get_what_without_backtrace() };

        let error_msg = unsafe { std::ffi::CStr::from_ptr(error_msg_ptr) };
        let owned_error = error_msg.to_owned().into_string()?;
        assert_eq!(
            owned_error,
            "Trying to create tensor with negative dimension -1: [-1, 5]"
        );

        Ok(())
    }

    #[test]
    fn test_flash_powder_narrow() -> StableTorchResult<()> {
        let mut t = Tensor::zeros(
            &[5, 5],
            &EmtpyOptions {
                ..Default::default()
            },
        )?;

        let mut view_mut = t.narrow_mut(0, 0, 3)?;
        view_mut.fill_tensor(&Tensor::from_f32(3.3)?)?;
        println!("view_mut: {:?}", view_mut.f32_ref()?);

        view_mut.fill_f64(3.0)?;
        println!("view_mut: {:?}", view_mut.f32_ref()?);
        // println!("t: {:?}", t.f32_ref()?);

        let view = t.narrow(0, 0, 3)?;
        println!("view: {:?}", view.f32_ref()?);
        println!("t: {:?}", t.f32_ref()?);

        let mut z = t.clone();
        z.fill_f64(5.5)?;
        println!("t: {:?}", t.f32_ref()?);
        println!("z: {:?}", z.f32_ref()?);

        Ok(())
    }
}

use torch_stable::aoti_torch::*;
use torch_stable::headeronly::core::ScalarType;
use torch_stable::stable::device::{Device, DeviceIndex};
use torch_stable::stable::ops::{EmtpyOptions, ToOptions};
use torch_stable::unsafe_call_dispatch_panic;
use torch_stable::{
    aoti_torch::{AtenTensorHandle, StableIValue, aoti_torch_zero_},
    stable::tensor::Tensor as StableTensor,
    unsafe_call_bail, unsafe_call_dispatch_bail,
};

use crate::{StableTorchResult, Ten, TenMut, Tensor, TensorAccess};
pub trait NativeFunctions: TensorAccess {
    fn narrow(&self, dim: usize, start: usize, end: usize) -> StableTorchResult<Ten<'_>> {
        // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489
        let mut stack: [StableIValue; 4] = [
            self.get_tensor().into(),
            dim.into(),
            start.into(),
            end.into(),
        ];
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());
        Ok(Ten::new(&self.get_tensor(), stack[0].try_into()?))
    }
}
impl NativeFunctions for Tensor {}
impl<'a> NativeFunctions for Ten<'a> {}
//impl<'a> NativeFunctions for TenMut<'a> {}

pub trait NativeFunctionsMut: TensorAccess {
    fn narrow_mut(
        &mut self,
        dim: usize,
        start: usize,
        end: usize,
    ) -> StableTorchResult<TenMut<'_>> {
        let mut stack: [StableIValue; 4] = [
            self.get_tensor().into(),
            dim.into(),
            start.into(),
            end.into(),
        ];
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());
        Ok(TenMut::new(&self.get_tensor(), stack[0].try_into()?))
    }

    // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L2724
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
impl NativeFunctionsMut for Tensor {}
impl<'a> NativeFunctionsMut for Ten<'a> {}
impl<'a> NativeFunctionsMut for TenMut<'a> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;
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

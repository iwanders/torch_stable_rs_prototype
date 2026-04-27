use torch_stable::aoti_torch::*;
use torch_stable::headeronly::core::{Layout, ScalarType};
use torch_stable::stable::device::Device;
use torch_stable::stable::ops::{EmtpyOptions, ToOptions};
use torch_stable::{
    aoti_torch::{StableIValue, aoti_torch_zero_},
    stable::tensor::Tensor as StableTensor,
    unsafe_call_bail, unsafe_call_dispatch_bail,
};

// This file has three traits:
// - NativeFunctions; Implemented for Tensor, Ten and Tenmut
// - NativeFunctionsMut; Implemented for Tensor and TenMut, so not for Ten.
// - NativeFunctionsOwned; Implemented only for Tensor
//
// These are strictly from the native functions yaml.

#[derive(Copy, Clone, Debug, Default)]
pub struct ZeroOptions {
    pub dtype: Option<ScalarType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ConvOptions<'b, B: TensorAccess> {
    pub bias: Option<&'b B>,
    pub stride: Option<i64>,
    pub padding: Option<i64>,
    pub dilation: Option<i64>,
    pub groups: Option<i64>,
}

use crate::{StableTorchResult, Ten, TenMut, Tensor, TensorAccess};
pub trait NativeFunctions: TensorAccess
where
    for<'a> &'a Self: Into<StableIValue>,
{
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

    // https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L8033
    fn to(&self, options: &ToOptions) -> StableTorchResult<Tensor> {
        const MAKE_COPY: bool = true;
        let mut stack: [StableIValue; 8] = [
            self.get_tensor().into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            options.non_blocking.into(),
            MAKE_COPY.into(),
            (&options.memory_format).into(),
        ];
        unsafe_call_dispatch_bail!("aten::to", "dtype_layout", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Tensor::new(r))
    }
    // https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1757
    fn conv2d<T: TensorAccess>(
        &self,
        weight: &T,
        options: &ConvOptions<T>,
    ) -> StableTorchResult<Tensor>
    where
        for<'a> &'a T: Into<StableIValue>,
    {
        let mut stack: [StableIValue; 7] = [
            self.into(),
            weight.into(),
            (&options.bias).into(),
            (&options.stride).into(),
            (&options.padding).into(),
            (&options.dilation).into(),
            (&options.groups).into(),
        ];
        unsafe_call_dispatch_bail!("aten::to", "dtype_layout", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Tensor::new(r))
    }
}
impl NativeFunctions for Tensor {}
impl<'a> NativeFunctions for Ten<'a> {}
impl<'a> NativeFunctions for TenMut<'a> {}

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

pub trait NativeFunctionsOwned: TensorAccess {
    fn empty(dimensions: &[usize], options: &EmtpyOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 6] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            (&options.memory_format).into(),
        ];
        // https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L2424
        unsafe_call_dispatch_bail!("aten::empty", "memory_format", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        unsafe_call_bail!(aoti_torch_zero_(r.get()));

        Ok(Tensor::new(r))
    }
    // https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L6800
    fn zeros(dimensions: &[usize], options: &ZeroOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 5] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
        ];
        unsafe_call_dispatch_bail!("aten::zeros", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Tensor::new(r))
    }
}
impl NativeFunctionsOwned for Tensor {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;
    #[test]
    fn test_flash_powder_narrow() -> StableTorchResult<()> {
        let mut t = Tensor::zeros(&[5, 5], &Default::default())?;

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
    #[test]
    fn test_flash_powder_aten_empty() -> StableTorchResult<()> {
        let _ = Tensor::empty(&[5, 5], &Default::default())?.fill_f64(0.0);

        Ok(())
    }

    #[test]
    fn test_flash_powder_to() -> StableTorchResult<()> {
        let t = Tensor::zeros(
            &[5, 5],
            &ZeroOptions {
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
}

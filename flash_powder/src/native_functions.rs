//! This holds functions and methods from the pytorch `native_functions.yaml` file.
//!
//! See [native_functions.yaml@v2.12.0-rc7](https://github.com/pytorch/pytorch/blob/v2.12.0-rc7/aten/src/ATen/native/native_functions.yaml)
//! and its [readme](https://github.com/pytorch/pytorch/blob/v2.12.0-rc7/aten/src/ATen/native/README.md).

// https://docs.pytorch.org/docs/2.11/tensor_view.html
// has a nice overview of what operators return views.

use crate::methods::TensorMethods;
use anyhow::bail;
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

/// Options to create zero tensors.
#[derive(Copy, Clone, Debug, Default)]
pub struct ZeroOptions {
    pub dtype: Option<ScalarType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
}

/// Options for conv2d.
#[derive(Copy, Clone, Debug)]
pub struct Conv2DOptions {
    pub stride: (i64, i64),
    pub padding: (i64, i64),
    pub dilation: (i64, i64),
    pub groups: i64,
}
impl Default for Conv2DOptions {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

use crate::{StableTorchResult, Ten, TenMut, Tensor, TensorAccess};

/// Native functions that require const access.
///
/// See the [`native_functions`][crate::native_functions] module for description of this trait's functionality.
pub trait NativeFunctions: TensorAccess + TensorMethods {
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
        assert_ne!(self.data_ptr(), r.data_ptr());

        Ok(Tensor::new(r))
    }
    /// A 2d convolution.
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1757)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.ao.nn.quantized.functional.conv2d.html#conv2d)
    fn conv2d<T: TensorAccess>(
        &self, // input
        weight: &T,
        bias: Option<&T>,
        options: &Conv2DOptions,
    ) -> StableTorchResult<Tensor> {
        // return wrap(dispatch_conv2d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
        let mut stack: [StableIValue; 7] = [
            self.get_tensor().into(),
            weight.get_tensor().into(),
            (&bias.map(|z| z.get_tensor())).into(),
            (&options.stride).into(),
            (&options.padding).into(),
            (&options.dilation).into(),
            (&options.groups).into(),
        ];
        unsafe_call_dispatch_bail!("aten::conv2d", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;

        Ok(Tensor::new(r))
    }

    /// View into a tensor
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L8362)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.view.html)
    ///
    fn view(&mut self, shape: &[usize]) -> StableTorchResult<Ten<'_>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), (shape).into()];
        unsafe_call_dispatch_bail!("aten::view", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.data_ptr(), r.data_ptr());
        Ok(Ten::new(self.get_tensor(), r))
    }
}
impl NativeFunctions for Tensor {}
impl<'a> NativeFunctions for Ten<'a> {}
impl<'a> NativeFunctions for TenMut<'a> {}

/// Native functions that require mutable access.
///
/// See the [`native_functions`][crate::native_functions] module for description of this trait's functionality.
pub trait NativeFunctionsMut: TensorAccess + TensorMethods {
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
        // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());

        Ok(TenMut::new(self.get_tensor_mut(), stack[0].try_into()?))
    }

    // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L2730
    fn fill_tensor<T: TensorAccess>(&mut self, value: &T) -> StableTorchResult<()> {
        let mut stack: [StableIValue; 2] =
            [(self.get_tensor()).into(), (value.get_tensor()).into()];
        unsafe_call_dispatch_bail!("aten::fill", "Tensor", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        let _ = Tensor::new(r);
        Ok(())
    }
    fn fill_f64(&mut self, value: f64) -> StableTorchResult<()> {
        unsafe_call_bail!(aoti_torch_aten_fill__Scalar(self.get_tensor().get(), value));
        Ok(())
    }

    /// View into a tensor
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L8362)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.view.html)
    ///
    fn view_mut(&mut self, shape: &[usize]) -> StableTorchResult<TenMut<'_>> {
        let mut stack: [StableIValue; 2] = [(self.get_tensor()).into(), (shape).into()];
        unsafe_call_dispatch_bail!("aten::view", "", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        assert_eq!(self.data_ptr(), r.data_ptr());
        Ok(TenMut::new(self.get_tensor_mut(), r))
    }
}
impl NativeFunctionsMut for Tensor {}
impl<'a> NativeFunctionsMut for Ten<'a> {}
impl<'a> NativeFunctionsMut for TenMut<'a> {}

/// Native functions that produce owned tensors.
///
/// See the [`native_functions`][crate::native_functions] module for description of this trait's functionality.
pub trait NativeFunctionsOwned: TensorAccess + TensorMethods {
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

    fn from_f32(value: f32) -> StableTorchResult<Tensor> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(
            torch_stable::aoti_torch::aoti_torch_scalar_to_tensor_float32(value, &mut handle_res)
        );
        Ok(Tensor::new(StableTensor::from_handle(handle_res)))
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
        //view_mut.fill_f64(3.3)?;
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

    #[test]
    fn test_flash_power_conv2d() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([1,4,4])
            INPUT_DATA_PY = list(d.view(-1).tolist())
            w = torch.tensor([[[1.0, 2.0],[3.0, 4.0]]]).unsqueeze(0)
            r = torch.nn.functional.conv2d(d, w)
        */

        let mut d = Tensor::zeros(&[1, 4, 4], &Default::default())?;
        for (i, v) in d.f32_mut()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }
        const INPUT_DATA_PY: &[f32] = &[
            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        assert_eq!(d.sizes(), &[1, 4, 4]); // #PYTHON list(d.shape))
        assert_eq!(
            d.f32_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        /*
            #|PYTHON

            R = 5
        */

        const R: u32 = 5;
        let mut w = Tensor::zeros(&[1, 1, 2, 2], &Default::default())?;
        for (i, v) in w.d_mut::<f32>()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }
        assert_eq!(w.sizes(), &[1, 1, 2, 2]);

        let r = d.conv2d(&w, None, &Default::default())?;
        assert_eq!(r.sizes(), &[1, 3, 3,]);

        assert_eq!(
            r.f32_ref()?,
            &[44.0f32, 54.0, 64.0, 84.0, 94.0, 104.0, 124.0, 134.0, 144.0]
        );

        /*
         *
         r = torch.nn.functional.conv2d(d, w, stride=(1, 2))
         print(r.shape)
         print(r.reshape(-1))
         """
         torch.Size([1, 3, 2])
         tensor([ 44,  64,  84, 104, 124, 144])
         """
        */
        let r = d.conv2d(
            &w,
            None,
            &Conv2DOptions {
                stride: (1, 2),
                ..Default::default()
            },
        )?;
        assert_eq!(r.sizes(), &[1, 3, 2]);
        assert_eq!(r.f32_ref()?, &[44.0f32, 64.0, 84.0, 104.0, 124.0, 144.0]);

        Ok(())
    }

    #[test]
    fn test_flash_power_view() -> StableTorchResult<()> {
        let mut d = Tensor::zeros(&[16], &Default::default())?;
        for (i, v) in d.f32_mut()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }

        let mut a = d.view_mut(&[4, 4])?;

        assert_eq!(a.sizes(), &[4, 4]);
        assert_eq!(
            a.f32_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        );
        a.f32_mut()?[0] = 50.0;

        assert_eq!(a.f32_mut()?[0], 50.0);
        assert_eq!(a.f32_ref()?[0], 50.0);

        let mut n = a.to_owned()?;
        // Currently lazy copy
        let old_n_ptr = n.const_data_ptr();
        assert_eq!(n.const_data_ptr(), a.const_data_ptr());

        // Verify n holds same data
        assert_eq!(n.f32_ref()?[0], 50.0);
        // Modify n, this performs the copy.
        n.f32_mut()?[0] = 20.0;

        // data pointer shouldn't be the same now.
        assert_ne!(n.const_data_ptr(), old_n_ptr);

        assert_eq!(d.f32_mut()?[0], 50.0);

        // Try a non owning view
        let v = d.view(&[16])?;
        let mut cv = v.to_owned()?;
        cv.f32_mut()?[0] = 10.0;
        assert_eq!(cv.f32_ref()?[0], 10.0);
        assert_eq!(v.f32_ref()?[0], 50.0);

        // Reshape to incorrect size.
        assert!(d.view(&[12]).is_err());

        Ok(())
    }
}

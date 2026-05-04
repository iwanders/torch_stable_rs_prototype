//! This holds the core methods on the tensor object.
//!
//! Most of them originate from the yaml `native_functions.yaml` file.
//! See [native_functions.yaml@v2.12.0-rc7](https://github.com/pytorch/pytorch/blob/v2.12.0-rc7/aten/src/ATen/native/native_functions.yaml)
//! and its [readme](https://github.com/pytorch/pytorch/blob/v2.12.0-rc7/aten/src/ATen/native/README.md).
//!
//! Its readme states;
//! > Tensor operations as methods are appropriate for "core" Tensor operations (e.g., add, sub, etc.), but not for more complicated neural network layers (e.g., conv2d)
//!
//! This module holds the methods that are considered core tensor operations.

// https://docs.pytorch.org/docs/2.11/tensor_view.html
// has a nice overview of what operators return views.
//
// Hmm... from https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/README.md
// We should probably follow that guidance and kick conv2d to a functional module.
//
// The foo_ underscore methods modify data in place, see https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/README.md#annotations

use crate::properties::TensorProperties;
use crate::{StableTorchResult, Ten, TenMut, Tensor, TensorAccess};
use torch_stable::aoti_torch::*;
use torch_stable::stable::ops::ToOptions;
use torch_stable::{
    aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor, unsafe_call_bail,
    unsafe_call_dispatch_bail,
};

/// Core methods that require const access.
///
/// See the [`core_methods`][crate::core_methods] module for description of this trait's functionality.
pub trait CoreMethods: TensorAccess + TensorProperties {
    /// Narrow view
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.narrow.html)
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

    /// To
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L8033)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.to.html)
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
impl CoreMethods for Tensor {}
impl<'a> CoreMethods for Ten<'a> {}
impl<'a> CoreMethods for TenMut<'a> {}

/// Core methods that require mutable access.
///
/// See the [`core_methods`][crate::core_methods] module for description of this trait's functionality.
pub trait CoreMethodsMut: TensorAccess + TensorProperties {
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

    /// Fill a tensor with another tensor.
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L2730)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.fill_.html)
    fn fill_tensor<T: TensorAccess>(&mut self, value: &T) -> StableTorchResult<()> {
        let mut stack: [StableIValue; 2] =
            [(self.get_tensor()).into(), (value.get_tensor()).into()];
        unsafe_call_dispatch_bail!("aten::fill_", "Tensor", stack.as_mut_slice());
        let r: StableTensor = stack[0].try_into()?;
        let retrieve = Tensor::new(r);
        assert_eq!(retrieve.data_ptr(), self.data_ptr());
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
impl CoreMethodsMut for Tensor {}
impl<'a> CoreMethodsMut for Ten<'a> {}
impl<'a> CoreMethodsMut for TenMut<'a> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_fill() -> StableTorchResult<()> {
        /*
            #|PYTHON
            t = torch.zeros([2,2])
            t.fill_(3.0)
            v = torch.tensor(5.0)
        */

        let mut t = Tensor::zeros(&[2, 2], &Default::default())?;
        assert_eq!(t.sizes(), &[2, 2]); // #PYTHON list(t.shape)

        let v = Tensor::from_f32(5.0)?;
        assert_eq!(v.f32_ref()?, &[5.0f32]); // #PYTHON list(v.view(-1).tolist())

        /*
            #|PYTHON
            t.fill_(v)
        */

        t.fill_tensor(&v)?;
        assert_eq!(t.f32_ref()?, &[5.0f32, 5.0, 5.0, 5.0]); // #PYTHON list(t.view(-1).tolist())

        Ok(())
    }

    #[test]
    fn test_flash_powder_narrow() -> StableTorchResult<()> {
        /*
            #|PYTHON
            t = torch.tensor(list(range(1,10)), dtype=torch.float).reshape([3,3])
            v = t.narrow(0, 0, 3)
            v.fill_(3.0)
            nv = t.narrow(0, 0, 3)
        */

        let mut t = Tensor::zeros(&[3, 3], &Default::default())?;
        assert_eq!(t.sizes(), &[3, 3]); // #PYTHON list(t.shape)

        let mut view_mut = t.narrow_mut(0, 0, 3)?;
        view_mut.fill_tensor(&Tensor::from_f32(3.0)?)?;
        assert_eq!(view_mut.sizes(), &[3, 3]); // #PYTHON list(v.shape)
        assert_eq!(
            view_mut.f32_ref()?,
            &[3.0f32, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        ); // #PYTHON list(v.view(-1).tolist())

        drop(view_mut);

        let view = t.narrow(0, 0, 3)?;
        assert_eq!(
            view.f32_ref()?,
            &[3.0f32, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        ); // #PYTHON list(nv.view(-1).tolist())

        Ok(())
    }
    #[test]
    fn test_flash_powder_aten_empty() -> StableTorchResult<()> {
        let _ = Tensor::empty(&[5, 5], &Default::default())?.fill_f64(0.0);
        Ok(())
    }

    #[test]
    fn test_flash_powder_to() -> StableTorchResult<()> {
        use crate::ScalarType;
        use crate::factory::ZeroOptions;
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
    fn test_flash_power_view() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([4,4])
        */
        let mut d = Tensor::zeros(&[16], &Default::default())?;
        for (i, v) in d.f32_mut()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }

        let mut a = d.view_mut(&[4, 4])?;

        assert_eq!(a.sizes(), &[4, 4]); // #PYTHON list(d.shape)
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

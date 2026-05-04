use crate::properties::TensorProperties;
use crate::{StableTorchResult, Ten, TenMut, Tensor, TensorAccess};
use torch_stable::aoti_torch::{AtenTensorHandle, aoti_torch_zero_};
use torch_stable::headeronly::core::{Layout, ScalarType};
use torch_stable::stable::device::Device;
pub use torch_stable::stable::ops::{EmtpyOptions, ToOptions};
use torch_stable::{
    aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor, unsafe_call_bail,
    unsafe_call_dispatch_bail,
};

/// Options to create zero tensors.
#[derive(Copy, Clone, Debug, Default)]
pub struct ZeroOptions {
    pub dtype: Option<ScalarType>,
    pub layout: Option<Layout>,
    pub device: Option<Device>,
    pub pin_memory: Option<bool>,
}

/// Native functions that produce owned tensors.
///
/// See the [`native_functions`][crate::native_functions] module for description of this trait's functionality.
pub trait TensorFactory: TensorAccess + TensorProperties {
    /// A new empty vector
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L2425)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.empty.html#torch.empty)
    ///
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
    /// A new empty vector
    ///
    ///
    /// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L6837)
    /// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.zeros.html)
    ///
    //
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
impl TensorFactory for Tensor {}

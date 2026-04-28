use torch_stable::unsafe_call_dispatch_panic;
use torch_stable::{
    StableTorchResult,
    aoti_torch::{AtenTensorHandle, StableIValue},
    stable::tensor::Tensor as StableTensor,
    unsafe_call_bail,
};
pub struct Tensor {
    tensor: StableTensor,
}
impl Clone for Tensor {
    fn clone(&self) -> Self {
        // Clone cannot throw... so we use a lazy clone; https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1278
        let mut stack: [StableIValue; 1] = [(&self.tensor).into()];
        unsafe_call_dispatch_panic!("aten::_lazy_clone", "", stack.as_mut_slice());
        let r: Tensor = Self::new(stack[0].try_into().unwrap());

        r
    }
}

impl Tensor {
    pub fn new(tensor: StableTensor) -> Self {
        Self { tensor }
    }

    pub fn from_f32(value: f32) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(
            torch_stable::aoti_torch::aoti_torch_scalar_to_tensor_float32(value, &mut handle_res)
        );
        Ok(Self {
            tensor: StableTensor::from_handle(handle_res),
        })
    }
}

pub struct Ten<'a> {
    // This is the backing tensor that shares data with the 'parent'.
    tensor: StableTensor,
    _parent: &'a StableTensor,
}
impl<'a> Ten<'a> {
    pub fn new(parent: &'a StableTensor, tensor: StableTensor) -> Self {
        Self {
            _parent: parent,
            tensor,
        }
    }
}
pub struct TenMut<'a> {
    // This is the backing tensor that shares data with the 'parent'.
    tensor: StableTensor,
    _parent: &'a StableTensor,
}
impl<'a> TenMut<'a> {
    pub fn new(parent: &'a StableTensor, tensor: StableTensor) -> Self {
        Self {
            _parent: parent,
            tensor,
        }
    }
}

/*

use crate::{StableTorchResult, Ten, TenMut, Tensor, TensorAccess};
pub trait NativeFunctions: TensorAccess
where
    for<'a> &'a Self: Into<StableIValue>,
{
// followed by
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
*/
pub trait TensorAccess {
    fn get_tensor(&self) -> &StableTensor;
}

impl<'a> TensorAccess for TenMut<'a> {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
}

impl<'a> TensorAccess for Ten<'a> {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
}
impl TensorAccess for Tensor {
    fn get_tensor(&self) -> &StableTensor {
        &self.tensor
    }
}

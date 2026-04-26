use crate::headeronly::core::ScalarType;
use crate::stable::device::{Device, DeviceIndex};
use crate::stable::ops::{EmtpyOptions, ToOptions};
use crate::util::unsafe_call_dispatch_panic;
use crate::{
    StableTorchResult,
    aoti_torch::{AtenTensorHandle, StableIValue, aoti_torch_zero_},
    stable::tensor::Tensor as StableTensor,
    unsafe_call_bail, unsafe_call_dispatch_bail,
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
    fn new(tensor: StableTensor) -> Self {
        Self { tensor }
    }

    fn get(&self) -> AtenTensorHandle {
        self.tensor.get()
    }
    pub fn dim(&self) -> usize {
        self.tensor.dim()
    }
    pub fn scalar_type(&self) -> ScalarType {
        self.tensor.scalar_type()
    }
    pub fn element_size(&self) -> usize {
        self.tensor.element_size()
    }
    pub fn device(&self) -> Device {
        self.tensor.device()
    }
    pub fn storage_offset(&self) -> usize {
        self.tensor.storage_offset()
    }
    pub fn numel(&self) -> usize {
        self.tensor.numel()
    }
    pub fn is_cpu(&self) -> bool {
        self.tensor.is_cpu()
    }

    pub fn zeros(dimensions: &[usize], options: &EmtpyOptions) -> StableTorchResult<Tensor> {
        let mut stack: [StableIValue; 6] = [
            (dimensions).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            (&options.memory_format).into(),
        ];
        unsafe_call_dispatch_bail!("aten::empty", "memory_format", stack.as_mut_slice());
        let r: Tensor = Self::new(stack[0].try_into()?);

        unsafe_call_bail!(aoti_torch_zero_(r.get()));

        Ok(r)
    }

    pub fn to(&self, options: &ToOptions) -> StableTorchResult<Tensor> {
        const MAKE_COPY: bool = true;
        let mut stack: [StableIValue; 8] = [
            (&self.tensor).into(),
            (&options.dtype).into(),
            (&options.layout).into(),
            (&options.device).into(),
            (&options.pin_memory).into(),
            options.non_blocking.into(),
            MAKE_COPY.into(),
            (&options.memory_format).into(),
        ];
        unsafe_call_dispatch_bail!("aten::to", "dtype_layout", stack.as_mut_slice());
        Ok(Self::new(stack[0].try_into()?))
    }

    pub fn const_data_ptr(&self) -> *const u8 {
        self.tensor.const_data_ptr()
    }

    pub fn mutable_data_ptr(&self) -> *mut u8 {
        self.tensor.mutable_data_ptr()
    }

    pub fn from_f32(value: f32) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(crate::aoti_torch::aoti_torch_scalar_to_tensor_float32(
            value,
            &mut handle_res
        ));
        Ok(Self {
            tensor: StableTensor::from_handle(handle_res),
        })
    }
    pub fn narrow(&self, dim: usize, start: usize, end: usize) -> StableTorchResult<Ten<'_>> {
        // https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L4489
        let mut stack: [StableIValue; 4] =
            [(&self.tensor).into(), dim.into(), start.into(), end.into()];
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());
        Ok(Ten::new(&self.tensor, stack[0].try_into()?))
    }
    pub fn narrow_mut(
        &mut self,
        dim: usize,
        start: usize,
        end: usize,
    ) -> StableTorchResult<TenMut<'_>> {
        let mut stack: [StableIValue; 4] =
            [(&self.tensor).into(), dim.into(), start.into(), end.into()];
        unsafe_call_dispatch_bail!("aten::narrow", "", stack.as_mut_slice());
        Ok(TenMut::new(&self.tensor, stack[0].try_into()?))
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

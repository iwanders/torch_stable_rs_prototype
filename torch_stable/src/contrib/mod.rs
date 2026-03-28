// This holds things that aren't actually part of the upstream stable API, but make sense.

use crate::aoti_torch::AtenTensorHandle;
use crate::aoti_torch::*;
use crate::stable::tensor::Tensor;
use crate::{StableTorchResult, unsafe_call_bail};

pub trait FromScalar {
    fn from_f32(value: f32) -> StableTorchResult<Tensor>;
}

impl FromScalar for Tensor {
    fn from_f32(value: f32) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_scalar_to_tensor_float32(value, &mut handle_res));
        Ok(Tensor::from_handle(handle_res))
    }
}

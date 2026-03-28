// This holds things that aren't actually part of the upstream stable API, but make sense.

use crate::aoti_torch::AtenTensorHandle;
use crate::aoti_torch::*;
use crate::stable::tensor::Tensor;
use crate::{StableTorchResult, unsafe_call_bail};

pub trait FromScalar {
    fn from_f32(value: f32) -> StableTorchResult<Tensor>;
    fn from_f64(value: f64) -> StableTorchResult<Tensor>;
}

impl FromScalar for Tensor {
    fn from_f32(value: f32) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_scalar_to_tensor_float32(value, &mut handle_res));
        Ok(Tensor::from_handle(handle_res))
    }
    fn from_f64(value: f64) -> StableTorchResult<Self> {
        let mut handle_res: AtenTensorHandle = std::ptr::null_mut();
        unsafe_call_bail!(aoti_torch_scalar_to_tensor_float64(value, &mut handle_res));
        Ok(Tensor::from_handle(handle_res))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::headeronly::core::{Layout, ScalarType};

    use crate::stable::device::{Device, DeviceIndex, DeviceType};
    #[test]
    fn test_tensor_from_scalar() {
        use crate::contrib::FromScalar;
        let t = Tensor::from_f32(std::f32::consts::PI).unwrap();
        assert_eq!(t.dim(), 0);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.sizes(), &[]);
        assert_eq!(t.strides(), &[]);
        assert_eq!(t.is_contiguous(), true);
        // assert_eq!(t.stride(0), 0); // no dimensions
        assert_eq!(t.get_device_index(), DeviceIndex(-1));
        assert_eq!(t.defined(), true);
        assert_eq!(t.scalar_type(), ScalarType::Float);
        assert_eq!(t.device(), Device::from_str("cpu").unwrap());
        let t = Tensor::from_f64(std::f64::consts::PI).unwrap();
        assert_eq!(t.scalar_type(), ScalarType::Double);
    }
}

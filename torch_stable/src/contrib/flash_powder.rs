// What is closest to a torch that works through oxidization? It's a flash powder, think magnesium powder burning.

// Strictly follow Rusts' semantics.
// Since it is; String / str
// It will be: Tensor / Ten
//
//

use crate::headeronly::core::ScalarType;
use crate::stable::device::{Device, DeviceIndex};
use crate::stable::ops::{EmtpyOptions, ToOptions};
use crate::{
    StableTorchResult,
    aoti_torch::{AtenTensorHandle, StableIValue, aoti_torch_zero_},
    stable::tensor::Tensor as StableTensor,
    unsafe_call_bail, unsafe_call_dispatch_bail,
};

pub struct Tensor {
    tensor: StableTensor,
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
}

pub struct Ten<'a> {
    // This is the backing tensor that shares data with the 'parent'.
    tensor: StableTensor,
    lifetime: &'a (),
}

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
}

// What is closest to a torch that works through oxidization? It's a flash powder, think magnesium powder burning.

// Strictly follow Rusts' semantics.
// Since it is; String / str
// It will be: Tensor / Ten
//
//

use anyhow::bail;
use zerocopy::{Immutable, IntoBytes, TryFromBytes};

use crate::aoti_torch::*;
use crate::headeronly::core::ScalarType;
use crate::stable::device::{Device, DeviceIndex};
use crate::stable::ops::{EmtpyOptions, ToOptions};
use crate::{
    StableTorchResult,
    aoti_torch::{AtenTensorHandle, StableIValue, aoti_torch_zero_},
    stable::tensor::Tensor as StableTensor,
    unsafe_call_bail, unsafe_call_dispatch_bail,
};

// We don't put the scalartype in the type, because we can multiply one type to another and have torch handle that.
// We don't put the device type in the type, because that can also be toggled with a global switch easily.

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
        unsafe_call_bail!(aoti_torch_scalar_to_tensor_float32(value, &mut handle_res));
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
    parent: &'a StableTensor,
}
impl<'a> Ten<'a> {
    pub fn new(parent: &'a StableTensor, tensor: StableTensor) -> Self {
        Self { parent, tensor }
    }
}
pub struct TenMut<'a> {
    // This is the backing tensor that shares data with the 'parent'.
    tensor: StableTensor,
    parent: &'a StableTensor,
}
impl<'a> TenMut<'a> {
    pub fn new(parent: &'a StableTensor, tensor: StableTensor) -> Self {
        Self { parent, tensor }
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
pub trait DataAccess: TensorAccess {
    fn data_ref(&self) -> StableTorchResult<&[u8]> {
        let z = self.get_tensor();
        let element_size = z.element_size();
        let elements = z.numel();
        let data_ptr = z.const_data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }
    fn f32_ref(&self) -> StableTorchResult<&[f32]> {
        let byte_ref = self.data_ref()?;
        use zerocopy::TryFromBytes;
        match <[f32]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn t_ref<T: IntoBytes + TryFromBytes + Immutable>(&self) -> StableTorchResult<&[T]> {
        let byte_ref = self.data_ref()?;
        match <[T]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
}
impl DataAccess for Tensor {}
impl<'a> DataAccess for Ten<'a> {}
impl<'a> DataAccess for TenMut<'a> {}

pub trait DataManipulationMut: TensorAccess {
    fn data_mut(&self) -> StableTorchResult<&mut [u8]> {
        let z = self.get_tensor();
        let element_size = z.element_size();
        let elements = z.numel();
        let data_ptr = z.mutable_data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts_mut(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    fn f32_mut(&self) -> StableTorchResult<&mut [f32]> {
        let byte_ref = self.data_mut()?;
        match <[f32]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn t_mut<T: IntoBytes + TryFromBytes + Immutable>(&mut self) -> StableTorchResult<&mut [T]> {
        let byte_ref = self.data_mut()?;
        match <[T]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

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
impl DataManipulationMut for Tensor {}
impl<'a> DataManipulationMut for Ten<'a> {}
impl<'a> DataManipulationMut for TenMut<'a> {}

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

        Ok(())
    }
}

use crate::data::DataMut;
use crate::native_functions::{EmtpyOptions, NativeFunctionsOwned};
use crate::tensor::Tensor;
use torch_stable::headeronly::core::ScalarType;
pub trait TensorScalar {
    fn tensor_scalar_type() -> ScalarType;
}
macro_rules! impl_tensor_scalar_trait {
    ($t:ty, $v:path) => {
        impl TensorScalar for $t {
            fn tensor_scalar_type() -> ScalarType {
                $v
            }
        }
    };
}

impl_tensor_scalar_trait!(f32, ScalarType::Float);
impl_tensor_scalar_trait!(f64, ScalarType::Double);
// impl_tensor_scalar_trait!(f16, ScalarType::Half);
// https://github.com/pytorch/pytorch/blob/6a357dd272853cb6567bb277da62750013c76b4a/torch/csrc/stable/stableivalue_conversions.h#L114
impl_tensor_scalar_trait!(u8, ScalarType::Byte);
impl_tensor_scalar_trait!(i8, ScalarType::Char);
impl_tensor_scalar_trait!(u16, ScalarType::UInt16);
impl_tensor_scalar_trait!(i16, ScalarType::Short);
impl_tensor_scalar_trait!(i32, ScalarType::Int);
impl_tensor_scalar_trait!(u32, ScalarType::UInt32);
impl_tensor_scalar_trait!(i64, ScalarType::Long);
impl_tensor_scalar_trait!(u64, ScalarType::UInt64);
impl_tensor_scalar_trait!(bool, ScalarType::Bool);

use zerocopy::{Immutable, IntoBytes, TryFromBytes};

impl<T: TensorScalar + Immutable + IntoBytes + TryFromBytes + Copy> TryInto<Tensor> for &[T] {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[self.len()],
            &EmtpyOptions {
                dtype: Some(T::tensor_scalar_type()),
                ..Default::default()
            },
        )?;
        v.d_mut::<T>()?.copy_from_slice(&self);
        Ok(v)
    }
}

impl<T: TensorScalar + Immutable + IntoBytes + TryFromBytes + Copy, const V: usize> TryInto<Tensor>
    for [T; V]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[V],
            &EmtpyOptions {
                dtype: Some(T::tensor_scalar_type()),
                ..Default::default()
            },
        )?;
        v.d_mut::<T>()?.copy_from_slice(&self);
        Ok(v)
    }
}
// And its ref;
impl<T: TensorScalar + Immutable + IntoBytes + TryFromBytes + Copy, const V: usize> TryInto<Tensor>
    for &[T; V]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[V],
            &EmtpyOptions {
                dtype: Some(T::tensor_scalar_type()),
                ..Default::default()
            },
        )?;
        v.d_mut::<T>()?.copy_from_slice(self);
        Ok(v)
    }
}

impl<T: TensorScalar + Immutable + IntoBytes + TryFromBytes + Copy, const C: usize, const R: usize>
    TryInto<Tensor> for [[T; C]; R]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[R, C],
            &EmtpyOptions {
                dtype: Some(T::tensor_scalar_type()),
                ..Default::default()
            },
        )?;
        v.u8_mut()?.copy_from_slice(&self.as_bytes());
        Ok(v)
    }
}
// and its ref;
impl<T: TensorScalar + Immutable + IntoBytes + TryFromBytes + Copy, const C: usize, const R: usize>
    TryInto<Tensor> for &[[T; C]; R]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[R, C],
            &EmtpyOptions {
                dtype: Some(T::tensor_scalar_type()),
                ..Default::default()
            },
        )?;
        v.u8_mut()?.copy_from_slice(&self.as_bytes());
        Ok(v)
    }
}

impl<
    T: TensorScalar + Immutable + IntoBytes + TryFromBytes + Copy,
    const C: usize,
    const R: usize,
    const D: usize,
> TryInto<Tensor> for [[[T; C]; R]; D]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[D, R, C],
            &EmtpyOptions {
                dtype: Some(T::tensor_scalar_type()),
                ..Default::default()
            },
        )?;
        v.u8_mut()?.copy_from_slice(&self.as_bytes());
        Ok(v)
    }
}
// and its ref;
impl<
    T: TensorScalar + Immutable + IntoBytes + TryFromBytes + Copy,
    const C: usize,
    const R: usize,
    const D: usize,
> TryInto<Tensor> for &[[[T; C]; R]; D]
{
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Tensor, Self::Error> {
        let mut v = Tensor::empty(
            &[D, R, C],
            &EmtpyOptions {
                dtype: Some(T::tensor_scalar_type()),
                ..Default::default()
            },
        )?;
        v.u8_mut()?.copy_from_slice(&self.as_bytes());
        Ok(v)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::StableTorchResult;
    use crate::data::DataRef;
    use crate::methods::TensorMethods;

    #[test]
    fn test_tensor_try_from() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor([5, 3])
        */
        let d: Tensor = [5i64, 3].try_into()?;
        assert_eq!(d.sizes(), &[2]); // #PYTHON list(d.shape)
        assert_eq!(
            d.u8_ref()?,
            &[5, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Long); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([5.0, 3.0])
        */
        let d: Tensor = (&[5.0f32, 3.0]).try_into()?;
        assert_eq!(d.sizes(), &[2]); // #PYTHON list(d.shape)
        assert_eq!(d.u8_ref()?, &[0, 0, 160, 64, 0, 0, 64, 64]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Float); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([[5.0, 3.0], [1.0, 2.0]])
        */
        let d: Tensor = [[5.0f32, 3.0], [1.0, 2.0]].try_into()?;
        assert_eq!(d.sizes(), &[2, 2]); // #PYTHON list(d.shape)
        assert_eq!(
            d.u8_ref()?,
            &[0, 0, 160, 64, 0, 0, 64, 64, 0, 0, 128, 63, 0, 0, 0, 64]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Float); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([1, 3, 4, 5], dtype=torch.int8)
        */
        let d: Tensor = [1i8, 3, 4, 5].try_into()?;
        assert_eq!(d.sizes(), &[4]); // #PYTHON list(d.shape)
        assert_eq!(d.u8_ref()?, &[1, 3, 4, 5]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Char); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([[[5]]])
        */
        let d: Tensor = [[[5i64]]].try_into()?;
        assert_eq!(d.sizes(), &[1, 1, 1]); // #PYTHON list(d.shape)
        assert_eq!(d.u8_ref()?, &[5, 0, 0, 0, 0, 0, 0, 0]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Long); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([True, False, True])
        */
        let d: Tensor = [true, false, true].try_into()?;
        assert_eq!(d.sizes(), &[3]); // #PYTHON list(d.shape)
        assert_eq!(d.u8_ref()?, &[1, 0, 1]); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Bool); // #PYTHON d.dtype

        // Non square
        /*
            #|PYTHON
            d = torch.tensor([[5.0, 3.0, 5.0], [1.0, 2.0, 0.0]])
        */
        let d: Tensor = [[5.0f32, 3.0, 5.0], [1.0, 2.0, 0.0]].try_into()?;
        assert_eq!(d.sizes(), &[2, 3]); // #PYTHON list(d.shape)
        assert_eq!(
            d.u8_ref()?,
            &[
                0, 0, 160, 64, 0, 0, 64, 64, 0, 0, 160, 64, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 0, 0
            ]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Float); // #PYTHON d.dtype

        /*
            #|PYTHON
            d = torch.tensor([[5.0, 3.0], [1.0, 2.0], [1.0, 2.0]])
        */
        let d: Tensor = [[5.0f32, 3.0], [1.0, 2.0], [1.0, 2.0]].try_into()?;
        assert_eq!(d.sizes(), &[3, 2]); // #PYTHON list(d.shape)
        assert_eq!(
            d.u8_ref()?,
            &[
                0, 0, 160, 64, 0, 0, 64, 64, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 128, 63, 0, 0, 0, 64
            ]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Float); // #PYTHON d.dtype

        // And with depth;
        /*
            #|PYTHON
            d = torch.tensor([[[1, 2],[3,4]], [[8, 1],[9,3]]])
        */
        let d: Tensor = [[[1i64, 2], [3, 4]], [[8, 1], [9, 3]]].try_into()?;
        assert_eq!(d.sizes(), &[2, 2, 2]); // #PYTHON list(d.shape)
        assert_eq!(
            d.u8_ref()?,
            &[
                1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
                0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0,
                3, 0, 0, 0, 0, 0, 0, 0
            ]
        ); // #PYTHON d.view(torch.uint8).view(-1).tolist()
        assert_eq!(d.scalar_type(), ScalarType::Long); // #PYTHON d.dtype

        Ok(())
    }
}

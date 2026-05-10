//! Accessors to the data in the Tensor.
//!
//! These only work if the Tensor is on the cpu.
//!
//! Types ending in 's' return a slice.
use anyhow::bail;
use zerocopy::{Immutable, IntoBytes, KnownLayout, TryFromBytes};

use crate::{
    properties::TensorProperties,
    tensor::{Ten, TenMut, Tensor, TensorAccess},
};
use torch_stable::StableTorchResult;

pub trait DataRef: TensorAccess + TensorProperties {
    fn u8s_ref(&self) -> StableTorchResult<&[u8]> {
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
    fn f32s_ref(&self) -> StableTorchResult<&[f32]> {
        let byte_ref = self.u8s_ref()?;
        use zerocopy::TryFromBytes;
        match <[f32]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    fn f64s_ref(&self) -> StableTorchResult<&[f64]> {
        let byte_ref = self.u8s_ref()?;
        use zerocopy::TryFromBytes;
        match <[f64]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn ds_ref<T: IntoBytes + TryFromBytes + Immutable>(&self) -> StableTorchResult<&[T]> {
        let byte_ref = self.u8s_ref()?;
        match <[T]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    fn d_ref<T: IntoBytes + TryFromBytes + Immutable + KnownLayout>(
        &self,
        index: &[usize],
    ) -> StableTorchResult<&T> {
        let mut offset = self.storage_offset();
        for (dim, index) in index.iter().enumerate() {
            offset += self.stride(dim) * index;
        }
        if std::mem::size_of::<T>() != self.element_size() {
            bail!(
                "indexing with element size of {} which does not match {}",
                std::mem::size_of::<T>(),
                self.element_size()
            )
        }
        let size = std::mem::size_of::<T>();
        let byte_ref = &self.u8s_ref()?[offset * size..size * (offset + 1)];
        match <T>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    fn f32_ref(&self, indices: &[usize]) -> StableTorchResult<&f32> {
        self.d_ref::<f32>(indices)
    }
}
impl DataRef for Tensor {}
impl<'a> DataRef for Ten<'a> {}
impl<'a> DataRef for TenMut<'a> {}

pub trait DataMut: TensorAccess + TensorProperties {
    fn u8s_mut(&mut self) -> StableTorchResult<&mut [u8]> {
        let z = self.get_tensor_mut();
        let element_size = z.element_size();
        let elements = z.numel();
        let data_ptr = z.mutable_data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts_mut(data_ptr, elements * element_size) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    fn f32s_mut(&mut self) -> StableTorchResult<&mut [f32]> {
        let byte_ref = self.u8s_mut()?;
        match <[f32]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn f64s_mut(&mut self) -> StableTorchResult<&mut [f64]> {
        let byte_ref = self.u8s_mut()?;
        match <[f64]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn ds_mut<T: IntoBytes + TryFromBytes + Immutable>(&mut self) -> StableTorchResult<&mut [T]> {
        let byte_ref = self.u8s_mut()?;
        match <[T]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    fn d_mut<T: IntoBytes + TryFromBytes + Immutable + KnownLayout>(
        &mut self,
        index: &[usize],
    ) -> StableTorchResult<&mut T> {
        let mut offset = self.storage_offset();
        for (dim, index) in index.iter().enumerate() {
            offset += self.stride(dim) * index;
        }
        if std::mem::size_of::<T>() != self.element_size() {
            bail!(
                "indexing with element size of {} which does not match {}",
                std::mem::size_of::<T>(),
                self.element_size()
            )
        }
        let size = std::mem::size_of::<T>();
        let byte_mut = &mut self.u8s_mut()?[offset * size..size * (offset + 1)];
        match <T>::try_mut_from_bytes(byte_mut) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }
    fn f32_mut(&mut self, indices: &[usize]) -> StableTorchResult<&mut f32> {
        self.d_mut::<f32>(indices)
    }
}
impl DataMut for Tensor {}
impl<'a> DataMut for TenMut<'a> {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_data_index() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([ 4,4])
            z = torch.narrow(d, 1, 0, 3)
        */

        let mut d = Tensor::from(&[
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ])?;
        assert_eq!(d.sizes(), &[4, 4]); // #PYTHON list(d.shape)
        assert_eq!(
            d.f32s_ref()?,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        assert_eq!(d.d_ref::<f32>(&[2, 2])?, &11.0); // #PYTHON d[2, 2].item()
        assert_eq!(d.d_ref::<f32>(&[2, 3])?, &12.0); // #PYTHON d[2, 3].item()
        assert_eq!(d.d_ref::<f32>(&[0, 3])?, &4.0); // #PYTHON d[0, 3].item()
        assert_eq!(d.d_ref::<f32>(&[3, 0])?, &13.0); // #PYTHON d[3, 0].item()

        println!("d: {d:?}");
        let mut z = d.narrow_mut(1, 0, 3)?;
        println!("z: {z:?}");

        assert_eq!(z.sizes(), &[4, 3]); // #PYTHON list(z.shape)
        assert_eq!(z.is_contiguous(), false);

        assert_eq!(z.d_ref::<f32>(&[0, 0])?, &1.0); // #PYTHON z[ 0,  0].item()
        assert_eq!(z.d_ref::<f32>(&[0, 1])?, &2.0); // #PYTHON z[ 0,  1].item()
        assert_eq!(z.d_ref::<f32>(&[0, 2])?, &3.0); // #PYTHON z[ 0,  2].item()
        assert_eq!(z.d_ref::<f32>(&[1, 0])?, &5.0); // #PYTHON z[ 1,  0].item()
        assert_eq!(z.d_ref::<f32>(&[1, 1])?, &6.0); // #PYTHON z[ 1,  1].item()
        assert_eq!(z.d_ref::<f32>(&[1, 2])?, &7.0); // #PYTHON z[ 1,  2].item()

        /*
            #|PYTHON
            z[0,2] = 50.0
            z[1,2] = 100.0
        */

        *(z.f32_mut(&[0, 2])?) = 50.0;
        *(z.f32_mut(&[1, 2])?) = 100.0;

        assert_eq!(z.d_ref::<f32>(&[0, 2])?, &50.0); // #PYTHON z[ 0,  2].item()
        assert_eq!(z.d_ref::<f32>(&[1, 2])?, &100.0); // #PYTHON z[ 1,  2].item()

        Ok(())
    }
}

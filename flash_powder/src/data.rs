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
use torch_stable::{contrib::TensorPropertiesContrib as _, StableTorchResult};

macro_rules! impl_slice_ref {
    ($t:ty, $v:ident) => {
        fn $v(&self) -> StableTorchResult<&[$t]> {
            self.ds_ref::<$t>()
        }
    };
}
macro_rules! impl_item_ref {
    ($t:ty, $v:ident) => {
        fn $v(&self, indices: &[usize]) -> StableTorchResult<&$t> {
            self.d_ref::<$t>(indices)
        }
    };
}

/// Access data in the tensor through a reference.
///
/// Methods ending with 's' return a slice.
///
/// These methods are only valid if the data is on the CPU.
pub trait DataRef: TensorAccess + TensorProperties {
    /// Direct access to the byte slice backing the tensor.
    /// This always provides the ENTIRE storage slice, and does materialize the data if we're in a COW situation.
    fn u8s_ref(&self) -> StableTorchResult<&[u8]> {
        let z = self.get_tensor();
        // https://github.com/pytorch/pytorch/blob/ec673ecd/c10/core/TensorImpl.h#L743-L756
        let data_ptr = z.data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts(data_ptr, z.storage_size()) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    /// Access to the slice spanning the entire tensor data.
    ///
    /// Zerocopy cast of [`Self::u8s_ref`] to `&[T]`.
    fn ds_ref<T: IntoBytes + TryFromBytes + Immutable>(&self) -> StableTorchResult<&[T]> {
        let byte_ref = self.u8s_ref()?;
        match <[T]>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    /// Element reference to element at the provided indices.
    fn d_ref<T: IntoBytes + TryFromBytes + Immutable + KnownLayout>(
        &self,
        index: &[usize],
    ) -> StableTorchResult<&T> {
        if index.len() > self.dim() {
            bail!(
                "indices provided {} dim, tensor is {} dim",
                index.len(),
                self.dim()
            )
        }

        let mut offset = 0;
        for (dim, index) in index.iter().enumerate() {
            if *index > self.sizes()[dim] {
                bail!(
                    "index {} for dimension {} exceeded size {}",
                    index,
                    dim,
                    self.sizes()[dim]
                );
            }
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
        if std::mem::size_of::<T>() != self.element_size() {
            bail!(
                "indexing with element size of {} which does not match {}",
                std::mem::size_of::<T>(),
                self.element_size()
            )
        }
        let byte_ref = &self.u8s_ref()?[offset * size..size * (offset + 1)];
        match <T>::try_ref_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    impl_slice_ref!(f32, f32s_ref);
    impl_slice_ref!(f64, f64s_ref);

    impl_slice_ref!(u16, u16s_ref);
    impl_slice_ref!(u32, u32s_ref);
    impl_slice_ref!(u64, u64s_ref);

    impl_slice_ref!(i8, i8s_ref);
    impl_slice_ref!(i16, i16s_ref);
    impl_slice_ref!(i32, i32s_ref);
    impl_slice_ref!(i64, i64s_ref);

    impl_item_ref!(f32, f32_ref);
    impl_item_ref!(f64, f64_ref);

    impl_item_ref!(u16, u16_ref);
    impl_item_ref!(u32, u32_ref);
    impl_item_ref!(u64, u64_ref);

    impl_item_ref!(i8, i8_ref);
    impl_item_ref!(i16, i16_ref);
    impl_item_ref!(i32, i32_ref);
    impl_item_ref!(i64, i64_ref);
}

impl DataRef for Tensor {}
impl<'a> DataRef for Ten<'a> {}
impl<'a> DataRef for TenMut<'a> {}

macro_rules! impl_slice_mut {
    ($t:ty, $v:ident) => {
        fn $v(&mut self) -> StableTorchResult<&mut [$t]> {
            self.ds_mut::<$t>()
        }
    };
}
macro_rules! impl_item_mut {
    ($t:ty, $v:ident) => {
        fn $v(&mut self, indices: &[usize]) -> StableTorchResult<&mut $t> {
            self.d_mut::<$t>(indices)
        }
    };
}

/// Access data in the tensor through a mutable reference.
///
/// Methods ending with 's' return a slice.
///
/// These methods are only valid if the data is on the CPU.
pub trait DataMut: TensorAccess + TensorProperties {
    /// Direct access to the mutable byte slice backing the tensor.
    fn u8s_mut(&mut self) -> StableTorchResult<&mut [u8]> {
        let z = self.get_tensor_mut();
        // https://github.com/pytorch/pytorch/blob/ec673ecd/c10/core/TensorImpl.h#L743-L756
        let data_ptr = z.data_ptr();
        if z.is_cpu() {
            Ok(unsafe { std::slice::from_raw_parts_mut(data_ptr, z.storage_size()) })
        } else {
            bail!("tensor must be on cpu to access slice")
        }
    }

    /// Access to the mutable slice spanning the entire tensor data.
    ///
    /// Zerocopy cast of [`Self::u8s_mut`] to `&mut [T]`.
    fn ds_mut<T: IntoBytes + TryFromBytes + Immutable>(&mut self) -> StableTorchResult<&mut [T]> {
        let byte_ref = self.u8s_mut()?;
        match <[T]>::try_mut_from_bytes(byte_ref) {
            Ok(e) => Ok(e),
            Err(z) => bail!("failed slice conversion: {z:?}"),
        }
    }

    /// Element mutable reference to element at the provided indices.
    fn d_mut<T: IntoBytes + TryFromBytes + Immutable + KnownLayout>(
        &mut self,
        index: &[usize],
    ) -> StableTorchResult<&mut T> {
        if index.len() > self.dim() {
            bail!(
                "indices provided {} dim, tensor is {} dim",
                index.len(),
                self.dim()
            )
        }

        let mut offset = 0;
        for (dim, index) in index.iter().enumerate() {
            if *index > self.sizes()[dim] {
                bail!(
                    "index {} for dimension {} exceeded size {}",
                    index,
                    dim,
                    self.sizes()[dim]
                );
            }
            offset += self.stride(dim) * index;
        }
        if std::mem::size_of::<T>() != self.element_size() {
            bail!(
                "indexing with element size of {} which does not match {}",
                std::mem::size_of::<T>(),
                self.element_size()
            )
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

    impl_slice_mut!(f32, f32s_mut);
    impl_slice_mut!(f64, f64s_mut);

    impl_slice_mut!(u16, u16s_mut);
    impl_slice_mut!(u32, u32s_mut);
    impl_slice_mut!(u64, u64s_mut);

    impl_slice_mut!(i8, i8s_mut);
    impl_slice_mut!(i16, i16s_mut);
    impl_slice_mut!(i32, i32s_mut);
    impl_slice_mut!(i64, i64s_mut);

    impl_item_mut!(f32, f32_mut);
    impl_item_mut!(f64, f64_mut);

    impl_item_mut!(u16, u16_mut);
    impl_item_mut!(u32, u32_mut);
    impl_item_mut!(u64, u64_mut);

    impl_item_mut!(i8, i8_mut);
    impl_item_mut!(i16, i16_mut);
    impl_item_mut!(i32, i32_mut);
    impl_item_mut!(i64, i64_mut);
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

        assert_eq!(d.f32_ref(&[2, 2])?, &11.0); // #PYTHON d[2, 2].item()
        assert_eq!(d.f32_ref(&[2, 3])?, &12.0); // #PYTHON d[2, 3].item()
        assert_eq!(d.f32_ref(&[0, 3])?, &4.0); // #PYTHON d[0, 3].item()
        assert_eq!(d.f32_ref(&[3, 0])?, &13.0); // #PYTHON d[3, 0].item()

        println!("d: {d:?}");
        let mut z = d.narrow_mut(1, 0, 3)?;
        println!("z: {z:?}");

        assert_eq!(z.sizes(), &[4, 3]); // #PYTHON list(z.shape)
        assert_eq!(z.is_contiguous(), false);
        assert_eq!(z.storage_offset(), 0);

        assert_eq!(z.f32_ref(&[0, 0])?, &1.0); // #PYTHON z[ 0,  0].item()
        assert_eq!(z.f32_ref(&[0, 1])?, &2.0); // #PYTHON z[ 0,  1].item()
        assert_eq!(z.f32_ref(&[0, 2])?, &3.0); // #PYTHON z[ 0,  2].item()
        assert_eq!(z.f32_ref(&[1, 0])?, &5.0); // #PYTHON z[ 1,  0].item()
        assert_eq!(z.f32_ref(&[1, 1])?, &6.0); // #PYTHON z[ 1,  1].item()
        assert_eq!(z.f32_ref(&[1, 2])?, &7.0); // #PYTHON z[ 1,  2].item()

        /*
            #|PYTHON
            z[0,2] = 50.0
            z[1,2] = 100.0
        */

        *(z.f32_mut(&[0, 2])?) = 50.0;
        *(z.f32_mut(&[1, 2])?) = 100.0;

        assert_eq!(z.f32_ref(&[0, 2])?, &50.0); // #PYTHON z[ 0,  2].item()
        assert_eq!(z.f32_ref(&[1, 2])?, &100.0); // #PYTHON z[ 1,  2].item()

        /*
            #|PYTHON
            y = torch.narrow(d, 1, 1, 3)
        */

        // Do something with a storage offset.
        let y = d.narrow(1, 1, 3)?;
        println!("y: {y:?}");
        assert_eq!(y.sizes(), &[4, 3]); // #PYTHON list(y.shape)
        assert_eq!(y.is_contiguous(), false);
        assert_eq!(y.storage_offset(), 1);
        assert_eq!(y.f32_ref(&[0, 2])?, &4.0); // #PYTHON y[ 0,  2].item()
        assert_eq!(y.f32_ref(&[1, 2])?, &8.0); // #PYTHON y[ 1,  2].item()

        /*
            #|PYTHON
            x = torch.narrow(d, 1, 1, 3).narrow(0, 1, 3)
        */

        // Do something with a storage offset.
        let x = d.narrow(1, 1, 3)?.narrow(0, 1, 3)?;
        println!("x: {x:?}");
        assert_eq!(x.sizes(), &[3, 3]); // #PYTHON list(x.shape)
        assert_eq!(x.is_contiguous(), false);
        assert_eq!(x.storage_offset(), 5);
        assert_eq!(x.f32_ref(&[0, 2])?, &8.0); // #PYTHON x[ 0,  2].item()
        assert_eq!(x.f32_ref(&[1, 2])?, &12.0); // #PYTHON x[ 1,  2].item()
        Ok(())
    }
}

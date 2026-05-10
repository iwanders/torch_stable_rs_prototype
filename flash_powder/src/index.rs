//! Indexing

use crate::core_methods::CoreMethods;
use crate::properties::TensorProperties;
use crate::tensor::{Ten, TenMut, Tensor};
use crate::torch;
use crate::{StableTorchResult, TensorAccess};
pub use torch_stable::stable::ops::{EmtpyOptions, ToOptions};
use torch_stable::stable::tensor::Tensor as StableTensor;

// https://docs.pytorch.org/docs/2.11/tensor_view.html
// torch just follows https://numpy.org/doc/stable/user/basics.indexing.html ?
//
// > PyTorch follows Numpy behaviors that basic indexing returns views, while advanced indexing returns a copy.
// > Assignment via either basic or advanced indexing is in-place.
// > See more examples in Numpy indexing documentation.
//
// :<
//
// Advanced indexing; https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
// Basically, if indexing is non tuple, like ndarray with bool/integer, or a tuple with sequence of int/bool.
// So basically whenever a select happens?
//
//
// https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/TensorIndexing.h#L88-L112
// This definitely has Slice(start, stop, step)
//
// Python slices have a step though :| Rust's ranges do not...
// Maybe we do something like (0..12,3) for 0:12:3 -> [0, 3, 6, 9]?
// We can probably implement Into TensorIndexOptions for both Range and (Range, usize) or something?
//
// Should we use isize or usize... usize feels more natural... but isize can walk backwards.

// Do we even need the step for indexing? Works in native lists.
// a = list(range(16))
// a[0:12:3] ->[0, 3, 6, 9]
//
// also works in numpy;
//
// np.array(list(range(16)))[0:12:3]
// array([0, 3, 6, 9])

// That https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/TensorIndexing.h#L438 looks very complex :/
//

pub enum TensorIndexOptions<'a> {
    Tensor(&'a StableTensor),
    Index(usize),
    Range(std::ops::Range<isize>),
    // Can we even do this?
    RangeWithStride {
        range: std::ops::Range<usize>,
        stride: isize,
    },
}
impl<'a> Into<TensorIndexOptions<'a>> for usize {
    fn into(self) -> TensorIndexOptions<'a> {
        TensorIndexOptions::Index(self)
    }
}
impl<'a> Into<TensorIndexOptions<'a>> for std::ops::Range<isize> {
    fn into(self) -> TensorIndexOptions<'a> {
        TensorIndexOptions::Range(self.clone())
    }
}

trait TensorIndexWorker: TensorAccess + TensorProperties + CoreMethods {
    fn do_the_real_indexing<'a, 'b>(
        &'b self,
        index: &[&TensorIndexOptions<'a>],
    ) -> StableTorchResult<Ten<'b>> {
        // Make a view into the tensor, we'll be updating this as we go through the indexing operations.
        let mut current = self.view(self.sizes())?;
        let mut dim = 0;
        for index_op_conv in index.iter() {
            let mut do_dim_add = true;
            match index_op_conv {
                TensorIndexOptions::Tensor(tensor) => todo!(),
                TensorIndexOptions::Index(index) => {
                    current = current.select(dim, *index)?;
                    do_dim_add = false;
                }
                TensorIndexOptions::Range(range) => {
                    let length = if range.start < 0 {
                        (self.sizes()[dim] as isize + range.start) as usize + 1
                    } else {
                        range.len()
                    };
                    current = current.narrow(dim, range.start, length)?;
                }
                TensorIndexOptions::RangeWithStride { range, stride } => todo!(),
            }
            if do_dim_add {
                dim += 1;
            }
        }
        Ok(current)
    }
}

impl TensorIndexWorker for Tensor {}

pub trait IndexSpec<T> {
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>>;
}
pub trait TensorIndex: TensorAccess + TensorProperties + CoreMethods + Sized {
    fn i<'a, I: IndexSpec<Self>>(&'a self, index: I) -> StableTorchResult<Ten<'a>> {
        index.do_index(self)
    }
}
impl TensorIndex for Tensor {}

impl<'a, A: Clone, T: TensorIndexWorker> IndexSpec<T> for A
where
    A: Into<TensorIndexOptions<'a>>,
{
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>> {
        let first: TensorIndexOptions<'_> = self.clone().into();
        tensor.do_the_real_indexing(&[&first])
    }
}

impl<'a, 'd, A: Clone, B: Clone, T: TensorIndexWorker> IndexSpec<T> for (A, B)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
{
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        tensor.do_the_real_indexing(&[&first, &second])
    }
}
impl<'a, A: Clone, B: Clone, C: Clone, T: TensorIndexWorker> IndexSpec<T> for (A, B, C)
where
    A: Into<TensorIndexOptions<'a>>,
    B: Into<TensorIndexOptions<'a>>,
    C: Into<TensorIndexOptions<'a>>,
{
    fn do_index<'b>(&self, tensor: &'b T) -> StableTorchResult<Ten<'b>> {
        let first: TensorIndexOptions<'_> = self.0.clone().into();
        let second: TensorIndexOptions<'_> = self.1.clone().into();
        let third: TensorIndexOptions<'_> = self.2.clone().into();
        tensor.do_the_real_indexing(&[&first, &second, &third])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_indexing() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([ 4,4])
        */

        let d = Tensor::from(&[
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

        let z = d.i((1..3, 0..1))?;
        /*
            #|PYTHON
            z = d[1:3, 0:1]
        */

        println!("z: {z:?}");
        assert_eq!(z.sizes(), &[2, 1]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.stride(1), 1); // #PYTHON  (z.stride(1))
        assert_eq!(z.f32_ref(&[0, 0])?, &5.0); // #PYTHON z[0,0].item()
        assert_eq!(z.f32_ref(&[1, 0])?, &9.0); // #PYTHON z[1,0].item()

        // Ah yes, now we need a tuple...
        //
        let z = d.i((1, 0..3))?;
        /*
            #|PYTHON
            z = d[1, 0:3]
        */

        println!("z: {z:?}");
        assert_eq!(z.sizes(), &[3]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 1); // #PYTHON  (z.stride(0))
        assert_eq!(z.f32_ref(&[0])?, &5.0); // #PYTHON z[0].item()
        assert_eq!(z.f32_ref(&[1])?, &6.0); // #PYTHON z[1].item()
        assert_eq!(z.f32_ref(&[2])?, &7.0); // #PYTHON z[2].item()

        let z = d.i((0..3, 1))?;
        /*
            #|PYTHON
            z = d[0:3, 1]
        */

        println!("z: {z:?}");
        assert_eq!(z.sizes(), &[3]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.f32_ref(&[0])?, &2.0); // #PYTHON z[0].item()
        assert_eq!(z.f32_ref(&[1])?, &6.0); // #PYTHON z[1].item()
        assert_eq!(z.f32_ref(&[2])?, &10.0); // #PYTHON z[2].item()

        let z = d.i((0..3, 1))?;
        /*
            #|PYTHON
            z = d[0:3, 1]
        */

        println!("z: {z:?}");
        assert_eq!(z.sizes(), &[3]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.f32_ref(&[0])?, &2.0); // #PYTHON z[0].item()
        assert_eq!(z.f32_ref(&[1])?, &6.0); // #PYTHON z[1].item()
        assert_eq!(z.f32_ref(&[2])?, &10.0); // #PYTHON z[2].item()

        let z = d.i((-3isize..3, -3isize..3))?;
        /*
            #|PYTHON
            z = d[-3:3, -3:3]
        */

        println!("z: {z:?}");
        assert_eq!(z.sizes(), &[2, 2]); // #PYTHON list(z.shape)
        assert_eq!(z.stride(0), 4); // #PYTHON  (z.stride(0))
        assert_eq!(z.f32_ref(&[0, 0])?, &6.0); // #PYTHON z[0,0].item()
        assert_eq!(z.f32_ref(&[1, 0])?, &10.0); // #PYTHON z[1,0].item()
        assert_eq!(z.f32_ref(&[1, 1])?, &11.0); // #PYTHON z[1,1].item()

        Ok(())
    }
}

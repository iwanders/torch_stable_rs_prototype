//! Indexing

use crate::core_methods::CoreMethods;
use crate::properties::TensorProperties;
use crate::tensor::{Ten, TenMut, Tensor};
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
    Index(isize),
    Range(std::ops::Range<usize>),
    RangeWithStride {
        range: std::ops::Range<usize>,
        stride: isize,
    },
}
impl<'a> Into<TensorIndexOptions<'a>> for isize {
    fn into(self) -> TensorIndexOptions<'a> {
        TensorIndexOptions::Index(self)
    }
}
impl<'a> Into<TensorIndexOptions<'a>> for &std::ops::Range<usize> {
    fn into(self) -> TensorIndexOptions<'a> {
        TensorIndexOptions::Range(self.clone())
    }
}
impl<'a> Into<TensorIndexOptions<'a>> for (std::ops::Range<usize>, isize) {
    fn into(self) -> TensorIndexOptions<'a> {
        TensorIndexOptions::RangeWithStride {
            range: self.0,
            stride: self.1,
        }
    }
}

pub trait TensorIndex: TensorAccess + TensorProperties + CoreMethods {
    fn i<'a, T>(&self, index: &'a [T]) -> StableTorchResult<Ten<'_>>
    where
        &'a T: Into<TensorIndexOptions<'a>>,
    {
        // Make a view into the tensor, we'll be updating this as we go through the indexing operations.
        let mut current = self.view(self.sizes())?;
        // let mut dim = self.dim();
        for (index_op_conv, dim) in index.iter().zip((0..index.len()).rev()) {
            // dim -= 1;
            let index_op: TensorIndexOptions<'_> = index_op_conv.into();
            match index_op {
                TensorIndexOptions::Tensor(tensor) => todo!(),
                TensorIndexOptions::Index(_) => todo!(),
                TensorIndexOptions::Range(range) => {
                    println!("dim {dim}, start {} len {}", range.start, range.len());
                    current = current.narrow(dim, range.start, range.len())?;
                }
                TensorIndexOptions::RangeWithStride { range, stride } => todo!(),
            }
        }
        Ok(current)
    }
}
impl TensorIndex for Tensor {}

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

        println!("d: {d:?}");
        let z = d.i(&[0..3usize, 0..1])?;
        println!("z: {z:?}");

        assert_eq!(z.f32s_ref()?, &[1.0, 5.0, 9.0]); // #PYTHON list(d[ 0:3, 0:1].view(-1).tolist())

        Ok(())
    }
}

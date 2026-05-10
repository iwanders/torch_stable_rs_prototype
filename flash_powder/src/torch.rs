//! This holds functions that pytorch puts into the torch module.
//!
//! This module is a bit sad atm... it holds select, but it not ever used because I needed specific selects in each
//! of the three principal types.
use crate::tensor::{Ten, TensorAccess};
use torch_stable::{
    StableTorchResult, aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor,
    unsafe_call_dispatch_bail,
};

/// Select an index in a dimension
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L5391)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.12/generated/torch.select.html)
pub fn select<T: TensorAccess>(input: &T, dim: usize, index: usize) -> StableTorchResult<Ten<'_>> {
    let mut stack: [StableIValue; 3] = [input.get_tensor().into(), dim.into(), index.into()];
    unsafe_call_dispatch_bail!("aten::select", "int", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;

    Ok(Ten::new(input.get_tensor(), r))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_powder_torch_select() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([4,4])
            r = torch.select(d, 0, 2);
            c = torch.select(d, 1, 2);
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
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        let r = select(&d, 0, 2)?;
        assert_eq!(r.sizes(), &[4]); // #PYTHON list(r.shape)
        assert_eq!(r.f32_ref(&[0])?, &9.0); // #PYTHON r[ 0].item()
        assert_eq!(r.f32_ref(&[1])?, &10.0); // #PYTHON r[ 1].item()
        assert_eq!(r.f32_ref(&[2])?, &11.0); // #PYTHON r[ 2].item()
        assert_eq!(r.f32_ref(&[3])?, &12.0); // #PYTHON r[ 3].item()

        let c = select(&d, 1, 2)?;
        assert_eq!(c.sizes(), &[4]); // #PYTHON list(c.shape)
        assert_eq!(c.f32_ref(&[0])?, &3.0); // #PYTHON c[ 0].item()
        assert_eq!(c.f32_ref(&[1])?, &7.0); // #PYTHON c[ 1].item()
        assert_eq!(c.f32_ref(&[2])?, &11.0); // #PYTHON c[ 2].item()
        assert_eq!(c.f32_ref(&[3])?, &15.0); // #PYTHON c[ 3].item()

        Ok(())
    }
}

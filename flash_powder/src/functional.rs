pub use torch_stable::stable::ops::{EmtpyOptions, ToOptions};
use torch_stable::{
    aoti_torch::StableIValue, stable::tensor::Tensor as StableTensor, unsafe_call_dispatch_bail,
};

use crate::{StableTorchResult, Tensor, TensorAccess};
/// Options for conv2d.
#[derive(Copy, Clone, Debug)]
pub struct Conv2DOptions {
    pub stride: (i64, i64),
    pub padding: (i64, i64),
    pub dilation: (i64, i64),
    pub groups: i64,
}
impl Default for Conv2DOptions {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

/// A 2d convolution.
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/native_functions.yaml#L1757)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.conv2d.html)
pub fn conv2d<T: TensorAccess>(
    input: &T,
    weight: &T,
    bias: Option<&T>,
    options: &Conv2DOptions,
) -> StableTorchResult<Tensor> {
    // return wrap(dispatch_conv2d(r.tensor(0), r.tensor(1), r.tensor(2), r.intlist(3), r.intlist(4), r.intlist(5), r.toInt64(6)));
    let mut stack: [StableIValue; 7] = [
        input.get_tensor().into(),
        weight.get_tensor().into(),
        (&bias.map(|z| z.get_tensor())).into(),
        (&options.stride).into(),
        (&options.padding).into(),
        (&options.dilation).into(),
        (&options.groups).into(),
    ];
    unsafe_call_dispatch_bail!("aten::conv2d", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;

    Ok(Tensor::new(r))
}

/// Relu
///
/// - [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.12.0-rc2/aten/src/ATen/native/native_functions.yaml#L5251)
/// - [pytorch equivalent](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.relu.html)
pub fn relu<T: TensorAccess>(input: &T) -> StableTorchResult<Tensor> {
    let mut stack: [StableIValue; 1] = [input.get_tensor().into()];
    unsafe_call_dispatch_bail!("aten::relu", "", stack.as_mut_slice());
    let r: StableTensor = stack[0].try_into()?;
    assert_ne!(input.get_tensor().data_ptr(), r.data_ptr());

    Ok(Tensor::new(r))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_flash_power_conv2d() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor(list(range(1,17)), dtype=torch.float).reshape([1,4,4])
            w = torch.tensor([[[1.0, 2.0],[3.0, 4.0]]]).unsqueeze(0)
            r = torch.nn.functional.conv2d(d, w)
        */

        let mut d = Tensor::zeros(&[1, 4, 4], &Default::default())?;
        for (i, v) in d.f32_mut()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }

        assert_eq!(d.sizes(), &[1, 4, 4]); // #PYTHON list(d.shape)
        assert_eq!(
            d.f32_ref()?,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0
            ]
        ); // #PYTHON list(d.view(-1).tolist())

        let mut w = Tensor::zeros(&[1, 1, 2, 2], &Default::default())?;
        for (i, v) in w.d_mut::<f32>()?.iter_mut().enumerate() {
            *v = (i + 1) as f32
        }
        assert_eq!(w.sizes(), &[1, 1, 2, 2]);
        assert_eq!(w.f32_ref()?, &[1.0f32, 2.0, 3.0, 4.0]); // #PYTHON list(w.view(-1).tolist())

        let r = conv2d(&d, &w, None, &Default::default())?;
        assert_eq!(r.sizes(), &[1, 3, 3]); // #PYTHON list(r.shape)

        assert_eq!(
            r.f32_ref()?,
            &[44.0f32, 54.0, 64.0, 84.0, 94.0, 104.0, 124.0, 134.0, 144.0]
        ); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            r = torch.nn.functional.conv2d(d, w, stride=(1, 2))
        */
        let r = conv2d(
            &d,
            &w,
            None,
            &Conv2DOptions {
                stride: (1, 2),
                ..Default::default()
            },
        )?;
        assert_eq!(r.sizes(), &[1, 3, 2]); // #PYTHON list(r.shape)
        assert_eq!(r.f32_ref()?, &[44.0f32, 64.0, 84.0, 104.0, 124.0, 144.0]); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            r = torch.nn.functional.conv2d(d, w, stride=(2, 2))
        */
        let r = conv2d(
            &d,
            &w,
            None,
            &Conv2DOptions {
                stride: (2, 2),
                ..Default::default()
            },
        )?;
        assert_eq!(r.sizes(), &[1, 2, 2]); // #PYTHON list(r.shape)
        assert_eq!(r.f32_ref()?, &[44.0f32, 64.0, 124.0, 144.0]); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            r = torch.nn.functional.conv2d(d, w, padding=(1, 2))
        */
        let r = conv2d(
            &d,
            &w,
            None,
            &Conv2DOptions {
                padding: (1, 2),
                ..Default::default()
            },
        )?;
        assert_eq!(r.sizes(), &[1, 5, 7]); // #PYTHON list(r.shape)
        assert_eq!(
            r.f32_ref()?,
            &[
                0.0f32, 4.0, 11.0, 18.0, 25.0, 12.0, 0.0, 0.0, 22.0, 44.0, 54.0, 64.0, 28.0, 0.0,
                0.0, 46.0, 84.0, 94.0, 104.0, 44.0, 0.0, 0.0, 70.0, 124.0, 134.0, 144.0, 60.0, 0.0,
                0.0, 26.0, 41.0, 44.0, 47.0, 16.0, 0.0
            ]
        ); // #PYTHON list(r.view(-1).tolist())

        /*
            #|PYTHON
            b = torch.tensor([5.0])
            r = torch.nn.functional.conv2d(d, w, b)
        */
        let mut b = Tensor::zeros(&[1], &Default::default())?;
        b.f32_mut()?.copy_from_slice(&[5.0]);
        assert_eq!(b.sizes(), &[1]); // #PYTHON list(b.shape)

        let r = conv2d(&d, &w, Some(&b), &Default::default())?;
        assert_eq!(r.sizes(), &[1, 3, 3]); // #PYTHON list(r.shape)
        assert_eq!(
            r.f32_ref()?,
            &[49.0f32, 59.0, 69.0, 89.0, 99.0, 109.0, 129.0, 139.0, 149.0]
        ); // #PYTHON list(r.view(-1).tolist())

        Ok(())
    }

    #[test]
    fn test_flash_power_relu() -> StableTorchResult<()> {
        /*
            #|PYTHON
            d = torch.tensor([-1.0, 0.0, 0.5, 1.0], dtype=torch.float)
            r = d.relu()
        */
        let d = Tensor::from(&[-1.0f32, 0.0, 0.5, 1.0])?;

        assert_eq!(d.sizes(), &[4]); // #PYTHON list(d.shape)
        let r = relu(&d)?;
        assert_eq!(r.f32_ref()?, &[0.0f32, 0.0, 0.5, 1.0]); // #PYTHON list(r.view(-1).tolist())

        Ok(())
    }
}

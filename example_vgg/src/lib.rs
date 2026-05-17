/// Quickly thrown together example of VGG11
///
/// `https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html#torchvision.models.vgg11`
/// Code:
/// `https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L91`.
///
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py

// Smallest is vgg11;
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L306-L329
// which is of the 'A' category.
//    weights = VGG11_Weights.verify(weights)
// return _vgg("A", False, weights, progress, **kwargs)
// https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L98
// VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
// batch norm = False
// Layers per 'A'.
// We're doing inference, we can skip the dropout.
use anyhow::bail;
use safetensors::SafeTensors;

use flash_powder as fp;
use flash_powder::functional;
use flash_powder::prelude::*;
use fp::{DType, Tensor};

// Bit of tooling...
trait ForwardLayer {
    fn forward(&self, tensor: &Tensor) -> Result<Tensor, anyhow::Error>;
}
type ForwardFun = dyn Fn(&Tensor) -> Result<Tensor, anyhow::Error>;
struct LambdaForward {
    f: Box<ForwardFun>,
}
impl LambdaForward {
    fn new<F>(f: F) -> Self
    where
        F: Fn(&Tensor) -> Result<Tensor, anyhow::Error> + 'static,
    {
        Self { f: Box::new(f) }
    }
}
impl ForwardLayer for LambdaForward {
    fn forward(&self, tensor: &Tensor) -> Result<Tensor, anyhow::Error> {
        (self.f)(tensor)
    }
}

// The config, as per https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L91
// but then as integers.
const CFG_A: &[u32] = &[
    64, 'M' as u32, 128, 'M' as u32, 256, 256, 'M' as u32, 512, 512, 'M' as u32, 512, 512,
    'M' as u32,
];

/// VGG struct with layers and classifier.
pub struct VGG {
    layers: Vec<Box<dyn ForwardLayer>>,
    classifier: Vec<Box<dyn ForwardLayer>>,
}
impl VGG {
    /// Create a new vgg config as per the layer specification and the provided weights.
    pub fn new(cfg: &[u32], tensors: &SafeTensors) -> Result<Self, anyhow::Error> {
        const PRINT: bool = false;

        if PRINT {
            let mut sorted = tensors.names().clone();
            sorted.sort();

            for name in sorted {
                println!("Tensor key: {}", name);

                if let Ok(view) = tensors.tensor(name) {
                    println!("  Shape: {:?}", view.shape());
                }
            }
        }

        // Okay... so we iterate over cfg... build the layers and then append them to ourselves? How hard can it be.
        let mut layers: Vec<Box<dyn ForwardLayer>> = vec![];

        // First, the group called 'features'.
        let mut feature_counter = 0;
        for v in cfg.iter() {
            let layer_name = format!("features.{feature_counter}");
            if *v == 'M' as u32 {
                // should be maxpool2d.
                layers.push(create_maxpool(2));
                feature_counter += 1;
            } else {
                layers.push(create_conv2d(tensors, &layer_name)?);
                // + 1 for conv2d, +1 for relu.
                feature_counter += 2;
            }
        }

        // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L43
        // and then the group called classifier
        let mut classifier: Vec<Box<dyn ForwardLayer>> = vec![];
        classifier.push(create_linear(tensors, &format!("classifier.0"))?);
        classifier.push(create_relu());
        // dropout

        // next linear block
        classifier.push(create_linear(tensors, &format!("classifier.3"))?);
        classifier.push(create_relu());
        // dropout

        // last linear.
        classifier.push(create_linear(tensors, &format!("classifier.6"))?);

        Ok(VGG { layers, classifier })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, anyhow::Error> {
        let mut r: Tensor = input.clone();
        // Run it through the layers.
        for f in self.layers.iter() {
            r = f.forward(&r)?;
        }
        // do some avgpool.
        r = fp::functional::adaptive_avg_pool2d(&r, (7, 7))?;
        // do a flatten.
        r = r.flatten(1, None)?;

        // Run it through the classifier.
        for f in self.classifier.iter() {
            r = f.forward(&r)?;
        }
        Ok(r)
    }
}

fn create_linear(
    tensors: &SafeTensors,
    layer_name: &str,
) -> Result<Box<dyn ForwardLayer>, anyhow::Error> {
    let weights = { safetensor_to_tensor(tensors, &format!("{layer_name}.weight"))? };
    let bias = {
        if let Ok(v) = safetensor_to_tensor(tensors, &format!("{layer_name}.bias")) {
            Some(v)
        } else {
            None
        }
    };
    Ok(Box::new(LambdaForward::new(move |input: &Tensor| {
        let bias_ref = bias.as_ref();
        let res = functional::linear(input, &weights, bias_ref)?;
        Ok(res)
    })))
}

fn create_conv2d(
    tensors: &SafeTensors,
    layer_name: &str,
) -> Result<Box<dyn ForwardLayer>, anyhow::Error> {
    let conv2d_options = functional::Conv2dOptions {
        stride: (1, 1),
        padding: (1, 1),
        ..Default::default()
    };
    //
    let weights = { safetensor_to_tensor(tensors, &format!("{layer_name}.weight"))? };
    let bias = {
        if let Ok(v) = safetensor_to_tensor(tensors, &format!("{layer_name}.bias")) {
            Some(v)
        } else {
            None
        }
    };
    Ok(Box::new(LambdaForward::new(move |t: &Tensor| {
        let bias_ref = bias.as_ref();
        let conv_res = functional::conv2d(t, &weights, bias_ref, &conv2d_options)?;
        let res = functional::relu(&conv_res)?;

        Ok(res)
    })))
}

fn create_maxpool(kernel_size: i64) -> Box<dyn ForwardLayer> {
    Box::new(LambdaForward::new(move |input: &Tensor| {
        fp::functional::max_pool2d(input, (kernel_size, kernel_size), &Default::default())
    }))
}
fn create_relu() -> Box<dyn ForwardLayer> {
    Box::new(LambdaForward::new(move |input: &Tensor| {
        fp::functional::relu(input)
    }))
}

fn safetensor_dtype_to_scalar_type(v: safetensors::Dtype) -> DType {
    match v {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F64 => DType::F64,
        _ => todo!("todo handle {v:?}"),
    }
}

fn safetensor_to_tensor(tensors: &SafeTensors, name: &str) -> Result<Tensor, anyhow::Error> {
    if let Ok(tensor_view) = tensors.tensor(name) {
        // Create a tensor of the correct shape and type
        let mut v = Tensor::zeros(
            tensor_view.shape(),
            &flash_powder::factory::TensorOptions {
                dtype: Some(safetensor_dtype_to_scalar_type(tensor_view.dtype())),
                ..Default::default()
            },
        )?;

        // Copy the bytes.
        v.data_mut()?.copy_from_slice(tensor_view.data());
        Ok(v)
    } else {
        bail!("could not find safetensor {name}")
    }
}

pub fn main() -> Result<(), anyhow::Error> {
    use std::path::PathBuf;
    let weights = PathBuf::from("data/vgg11-8a719046.safetensors");
    if !weights.is_file() {
        eprintln!(
            "Run this binary from the 'example_vgg' directory, it looks for  \
            {}, if that doesn't exist:\n Download it from https://download.pytorch.org/models/vgg11-8a719046.pth,\
            convert it to safetensors with ./convert_pth.py",
            weights.display()
        );
        //
        bail!("missing necessary file, bailing out")
    }

    let data = std::fs::read(weights).expect("Unable to read file");

    let tensors = SafeTensors::deserialize(&data)?;

    let vgg = VGG::new(&CFG_A, &tensors)?;

    println!(
        "It's just label index output for now... use \
        https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/_meta.py#L7 to look them up"
    );

    for argument in std::env::args().skip(1) {
        let img = image::ImageReader::open(&argument)?.decode()?.to_rgb8();

        // Lets first just tensorify the image.
        let mut t = Tensor::zeros(
            &[img.height() as usize, img.width() as usize, 3],
            &flash_powder::factory::TensorOptions {
                dtype: Some(DType::U8),
                ..Default::default()
            },
        )?;
        t.data_mut()?.copy_from_slice(img.as_raw().as_slice());

        // Convert that into a float tensor and fix the whole 255 situation.
        let img_float = t.to(&flash_powder::factory::ToOptions {
            dtype: Some(DType::F32),
            ..Default::default()
        })?;
        let divisor: Tensor = (255.0,).try_into()?;
        let img_tensor_ready = img_float.div(&divisor)?;

        // println!("img_tensor_ready.shape: {:?}", img_tensor_ready.shape());

        // Swap [h, w, 3] into [3, h, w]

        // I don't have dimension permutate yet... so this uglyness works.
        let w = img_tensor_ready.shape()[1];
        let h = img_tensor_ready.shape()[0];

        let channels_stacked = img_tensor_ready
            .permute(&[2, 0, 1])?
            .view(&[1, 3, h, w])?
            .to_owned()?;

        // println!("channels_stacked: {:?}", channels_stacked.shape());
        let r = vgg.forward(&channels_stacked)?;
        // https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/_meta.py#L7
        // Find the highest value, using some typical

        let max_item = r
            .f32s_ref()?
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap());

        println!(
            "{argument: >50}: max_item: {max_item: >10?}, which in _meta.py#L7 is line number: {}",
            max_item.unwrap().0 + 8
        );
    }

    Ok(())
}

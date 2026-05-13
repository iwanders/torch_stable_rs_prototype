# example_vgg

This implements an equivalent to [vgg11](https://github.com/pytorch/vision/blob/499ca5103b5c6abdf1973651d6eb3db9dfecdfbd/torchvision/models/vgg.py#L306-L329) using the `flash_powder` bindings.

To run this, download the [weights](https://download.pytorch.org/models/vgg11-8a719046.pth) and put them into the `data` directory.

Use the [convert_pth.py](./convert_pth.py) script to convert the pth file to a safetensors file.

From this directory run `cargo r -- ./data/*.JPEG`.

Use `./pytorch_vgg11.py ./data/*.JPEG` to do the same using `torchvision` and `pytorch`.

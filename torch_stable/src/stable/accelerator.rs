// https://github.com/pytorch/pytorch/blob/f2b47323ac2c438722c2db58aa31d9222676509d/torch/csrc/stable/accelerator.h#L28

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DeviceIndex(pub i32);

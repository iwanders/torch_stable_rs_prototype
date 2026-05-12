#!/usr/bin/env python3

import torch

# pip install safetensors
from safetensors.torch import save_file

state_dict = torch.load(
    "./data/vgg11-8a719046.pth", map_location="cpu", weights_only=True
)

save_file(state_dict, "./data/vgg11-8a719046.safetensors")

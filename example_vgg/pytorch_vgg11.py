#!/usr/bin/env python3

import torch
from torchvision import models

import torchvision.io as io
import sys

#model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
model = models.vgg11()
model.load_state_dict(torch.load("./data/vgg11-8a719046.pth", weights_only=True))

model.eval()


for f in sys.argv[1:]:

    # Loads image as a uint8 tensor [C, H, W]
    image_tensor = io.read_image(f)
    image_tensor = image_tensor.to(dtype=torch.float)

    # Scale to rgb 
    image_tensor = image_tensor / 255.0

    # Expand dimension to add the batch.
    image_tensor = image_tensor.reshape((1, 3, image_tensor.shape[1], image_tensor.shape[2]))
 
    # Run the model.
    with torch.no_grad():
        output = model(image_tensor)

    # Collect the best scoring index.
    best = (0, -float("inf"))
    for i, v in enumerate(output.tolist()[0]):
        if v > best[1]:
            best = (i,v)

    index, score = best

    print(f"{f: >50} {index: >10}  {score: >10.5f} which in _meta.py#L7 is line number {index + 8}")

import torch
from wtconvnext import WTConvNeXt
layer = WTConvNeXt(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))
rand_tensor = torch.randn(2, 3, 1024, 1024)
x = layer(rand_tensor)
print("x.shape")
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch, channels, 1, 1)
        return x * y.expand_as(x)

class ImprovedFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(ImprovedFPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            l_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            se_block = SEBlock(out_channels)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.se_blocks.append(se_block)

    def forward(self, inputs):
        # Assume inputs is a list of feature maps from the backbone
        c3, c4, c5 = inputs
        p5 = self.lateral_convs[2](c5)
        p4 = self.lateral_convs[1](c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_convs[0](c3) + F.interpolate(p4, scale_factor=2, mode='nearest')

        p5 = self.se_blocks[2](self.fpn_convs[2](p5))
        p4 = self.se_blocks[1](self.fpn_convs[1](p4))
        p3 = self.se_blocks[0](self.fpn_convs[0](p3))

        return p3, p4, p5

# Example usage
# Assuming backbone outputs are feature maps with 256, 512, and 1024 channels
fpn = ImprovedFPN(in_channels_list=[256, 512, 1024], out_channels=256)
# inputs = [c3, c4, c5] from backbone
# 定义backbone输出的特征图尺寸
c3_size = (1, 256, 128, 128)  # batch_size, channels, height, width
c4_size = (1, 512, 64, 64)
c5_size = (1, 1024, 32, 32)

# 随机生成backbone输出的特征图
c3 = torch.randn(c3_size)
c4 = torch.randn(c4_size)
c5 = torch.randn(c5_size)

# 打印生成的tensor的形状
print("c3 shape:", c3.shape)
print("c4 shape:", c4.shape)
print("c5 shape:", c5.shape)

# 使用ImprovedFPN
fpn = ImprovedFPN(in_channels_list=[256, 512, 1024], out_channels=256)

# 将随机生成的特征图输入到FPN中
outputs = [fpn([c3, c4, c5])]

print("hello")


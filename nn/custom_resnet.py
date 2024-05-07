'''
ResNet Implementation in PyTorch with Transpose Option.

This implementation extends traditional ResNet models to support both
standard and transposed convolutions, enabling its use in applications
like image reconstruction where upscaling is required. The model architecture
can dynamically adjust based on input parameters to fit various image sizes
and dimensions.

The model architecture is adaptable to ResNet variants including ResNet18,
ResNet34, ResNet50, ResNet101, and ResNet152 by specifying the block type
(BasicBlock or Bottleneck) and the sequence of layers.

References:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun,
        "Deep Residual Learning for Image Recognition." arXiv:1512.03385.
        The foundational paper introduces the concept of residual learning
        and provides the basis for various ResNet architectures.

This code provides flexible utility functions to instantiate various ResNet
configurations depending on whether transposing layers are required or not,
making it suitable for a wide range of vision-based tasks from classic
classification to advanced image synthesis.

Authors:
    - Kaiming He et al. for the original ResNet model and concept.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, transpose, stride=1):
        super(BasicBlock, self).__init__()
        kernel_size3 = 3
        kernel_size1 = 1
        if transpose is True and stride != 1:
            if stride == 3:
                Conv2d = nn.ConvTranspose2d
                kernel_size3 += 2
                kernel_size1 += 2
            else:
                Conv2d = nn.ConvTranspose2d
                kernel_size3 += 1
                kernel_size1 += 1
        else:
            Conv2d = nn.Conv2d

        self.conv1 = Conv2d(
            in_planes, planes, kernel_size=kernel_size3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, planes,
                          kernel_size=kernel_size1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, transpose, stride=1):
        super(Bottleneck, self).__init__()

        if transpose is True:
            Conv2d = nn.ConvTranspose2d

        else:
            Conv2d = nn.Conv2d
        
        kernel_size1 = 1
        kernel_size3 = 3
        if stride > 1:
            kernel_size3 += 1
            kernel_size1 += 1

        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=kernel_size3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes,
                          kernel_size=kernel_size1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, network_params, block, transpose):
        super(ResNet, self).__init__()
        obs_shape = network_params.obs_shape
        d_model = network_params.d_model
        self.expansion = 4
        self.num_blocks = 2
        # Compute feasible number of layers based on image dimensions
        num_channels, h, w = obs_shape

        ninp = d_model if transpose else num_channels
        noutp = num_channels if transpose else d_model

        # Limit the number of layers to either the calculated minimum or the user-defined maximum
        num_layers = network_params.num_layers
        self.initial_w = max(math.ceil(w/2**num_layers), 1)
        self.initial_h = max(math.ceil(h/2**num_layers), 1)

        layers = []
        current_d_model = ninp
        for i in range(num_layers):
            if not transpose:
                plane = d_model //  (2 ** (num_layers - i - 1))
            else:
                plane = d_model // (2 ** i)
            layer = self._make_layer(block, current_d_model, plane, self.num_blocks, transpose, stride=2)
            layers.append(layer)
            current_d_model = plane

        self.layers = nn.Sequential(*layers)

        if transpose:
            final = nn.Sequential(nn.Conv2d(current_d_model* block.expansion, noutp, 1), nn.Tanh())
        else:
            final = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(current_d_model * block.expansion, noutp)]
        self.final_layer = nn.Sequential(*final)

        self.d_model = d_model
        self.transpose = transpose      
        self.h, self.w = h, w  

    def _make_layer(self, block, in_planes, out_plaines, num_blocks, transpose, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(block(in_planes, out_plaines, transpose, stride))
            self.in_planes = out_plaines * block.expansion
            in_planes = out_plaines
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.transpose:
            x = x.view([x.size(0), -1, 1, 1])  # Ensuring it matches expected input dimensions for the initial 
            x = x.repeat([1, 1, self.initial_h, self.initial_w])
        x = self.layers(x)
        if self.transpose:
            x = F.interpolate(x, size=(self.h, self.w), mode='bilinear', align_corners=False)
        x = self.final_layer(x)
        return x

def cnn_ResNet(network_params):
    return ResNet(network_params, BasicBlock, transpose = False)

def transpose_cnn_ResNet(network_params):
    return ResNet(network_params, BasicBlock, transpose = True)

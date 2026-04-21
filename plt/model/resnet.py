import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_type="batch", affine=True, track_running_stats=True, momentum=0.1):
        super().__init__()
        if norm_type == "batch":
            norm_ = nn.BatchNorm2d
        elif norm_type == "instance":
            norm_= nn.InstanceNorm2d

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_(planes, affine=affine, track_running_stats=track_running_stats, momentum=momentum)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_(planes, affine=affine, track_running_stats=track_running_stats, momentum=momentum)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_(planes * self.expansion, affine=affine, track_running_stats=track_running_stats, momentum=momentum) 

        self.relu = nn.ReLU(inplace=False)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", norm_(planes * self.expansion, affine=affine, track_running_stats=track_running_stats, momentum=momentum)) 
            ]))


    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    


class ResNet(nn.Module):
    def __init__(self, block="bottleneck", layers: list = (3, 4, 23, 3), input_shape=None, norm_type="batch", output_dim=None, regression=False, features=False, return_all_features=False, affine=True, track_running_stats=True, momentum=0.1):
        self.inplanes = 64
        self.norm_type = norm_type
        self.momentum = momentum if momentum else None

        super().__init__()

        if block == "bottleneck":
            block = Bottleneck
        elif block == "basic":
            block = BasicBlock
        elif block == "affine-bottleneck":
            block = AffineBottleneck

        if input_shape is not None:
            channels_in = input_shape
        else:
            channels_in = 3

        self.is_regression = regression
        self.features = features
        self.return_all_features = return_all_features
        self.affine = affine 
        self.track_running_stats = track_running_stats
        self.layers = layers

        self.conv1 = nn.Conv2d(channels_in, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        
        if self.norm_type == "batch":
            self.bn1 = nn.BatchNorm2d(64, affine=affine, track_running_stats=track_running_stats, momentum=momentum) 
        elif self.norm_type == "instance":
            self.bn1 = nn.InstanceNorm2d(64, affine=affine, track_running_stats=track_running_stats, momentum=momentum)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        if not self.features:
            self.fc = nn.Linear(512 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=self.affine))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_type=self.norm_type, affine=self.affine, track_running_stats=self.track_running_stats, momentum=self.momentum))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None, norm_type=self.norm_type, affine=self.affine, track_running_stats=self.track_running_stats, momentum=self.momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if x.shape[-2:] != (1, 1):
            x = nn.AvgPool2d(x.shape[2:])(x)
            
        x = x.view(x.size(0), -1)

        if not self.features:
            x = self.fc(x)

        return x

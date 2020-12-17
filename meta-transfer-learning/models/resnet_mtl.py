import torch.nn as nn
from models.conv2d_mtl import Conv2dMtl, conv1x1, conv3x3, conv3x3mtl, conv1x1mtl

__all__ = ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, mtl=False):
        super(BasicBlock, self).__init__()
        if mtl:
            self.conv1 = conv3x3mtl(inplanes, planes, stride)
            self.conv2 = conv3x3mtl(planes, planes)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, mtl=False):
        super(Bottleneck, self).__init__()
        if mtl:
            self.conv1 = conv1x1mtl(inplanes, planes)
            self.conv2 = conv3x3mtl(planes, planes, stride)
            self.conv3 = conv1x1mtl(planes, planes * self.expansion)
        else:
            self.conv1 = conv1x1(inplanes, planes)
            self.conv2 = conv3x3(planes, planes, stride)
            self.conv3 = conv1x1(planes, planes * self.expansion)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], mtl=False, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        if mtl:
            self.Conv2d = Conv2dMtl
        else:
            self.Conv2d = nn.Conv2d

        self.conv1 = self.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], mtl=mtl)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, mtl=mtl)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, mtl=mtl)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, mtl=mtl)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, mtl=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, mtl=mtl)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, mtl=mtl))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet(model_type, **kwargs):
    if model_type == 'resnet10':
        return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    elif model_type == 'resnet12':
        return ResNet(Bottleneck, [1, 1, 1, 1], **kwargs)
    elif model_type == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_type == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif model_type == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet12(**kwargs):
    """Construct a Resnet-12 model
    """
    model = ResNet(Bottleneck, [1, 1, 1, 1], *kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    import torch
    resnet = resnet(model_type='resnet18', mtl=False)
    input = torch.randn(1, 3, 80, 80)
    out = resnet(input)
    print(out.size())   # [1, 512, 5, 5]

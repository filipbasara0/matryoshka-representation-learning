import torch.nn as nn
from torchvision import models


# adapted from https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/resnet_hacks.py
def _adapt_resnet_model(model):
    """
    Modifies some layers to handle the smaller CIFAR images, following
    the SimCLR paper. Specifically, replaces the first conv layer with 
    a smaller 3x3 kernel and 1x1 strides and removes the max pooling layer.
    """
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
    model.conv1 = conv1
    model.maxpool = nn.Identity()
    return model


class Squeeze(nn.Module):

    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)


def _prep_encoder(model):
    modules = list(model.children())[:-1]
    modules.append(nn.AdaptiveAvgPool2d(1))
    modules.append(Squeeze())

    return nn.Sequential(*modules)


def resnet18(modify_model=False):
    resnet = models.resnet18(weights=None)
    if modify_model:
        resnet = _adapt_resnet_model(resnet)
    return _prep_encoder(resnet)


def resnet50(modify_model=False):
    resnet = models.resnet50(weights=None)
    if modify_model:
        resnet = _adapt_resnet_model(resnet)
    return _prep_encoder(resnet)

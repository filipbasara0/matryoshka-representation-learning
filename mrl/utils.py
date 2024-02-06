import torch
from torchvision.datasets import CIFAR10, STL10

from mrl.aug import get_training_transforms, ViewGenerator
from mrl.encoders import resnet18, resnet50
from mrl.custom_datasets import tiny_imagenet, food101, imagenet1k


def get_dataset(args):
    dataset_name, dataset_path = args.dataset_name, args.dataset_path
    if dataset_name == "cifar10":
        return CIFAR10(dataset_path,
                       train=True,
                       download=True,
                       transform=ViewGenerator(get_training_transforms(32), 2))
    elif dataset_name == "stl10":
        return STL10(dataset_path,
                     split='unlabeled',
                     download=True,
                     transform=ViewGenerator(get_training_transforms(96), 2))
    elif dataset_name == "tiny_imagenet":
        return tiny_imagenet(transform=ViewGenerator(get_training_transforms(64), 2))
    elif dataset_name == "food101":
        return food101(transform=ViewGenerator(get_training_transforms(192), 2))
    elif dataset_name == "imagenet1k":
        return imagenet1k(transform=ViewGenerator(get_training_transforms(192), 2))

    raise Exception("Invalid dataset name - options are [cifar10, stl10]")


def get_encoder(model_name, modify_model=False):
    if model_name == "resnet18":
        return resnet18(modify_model)
    elif model_name == "resnet50":
        return resnet50(modify_model)
    raise Exception(
        "Invalid model name - options are [resnet18, resnet50]")


def accuracy(output, target, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.inference_mode
def get_feature_size(encoder):
    """Get the feature size from the encoder using a dummy input."""
    encoder.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = encoder(dummy_input)
    return output.shape[1]

from datasets import load_dataset

from mrl.custom_datasets.huggingface_dataset import HuggingfaceDataset


def tiny_imagenet(transform, split="train"):
    data = load_dataset('Maysee/tiny-imagenet', split=split)
    return HuggingfaceDataset(data, transform)


def food101(transform, split="train"):
    data = load_dataset('food101', split=split)
    return HuggingfaceDataset(data, transform)


def imagenet1k(transform, split="train"):
    data = load_dataset("imagenet-1k", split=split)
    return HuggingfaceDataset(data, transform)

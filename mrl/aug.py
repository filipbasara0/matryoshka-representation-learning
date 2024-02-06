import torchvision.transforms as transforms

import numpy as np

np.random.seed(42)


class ViewGenerator(object):

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def _grayscale_to_rgb(img):
    if img.mode == "L" or img.mode != "RGB":
        return img.convert("RGB")
    return img


def _round_up_to_odd(num):
    return np.ceil(num) // 2 * 2 + 1


def get_training_transforms(image_size):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.Lambda(_grayscale_to_rgb),
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.GaussianBlur(_round_up_to_odd(int(image_size * 0.1))),
        transforms.RandomSolarize(127, 0.5),
        transforms.ToTensor()
    ])


def get_inference_transforms(image_size=(96, 96)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.Lambda(_grayscale_to_rgb),
        transforms.ToTensor()
    ])

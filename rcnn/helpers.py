import detection.transforms as T
import torch
import torchvision

from PIL import Image


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def apply_random_crop(path_image):
    torch.manual_seed(17)
    im = Image.open(path_image)

    return torchvision.transforms.RandomCrop(size=(400, 400))(im)

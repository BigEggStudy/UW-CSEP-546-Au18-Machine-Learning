from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
import random

training_image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=30, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.CenterCrop(size=24),
    transforms.ToTensor(),
])

def Featurize(images, labels, contain_mirror = False, transform_batch_size = 0, shuffle = False):
    x = torch.stack([ transforms.ToTensor()(image) for image in images ])
    y = torch.Tensor([ [ yValue ] for yValue in labels ])

    if contain_mirror:
        mirror_x = torch.stack([ transforms.ToTensor()(image.transpose(Image.FLIP_LEFT_RIGHT)) for image in images ])
        mirror_y = torch.Tensor([ [ yValue ] for yValue in labels ])
        x = torch.cat((x, mirror_x), 0)
        y = torch.cat((y, mirror_y), 0)

    for i in range(transform_batch_size):
        transform_x = torch.stack([ training_image_transforms(image) for image in images ])
        transform_y = torch.Tensor([ [ yValue ] for yValue in labels ])
        x = torch.cat((x, transform_x), 0)
        y = torch.cat((y, transform_y), 0)
        if contain_mirror:
            mirror_x = torch.stack([ training_image_transforms(image.transpose(Image.FLIP_LEFT_RIGHT)) for image in images ])
            mirror_y = torch.Tensor([ [ yValue ] for yValue in labels ])
            x = torch.cat((x, mirror_x), 0)
            y = torch.cat((y, mirror_y), 0)

    if shuffle:
        indexes = torch.randperm(len(x))
        x_perm = x[indexes]
        y_perm = y[indexes]
        x, y = x_perm, y_perm

    return (x, y)

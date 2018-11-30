from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
import random

training_image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=30),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=24),
    transforms.ToTensor(),
    transforms.Normalize([127], [255]),
])

def Featurize(images, labels, transforms_data = False, batch_size = 4, shuffle = True, mirror = True):
    if not transforms_data:
        return ( torch.stack([ transforms.ToTensor()(image) for image in images ]), torch.Tensor([ [ yValue ] for yValue in labels ]) )
    else:
        data = []
        for i in range(batch_size):
            for (image, label) in zip(images, labels):
                data.append(( image, label ))
                if mirror:
                    data.append(( image.transpose(Image.FLIP_LEFT_RIGHT), label ))

        if shuffle:
            random.shuffle(data)

        return ( torch.stack([ training_image_transforms(image) for image, _ in data ]), torch.Tensor([ [ label ] for _, label in data ]) )

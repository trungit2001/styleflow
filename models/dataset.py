import os

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from models.images import (
    get_data_shuffle, 
    image_loader
)

class ImagePairDataset(Dataset):
    def __init__(
        self,
        args: dict, 
        transform: torchvision.transforms,
    ) -> None:
        super(ImagePairDataset, self).__init__()

        self.input_content_path = args.input_content_path
        self.input_style_path = args.input_style_path

        # load and shuffle data from file images pathx
        content_file_path = os.path.join(self.input_content_path, args.content_file)
        style_file_path = os.path.join(self.input_style_path, args.style_file)
        data_shuffled = get_data_shuffle(content_file_path, style_file_path)

        self.content_images = data_shuffled["content_images_shuffled"]
        self.style_images = data_shuffled["style_image_shuffled"]
        self.transform = transform

        print("Number of content images:", len(self.content_images))
        print("Number of style images:", len(self.style_images))

    def __getitem__(self, index):
        content_image_path = os.path.join(self.input_content_path, self.content_images[index])
        style_image_path = os.path.join(self.input_style_path, self.style_images[index])

        content_image = image_loader(content_image_path)
        style_image = image_loader(style_image_path)

        if self.transform is not None:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

        return content_image, style_image

    def __len__(self):
        return len(self.content_images)


def get_dataset(args: dict):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((args.height, args.width), scale=(0.7, 1.0)),
        transforms.ToTensor()
    ])

    dataset = ImagePairDataset(args, transform)

    return dataset

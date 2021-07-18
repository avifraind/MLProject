from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class PhotoMonetDataset(Dataset):
    def __init__(self, root_photo, root_monet, transform=None):
        self.root_photo = root_photo
        self.root_monet = root_monet
        self.transform = transform

        self.photo_images = os.listdir(root_photo)
        self.monet_images = os.listdir(root_monet)
        self.length_dataset = max(len(self.photo_images), len(self.monet_images)) # 1000, 1500
        self.photo_len = len(self.photo_images)
        self.monet_len = len(self.monet_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        photo_img = self.photo_images[index % self.photo_len]
        monet_img = self.monet_images[index % self.monet_len]

        photo_path = os.path.join(self.root_photo, photo_img)
        monet_path = os.path.join(self.root_monet, monet_img)

        photo_img = np.array(Image.open(photo_path).convert("RGB"))
        monet_img = np.array(Image.open(monet_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=photo_img, image0=monet_img)
            photo_img = augmentations["image"]
            monet_img = augmentations["image0"]

        return photo_img, monet_img

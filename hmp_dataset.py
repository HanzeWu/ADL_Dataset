import config
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from glob import glob
from PIL import Image

label_dict = dict(zip(config.ACTIONS, range(0, len(config.ACTIONS))))


class HMP(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        # get the address of all of the images in target folder and subfolder
        self.imgs_path = glob(os.path.join(config.HMP_RP, '*', '**.*'))
        self.imgs_path = glob(os.path.join(config.HMP_RP, '*', '**.*'))
        # print(self.imgs_path)
        self.data = [self.transforms(Image.open(img_path).convert("RGB")) for img_path in self.imgs_path]
        self.label = [label_dict[img_path.split('\\')[1]] for img_path in self.imgs_path]

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = label_dict[img_path.split('\\')[1]]
        data = Image.open(img_path).convert("RGB")
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs_path)

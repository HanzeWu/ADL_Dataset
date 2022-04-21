import torch
import torchvision.transforms as transforms
import os
from glob import glob
from PIL import Image
import config

model_path = "./model/" + "model-0.662577.pth"
png_path = "./data/HMP_pred/" + "Climb_stairs0.png"

# img aug
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.14846641, 0.11407742, 0.10433987], [0.13520855, 0.11859692, 0.11858473])
])

data = torch.tensor(transform(Image.open(png_path).convert("RGB"))).resize(1, 3, 224, 224)

from hmp_classification import CnnNet

model = CnnNet()

model.load_state_dict(torch.load(model_path))

y_pred = model(data)

_, pred = torch.max(y_pred, dim=1)
print(pred)
print(config.ACTIONS[pred])

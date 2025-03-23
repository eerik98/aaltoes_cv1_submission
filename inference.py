import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models.segmentation as segmentation
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import numpy as np
from own_dataset import OwnDataset
import time
import cv2
import yaml
import sys
from nets.EITLnet import SegFormer


model = SegFormer(num_classes=2, phi='b2', dual=True)

result_dir='/home/eerik/data_storage/DATA/aaltoes-2025-computer-vision-v-1/test/test/preds'
image_dir='/home/eerik/data_storage/DATA/aaltoes-2025-computer-vision-v-1/test/test/images'

checkpoint_path='checkpoints/21.pth'
checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

model.load_state_dict(torch.load(checkpoint_path,weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #IMAGENET normalization
])

#test_dataset = OwnDataset(path=img, img_transform=img_transform)
#test_loader=DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)


for image_file in os.listdir(image_dir):
    image = Image.open(os.path.join(image_dir,image_file)).convert("RGB")
    image_tensor = img_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device) #add batch_dim

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.argmin(output, dim=1).squeeze(0)
        output = output.cpu().numpy()
        cv2.imwrite(os.path.join(result_dir,image_file),(output*255).astype('uint8'))
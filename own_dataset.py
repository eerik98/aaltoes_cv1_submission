from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import random


class OwnDataset(Dataset):
    def __init__(self, path, img_transform, label_transform):
        forged_dir=os.path.join(path,'images')
        label_dir=os.path.join(path,'masks')
        original_dir=os.path.join(path,'originals')
        self.forged_dir = forged_dir
        self.label_dir = label_dir
        self.original_dir = original_dir
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(os.listdir(self.forged_dir))
        #
        #return 100

    def __getitem__(self, idx):
        forged_path = os.path.join(self.forged_dir,'image_'+str(idx)+'.png')
        original_path = os.path.join(self.original_dir,'image_'+str(idx)+'.png')
        label_path = os.path.join(self.label_dir,'image_'+str(idx)+'.png')  # Assuming masks have the same filename as images
        
        #try to get orignal 30 % of the time
        if os.path.exists(original_path) and random.random()<0.3:
            image = Image.open(original_path).convert("RGB")
            label = Image.new("L", (256, 256), (0, 0, 0))

        else:
            image = Image.open(forged_path).convert("RGB")
            label = Image.open(label_path).convert("L")  # Grayscale mask (foreground/background)

        
        image = self.img_transform(image)
        label = self.label_transform(label)


        label = torch.where(label > 0, 1, 0)  # Binarize mask (0: background, 1: foreground)
        label = torch.cat((label,1-label),dim=0).float()

        return image,label

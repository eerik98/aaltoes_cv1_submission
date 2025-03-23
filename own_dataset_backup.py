from torch.utils.data import Dataset
import torch
import os
from PIL import Image


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
        
        forged = Image.open(forged_path).convert("RGB")
        label_forged = Image.open(label_path).convert("L")  # Grayscale mask (foreground/background)

        #try to get orignal, if not found fully forged. In this case return the forged twice
        try:    
            original = Image.open(original_path).convert("RGB")
            label_original = Image.new("L", (256, 256), (0, 0, 0))
        except:
            original = forged
            label_original = label_forged

        original = self.img_transform(original)
        forged = self.img_transform(forged)
        label_forged = self.label_transform(label_forged)
        label_original = self.label_transform(label_original)

        label_forged = torch.where(label_forged > 0, 1, 0)  # Binarize mask (0: background, 1: foreground)
        label_original = torch.where(label_original > 0, 1, 0)  # Binarize mask (0: background, 1: foreground)
        label_original = torch.cat((label_original,1-label_original),dim=0).float()
        label_forged = torch.cat((label_forged,1-label_forged),dim=0).float()

        return original,label_original,forged,label_forged

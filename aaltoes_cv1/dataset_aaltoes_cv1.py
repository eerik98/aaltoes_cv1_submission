from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import random


class DatasetAaltoesCV1(Dataset):
    def __init__(self, path, img_transform, label_transform, only_inference=False):
        self.forged_dir=os.path.join(path,'images')
        self.label_dir=os.path.join(path,'masks')
        self.original_dir=os.path.join(path,'originals')
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.only_inference = only_inference
        
        dict1 = dict([(int(img.split('_')[1].split('.')[0]), img) for img in os.listdir(self.forged_dir)])
        if not only_inference:
            dict2 = dict([(int(img.split('_')[1].split('.')[0]), img) for img in os.listdir(self.label_dir)])
            dict3 = dict([(int(img.split('_')[1].split('.')[0]), img) for img in os.listdir(self.original_dir)])

        if only_inference:
            ks = set(dict1.keys())
        else:
            ks = set(dict1.keys()).union(set(dict2.keys()), set(dict3.keys()))
        self.img_idxs = list(range(len(ks)))
        self.img_to_idxs = dict(zip(sorted(ks), self.img_idxs))

        self.dataset = dict((self.img_to_idxs[k], [None, None, None]) for k in ks)
        for idx, img in dict1.items(): self.dataset[self.img_to_idxs[idx]][0] = img
        if not only_inference:
            for idx, img in dict2.items(): self.dataset[self.img_to_idxs[idx]][1] = img
            for idx, img in dict3.items(): self.dataset[self.img_to_idxs[idx]][2] = img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        forged_fn, label_fn, original_fn = self.dataset[idx]

        if self.only_inference:
            forged_path = os.path.join(self.forged_dir, forged_fn)
            image = Image.open(forged_path).convert("RGB")
            image = self.img_transform(image)
            return image, forged_fn

        # forged_path = os.path.join(self.forged_dir,'image_'+str(idx)+'.png')
        # original_path = os.path.join(self.original_dir,'image_'+str(idx)+'.png')
        # label_path = os.path.join(self.label_dir,'image_'+str(idx)+'.png')  # Assuming masks have the same filename as images
        forged_path = os.path.join(self.forged_dir, forged_fn)
        label_path = os.path.join(self.label_dir, label_fn)

        #try to get orignal 30 % of the time
        if original_fn is not None and random.random()<0.3:
            original_path = os.path.join(self.original_dir, original_fn)
            image = Image.open(original_path).convert("RGB")
            label = Image.new("L", (256, 256), 0)
        else:
            image = Image.open(forged_path).convert("RGB")
            label = Image.open(label_path).convert("L")  # Grayscale mask (foreground/background)

        image = self.img_transform(image)
        label = self.label_transform(label)

        label = torch.where(label > 0, 1, 0)  # Binarize mask (0: background, 1: foreground)
        label = torch.cat((label,1-label),dim=0).float()
        return image,label

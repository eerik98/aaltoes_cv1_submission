from nets.EITLnet import SegFormer
import torch
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
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml
import shutil
import cv2


#log_dir=os.path.join('./','logs')
#checkpoint_dir=os.path.join(log_dir,'checkpoints')

eval_file=open(os.path.join('checkpoints','eval.txt'),"w")

dataset_path='/home/eerik/data_storage/DATA/aaltoes-2025-computer-vision-v-1'
batch_size=4
device='cuda'
n_epochs=100
val_size=0.1

# Define transformations
img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #IMAGENET normalization
])
label_transform = T.Compose([
    T.ToTensor()
])

full_dataset = OwnDataset(path=os.path.join(dataset_path,'train/train'), img_transform=img_transform,label_transform=label_transform)

indices = np.arange(len(full_dataset))
train_indices, val_indices = train_test_split(indices, test_size=val_size, shuffle=True)

train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)


model = SegFormer(num_classes=2, phi='b2', dual=True)
checkpoint_path='checkpoints/21.pth'
checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(torch.load(checkpoint_path,weights_only=True))


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

model = model.to(device)

jaccard_index = torchmetrics.JaccardIndex(task='binary',num_classes=1)
jaccard_index.to(device)

for epoch in range(n_epochs):

    model.train()
    running_loss = 0.0
    
    for images,labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        preds = torch.argmin(outputs, dim=1).unsqueeze(1)
   
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    model.eval()
    jaccard_index.reset()
    with torch.no_grad():
        for images,labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            labels = (labels[:,0,:,:]).unsqueeze(1)
            outputs = model(images)
            preds = torch.argmin(outputs, dim=1).unsqueeze(1)
            jaccard_index.update(preds, labels)

    avg_jaccard = jaccard_index.compute()

    print(avg_jaccard)

    eval_file.write(f"Validation (IoU): {avg_jaccard:.4f}\n")
    eval_file.flush()

    torch.save(model.state_dict(), os.path.join('checkpoints',str(epoch)+'.pth'))

eval_file.close()












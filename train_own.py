from nets.EITLnet import SegFormer
import torch
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from own_dataset import OwnDataset
from tqdm import tqdm

eval_file=open(os.path.join('checkpoints','eval.txt'),"w")

# dataset_path='/home/eerik/data_storage/DATA/aaltoes-2025-computer-vision-v-1'
dataset_train_path=os.path.expanduser('~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1/train')
# dataset_train_path=os.path.expanduser('~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1_proper/train')
dataset_validation_path=os.path.expanduser('~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1_proper/validation')
batch_size=16
device='cuda'
n_epochs=100
workers=12
momentum=0.9
weight_decay=1e-2

if device != 'cpu':
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = False
    cudnn.enabled = False
    # cudnn.enabled = True

# Define transformations
img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #IMAGENET normalization
])
label_transform = T.Compose([
    T.ToTensor()
])

train_dataset = OwnDataset(path=dataset_train_path, img_transform=img_transform,label_transform=label_transform)
validation_dataset = OwnDataset(path=dataset_validation_path, img_transform=img_transform,label_transform=label_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=workers)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)

model = SegFormer(num_classes=2, phi='b2', dual=True)

start_epoch = 23
checkpoint_path=f'checkpoints/{start_epoch}.pth'
checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(torch.load(checkpoint_path,weights_only=True))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, betas=(momentum, 0.999), weight_decay=weight_decay)

model = model.to(device)

jaccard_index = torchmetrics.JaccardIndex(task='binary',num_classes=1)
jaccard_index.to(device)

for epoch in range(start_epoch+1, n_epochs):
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

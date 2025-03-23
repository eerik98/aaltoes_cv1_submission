import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from EITLNet.nets.EITLnet import SegFormer
from aaltoes_cv1.dataset_aaltoes_cv1 import DatasetAaltoesCV1
from aaltoes_cv1.utils import get_transforms, get_config

config = get_config('./config.yml')

eval_file=open(os.path.join('checkpoints','eval.txt'),"a")

# dataset_path='/home/eerik/data_storage/DATA/aaltoes-2025-computer-vision-v-1'
dataset_train_path=os.path.expanduser('~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1/train')
# dataset_train_path=os.path.expanduser('~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1_proper/train')
dataset_validation_path=os.path.expanduser('~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1_proper/validation')
batch_size=16
device='cuda'
n_epochs=100
workers=12
# momentum=0.9
# weight_decay=1e-2
# , betas=(momentum, 0.999), weight_decay=weight_decay

if device != 'cpu':
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    cudnn.deterministic = False
    cudnn.enabled = False
    # cudnn.enabled = True

img_transform, label_transform = get_transforms()

train_dataset = DatasetAaltoesCV1(path=dataset_train_path, img_transform=img_transform,label_transform=label_transform)
validation_dataset = DatasetAaltoesCV1(path=dataset_validation_path, img_transform=img_transform,label_transform=label_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=workers)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=workers)

model = SegFormer(num_classes=2, phi='b2', dual=True)

start_epoch = 23
checkpoint_path=f'checkpoints/{start_epoch}.pth'
checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(torch.load(checkpoint_path,weights_only=True))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

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

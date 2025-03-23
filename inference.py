import os
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from EITLNet.nets.EITLnet import SegFormer
from aaltoes_cv1.dataset_aaltoes_cv1 import DatasetAaltoesCV1
from aaltoes_cv1.utils import get_transforms, get_config

config = get_config('./config/config.yml')
checkpoint=config['train']['best_epoch']
test_dataset_path=os.path.expanduser(config['test_dataset'])

result_dir='preds'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

model = SegFormer(num_classes=2, phi='b2', dual=True)

checkpoint_path=f'checkpoints/{checkpoint}.pth'
checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

model.load_state_dict(torch.load(checkpoint_path,weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

img_transform, _ = get_transforms()

test_dataset = DatasetAaltoesCV1(path=test_dataset_path, img_transform=img_transform, label_transform=None, only_inference=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=12)

for images, filenames in tqdm(test_loader, desc="Inference"):
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmin(outputs, dim=1)
        predictions = predictions.cpu().numpy()

    for pred, filename in zip(predictions, filenames):
        cv2.imwrite(os.path.join(result_dir, filename), (pred * 255).astype('uint8'))

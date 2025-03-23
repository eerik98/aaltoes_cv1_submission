import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from aaltoes_cv1.dataset_aaltoes_cv1 import DatasetAaltoesCV1
import cv2
from EITLNet.nets.EITLnet import SegFormer
from tqdm import tqdm


model = SegFormer(num_classes=2, phi='b2', dual=True)

result_dir='preds'
image_dir=os.path.expanduser("~/workspace/aaltoes_cv1/kaggle_aaltoes_cv1/test")

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

checkpoint_path='checkpoints/23.pth'
checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

model.load_state_dict(torch.load(checkpoint_path,weights_only=True))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #IMAGENET normalization
])

test_dataset = DatasetAaltoesCV1(path=image_dir, img_transform=img_transform, label_transform=None, only_inference=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=12)


for images, filenames in tqdm(test_loader, desc="Inference"):
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        # Assuming the model's output has shape [batch, num_classes, H, W]
        predictions = torch.argmin(outputs, dim=1)  # shape: [batch, H, W]
        predictions = predictions.cpu().numpy()

    # Save each prediction with its corresponding filename.
    for pred, filename in zip(predictions, filenames):
        # Multiply by 255 if the mask is binary (0/1).
        cv2.imwrite(os.path.join(result_dir, filename), (pred * 255).astype('uint8'))

# for image_file in os.listdir(image_dir):
#     image = Image.open(os.path.join(image_dir,image_file)).convert("RGB")
#     image_tensor = img_transform(image)
#     image_tensor = image_tensor.unsqueeze(0).to(device) #add batch_dim

#     with torch.no_grad():
#         output = model(image_tensor)
#         output = torch.argmin(output, dim=1).squeeze(0)
#         output = output.cpu().numpy()
#         cv2.imwrite(os.path.join(result_dir,image_file), (output*255).astype('uint8'))

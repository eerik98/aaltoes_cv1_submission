import yaml
import torchvision.transforms as T

def get_config(config_fp='./config/config.yml'):
    with open(config_fp, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_transforms():
    img_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #IMAGENET normalization
    ])
    label_transform = T.Compose([
        T.ToTensor()
    ])
    return img_transform, label_transform

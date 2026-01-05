import torch
import torchvision.transforms as transforms
from PIL import Image

# ImageNet Normalization (Same as training)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def get_transform():
    """Returns the preprocessing transform for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

def process_image(image_file):
    """Opens an image file and applies transforms."""
    image = Image.open(image_file).convert('RGB')
    transform = get_transform()
    return transform(image).unsqueeze(0)  # Add batch dimension
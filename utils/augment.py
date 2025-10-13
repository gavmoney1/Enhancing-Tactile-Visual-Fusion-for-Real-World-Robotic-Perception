import os
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import random
import torch

# Paths
input_folder = "/home/gmoney/Enhancing-Tactile-Visual-Fusion-for-Real-World-Robotic-Perception/datasets/butterflies/butterflies_224"
output_folder = "/home/gmoney/Enhancing-Tactile-Visual-Fusion-for-Real-World-Robotic-Perception/datasets/butterflies/butterflies_224_aug"
os.makedirs(output_folder, exist_ok=True)

# Basic augmentations
basic_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),  # adjust size to your model
])

# Color jitter (small changes)
color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

# Gaussian blur
def random_blur(img):
    if random.random() < 0.5:
        radius = random.uniform(0.5, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius))
    return img

# Gaussian noise
def add_gaussian_noise(img):
    if random.random() < 0.5:
        tensor = transforms.ToTensor()(img)
        noise = torch.randn_like(tensor) * 0.05  # adjust noise level
        tensor = torch.clamp(tensor + noise, 0, 1)
        return transforms.ToPILImage()(tensor)
    return img

# Discrete rotation: 0°, 90°, 180°, 270°
def discrete_rotation(img):
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    return img.rotate(angle)

# Full augmentation pipeline
def augment_image(img):
    img = basic_aug(img)
    img = discrete_rotation(img)
    img = color_jitter(img)
    img = random_blur(img)
    img = add_gaussian_noise(img)
    return img

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGB")

        for i in range(3):  # number of augmentations per image
            aug_img = augment_image(img)
            aug_img.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug{i}.png"))

print("Augmentation complete!")

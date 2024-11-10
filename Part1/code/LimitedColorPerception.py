import torch
from torch.utils.data import Dataset
from torchvision import transforms
from RGBAndContrastTransform import RGBAndContrastTransform

from PIL import Image

class LimitedColorPerceptionDataset(Dataset):
    def __init__(self, dataset, month_age, max_age=12.0):
        """
        Initializes the dataset with limited color perception based on infant color development.

        Args:
            dataset (Dataset): The original dataset to wrap.
            month_age (float): The age in months of the subject.
            max_age (float, optional): The age in months when full color perception is reached. Defaults to 12.0.
        """
        self.dataset = dataset
        self.month_age = month_age
        self.max_age = max_age

         # Define the saturation value as a linear function of month_age
        linear_value = 0.1 + (0.9 * (self.month_age / self.max_age))
        
        # Set up transformation with the calculated saturation value
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            RGBAndContrastTransform(max_value=1, channel=0, contrast_factor=1.5), # red channel
            RGBAndContrastTransform(max_value=1, channel=1, contrast_factor=1.5), # green channel
            RGBAndContrastTransform(max_value=1, channel=2, contrast_factor=1.5) # blue channel
        ])
                
    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Ensure the image is in RGB format
        if isinstance(image, Image.Image):
            image = image.convert('RGB')

        # Apply the transformation
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)
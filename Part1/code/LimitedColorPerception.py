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

        self.settings = {
            0: {'red': 0.1, 'green': 0.05, 'blue': 0.05, 'contrast': 0.1},
            1: {'red': 0.3, 'green': 0.1, 'blue': 0.1, 'contrast': 0.2},
            2: {'red': 0.5, 'green': 0.2, 'blue': 0.15, 'contrast': 0.3},
            3: {'red': 0.7, 'green': 0.5, 'blue': 0.3, 'contrast': 0.4},
            4: {'red': 0.8, 'green': 0.6, 'blue': 0.4, 'contrast': 0.6},
            5: {'red': 0.9, 'green': 0.8, 'blue': 0.6, 'contrast': 0.7},
            6: {'red': 1.0, 'green': 0.9, 'blue': 0.8, 'contrast': 0.8},
            7: {'red': 1.0, 'green': 0.95, 'blue': 0.85, 'contrast': 0.8},
            8: {'red': 1.0, 'green': 1.0, 'blue': 0.9, 'contrast': 0.9},
            9: {'red': 1.0, 'green': 1.0, 'blue': 1.0, 'contrast': 1.0},
            10: {'red': 1.0, 'green': 1.0, 'blue': 1.0, 'contrast': 1.0},
            11: {'red': 1.0, 'green': 1.0, 'blue': 1.0, 'contrast': 1.0},
            12: {'red': 1.0, 'green': 1.0, 'blue': 1.0, 'contrast': 1.0},
        }

        # Set up transformation with the calculated saturation value
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            RGBAndContrastTransform(max_value=self.settings[self.month_age]['red'], channel=0, contrast_factor=self.settings[self.month_age]['contrast']), # red channel
            RGBAndContrastTransform(max_value=self.settings[self.month_age]['green'], channel=1, contrast_factor=self.settings[self.month_age]['contrast']), # green channel
            RGBAndContrastTransform(max_value=self.settings[self.month_age]['blue'], channel=2, contrast_factor=self.settings[self.month_age]['contrast']) # blue channel
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
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class VisualAcuityDataset(Dataset):
    def __init__(self, dataset, month_age, max_age=5.0, max_sigma=5.0):
        """
        Initializes the dataset with flexible visual acuity.

        Args:
            dataset (Dataset): The original dataset to wrap.
            month_age (float): The age in months of the subject.
            max_age (float, optional): The age in months when full visual acuity is reached. Defaults to 12.0.
            max_sigma (float, optional): The maximum sigma value for Gaussian blur. Defaults to 5.0.
        """
        self.dataset = dataset
        self.month_age = month_age
        self.max_age = max_age
        self.max_sigma = max_sigma

        # Calculate the visual acuity (VA) based on the month age
        self.VA = 600 - (580 * (self.month_age / self.max_age))
        self.VA = max(20, min(600, self.VA))  # Clamp VA between 20 and 600

        # Map the visual acuity to a sigma value for Gaussian blur
        self.sigma = self.max_sigma * ((self.VA - 20) / 580)
        self.sigma = max(0.0, self.sigma)  # Ensure sigma is non-negative

        # Determine the kernel size for the Gaussian blur
        if self.sigma > 0:
            self.kernel_size = int(2 * round(3 * self.sigma) + 1)
            if self.kernel_size % 2 == 0:
                self.kernel_size += 1
        else:
            self.kernel_size = 1  # No blur needed

        # Define the image transformation
        if self.sigma > 0:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
            ])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index after applying the visual acuity transformation.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: (transformed image, label)
        """
        image, label = self.dataset[idx]

        # Ensure the image is in RGB format
        if isinstance(image, Image.Image):
            image = image.convert('RGB')

        # Apply the transformation
        image = self.transform(image)
        return image, label
    

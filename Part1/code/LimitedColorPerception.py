import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

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

    def limited_color_perception(self, img):
        """
        Applies color perception limitations based on the month age.

        Args:
            img (PIL Image): The image to transform.

        Returns:
            PIL Image: The transformed image with limited color perception.
        """
        img = np.array(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Define weights based on month age stages
        if self.month_age < 3:
            # Emphasize red only
            red_weight = 1.0
            blue_weight = 0.0
            green_weight = 0.0
        elif self.month_age < 6:
            # Emphasize red and blue, partially green
            red_weight = 1.0
            blue_weight = 1.0
            green_weight = 0.5
        else:
            # Emphasize all colors (closer to full spectrum)
            red_weight = 1.0
            blue_weight = 1.0
            green_weight = 1.0

        # Adjust saturation based on these weights
        h_red = ((h < 60) | (h > 240))
        h_blue = ((h >= 120) & (h <= 240))
        h_green = ((h >= 60) & (h < 120))

        s[h_red] = np.clip(s[h_red] * red_weight, 0, 255).astype(np.uint8)
        s[h_blue] = np.clip(s[h_blue] * blue_weight, 0, 255).astype(np.uint8)
        s[h_green] = np.clip(s[h_green] * green_weight, 0, 255).astype(np.uint8)

        # Merge the adjusted channels
        hsv = cv2.merge([h, s, v])

        # Apply slight contrast adjustment
        contrast = 0.9 + (0.1 * (self.month_age / self.max_age))
        hsv = cv2.convertScaleAbs(hsv, alpha=contrast)

        # Convert back to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index after applying the limited color perception transformation.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: (transformed image, label)
        """
        image, label = self.dataset[idx]

        # Ensure the image is in RGB format
        if isinstance(image, Image.Image):
            image = image.convert('RGB')

        # Apply the limited color perception transformation
        transformed_image = self.limited_color_perception(image)
        transformed_image = transforms.ToTensor()(transformed_image)

        return transformed_image, label

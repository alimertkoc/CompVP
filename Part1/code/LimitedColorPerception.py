from torch.utils.data import Dataset
from torchvision import transforms
from RGBAndContrastTransform import RGBAndContrastTransform
from PIL import Image


class LimitedColorPerceptionDataset(Dataset):
    def __init__(self, dataset, month_age, max_age=6):
        """
        Initializes the dataset with limited color perception based on infant color development.

        Args:
            dataset (Dataset): The original dataset to wrap.
            month_age (float): The age in months of the subject.
            max_age (float, optional): The age in months when full color perception is reached. Defaults to 6.
        """
        self.dataset = dataset
        self.month_age = month_age
        self.max_age = max_age

        # Initialize the color properties
        r, g, b, c = 0.8, 0.6, 0.4, 0.7

        # Define the color properties for each month age
        self.color_properties = {
            i: {
                "red": min(1, r + (1 - r) / 4 * i),
                "green": min(1, g + (1 - g) / 4 * i),
                "blue": min(1, b + (1 - b) / 4 * i),
                "contrast": min(1, c + (1 - c) / 4 * i),
            }
            for i in range(self.max_age + 1)
        }

        # Set up transformation with the calculated saturation value
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                RGBAndContrastTransform(
                    max_value=self.color_properties[self.month_age]["red"],
                    channel=0,
                    contrast_factor=self.color_properties[self.month_age]["contrast"],
                ),  # red channel
                RGBAndContrastTransform(
                    max_value=self.color_properties[self.month_age]["green"],
                    channel=1,
                    contrast_factor=self.color_properties[self.month_age]["contrast"],
                ),  # green channel
                RGBAndContrastTransform(
                    max_value=self.color_properties[self.month_age]["blue"],
                    channel=2,
                    contrast_factor=self.color_properties[self.month_age]["contrast"],
                ),  # blue channel
            ]
        )

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Ensure the image is in RGB format
        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Apply the transformation
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

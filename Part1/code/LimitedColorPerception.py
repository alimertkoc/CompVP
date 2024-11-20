from torch.utils.data import Dataset
from torchvision import transforms
from RGBAndContrastTransform import RGBAndContrastTransform
from PIL import Image


class LimitedColorPerceptionDataset(Dataset):
    def __init__(self, dataset, month_age, max_age=6) -> None:
        """
        Args:
            dataset: The dataset to be wrapped and modified.
            month_age (int): The age in months of the person.
            max_age (int): The age in months when full color perception is reached.
        """
        self.dataset = dataset
        self.month_age = month_age
        self.max_age = max_age
        # Max age value is choosen as a result of trial and error

        # Inital color properties (red, green, blue, and contrast)
        red, green, blue, contrast = 0.8, 0.6, 0.4, 0.7

        # Color properties for each month age
        # 4 is the number of steps to reach the maximum value (based on papers' results)
        # Increment value (e.g., (1 - r) / 4) is choosen as if it's equal for each step
        self.color_properties = {
            i: {
                "red": min(1, red + (1 - red) / 4 * i),
                "green": min(1, green + (1 - green) / 4 * i),
                "blue": min(1, blue + (1 - blue) / 4 * i),
                "contrast": min(1, contrast + (1 - contrast) / 4 * i),
            }
            for i in range(self.max_age + 1)
        }

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                RGBAndContrastTransform(
                    max_value=self.color_properties[self.month_age]["red"],
                    channel=0,
                    contrast_factor=self.color_properties[self.month_age]["contrast"],
                ),
                RGBAndContrastTransform(
                    max_value=self.color_properties[self.month_age]["green"],
                    channel=1,
                    contrast_factor=self.color_properties[self.month_age]["contrast"],
                ),
                RGBAndContrastTransform(
                    max_value=self.color_properties[self.month_age]["blue"],
                    channel=2,
                    contrast_factor=self.color_properties[self.month_age]["contrast"],
                ),
            ]
        )

    def __getitem__(self, idx):  # required method for Dataset
        image, label = self.dataset[idx]

        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        image = self.transform(image)
        return image, label

    def __len__(self):  # required method for Dataset
        return len(self.dataset)

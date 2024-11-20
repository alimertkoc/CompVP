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

        # Property values are choosen based on limited color perception paper and trial and error
        self.color_properties = {
            0: {"red": 0.5, "green": 0.47, "blue": 0.32, "contrast": 0.62},
            1: {"red": 0.66, "green": 0.55, "blue": 0.35, "contrast": 0.65},
            2: {"red": 0.72, "green": 0.65, "blue": 0.45, "contrast": 0.68},
            3: {"red": 0.78, "green": 0.73, "blue": 0.55, "contrast": 0.7},
            4: {"red": 0.84, "green": 0.8, "blue": 0.65, "contrast": 0.73},
            5: {"red": 0.9, "green": 0.85, "blue": 0.76, "contrast": 0.76},
            6: {"red": 0.96, "green": 0.89, "blue": 0.85, "contrast": 0.8},
            7: {"red": 0.97, "green": 0.92, "blue": 0.95, "contrast": 0.84},
            8: {"red": 0.98, "green": 0.95, "blue": 0.97, "contrast": 0.88},
            9: {"red": 0.99, "green": 0.97, "blue": 0.98, "contrast": 0.9},
            10: {"red": 1.0, "green": 1.0, "blue": 0.99, "contrast": 0.92},
            11: {"red": 1.0, "green": 1.0, "blue": 1.0, "contrast": 0.94},
            12: {"red": 1.0, "green": 1.0, "blue": 1.0, "contrast": 0.96},
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

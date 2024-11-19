from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class VisualAcuityDataset(Dataset):
    def __init__(self, dataset, month_age, max_age=5, max_sigma=5):
        """
        Args:
            month_age (float): The age in months of the person.
            max_age (float, optional): The age in months when full visual acuity is reached. Defaults to 5.0.
            max_sigma (float, optional): The maximum sigma value for Gaussian blur. Defaults to 5.0.
        """
        self.dataset = dataset
        self.month_age = month_age
        self.max_age = max_age
        self.max_sigma = max_sigma
        # Max age and sigma values choosen as a result of trial and error

        # Visual acuity (VA) based on the month age
        # Max and min VA values are based on the paper (20/600 and 20/20)
        self.VA = 600 - (580 * (self.month_age / self.max_age))
        self.VA = max(20, min(600, self.VA))

        # Sigma value for Gaussian blur by using VA
        self.sigma = max(0.0, self.max_sigma * ((self.VA - 20) / 580))

        # Kernel size for the Gaussian blur
        if self.sigma > 0:
            self.kernel_size = int(2 * round(3 * self.sigma) + 1)
            if self.kernel_size % 2 == 0:
                self.kernel_size += 1
        elif self.sigma == 0:
            self.kernel_size = 1  # No blur

        if self.sigma > 0:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.GaussianBlur(
                        kernel_size=self.kernel_size, sigma=self.sigma
                    ),
                ]
            )
        elif self.sigma == 0:
            self.transform = transforms.ToTensor()

    def __len__(self):  # required method for Dataset
        return len(self.dataset)

    def __getitem__(self, idx):  # required method for Dataset
        image, label = self.dataset[idx]

        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        image = self.transform(image)
        return image, label

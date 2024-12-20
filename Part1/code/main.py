from VisualAcuityDataset import VisualAcuityDataset
from LimitedColorPerceptionDataset import LimitedColorPerceptionDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import STL10

# STL10 dataset from torchvision.datasets 
dataset = STL10("./data", split="train", download=True)
month_ages = list(range(0, 7))

# Datasets for both properties (Visual Acuity and Limited Color Perception)
va_dataset = [VisualAcuityDataset(dataset, month_age=month_age) for month_age in month_ages]
lcp_dataset = [LimitedColorPerceptionDataset(dataset, month_age=month_age) for month_age in month_ages]

x = input('Property ("lcp" for Color Perception | "va" for Visual Acuity): ').lower().strip()

# DataLoaders for each month for the selected property
if x == "va":
    data_loaders = [DataLoader(va_dataset[month_age], batch_size=32, shuffle=False) for month_age in month_ages]
elif x == "lcp":
    data_loaders = [DataLoader(lcp_dataset[month_age], batch_size=32, shuffle=False) for month_age in month_ages]


# The vision progression in 6 months for the selected property 
def show_images_in_sequence(data_loaders):
    plt.figure(figsize=(25, 25))
    for month_age, data_loader in enumerate(data_loaders):
        for images, _ in data_loader:
            first_image = images[15].permute(1, 2, 0).numpy() * 255
            first_image = first_image.astype(np.uint8)

            plt.subplot(1, len(data_loaders), month_age + 1)
            plt.imshow(first_image)
            plt.axis("off")
            plt.title(f"{month_age} months")
            break

    plt.show()


show_images_in_sequence(data_loaders)

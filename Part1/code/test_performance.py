import time
from RGBAndContrastTransform import RGBAndContrastTransform
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root_dir = "./data"
# Set up a sample transformation (e.g., resizing and normalizing)
transform_LC = transforms.Compose(
    [
        transforms.ToTensor(),
        RGBAndContrastTransform(max_value=0.75, channel=0, contrast_factor=1.5),
        RGBAndContrastTransform(max_value=0.75, channel=1, contrast_factor=1.5),
        RGBAndContrastTransform(max_value=0.75, channel=2, contrast_factor=1.5),
    ]
)

transform_VA = transforms.Compose(
    [transforms.ToTensor(), transforms.GaussianBlur(kernel_size=1, sigma=(0.1, 2.0))]
)

# Load dataset with and without transformations
dataset_LC_with_transform = datasets.STL10(root=root_dir, transform=transform_LC)
dataset_VA_with_transform = datasets.STL10(root=root_dir, transform=transform_VA)
dataset_without_transform = datasets.STL10(
    root=root_dir, transform=transforms.ToTensor()
)  # only converting to tensor, minimal transformation

# Create DataLoader instances
batch_size = 100
loader_with_transform_LC = DataLoader(
    dataset_LC_with_transform, batch_size=batch_size, shuffle=False
)
loader_with_transform_VA = DataLoader(
    dataset_VA_with_transform, batch_size=batch_size, shuffle=False
)

loader_without_transform = DataLoader(
    dataset_without_transform, batch_size=batch_size, shuffle=False
)


# Function to measure the time for loading a batch of images
def measure_loading_time(loader):
    start_time = time.time()
    # Load one batch
    for _ in loader:
        break
    end_time = time.time()
    return end_time - start_time


# Measure loading time with transformation
time_with_transform_LC = measure_loading_time(loader_with_transform_LC)
time_with_transform_VA = measure_loading_time(loader_with_transform_VA)

# Measure loading time without transformation
time_without_transform = measure_loading_time(loader_without_transform)

print(f"Loading time with transformations LC: {time_with_transform_LC:.4f} seconds")
print(f"Loading time with transformations VA: {time_with_transform_VA:.4f} seconds")
print(f"Loading time without transformations: {time_without_transform:.4f} seconds")

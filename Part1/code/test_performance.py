import time
from RGBAndContrastTransform import RGBAndContrastTransform
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

root_dir = "./data"

# Sample transformation for Limited Color Perception (Random argument values)
transform_LCP = transforms.Compose(
    [
        transforms.ToTensor(),
        RGBAndContrastTransform(max_value=0.75, channel=0, contrast_factor=1.5),
        RGBAndContrastTransform(max_value=0.75, channel=1, contrast_factor=1.5),
        RGBAndContrastTransform(max_value=0.75, channel=2, contrast_factor=1.5),
    ]
)

# Sample transformation for Visual Acuity (Random argument values)
transform_VA = transforms.Compose(
    [transforms.ToTensor(), transforms.GaussianBlur(kernel_size=1, sigma=(0.1, 2.0))]
)

# Datasets with and without transformation
dataset_LCP_with_transform = datasets.STL10(root=root_dir, transform=transform_LCP)
dataset_VA_with_transform = datasets.STL10(root=root_dir, transform=transform_VA)
dataset_without_transform = datasets.STL10(root=root_dir, transform=transforms.ToTensor())

# Dataloaders for each dataset (100 images per batch)
batch_size = 100
loader_with_transform_LCP = DataLoader(dataset_LCP_with_transform, batch_size=batch_size, shuffle=False)
loader_with_transform_VA = DataLoader(dataset_VA_with_transform, batch_size=batch_size, shuffle=False)
loader_without_transform = DataLoader(dataset_without_transform, batch_size=batch_size, shuffle=False)


# The total time for loading a batch of images
# The measuring function is based on a medium article by `Benjamin Bodner` and PyTorch documentation
def measure_loading_time(loader):
    start_time = time.time()
    # Load one batch
    for _ in loader:
        break
    end_time = time.time()
    return end_time - start_time


# Loading time with transformation
time_with_transform_LCP = measure_loading_time(loader_with_transform_LCP)
time_with_transform_VA = measure_loading_time(loader_with_transform_VA)

# Loading time without transformation
time_without_transform = measure_loading_time(loader_without_transform)

print(f"Loading time with transformations LC: {time_with_transform_LCP:.4f} seconds")
print(f"Loading time with transformations VA: {time_with_transform_VA:.4f} seconds")
print(f"Loading time without transformations: {time_without_transform:.4f} seconds")

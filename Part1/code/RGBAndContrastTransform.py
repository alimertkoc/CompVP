import torch
import torchvision.transforms.functional as F


class RGBAndContrastTransform:
    def __init__(
        self,
        min_value=0.0,
        max_value=1.0,
        channel=None,
        contrast_factor=1.0,
    ):
        """
        Transforms color property values (color channels and contrast) of an image tensor.

        Args:
        - min_value (float): The minimum value to clamp the color property to.
        - max_value (float): The maximum value to clamp the color property to.
        - channel (int or None): The channel index to apply clamping to. (`None` for all channels)
        - contrast_factor (float): The factor to adjust the contrast of the image.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.channel = channel
        self.contrast_factor = contrast_factor

    def __call__(self, img_tensor: torch.Tensor):
        img_tensor = F.adjust_contrast(img_tensor, self.contrast_factor)
        if self.channel is not None:
            # Color property value change of the selected channel
            img_tensor[self.channel] = torch.clamp(
                img_tensor[self.channel],
                self.min_value,
                self.max_value,
            )
        else:
            # Color property value change of all channels
            img_tensor = torch.clamp(img_tensor, self.min_value, self.max_value)
        return img_tensor

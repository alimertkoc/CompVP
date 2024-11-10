import torch
from torchvision import transforms

class ClampRGBTransform:
    def __init__(self, min_value=0.0, max_value=1.0, channel=None, increment=0.0):
        """
        Custom transform to clamp values of an image tensor.

        Args:
        - min_value (float): Minimum value for clamping.
        - max_value (float): Maximum value for clamping.
        - channel (int or None): The channel index to apply clamping to (e.g., 0 for Red).
          If None, applies to all channels.
        - increment (float): The value to increment the channel(s) before clamping.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.channel = channel
        self.increment = increment

    def __call__(self, img_tensor):
        if self.channel is not None:
            # Apply increment and clamp only to the specified channel
            img_tensor[self.channel] = torch.clamp(img_tensor[self.channel] + self.increment, self.min_value, self.max_value)
        else:
            # Apply increment and clamp to all channels
            img_tensor = torch.clamp(img_tensor + self.increment, self.min_value, self.max_value)
        return img_tensor

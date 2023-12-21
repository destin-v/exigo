import torch
from torch import nn


class BasicCNN(nn.Module):
    """Example Convolutional Neural Network (CNN) for MNIST identification."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding="valid")
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=7, kernel_size=3, padding="valid")
        self.conv3 = nn.Conv2d(in_channels=7, out_channels=5, kernel_size=3, padding="valid")
        self.conv4 = nn.Conv2d(in_channels=5, out_channels=1, kernel_size=3, padding="valid")

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()

        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear1 = nn.Linear(in_features=400, out_features=10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns prediction as well as the embedding (one layer before final prediction).  The embedding is used for visualizing the projections.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Prediction and embedding tensors.
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        e = self.flatten(x)
        x = self.linear1(e)

        return x, e

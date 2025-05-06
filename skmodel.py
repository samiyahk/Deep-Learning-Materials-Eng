import torch
import torch.nn as nn

__all__ = ['CNN']

class CNN(nn.Module):
    def __init__(self,
                 num_classes=31,
                 conv1_channels=128,
                 conv2_channels=512):
        super(CNN, self).__init__()
        # Convolutional feature extractor with tunable channel sizes
        self.features = nn.Sequential(
            # First conv block: input 1 channel -> conv1_channels
            nn.Conv2d(1, conv1_channels, kernel_size=5, stride=1, padding=2),  # N x 50 x 50 -> N x conv1 x 50 x 50
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # N x conv1 x 25 x 25

            # Second conv block: conv1_channels -> conv2_channels
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1), # N x conv2 x 25 x 25
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # N x conv2 x 12 x 12

            # Third conv block: conv2_channels -> conv2_channels (kept same)
            nn.Conv2d(conv2_channels, conv2_channels, kernel_size=3, padding=1), # N x conv2 x 12 x 12
            nn.ReLU(inplace=True),

            # Fourth conv block: conv2_channels -> 1024
            nn.Conv2d(conv2_channels, 1024, kernel_size=3, padding=1),           # N x 1024 x 12 x 12
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                              # N x 1024 x 6 x 6
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))  # N x 1024 x 3 x 3

        # Classifier head remains unchanged
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        # x: N x 1 x H x W
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Test function to verify output shape
if __name__ == "__main__":
    x = torch.randn(1, 1, 50, 50)
    model = CNN(num_classes=31)
    print(model(x).shape)  # should be torch.Size([1, 31])

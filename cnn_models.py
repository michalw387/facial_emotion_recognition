from torch import nn

import config


class Model3D(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8, output_dim=config.NUM_EMOTIONS):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequence = nn.Sequential(
            # 3 x 100 x 100
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=hidden_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 16 x 50 x 50
            nn.Conv2d(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 32 x 25 x 25
            nn.Conv2d(
                in_channels=hidden_dim * 4,
                out_channels=hidden_dim * 8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5),
            # 64 x 5 x 5
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8 * 5 * 5, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.sequence(x)


class Model2D(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=config.NUM_EMOTIONS):
        super().__init__()
        self.flatten = nn.Flatten()
        self.sequence = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim * 2, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 X 50 X 50
            nn.Conv2d(
                hidden_dim * 2, hidden_dim * 4, kernel_size=2, stride=1, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 X 25 X 25
            nn.Conv2d(
                hidden_dim * 4, hidden_dim * 8, kernel_size=2, stride=1, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(),
            nn.MaxPool2d(5, 5),  # 64 X 5 X 5
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8 * 5 * 5, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.sequence(x)

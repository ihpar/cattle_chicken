import torch.nn as nn


class SensorRegressor(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )

        self.flattened_size = 8 * input_dim

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

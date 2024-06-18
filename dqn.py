import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(15, 15), stride=(5, 5))  # 400 x 300 -> 67 x 57
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))  # 67 x 57 -> 31 x 26
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))  # 31 x 26 -> 29 x 24
        self.xfc = nn.Linear(64*25*35, 1024)  # append survive time
        self.yfc = nn.Linear(1, 512)
        self.output = nn.Linear(1536, n_actions)

    def forward(self, x, y):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = F.relu(self.xfc(x))
        y = F.relu(self.yfc(y))
        z = torch.cat((x, y), 1)
        return self.output(z)

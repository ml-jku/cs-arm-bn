import torch
import torch.nn as nn


class ContextNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (kernel_size - 1) // 2

        self.context_net = nn.Sequential(
                                nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=padding),
                                nn.BatchNorm2d(hidden_dim),
                                nn.ReLU(),
                                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                                nn.BatchNorm2d(hidden_dim),
                                nn.ReLU(),
                                nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=padding)
                            )


    def forward(self, x):
        out = self.context_net(x)
        return out
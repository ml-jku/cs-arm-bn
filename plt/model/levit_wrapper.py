import torch
import torch.nn as nn
from transformers import LevitForImageClassification

class LeViTLogitsWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LevitForImageClassification(config)

    def forward(self, x):
        # forward pass
        outputs = self.model(x)
        # return logits tensor directly
        return outputs.logits
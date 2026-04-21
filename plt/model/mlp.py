import torch
from torch import nn
from torch.nn.utils import weight_norm as wn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bias=True, batchnorm=False, layernorm=False, dropout=0.5, norm_reduce=False):
        super().__init__()
        self.norm_reduce = norm_reduce

        self.layers = nn.ModuleList()

        batchnorm = nn.BatchNorm1d(hidden_dim) if batchnorm else nn.Identity()
        layernorm = nn.LayerNorm(hidden_dim) if layernorm else nn.Identity()

        for layer in range(n_layers):
            dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                               nn.Linear(dim, hidden_dim, bias=bias),
                               batchnorm,
                               layernorm,
                               nn.Dropout(p=dropout),
                               nn.ReLU()
            ))


        dim = input_dim if n_layers == 0 else hidden_dim

        self.fc = nn.Linear(dim, output_dim, bias=bias)

                        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.fc(x)
        
        if self.norm_reduce:
            x = torch.norm(x)

        return x

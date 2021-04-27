import torch

import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Linear, self).__init__()
        # self.mlp = nn.Conv2d(in_dims, out_dims, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        self.mlp = nn.Linear(in_dims, out_dims)

    def forward(self, x):
        return self.mlp(x)
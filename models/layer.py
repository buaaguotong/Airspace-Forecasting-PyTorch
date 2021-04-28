import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Linear, self).__init__()
        self.mlp = nn.Conv2d(in_dims, out_dims, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class DiffGraphConv(nn.Module):

    def __init__(self, in_dims, out_dims, adj_mats, orders=2, enable_bias=True):
        super(DiffGraphConv, self).__init__()
        self.in_dims = in_dims*(1+orders*len(adj_mats))
        self.out_dims = out_dims
        self.adj_mats = adj_mats
        self.enable_bias = enable_bias
        self.orders = orders

        self.linear = Linear(self.in_dims, out_dims)

    def forward(self, x):

        output = [x]
        for adj in self.adj_mats:
            x_mul = torch.einsum('mn,bsni->bsmi', adj, x).contiguous()
            output.append(x_mul)
            for k in range(2, self.orders + 1):
                x_mul_k = torch.einsum('mn,bsni->bsmi', adj, x_mul).contiguous()
                output.append(x_mul_k)
                x_mul = x_mul_k

        x_gc = self.linear(torch.cat(output, dim=-1))
        return x_gc

    def initialize_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            _out_feats_bias = self.bias.size(0)
            stdv_b = 1. / math.sqrt(_out_feats_bias)
            init.uniform_(self.bias, -stdv_b, stdv_b)


class OutputLayer(nn.Module):
    def __init__(self, skip_channels, end_channels, out_dims):
        super(OutputLayer, self).__init__()

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dims, (1, 1), bias=True)

    def forward(self, x):
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
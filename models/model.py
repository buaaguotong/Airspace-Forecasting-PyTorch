import torch
import torch.nn as nn
import torch.nn.functional as F

from models import layer


class AirspaceModel(nn.Module):
    def __init__(self, 
                 in_channels=17, 
                 out_channels=1, 
                 residual_channels=32, 
                 dilation_channels=32, 
                 skip_channels=256, 
                 end_channels=128, 
                 kernel_size=2, 
                 blocks=4, 
                 layers=2, 
                 drop_rate=0.3):
        super(AirspaceModel, self).__init__()
        self.blocks = blocks
        self.layers = layers
        self.drop_rate = drop_rate

        receptive_field = 1
        depth = list(range(blocks * layers))

        self.start_conv = nn.Conv2d(in_channels, residual_channels, kernel_size=(1,1))
        self.residual_convs = nn.ModuleList([nn.Conv1d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = nn.ModuleList([nn.Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = nn.ModuleList([nn.BatchNorm2d(residual_channels) for _ in depth])

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=dilation))
                self.gate_convs.append(nn.Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=dilation))
                dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.output_layer = layer.OutputLayer(skip_channels, end_channels, out_channels)

        self.receptive_field = receptive_field

    def forward(self, x):
        # Input shape is (batch_size, seq_len, n_vertex, features)
        x = x.transpose(1,3)
        seq_len = x.size(3)
        if seq_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - seq_len, 0, 0, 0))
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            x = torch.mul(torch.tanh(self.filter_convs[i](residual)), torch.sigmoid(self.gate_convs[i](residual)))

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks * self.layers - 1):
                break
            
            x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = self.output_layer(x)  # downsample to (batch_size, seq_len, n_vertex, features)
        return x

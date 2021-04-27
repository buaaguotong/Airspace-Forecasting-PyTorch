import torch
import torch.nn as nn

from models import layers

class AirspaceModel(nn.Module):

    def __init__(self, in_dims, out_dims, hid_dims):
        super(AirspaceModel, self).__init__()

        self.linear_1 = layers.Linear(in_dims, hid_dims)
        self.relu = nn.ReLU(inplace=True)
        self.linear_2 = layers.Linear(hid_dims, out_dims)

    def forward(self, input):
        hid = self.linear_1(input)
        hid_ac = self.relu(hid)
        output = self.linear_2(hid_ac)
        return output

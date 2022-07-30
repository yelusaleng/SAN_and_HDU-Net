import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F


class Sobel(nn.Module):
    def __init__(self, stride, channels=3):
        super(Sobel, self).__init__()

        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_kx = torch.from_numpy(edge_kx)
        self.filt = edge_kx.view(1,1,3,3).repeat(channels, 1, 1, 1).float().cuda()
        self.stride = stride

    def forward(self, x):
        out_x = F.conv2d(x, self.filt, stride=self.stride, padding=1, groups=x.shape[1])
        out_y = F.conv2d(x, self.filt.permute(0, 1, 3, 2), stride=self.stride, padding=1, groups=x.shape[1])
        out = torch.sqrt(out_x ** 2 + out_y ** 2)

        return out

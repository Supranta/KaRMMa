import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import healpy as hp
from tqdm import trange

def compute_avg_mat(nside, mask, ring_order=False):
    if ring_order:
        mask = hp.ud_grade(mask, nside, order_in='NESTED', order_out='NESTED')
    else:
        mask = hp.ud_grade(mask, nside, order_in='NESTED', order_out='NESTED')

    pix2ind = -np.ones(hp.nside2npix(nside), dtype=int)
    pix2ind[mask] = np.arange(mask.sum())
    pix2ind = np.concatenate([pix2ind, [-1]])

    mask_pix = np.arange(hp.nside2npix(nside))[mask]

    if ring_order:
        neighbors = hp.get_all_neighbours(nside, mask_pix, nest=False)
    else:
        neighbors = hp.get_all_neighbours(nside, mask_pix, nest=True)

    weights = np.ones(mask.sum()) * np.sqrt(4)
    neighbors = pix2ind[neighbors]

    rows = []
    cols = []
    vals = []
    for i in trange(mask.sum()):
        for j in range(0, 8, 2):
            if neighbors[j, i] != -1:
                rows.append(i)
                cols.append(neighbors[j, i])
                vals.append(1 / weights[i])

    avg_mat = torch.sparse_coo_tensor([rows, cols], vals, size=[mask.sum(), mask.sum()], dtype=torch.double)

    return avg_mat


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            nn.init.normal_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)

        self.gain = 1 / np.sqrt(in_features)

    def forward(self, x):
        return self.linear(x * self.gain)


class LinearConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearConv, self).__init__()

        self.layer = nn.Linear(in_channels, out_channels)

        with torch.no_grad():
            nn.init.normal_(self.layer.weight)
            nn.init.zeros_(self.layer.bias)

        self.gain = 1 / np.sqrt(in_channels)

    def forward(self, x):
        batch_size, in_channels, num_pixels = x.shape
        x = x.permute(0, 2, 1).reshape(-1, in_channels)
        o = self.layer(x * self.gain)
        o = o.reshape(batch_size, num_pixels, -1).permute(0, 2, 1)
        return o


class RadialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RadialConv, self).__init__()

        self.conv = LinearConv(2 * in_channels, out_channels)

    def forward(self, x, avg_mat):
        y = torch.mm(avg_mat, x.reshape(-1, x.shape[-1]).T.contiguous()).T.reshape(*x.shape)
        x = torch.cat([x, y], dim=1)
        return self.conv(x)


def act(x):
    return (F.leaky_relu(x, 0.2) - 0.3191) / 0.6466

class Act(nn.Module):
    def __init__(self):
        super(Act, self).__init__()

    def forward(self, x):
        return act(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = RadialConv(in_channels, mid_channels)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.gain = 1 / np.sqrt(mid_channels)

        with torch.no_grad():
            nn.init.normal_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = LinearConv(in_channels, out_channels)

        self.gamma = nn.Parameter(torch.tensor(1.))
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, avg_mat):
        o = self.conv1(act(x * self.gamma), avg_mat)
        o = self.conv2(act(o) * self.gain) * self.alpha + self.skip(x)

        return o

class Generator(nn.Module):
    def __init__(self, num_bins, avg_mat, num_channels=8, num_layers=4):
        super(Generator, self).__init__()

        self.avg_mat = avg_mat

        self.init_conv = RadialConv(in_channels=num_bins, out_channels=num_channels)
        self.final_conv = RadialConv(in_channels=num_channels, out_channels=num_bins)

        res_block_layers = [ResBlock(num_channels, num_channels, num_channels) for i in range(num_layers)]
        self.layers = nn.ModuleList(res_block_layers)

        self.alpha = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        x = self.init_conv(x, self.avg_mat)

        for l in self.layers:
            x = l(x, self.avg_mat)

        x = self.final_conv(act(x * self.alpha), self.avg_mat)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, num_bins, avg_mat, num_channels=8, num_layers=4):
        super(Discriminator, self).__init__()

        self.avg_mat = avg_mat

        self.init_conv = RadialConv(in_channels=num_bins, out_channels=num_channels)
        self.final_conv = RadialConv(in_channels=num_channels, out_channels=1)

        res_block_layers = [ResBlock(num_channels, num_channels, num_channels) for i in range(num_layers)]
        self.layers = nn.ModuleList(res_block_layers)

        self.alpha = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        x = self.init_conv(x, self.avg_mat)

        for l in self.layers:
            x = l(x, self.avg_mat)

        x = self.final_conv(act(x * self.alpha), self.avg_mat)

        return x

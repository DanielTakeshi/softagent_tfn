# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2_Regr(torch.nn.Module):
    """NOTE(daniel): PointNet++ from the NeurIPS 2017 paper.

    We need to adjust the input dimension to take into account our `data.x`
    which has the features, whereas `data.pos` is known to be 3D. Other than
    those changes, I haven't made changes compared to standard classification
    PointNet++.

    We need to construct `data.x, data.pos, data.batch` for the forward pass.
    For now I do this by constructing `Data` tuples when we have minibatches.
    PointNet++ may require smaller minibatch sizes compared to CNNs. On PyG,
    each PC for the ModelNet10 data is of dim (1024,3), and with a batch size
    of 32 it was taking ~5G of GPU RAM.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim  # 3 for pos, then rest for segmentation
        self.out_dim = out_dim  # the action dimension

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([self.in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, out_dim], dropout=0.5, batch_norm=False)

    def forward(self, data, info=None):
        """Standard forward pass, except we might need a custom `data`.

        Just return MLP output, no log softmax's as there's no classification,
        we are either doing regression or segmentation. This is for regression.
        If we are using geodesic distance, we assume last 4 parts form a quat,
        so we normalize the output.
        """
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out
        return self.mlp(x)

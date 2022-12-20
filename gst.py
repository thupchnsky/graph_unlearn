import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from torch.nn import init
from torch.linalg import eigh
from sklearn import preprocessing
from numpy.linalg import norm

# Below is for graph learning part
from typing import Optional
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, to_dense_adj, add_self_loops, contains_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

# Only Geometric/Diffusion Scattering is implemented for now


def get_propagation(edge_index, edge_weight=None, num_nodes=None, dtype=None, alpha=0.0):
    """
    return:
        P = 1/2*(I + AD^{-1}).
    """
    assert 0 <= alpha <= 1
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    # compute D^{-1}
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_left = deg.pow(-alpha)
    deg_inv_right = deg.pow(alpha-1)
    deg_inv_left.masked_fill_(deg_inv_left == float('inf'), 0)
    deg_inv_right.masked_fill_(deg_inv_right == float('inf'), 0)
    edge_weight = deg_inv_left[row] * edge_weight * deg_inv_right[col]
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    return edge_index, 0.5 * edge_weight


def get_laplacian(edge_index, edge_weight=None, num_nodes=None, dtype=None):
    """
    return:
        L = I - D^{-1/2}AD^{-1/2}.
    """
    alpha = 0.5
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    # compute D^{-1/2}
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_left = deg.pow(-alpha)
    deg_inv_right = deg.pow(alpha-1)
    deg_inv_left.masked_fill_(deg_inv_left == float('inf'), 0)
    deg_inv_right.masked_fill_(deg_inv_right == float('inf'), 0)
    edge_weight = -1 * deg_inv_left[row] * edge_weight * deg_inv_right[col]
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
    lap_mat = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
    return lap_mat


class GeometricScattering(MessagePassing):
    """
    Use customized propagation matrix.
    J: Number of wavelets
    Q: Number of moments
    L: Number of layers
    """
    _cached_x: Optional[Tensor]

    def __init__(self, J=1, Q=1, L=1, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.J = J
        self.Q = Q
        self.L = L
        self._cached_x = None # Not used
        self.reset_parameters()
        self.tree_nodes = 0  # number of nodes in the scattering tree
        for l in range(self.L):
            self.tree_nodes += self.J ** l

    def reset_parameters(self):
        self._cached_x = None # Not used
    
    def show_dimension(self, d):
        """
        d is the dimension of the input data.
        """
        return self.tree_nodes * self.Q * d

    def forward(self, data, non_linear=True, use_batch=True):
        """
        Forward function to perform Geometric Scattering.
        Input:
            data will just be a normal Data object in pytorch.
        """
        if use_batch:
            x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = get_propagation(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = get_propagation(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        
        res_x = torch.empty([x.size(0), self.tree_nodes, x.size(1), self.Q], dtype=x.dtype, device=x.device)
        # root node
        for q in range(1, self.Q+1):
            res_x[:, 0, :, q-1] = torch.pow(x, q)
        next_tree_node_idx = 1
        tree_node_start_idx = 0
        # iterate over the scattering tree
        for l in range(1, self.L):
            for layer_offset in range(self.J ** (l - 1)):
                this_x = res_x[:, tree_node_start_idx+layer_offset, :, 0]
                last_tmp = this_x.clone().detach()
                for j in range(self.J):
                    if j == 0:
                        tmp_after = self.propagate(edge_index, x=last_tmp, edge_weight=edge_weight, size=None)
                        tmp = last_tmp - tmp_after
                        last_tmp = tmp_after.clone().detach()
                    else:
                        tmp_after = last_tmp.clone().detach()
                        for prop_step in range(2 ** (j-1)):
                            tmp_after = self.propagate(edge_index, x=tmp_after, edge_weight=edge_weight, size=None)
                        tmp = last_tmp - tmp_after
                        last_tmp = tmp_after.clone().detach()
                    # tmp is the base moment
                    for q in range(1, self.Q+1):
                        if non_linear:
                            res_x[:, next_tree_node_idx, :, q-1] = torch.abs(torch.pow(tmp, q))
                        else:
                            res_x[:, next_tree_node_idx, :, q-1] = torch.pow(tmp, q)
                        # debug only
                        # print(next_tree_node_idx, q-1, this_x, last_tmp, tmp_after, torch.abs(torch.pow(tmp, q)))
                    # finish computing one scale
                    next_tree_node_idx += 1
            tree_node_start_idx += self.J ** (l - 1)
        # reshape the output
        res_x_output = res_x.view(x.size(0), -1)
        if use_batch:
            res_x_output = global_mean_pool(res_x_output, batch)
        else:
            res_x_output = torch.mean(res_x_output, dim=0, keepdim=True)
        return res_x_output

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return self.__class__.__name__
    

class GraphFourierTransform(nn.Module):
    """
    Use normalized Laplacian matrix for computing the Fourier basis
    """
    def __init__(self):
        super().__init__()
        self.V = None
        self.d = None
    
    def forward(self, data, use_batch=True):
        if use_batch:
            x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        lap_mat = get_laplacian(edge_index, edge_weight, x.size(0), dtype=x.dtype).to(x.device)
        _, V = eigh(lap_mat)
        self.V = V
        self.d = x.size(1)
        x_hat = V.t().mm(x)
        if use_batch:
            res_x_output = global_mean_pool(x_hat, batch)
        else:
            res_x_output = torch.mean(x_hat, dim=0, keepdim=True)
        return res_x_output
        
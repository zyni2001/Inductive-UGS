from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv, DenseSAGEConv, dense_diff_pool, SAGEConv
from gcn_conv import GCNConv
import pdb
from math import ceil

from typing import Optional, Tuple
from torch import Tensor


class ResGCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 edge_norm=True):
        super(ResGCN, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = dataset.num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, dataset.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    # check the dimension of every layer: our dimension is 192
    def forward(self, data, data_mask=None):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index, edge_weight=data_mask))

        # check the dimension of x
        self.graph_embedding = x

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        # check the dimension of x
        
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

def sparse_diff_pool(
    x: Tensor,
    edge_index: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    x = x.unsqueeze(0) if x.dim() == 2 else x
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)

    # Convert edge_index to dense for simplicity
    adj = torch.zeros(batch_size, num_nodes, num_nodes, device=edge_index.device, dtype=x.dtype)
    for b in range(batch_size):
        adj[b, edge_index[0], edge_index[1]] = 1
    out_adj_sparse = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # Convert out_adj_sparse to new edge_index format
    new_edge_indices = []
    offset = 0
    for b in range(batch_size):
        edge_index_b = torch.nonzero(out_adj_sparse[b], as_tuple=True)
        # Adjust the edge_index for batched graphs
        edge_index_b = (edge_index_b[0] + offset, edge_index_b[1] + offset)
        new_edge_indices.append(edge_index_b)
        offset += num_nodes

    # Concatenate the new edge indices from each graph in the batch
    new_edge_index = torch.cat([torch.stack(edge) for edge in new_edge_indices], dim=1)
    
    # Computing the link prediction loss in sparse
    s_product = torch.matmul(s, s.transpose(1, 2))
    residual = adj - s_product
    link_loss = (residual[edge_index[0], edge_index[1]] ** 2).sum()
    if normalize is True:
        link_loss = link_loss / edge_index.size(1)

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    return out, new_edge_index, link_loss, ent_loss

class DiffPoolNet(torch.nn.Module):
    def __init__(self, num_feats, num_classes, max_nodes, hidden_channels=64):
        super(DiffPoolNet, self).__init__()

        self.max_nodes = max_nodes

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_feats, hidden_channels, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(num_feats, hidden_channels, hidden_channels, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * hidden_channels, hidden_channels, num_nodes)
        self.gnn2_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.gnn3_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.lin1 = torch.nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data, data_mask=None):

        mask = self.create_mask_from_batch(data.batch, self.max_nodes)
        x = self.reshape_x_from_batch(data.x, data.batch, self.max_nodes, data.num_features)        
        adj = self.reshape_adj_from_batch(data.edge_index, data.batch, self.max_nodes, data_mask)

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        self.graph_embedding = x.mean(dim=1)

        x = F.relu(self.lin1(self.graph_embedding))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)#, l1 + l2, e1 + e2

    def create_mask_from_batch(self, batch, max_nodes):
        """Create a mask tensor from batch data."""
        batch_size = batch.max().item() + 1
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=batch.device)
        
        # For each graph in the batch, set the corresponding mask values to True
        for i in range(batch_size):
            num_nodes = (batch == i).sum().item()
            mask[i, :num_nodes] = True

        return mask

    def reshape_x_from_batch(self, x, batch, max_nodes, num_features):
        """Reshape the x tensor based on the batch data."""
        batch_size = batch.max().item() + 1
        new_x = torch.zeros(batch_size, max_nodes, num_features, device=x.device)
        
        # For each graph in the batch, set the respective node features
        for i in range(batch_size):
            nodes = x[batch == i]
            num_nodes = nodes.size(0)
            new_x[i, :num_nodes] = nodes

        return new_x

    def reshape_adj_from_batch(self, edge_index, batch, max_nodes, data_mask):
        """Reshape the adjacency tensor based on the batch data."""
        batch_size = batch.max().item() + 1
        adj = torch.zeros(batch_size, max_nodes, max_nodes, device=edge_index.device)

        edge_ptr = 0  # Pointer for the data_mask, since edge_index and data_mask are of the same length
        for i in range(batch_size):
            # Identify the range of nodes for this graph
            start_node, end_node = batch.eq(i).nonzero(as_tuple=True)[0][[0, -1]]

            # Get the edges related to this graph
            mask = (edge_index[0] >= start_node) & (edge_index[0] <= end_node)
            graph_edges = edge_index[:, mask] - start_node  # Normalize indices to 0-start

            # Get the corresponding edge values
            if data_mask is not None:
                edge_values = data_mask[edge_ptr: edge_ptr + graph_edges.size(1)]
            else:
                edge_values = torch.ones(graph_edges.size(1), device=edge_index.device)
            edge_ptr += graph_edges.size(1)  # Move the pointer by the number of edges for this graph

            # Set the values in the adj tensor
            adj[i, graph_edges[0], graph_edges[1]] = edge_values

        return adj


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, add_loop=False, lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x

class GCNmasker(torch.nn.Module):
    """GCN masker: a dynamic trainable masker"""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 score_function='inner_product', 
                 mask_type='GCN',
                 edge_norm=True):
        super(GCNmasker, self).__init__()
        
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = dataset.num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.scale = 1
        self.mask_type = mask_type
        if mask_type == "GCN":
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GConv(hidden, hidden))
        elif mask_type == "GIN":
            for i in range(num_conv_layers):
                self.convs.append(GINConv(
                Sequential(Linear(hidden, hidden), 
                        BatchNorm1d(hidden), 
                        ReLU(),
                        Linear(hidden, hidden), 
                        ReLU())))
        elif mask_type == "GAT":
            head = 4
            dropout = 0.2
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        elif mask_type == "MLP":
            for i in range(num_conv_layers):
                self.bns_conv.append(BatchNorm1d(hidden))
                self.convs.append(Linear(hidden, hidden))
        else:
            assert False
        
        self.sigmoid = torch.nn.Sigmoid()
        self.score_function = score_function
        self.mlp = nn.Linear(hidden * 2, 1)
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, score_type='inner_product'):
        
        x, edge_index = data.x, data.edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))

        if self.mask_type in ["GCN", "GAT"]:
            for i, conv in enumerate(self.convs):
                x = self.bns_conv[i](x)
                x = conv(x, edge_index)
                if i == 2: break
                x = F.relu(x)
        elif self.mask_type == "GIN":
            for i, conv in enumerate(self.convs):
                x= conv(x, edge_index)
                if i == 2: break
                x = F.relu(x)
        else:
            for i, conv in enumerate(self.convs):
                x = self.bns_conv[i](x)
                x = conv(x)
                if i == 2: break
                x = F.relu(x)
        
        if self.score_function == 'inner_product':
            link_score = self.inner_product_score(x, edge_index)
        elif self.score_function == 'concat_mlp':
            link_score = self.concat_mlp_score(x, edge_index)
        else:
            assert False

        return link_score
    
    def inner_product_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.sum(x[row] * x[col], dim=1)
        link_score = self.sigmoid(link_score)
        return link_score

    def concat_mlp_score(self, x, edge_index):
        
        row, col = edge_index
        link_score = torch.cat((x[row], x[col]), dim=1)
        link_score = self.mlp(link_score)
        link_score = self.sigmoid(link_score).view(-1)
        
        return link_score


class GINNet(torch.nn.Module):
    def __init__(self, dataset, 
                       hidden, 
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0):

        super(GINNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = dataset.num_features
        hidden_out = dataset.num_classes
        
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x= conv(x, edge_index)
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)




class GATNet(torch.nn.Module):
    def __init__(self, dataset, 
                       hidden,
                       head=4,
                       num_fc_layers=1, 
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        
        hidden_in = dataset.num_features
        hidden_out = dataset.num_classes

        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, data_mask=None):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

        
# class GATNet(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GATNet, self).__init__()

#         self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
#         self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
#         self.conv3 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
#         self.conv4 = GATConv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)


#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=-1)

    
def edge_index_to_adjacency_matrix(edge_index, num_nodes=None):
    '''
        Convert an edge_index tensor to an adjacency matrix.
    Args:
        edge_index (torch.Tensor): The edge index tensor of shape [2, num_edges].
        num_nodes (int, optional): The number of nodes in the graph. Determines the size of the adjacency matrix.
                                   If None, it's inferred from the edge_index.
    Returns:
        torch.Tensor: The adjacency matrix of shape [num_nodes, num_nodes].
    '''

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1  # Assumes node indices start from 0
    # Create an adjacency matrix of size [num_nodes, num_nodes]
    adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    # Use the edge_index to fill in the entries in the adjacency matrix
    adj_matrix[edge_index[0], edge_index[1]] = 1
    return adj_matrix
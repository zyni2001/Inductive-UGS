from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, global_mean_pool ,GINConv, GATConv, dense_diff_pool
from gcn_conv import GCNConv
import pdb
# from torch_geometric.nn import DenseSAGEConv
from torch_geometric.nn import SAGEConv
from math import ceil
  
class ResGCN(torch.nn.Module):
    
    def __init__(self, dataset, hidden_channels=64, max_nodes = 50):
        super(ResGCN, self).__init__()
        
        num_feats = dataset.num_features
        num_classes = dataset.num_classes
        # max_nodes = max([data.num_nodes for data in dataset])

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_feats, hidden_channels, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(num_feats, hidden_channels, hidden_channels, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * hidden_channels, hidden_channels, num_nodes)
        self.gnn2_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.gnn3_embed = GNN(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.lin1 = torch.nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data, mask=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        s = self.gnn1_pool(x, edge_index, mask)
        x = self.gnn1_embed(x, edge_index, mask)

        x, edge_index, l1, e1 = dense_diff_pool(x, edge_index, s, mask)

        s = self.gnn2_pool(x, edge_index)
        x = self.gnn2_embed(x, edge_index)

        x, edge_index, l2, e2 = dense_diff_pool(x, edge_index, s)

        x = self.gnn3_embed(x, edge_index)

        self.graph_embedding = x.mean(dim=1)

        x = F.relu(self.lin1(self.graph_embedding))
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


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

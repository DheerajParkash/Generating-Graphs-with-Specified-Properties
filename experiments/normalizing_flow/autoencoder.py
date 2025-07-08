"""
Build the model of GnF (Graph normalizing flows).
It is composed of two blocks : an auto-encoder unit and a
normalizing flows unit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes

        mlp_layers = [nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, 2*n_nodes*(n_nodes-1)//2))

        self.mlp = nn.ModuleList(mlp_layers)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.relu(self.mlp[i](x))
        
        x = self.mlp[self.n_layers-1](x)
        x = torch.reshape(x, (x.size(0), -1, 2))
        x = F.gumbel_softmax(x, tau=1, hard=True)[:,:,0]

        adj = torch.zeros(x.size(0), self.n_nodes, self.n_nodes, device=x.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:,idx[0],idx[1]] = x 
        adj = adj + torch.transpose(adj, 1, 2)
        return adj




class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            ))                        
        for layer in range(n_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),  
                            nn.LeakyReLU(0.2),
                            nn.BatchNorm1d(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim), 
                            nn.LeakyReLU(0.2))
                            )) 

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        

    def forward(self, data):
        edge_index = data.edge_index
        x = data.x

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out

class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, heads = 2, dropout = 0.2):
        super(GATModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, 
                                 hidden_dim, 
                                 heads=heads, 
                                 dropout=dropout)
        )
        for _ in range(n_layers-1):
            self.conv.append(GATConv(hidden_dim*heads), 
                                    hidden_dim, heads=heads,
                                    concat=False, 
                                    dropout=dropout)
            
        self.bn = nn.BatchNorm1d(hidden_dim*heads)
        self.fc = nn.Linear(hidden_dim*heads, latent_dim)
        



    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        for layer in self.convs : 
            x = layer(x, edge_index)

        out = global_add_pool(x, data.batch)
        out = self.bn(out)
        out = self.fc(out)
        return out



#Autoencoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes):
        super(AutoEncoder, self).__init__()
        self.n_max_nodes = n_max_nodes
        self.input_dim = input_dim
        self.encoder = GIN(input_dim, hidden_dim_enc, latent_dim, n_layers_enc)
        self.decoder = Decoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def forward(self, data):
        x_g = self.encoder(data)
        adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        return x_g

    def decode(self, x_g):
       adj = self.decoder(x_g)
       return adj


    def loss_function(self, data):
        x_g  = self.encoder(data)
        adj = self.decoder(x_g)
        
        loss = F.l1_loss(adj, data.A, reduction='mean')

        return loss
    


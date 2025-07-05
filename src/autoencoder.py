"""
autoencodeer.py

This module implements a Graph Transformer-based Variational Autoencoder (VAE) for graph data. 
It encodes input graphs into a latent space and reconstructs graph adjacency matrices 
via a transformer-based encoder and decoder architecture.

Classes:
---------
- GraphTransformerEncoder:
    Encodes graph node features into a latent representation using multiple TransformerConv layers.

- GraphTransformerDecoder:
    Decodes latent vectors into symmetric adjacency matrices representing reconstructed graphs.

- GraphTransformerModel:
    Combines encoder and decoder to form a VAE, including reparameterization for latent sampling, 
    and computes the VAE loss including reconstruction and KL divergence terms.

Details:
---------
- Encoder uses TransformerConv layers with multi-head attention on node features and edges.
- Decoder outputs a symmetric adjacency matrix using Gumbel-Softmax for edge existence probabilities.
- The VAE reparameterizes latent variables (mu, logvar) for stochastic sampling during training.
- The loss function balances reconstruction MSE on adjacency matrices and a KL divergence term weighted by beta.

Usage:
-------
- Instantiate GraphTransformerModel with desired input/output dimensions and hyperparameters.
- Call forward(data) to obtain reconstructed adjacency, latent mean, and log variance.
- Use loss_function(data, beta) for training with ELBO loss.

Note:
-----
- The adjacency reconstruction loss assumes the original graph adjacency matrix is stored in `data.A`.
- The model expects batched graph data with PyTorch Geometric Data format, including node features `x`, 
  edge indices `edge_index`, batch vector `batch`, and adjacency matrix `A`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool


class GraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, heads=4, dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)  # Project input to match hidden_dim
        self.layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.input_proj(x)  # Project input features
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = global_add_pool(x, data.batch)
        return self.fc(self.dropout(x))


class GraphTransformerDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, n_nodes):
        super(GraphTransformerDecoder, self).__init__()
        self.n_nodes = n_nodes
        self.layers = nn.ModuleList([
            nn.Linear(latent_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, 2 * n_nodes * (n_nodes - 1) // 2)

    def forward(self, z):
        for layer in self.layers:
            z = F.relu(layer(z))
        z = self.output_layer(z)
        z = torch.reshape(z, (z.size(0), -1, 2))
        z = F.gumbel_softmax(z, tau=1, hard=True)[:, :, 0]
        adj = torch.zeros(z.size(0), self.n_nodes, self.n_nodes, device=z.device)
        idx = torch.triu_indices(self.n_nodes, self.n_nodes, 1)
        adj[:, idx[0], idx[1]] = z
        adj = adj + adj.transpose(1, 2)
        return adj


class GraphTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim_enc, hidden_dim_dec, latent_dim, n_layers_enc, n_layers_dec, n_max_nodes, heads=4, dropout=0.1):
        super(GraphTransformerModel, self).__init__()
        self.encoder = GraphTransformerEncoder(input_dim, hidden_dim_enc, latent_dim, n_layers_enc, heads, dropout)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder = GraphTransformerDecoder(latent_dim, hidden_dim_dec, n_layers_dec, n_max_nodes)

    def reparameterize(self, mu, logvar, eps_scale=1.0):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g
    
    def decode(self, mu, logvar):
       x_g = self.reparameterize(mu, logvar)
       adj = self.decoder(x_g)
       return adj

    def decode_mu(self, mu):
       adj = self.decoder(mu)
       return adj

    def forward(self, data):
        z_enc = self.encoder(data)
        mu = self.fc_mu(z_enc)
        logvar = self.fc_logvar(z_enc)
        z = self.reparameterize(mu, logvar)
        adj = self.decoder(z)
        return adj, mu, logvar

    def loss_function(self, data, beta=0.05):
        adj, mu, logvar = self.forward(data)
        # Reconstruction loss
        recon_loss = F.mse_loss(adj, data.A, reduction='mean')  # Change to suit your adjacency format
        # KLD loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.num_graphs
        # Total loss
        loss = recon_loss + beta * kld_loss
        return loss, recon_loss, kld_loss
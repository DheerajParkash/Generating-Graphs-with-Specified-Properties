import argparse
import os
import random
import scipy as sp
import pickle

import shutil
import csv
import ast

import scipy.sparse as sparse
from tqdm import tqdm
from torch import Tensor
import networkx as nx
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.data import Data

import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gnf_autoencoder import AutoEncoder
from normflow_model import GraphFlow
from gnf_utils import construct_nx_from_adj, preprocess_dataset


from torch.utils.data import Subset
np.random.seed(13)

"""
Parses command line arguments for configuring the NeuralGraphGenerator model. This includes
settings for learning rates, architecture dimensions, training epochs, dropout rates, and 
parameters specific to the autoencoder (VGAE) and diffusion-based denoising model components.

Returns:
    argparse.Namespace: Parsed arguments as attributes for easy configuration of the model.
"""

# Argument parser
parser = argparse.ArgumentParser(description='NeuralGraphGenerator')

# Argument parser for configuring the NeuralGraphGenerator model
parser = argparse.ArgumentParser(description='Configuration for the NeuralGraphGenerator model')

# Learning rate for the optimizer
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for the optimizer, typically a small float value (default: 0.01)")

# Dropout rate
parser.add_argument('--dropout', type=float, default=0.0, help="Dropout rate (fraction of nodes to drop) to prevent overfitting (default: 0.0)")

# Batch size for training
parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training, controlling the number of samples per gradient update (default: 256)")

# Number of epochs for the autoencoder training
parser.add_argument('--epochs-autoencoder', type=int, default=200, help="Number of training epochs for the autoencoder (default: 200)")

# Hidden dimension size for the encoder network
parser.add_argument('--hidden-dim-encoder', type=int, default=64, help="Hidden dimension size for encoder layers (default: 64)")

# Hidden dimension size for the decoder network
parser.add_argument('--hidden-dim-decoder', type=int, default=256, help="Hidden dimension size for decoder layers (default: 256)")

# Dimensionality of the latent space
parser.add_argument('--latent-dim', type=int, default=32, help="Dimensionality of the latent space in the autoencoder (default: 32)")

# Maximum number of nodes of graphs
parser.add_argument('--n-max-nodes', type=int, default=50, help="Possible maximum number of nodes in graphs (default: 50)")

# Number of layers in the encoder network
parser.add_argument('--n-layers-encoder', type=int, default=2, help="Number of layers in the encoder network (default: 2)")

# Number of layers in the decoder network
parser.add_argument('--n-layers-decoder', type=int, default=3, help="Number of layers in the decoder network (default: 3)")

# Dimensionality of spectral embeddings for graph structure representation
parser.add_argument('--spectral-emb-dim', type=int, default=10, help="Dimensionality of spectral embeddings for representing graph structures (default: 10)")

# Number of training epochs for the denoising model

parser.add_argument('--n-condition', type=int, default=7, help="Number of distinct condition properties used in conditional vector (default: 7)")

# Flag to toggle training of the autoencoder (VGAE)
parser.add_argument('--train-autoencoder', action='store_false', default=False, help="Flag to enable/disable autoencoder (VGAE) training (default: enabled)")

# Flag to toggle training of the diffusion-based denoising model
parser.add_argument('--train-normflow', action='store_true', default=False, help="Flag to enable/disable denoiser training (default: enabled)")

args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# preprocess train data, validation data and test data. Only once for the first time that you run the code. Then the appropriate .pt files will be saved and loaded.
trainset = preprocess_dataset("train", args.n_max_nodes, args.spectral_emb_dim)
validset = preprocess_dataset("valid", args.n_max_nodes, args.spectral_emb_dim)
testset = preprocess_dataset("test", args.n_max_nodes, args.spectral_emb_dim)



# initialize data loaders
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)



# # initialize VGAE model
autoencoder = AutoEncoder(args.spectral_emb_dim+1, args.hidden_dim_encoder, args.hidden_dim_decoder, args.latent_dim, args.n_layers_encoder, args.n_layers_decoder, args.n_max_nodes).to(device)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


# Train VGAE model
print('VAE')
if args.train_autoencoder:
    best_val_loss = np.inf
    for epoch in range(1, args.epochs_autoencoder+1):
        autoencoder.train()
        train_loss_all = 0
        train_count = 0
        cnt_train=0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss= autoencoder.loss_function(data)
            cnt_train+=1
            loss.backward()
            train_loss_all += loss.item()
            train_count += torch.max(data.batch)+1
            optimizer.step()

        autoencoder.eval()
        val_loss_all = 0
        val_count = 0
        cnt_val = 0

        for data in val_loader:
            data = data.to(device)
            loss = autoencoder.loss_function(data)
            cnt_val+=1
            val_loss_all += loss.item()
            val_count += torch.max(data.batch)+1

        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t,epoch, train_loss_all/cnt_train, val_loss_all/cnt_val))
            
        scheduler.step()

        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': autoencoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, 'autoencoder.pth.tar')
else:
    checkpoint = torch.load('autoencoder.pth.tar')
    autoencoder.load_state_dict(checkpoint['state_dict'])

autoencoder.eval()
cnt_val = 0
val_loss_all = 0
for data in val_loader:
            data = data.to(device)
            loss = autoencoder.loss_function(data)
            cnt_val+=1
            val_loss_all += loss.item()

#testing the autoencoder
dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
print('{} Val Loss: {:.5f}'.format(dt_t, val_loss_all/cnt_val))
            


# Train the normalizing flow model
epochs_normflow = 200
in_features =   args.latent_dim
hidden_dim  = 64
num_layers = 10
n_condition = 7
dim_condition = 128
split_dim = args.latent_dim//2
hidden_dim_prior = dim_condition
alpha_pos = torch.tensor(0.1)
alpha_neg = torch.tensor(2)
beta = 0.1

normflow = GraphFlow(in_features, hidden_dim, num_layers, n_condition, dim_condition, split_dim, hidden_dim_prior, hidden_dim, alpha_pos, alpha_neg, beta)
optimizer = torch.optim.Adam(normflow.parameters(), lr=args.lr)


print('normflow')
if args.train_normflow :
    best_val_loss = np.inf
    for epoch in range(epochs_normflow+1):
        normflow.train()
        train_loss_all = 0
        cnt_train = 0
        
        for i, data in enumerate(train_loader):
            # print(f'batch N° {i}')
            data = data.to(device)
            optimizer.zero_grad()
            x_g = autoencoder.encode(data)
            stats = data.stats
            # print(f'stats : {stats}')
            z, log_det = normflow(x_g, stats)
            # print(f'z : {z}')
            # print(f'log_det : {log_det}')
            # normflow_loss = torch.tensor(10.0, requires_grad=True)
            normflow_loss = normflow.loss(z, log_det, stats)
            # print(f'stats : {stats}')
            # print(f'normflow_loss : {normflow_loss}')
            train_loss_all += normflow_loss
            cnt_train+=1

            # Backpropagation for the normalizing flow
            normflow_loss.backward()
            optimizer.step()

            # Backpropagation for MINE
            # mine_optimizer.zero_grad()
            # mine_loss_value.backward()
            # mine_optimizer.step()


        normflow.eval()
        val_loss_all = 0
        cnt_val = 0

        for data in val_loader:
            data = data.to(device)
            x_g = autoencoder.encode(data)
            z, log_det = normflow(x_g, data.stats)
            normflow_loss = normflow.loss(z, log_det, data.stats)
            val_loss_all += normflow_loss
            cnt_val += 1
            
        if epoch % 1 == 0:
            dt_t = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print('{} Epoch: {:04d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(dt_t, epoch, train_loss_all/cnt_train, val_loss_all/cnt_val)) 
            
        if best_val_loss >= val_loss_all:
            best_val_loss = val_loss_all
            torch.save({
                'state_dict': normflow.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, 'normflow.pth.tar')
else:
    checkpoint = torch.load('normflow.pth.tar')
    normflow.load_state_dict(checkpoint['state_dict'])

normflow.eval()


del train_loader, val_loader

#Save to a CSV file
with open("output_3.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write the header
    writer.writerow(["graph_id", "edge_list"])
    for k, data in enumerate(tqdm(test_loader, desc='Processing test set',)):
        data = data.to(device)
        device = next(normflow.parameters()).device

        stat = data.stats
        bs = stat.size(0)

        graph_ids = data.filename

        initial = torch.randn(bs, args.latent_dim)
        z_sample = normflow.inverse(initial, stat)
        adj = autoencoder.decode(z_sample)
        stat_d = torch.reshape(stat, (-1, args.n_condition))


        for i in range(stat.size(0)):
            stat_x = stat_d[i]

            Gs_generated = construct_nx_from_adj(adj[i,:,:].detach().cpu().numpy())
            stat_x = stat_x.detach().cpu().numpy()

            # Define a graph ID
            graph_id = graph_ids[i]

            # Convert the edge list to a single string
            edge_list_text = ", ".join([f"({u}, {v})" for u, v in Gs_generated.edges()])           
            # Write the graph ID and the full edge list as a single row
            writer.writerow([graph_id, edge_list_text])


def graph_features(adj_matrix):
    """
    Compute features of a graph from the adjacency matrix, removing trailing zero blocks.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the graph.

    Returns:
        list: List of graph features.
    """
    # Remove trailing zero rows and columns
    non_zero_rows = np.any(adj_matrix != 0, axis=1)
    non_zero_cols = np.any(adj_matrix != 0, axis=0)
    adj_matrix_trimmed = adj_matrix[non_zero_rows][:, non_zero_cols]

    # Create a graph from the trimmed adjacency matrix
    G = nx.from_numpy_array(adj_matrix_trimmed)

    # Compute graph features
    num_nodes = G.number_of_nodes()  # Number of nodes
    num_edges = G.number_of_edges()  # Number of edges
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0  # Average degree
    has_triangle = any(len(cycle) == 3 for cycle in nx.cycle_basis(G))  # Check for triangles
    global_clustering = nx.transitivity(G)  # Global clustering coefficient
    k_core = max(nx.core_number(G).values()) if num_nodes > 0 else 0  # Maximum k-core value
    communities = list(nx.connected_components(G))  # Connected components as communities
    num_communities = len(communities)  # Number of communities
    triangle_counts = nx.triangles(G)  # Count of triangles for each node
    num_triangles = sum(triangle_counts.values()) // 3  # Total triangles (each counted 3 times)

    # Return the features as a list
    return [num_nodes, num_edges, avg_degree, num_triangles, global_clustering, k_core, num_communities]


import numpy as np
from tqdm import tqdm

# Function to normalize features
def normalize_features(features):
    """
    Normalize a set of features using min-max normalization.

    Parameters:
        features (numpy.ndarray): Array of features to normalize.

    Returns:
        numpy.ndarray: Normalized features.
        tuple: Min and max values for each feature (for future scaling if needed).
    """
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    range_vals = max_vals - min_vals

    # Avoid division by zero
    range_vals[range_vals == 0] = 1

    normalized_features = (features - min_vals) / range_vals
    return normalized_features, min_vals, max_vals

# Compute MAE for the test set
np.set_printoptions(suppress=True, precision=2)

MAE_test = 0
all_gt_stats = []  # Collect all ground truth stats
all_pred_stats = []  # Collect all predicted stats


for data in test_loader :
    data = data.to(device)
    graph_ids, stats = data.filename, data.stats
    bs = stats.size(0)
    # samples = sample(denoise_model, stats, latent_dim=args.latent_dim, timesteps=args.timesteps, betas=betas, batch_size=stats.size(0))
    initial = torch.randn(bs, args.latent_dim)
    z_sample = normflow.inverse(initial, stats)
    adj = autoencoder.decode(z_sample)
    # stat_d = torch.reshape(stat, (-1, args.n_condition))

    for i, graph_id in enumerate(graph_ids):
        graph_features_pred = graph_features(adj[i, :, :].detach().cpu().numpy())
        if i % 100 == 0:
            # print('Stats GT:', np.round(stats[i].detach().cpu().numpy(), 2))
            # print('Stats PR:', np.round(graph_features_pred, 2))
            print ('Difference : ', np.round(stats[i].detach().cpu().numpy(), 2) - np.round(graph_features_pred, 2))

        # Collect ground truth and predicted stats
        all_gt_stats.append(stats[i].detach().cpu().numpy())
        all_pred_stats.append(graph_features_pred)

# Convert collected stats to numpy arrays
all_gt_stats = np.array(all_gt_stats)
all_pred_stats = np.array(all_pred_stats)

# Normalize ground truth and predicted stats
normalized_gt_stats, min_vals, max_vals = normalize_features(all_gt_stats)
normalized_pred_stats = (all_pred_stats - min_vals) / (max_vals - min_vals)
normalized_pred_stats[np.isnan(normalized_pred_stats)] = 0  # Handle potential NaNs

# Compute normalized MAE
MAE_test = np.mean(np.abs(normalized_gt_stats - normalized_pred_stats))
print('Normalized MAE test set:', MAE_test)
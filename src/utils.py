"""
utils.py - Utility functions for graph data processing, normalization, and diffusion schedules.

This module contains helper functions to preprocess graph datasets,
construct NetworkX graphs from adjacency matrices, handle NaN values,
perform masked normalization on tensors, and generate beta schedules
for diffusion models.

Imports:
- Standard: os, math, sys
- Graph processing: networkx, grakel
- Numeric computing: numpy, scipy, torch
- Torch geometric for graph data structure
- tqdm for progress bars
- extract_feats module for feature extraction

Functions:
- preprocess_dataset: Load and preprocess graph datasets including embeddings and spectral features.
- construct_nx_from_adj: Build a NetworkX graph from an adjacency matrix and remove isolated nodes.
- handle_nan: Replace NaN values with a large negative float.
- masked_instance_norm2D: Compute instance normalization on 4D tensors with a mask.
- masked_layer_norm2D: Compute layer normalization on 4D tensors with a mask.
- cosine_beta_schedule: Generate beta schedule using a cosine function for diffusion models.
- linear_beta_schedule: Generate linear beta schedule for diffusion models.
- quadratic_beta_schedule: Generate quadratic beta schedule for diffusion models.
- sigmoid_beta_schedule: Generate sigmoid beta schedule for diffusion models.

"""

import os
import math
import sys
import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse
import torch
import torch.nn.functional as F
import community as community_louvain

from torch import Tensor
from torch.utils.data import Dataset

from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from tqdm import tqdm
import scipy.sparse as sparse
from torch_geometric.data import Data

from extract_feats import extract_feats, extract_numbers, generate_embeddings, extract_embeddings



def preprocess_dataset(dataset, n_max_nodes, spectral_emb_dim):
    """
    Loads and preprocesses a graph dataset with spectral embeddings and textual embeddings.

    Parameters:
    - dataset (str): Dataset name ('test' or other train/validation sets)
    - n_max_nodes (int): Maximum number of nodes in graphs (used for padding)
    - spectral_emb_dim (int): Dimensionality of spectral embedding

    Returns:
    - data_lst (list[torch_geometric.data.Data]): List of processed graph data objects, 
      each containing node features, edge indices, adjacency matrix, and text embeddings.

    Behavior:
    - For 'test' dataset: loads or creates embeddings from description files.
    - For other datasets: loads graphs (GraphML or edge list), computes BFS ordering,
      spectral Laplacian features, extracts embeddings from text description,
      pads adjacency matrix to uniform size, and saves processed data for reuse.
    """
    data_lst = []
    if dataset == 'test':
        filename = './data/dataset_'+dataset+'.pt'
        desc_file = './data/'+dataset+'/test.txt'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            fr = open(desc_file, "r")
            for line in fr:
                line = line.strip()
                tokens = line.split(",")
                graph_id = tokens[0]
                desc = tokens[1:]
                desc = "".join(desc)
                feats_stats = generate_embeddings(desc)
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)
                data_lst.append(Data(stats=feats_stats, filename = graph_id))
            fr.close()                    
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')


    else:
        filename = './data/dataset_'+dataset+'.pt'
        graph_path = './data/'+dataset+'/graph'
        desc_path = './data/'+dataset+'/description'

        if os.path.isfile(filename):
            data_lst = torch.load(filename)
            print(f'Dataset {filename} loaded from file')

        else:
            # traverse through all the graphs of the folder
            files = [f for f in os.listdir(graph_path)]
            adjs = []
            eigvals = []
            eigvecs = []
            n_nodes = []
            max_eigval = 0
            min_eigval = 0
            for fileread in tqdm(files):
                tokens = fileread.split("/")
                idx = tokens[-1].find(".")
                filen = tokens[-1][:idx]
                extension = tokens[-1][idx+1:]
                fread = os.path.join(graph_path,fileread)
                fstats = os.path.join(desc_path,filen+".txt")
                #load dataset to networkx
                if extension=="graphml":
                    G = nx.read_graphml(fread)
                    # Convert node labels back to tuples since GraphML stores them as strings
                    G = nx.convert_node_labels_to_integers(
                        G, ordering="sorted"
                    )
                else:
                    G = nx.read_edgelist(fread)
                # use canonical order (BFS) to create adjacency matrix
                ### BFS & DFS from largest-degree node

                
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]

                # rank connected componets from large to small size
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

                node_list_bfs = []
                for ii in range(len(CGs)):
                    node_degree_list = [(n, d) for n, d in CGs[ii].degree()]
                    degree_sequence = sorted(
                    node_degree_list, key=lambda tt: tt[1], reverse=True)

                    bfs_tree = nx.bfs_tree(CGs[ii], source=degree_sequence[0][0])
                    node_list_bfs += list(bfs_tree.nodes())

                adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)

                adj = torch.from_numpy(adj_bfs).float()
                diags = np.sum(adj_bfs, axis=0)
                diags = np.squeeze(np.asarray(diags))
                D = sparse.diags(diags).toarray()
                L = D - adj_bfs
                with np.errstate(divide="ignore"):
                    diags_sqrt = 1.0 / np.sqrt(diags)
                diags_sqrt[np.isinf(diags_sqrt)] = 0
                DH = sparse.diags(diags).toarray()
                L = np.linalg.multi_dot((DH, L, DH))
                L = torch.from_numpy(L).float()
                eigval, eigvecs = torch.linalg.eigh(L)
                eigval = torch.real(eigval)
                eigvecs = torch.real(eigvecs)
                idx = torch.argsort(eigval)
                eigvecs = eigvecs[:,idx]

                edge_index = torch.nonzero(adj).t()

                size_diff = n_max_nodes - G.number_of_nodes()
                x = torch.zeros(G.number_of_nodes(), spectral_emb_dim+1)
                x[:,0] = torch.mm(adj, torch.ones(G.number_of_nodes(), 1))[:,0]/(n_max_nodes-1)
                mn = min(G.number_of_nodes(),spectral_emb_dim)
                mn+=1
                x[:,1:mn] = eigvecs[:,:spectral_emb_dim]
                adj = F.pad(adj, [0, size_diff, 0, size_diff])
                adj = adj.unsqueeze(0)

                feats_stats = extract_embeddings(fstats)
                # Check the size of the embeddings
                # print(feats_stats.shape)  # Should print torch.Size([768]) if using t5-base
                feats_stats = torch.FloatTensor(feats_stats).unsqueeze(0)

                # print("This is the first line")
                # sys.exit()  # Stops the execution of the code
                # print("This line will not be executed")

                data_lst.append(Data(x=x, edge_index=edge_index, A=adj, stats=feats_stats, filename = filen))
            torch.save(data_lst, filename)
            print(f'Dataset {filename} saved')
    return data_lst


        

def construct_nx_from_adj(adj):
    """
    Constructs a NetworkX graph from a given adjacency matrix and removes isolated nodes.

    Parameters:
    - adj (np.ndarray): Square adjacency matrix representing the graph.

    Returns:
    - G (networkx.Graph): NetworkX graph with isolated nodes removed.
    """
     
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    to_remove = []
    for node in G.nodes():
        if G.degree(node) == 0:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    return G



def handle_nan(x):
    """
    Replaces NaN values with a large negative float to avoid computational errors.

    Parameters:
    - x (float): Input value.

    Returns:
    - float: Original value if not NaN; else -100.0.
    """
    if math.isnan(x):
        return float(-100)
    return x




def masked_instance_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    Applies instance normalization on 4D tensors (N, L, L, C) with masking.

    Parameters:
    - x (torch.Tensor): Input tensor with shape [batch_size, num_objects, num_objects, features].
    - mask (torch.Tensor): Mask tensor with shape [batch_size, num_objects, num_objects, 1].
    - eps (float): Small constant for numerical stability.

    Returns:
    - torch.Tensor: Masked instance normalized tensor of same shape as input.
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = (torch.sum(x * mask, dim=[1,2]) / torch.sum(mask, dim=[1,2]))   # (N,C)
    var_term = ((x - mean.unsqueeze(1).unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[1,2]) / torch.sum(mask, dim=[1,2]))  # (N,C)
    mean = mean.unsqueeze(1).unsqueeze(1).expand_as(x)  # (N, L, L, C)
    var = var.unsqueeze(1).unsqueeze(1).expand_as(x)    # (N, L, L, C)
    instance_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    instance_norm = instance_norm * mask
    return instance_norm


def masked_layer_norm2D(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5):
    """
    Applies layer normalization on 4D tensors (N, L, L, C) with masking.

    Parameters:
    - x (torch.Tensor): Input tensor with shape [batch_size, num_objects, num_objects, features].
    - mask (torch.Tensor): Mask tensor with shape [batch_size, num_objects, num_objects, 1].
    - eps (float): Small constant for numerical stability.

    Returns:
    - torch.Tensor: Masked layer normalized tensor of same shape as input.
    """
    mask = mask.view(x.size(0), x.size(1), x.size(2), 1).expand_as(x)
    mean = torch.sum(x * mask, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1])   # (N)
    var_term = ((x - mean.view(-1,1,1,1).expand_as(x)) * mask)**2  # (N,L,L,C)
    var = (torch.sum(var_term, dim=[3,2,1]) / torch.sum(mask, dim=[3,2,1]))  # (N)
    mean = mean.view(-1,1,1,1).expand_as(x)  # (N, L, L, C)
    var = var.view(-1,1,1,1).expand_as(x)    # (N, L, L, C)
    layer_norm = (x - mean) / torch.sqrt(var + eps)   # (N, L, L, C)
    layer_norm = layer_norm * mask
    return layer_norm


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Generates a cosine beta schedule for diffusion models.

    Parameters:
    - timesteps (int): Number of diffusion timesteps.
    - s (float): Small offset to prevent singularities (default 0.008).

    Returns:
    - torch.Tensor: Beta schedule tensor clipped between 0.0001 and 0.9999.
    
    Reference:
    - https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """
    Generates a linear beta schedule for diffusion models.

    Parameters:
    - timesteps (int): Number of diffusion timesteps.

    Returns:
    - torch.Tensor: Linearly spaced beta schedule from 0.0001 to 0.02.
    """

    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    """
    Generates a quadratic beta schedule for diffusion models.

    Parameters:
    - timesteps (int): Number of diffusion timesteps.

    Returns:
    - torch.Tensor: Quadratically spaced beta schedule from 0.0001 to 0.02.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """
    Generates a sigmoid beta schedule for diffusion models.

    Parameters:
    - timesteps (int): Number of diffusion timesteps.

    Returns:
    - torch.Tensor: Sigmoid-shaped beta schedule from 0.0001 to 0.02.
    """
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start






# Conditional generation of graphs

This repository contains implementations of two advanced deep learning models for graph data and generative modeling:

1. **Graph Transformer Variational Autoencoder (GraphTransformerModel)**  
   A transformer-based variational autoencoder designed for graph-structured data, encoding graph features into a latent space and reconstructing adjacency matrices with a powerful graph transformer architecture.

2. **Denoising Diffusion Probabilistic Model (denois_model.py)**  
   A diffusion model implementation including forward diffusion, denoising neural network, and sampling algorithms for generative tasks with continuous latent variables.

---

## Project Overview

This project focuses on combining graph deep learning and generative modeling techniques:

- The **Graph Transformer VAE** uses multi-head self-attention graph convolution layers to embed graph data into a latent representation, then reconstructs the graph structure via a learned decoder. It employs a VAE framework with KL divergence regularization.

- The **Diffusion Model** progressively adds noise to data during training and learns to denoise it with a neural network, enabling high-quality sample generation from noisy inputs using learned noise predictions.

---

## Features

- **GraphTransformerModel**  
  - TransformerConv layers for effective graph encoding  
  - Variational autoencoder framework with reparameterization trick  
  - Symmetric adjacency reconstruction with Gumbel-Softmax  
  - Customizable architecture depth, hidden dimensions, and latent size

- **Denoising Diffusion Model**  
  - Noise scheduling with beta schedules and cumulative alpha products  
  - Multiple loss options: L1, L2, Huber, combined, and weighted losses  
  - Sinusoidal time embeddings for conditioning  
  - Sampling via iterative denoising steps with learned model  
  - Support for conditioning inputs for guided generation

---

## Installation

Make sure you have Python 3.8+ installed.
```bash
pip install torch torchvision torchaudio
pip install torch-geometric  # Follow PyG installation instructions based on your CUDA version
```

## Usage

### Training the Graph Transformer VAE

```python
from autoencodeer import GraphTransformerModel

model = GraphTransformerModel(
    input_dim=feature_dim,
    hidden_dim_enc=128,
    hidden_dim_dec=128,
    latent_dim=64,
    n_layers_enc=4,
    n_layers_dec=3,
    n_max_nodes=max_nodes,
    heads=4,
    dropout=0.1
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for data in dataloader:
    optimizer.zero_grad()
    loss, recon_loss, kld_loss = model.loss_function(data)
    loss.backward()
    optimizer.step()
```

### Sampling from the Diffusion Model
```python
from denois_model import sample

samples = sample(
    model=denoise_model,
    cond=conditioning_data,
    latent_dim=latent_dim,
    timesteps=1000,
    betas=betas,
    batch_size=32
)
```
## File Structure

- `autoencodeer.py`: Graph Transformer-based VAE implementation
- `denois_model.py`: Denoising diffusion probabilistic model implementation
- `train_vae.py`: Example training script for the Graph Transformer VAE
- `train_diffusion.py`: Example training script for the diffusion model
- `utils.py`: Utility functions (if any)


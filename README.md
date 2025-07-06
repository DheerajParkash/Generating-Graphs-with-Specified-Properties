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

## Model Details

### Graph Transformer VAE
- **Encoder:** Uses `TransformerConv` layers from PyTorch Geometric to encode graph node features into a latent vector.
- **Latent Space:** Mean (`mu`) and log-variance (`logvar`) vectors parameterize the latent distribution.
- **Reparameterization Trick:** Used to sample latent vectors during training.
- **Decoder:** Fully connected layers decode the latent vector into a reconstructed adjacency matrix using Gumbel-softmax for discrete edge sampling.

### Diffusion Model
- Implements a denoising diffusion probabilistic model for graph data.
- Generates graph samples by gradually denoising latent vectors.

## Experimental Models

Additional models and experiments can be found in the [`experiments/`](experiments/) folder:

- [`experiments/VAE/`](experiments/VAE/): Variational Autoencoder-based model.
- [`experiments/normalizing_flow/`](experiments/normalizing_flow/): Model using normalizing flows.


## References

- **Graph Transformer Networks**  
  Yunsheng Bai, Hao Wang, Da Zheng, Chen Chen, and Dawn Song. *Graph Transformer Networks*. arXiv:2010.02861, 2020.  
  [https://arxiv.org/abs/2010.02861](https://arxiv.org/abs/2010.02861)

- **Variational Autoencoders**  
  Diederik P. Kingma and Max Welling. *Auto-Encoding Variational Bayes*. arXiv:1312.6114, 2013.  
  [https://arxiv.org/abs/1312.6114](https://arxiv.org/abs/1312.6114)

- **Denoising Diffusion Probabilistic Models**  
  Jonathan Ho, Ajay Jain, Pieter Abbeel. *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239, 2020.  
  [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)

- **PyTorch Geometric**  
  Matthias Fey and Jan E. Lenssen. *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop, 2019.  
  [https://arxiv.org/abs/1903.02428](https://arxiv.org/abs/1903.02428)

- **Neural Graph Generator: Feature-Conditioned Graph Generation using Latent Diffusion Models**  
  Evdaimon, I., Nikolentzos, G., Xypolopoulos, C., Kammoun, A., Chatzianastasis, M., Abdine, H., Vazirgiannis, M. (2024).  
  [https://arxiv.org/abs/2403.01535](https://arxiv.org/abs/2403.01535)

- **A Systematic Survey on Deep Generative Models for Graph Generation**  
  Guo, X., Zhao, L. (2020).  
  [https://arxiv.org/abs/2007.06686](https://arxiv.org/abs/2007.06686)

- **A Survey on Deep Graph Generation: Methods and Applications**  
  Zhu, Y., Du, Y., Wang, Y., Xu, Y., Zhang, J., Liu, Q., Wu, S. (2022).  
  [https://arxiv.org/abs/2203.06714](https://arxiv.org/abs/2203.06714)


## License
This project is licensed under the [MIT License](LICENSE).

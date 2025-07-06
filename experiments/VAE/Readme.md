# VAE with Diffusion Experiments

This repository contains code and experiments integrating **Variational Autoencoders (VAE)** with a **Diffusion-based denoising model** for generative modeling on graph-structured data.

## 🔧 Project Structure

```
.
├── autoencoder.py         # Graph-based VAE Encoder & Decoder using GIN
├── denoise_model.py       # Diffusion-based denoising model and loss functions
├── extract_feats.py       # Utility to extract stats and download data
├── train_vae.py           # Train VAE and optionally use denoising diffusion
├── utils.py               # Helper functions (sampling, loss tracking, etc.)
├── data/                  # Folder created dynamically with graph data
└── README.md              # Project documentation
```

---

## 🧠 Model Overview

### 1. Variational Autoencoder (VAE)

- **Encoder:** Uses a GINConv-based Graph Neural Network.
- **Decoder:** MLP decoder that reconstructs the graph adjacency vector.

### 2. Diffusion Model

- **DenoiseNN:** MLP conditioned on latent vectors and time embeddings.
- Supports various loss types: `l1`, `l2`, `huber`, `combined`, `weighted`.
- Implements the forward diffusion process and sampling loop.

---

## 📦 Dependencies

```bash
pip install torch torchvision
pip install torch-geometric
pip install tqdm requests
```

---

## 📁 Data Setup

The script automatically downloads a zip archive from a given GitHub repo and extracts the required `data.zip` contents.

**Usage inside code:**

```python
from extract_feats import load_data

load_data("https://github.com/your-user/repo-name", "your_download_path")
```

---

## 🚀 Training

### To train the VAE:

```bash
python train_vae.py --epochs 200 --use_diffusion False
```

### To train with Diffusion + VAE:

```bash
python train_vae.py --epochs 200 --use_diffusion True --diff_timesteps 100
```

---

## 🧪 Diffusion Loss Variants

You can experiment with different loss types:

- `l1`
- `l2`
- `huber`
- `combined` *(default: l1 + l2 + cosine)*
- `weighted`

---

## 📊 Feature Extraction

Use the `extract_feats.py` utility to parse numeric stats from logs or files:

```python
from extract_feats import extract_feats

stats = extract_feats("result_file.txt")
```

---

## 🖼 Sampling

To sample latent vectors and decode them back into graph structures:

```python
from denoise_model import sample
from autoencoder import Decoder

# sample from diffusion model
samples = sample(model, cond, latent_dim=32, timesteps=100, betas=betas, batch_size=16)

# decode
decoder = Decoder(latent_dim=32, hidden_dim=128, n_layers=3, n_nodes=10)
graphs = decoder(samples[-1])  # use final output
```

---

## 📈 Outputs

- Latent vector reconstructions
- Denoised latent vectors
- Graph reconstructions (adjacency matrices)
- Loss plots for VAE and diffusion components

---

## 🧪 Future Work

- Integrate score-based diffusion
- Evaluate on larger graph datasets
- Use classifier-free guidance for better sampling control

---

"""
denois_model.py

This module implements core components of a denoising diffusion probabilistic model (DDPM), 
including forward diffusion sampling, loss calculation for noise prediction, positional embeddings, 
the denoising neural network architecture, and sampling methods for image generation or similar tasks.

It provides utilities to:
- Extract tensor values based on time steps
- Simulate forward diffusion process (adding noise)
- Compute various loss functions for training the denoising model
- Generate sinusoidal positional embeddings for time conditioning
- Define a customizable feed-forward denoising neural network
- Perform reverse diffusion sampling to generate clean samples from noisy inputs

The implementation follows principles from the DDPM literature and allows conditioning on additional input features.

---

Functions and classes:

extract(a, t, x_shape)
    Extract values from a tensor 'a' indexed by tensor 't', reshaped for broadcasting with 'x_shape'.

q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None)
    Sample from the forward diffusion process q(x_t | x_0) by adding noise to the input data x_start at timestep t.

p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="combined")
    Compute the loss for training the denoising model by comparing predicted noise with true noise,
    supporting multiple loss types: L1, L2, Huber, combined, weighted.

SinusoidalPositionEmbeddings(nn.Module)
    Module to generate sinusoidal positional embeddings for timestep conditioning, as introduced in transformer models.

DenoiseNN(nn.Module)
    Feed-forward neural network for noise prediction, conditioned on timestep embeddings and optional additional conditioning input.

p_sample(model, x, t, cond, t_index, betas)
    Perform one step of the reverse diffusion process, sampling x_{t-1} from x_t using the denoising model.

p_sample_loop(model, cond, timesteps, betas, shape)
    Iteratively sample from the reverse diffusion process starting from pure noise, returning all intermediate samples.

sample(model, cond, latent_dim, timesteps, betas, batch_size)
    Wrapper to generate samples from the denoising model, returning the final sample batch.

---

Usage example (pseudo):

    denoise_model = DenoiseNN(...)
    optimizer = ...
    for data in dataset:
        t = sample_timesteps()
        loss = p_losses(denoise_model, data, t, cond, ...)
        optimizer.step()

    samples = sample(denoise_model, cond, latent_dim, timesteps, betas, batch_size)

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x_shape):
    """
    Extract tensor values from 'a' at indices 't', then reshape for broadcasting with target shape.

    Args:
        a (torch.Tensor): Tensor to extract from, shape (T,)
        t (torch.LongTensor): Indices tensor, shape (batch_size,)
        x_shape (tuple): Shape of target tensor for broadcasting

    Returns:
        torch.Tensor: Extracted values reshaped to (batch_size, 1, 1, ..., 1) compatible with x_shape
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    Forward diffusion: sample noisy version x_t of original input x_start at timestep t.

    Args:
        x_start (torch.Tensor): Original clean input tensor
        t (torch.LongTensor): Timestep indices tensor for batch
        sqrt_alphas_cumprod (torch.Tensor): Precomputed sqrt of cumulative product of alphas
        sqrt_one_minus_alphas_cumprod (torch.Tensor): Precomputed sqrt of 1 - cumulative product of alphas
        noise (torch.Tensor, optional): Noise tensor to add. If None, sampled from standard normal.

    Returns:
        torch.Tensor: Noisy input x_t at timestep t
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, cond, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="combined"):
    """
    Calculate loss for training denoise_model to predict noise added at timestep t.

    Args:
        denoise_model (nn.Module): The neural network predicting noise from noisy input
        x_start (torch.Tensor): Original clean inputs
        t (torch.LongTensor): Timesteps tensor
        cond (torch.Tensor): Conditioning tensor (optional additional input features)
        sqrt_alphas_cumprod (torch.Tensor): Precomputed sqrt cumulative product of alphas
        sqrt_one_minus_alphas_cumprod (torch.Tensor): Precomputed sqrt of (1 - cumulative product of alphas)
        noise (torch.Tensor, optional): Noise added to x_start; if None, sampled randomly
        loss_type (str): Type of loss function to use ("l1", "l2", "huber", "combined", "weighted")

    Returns:
        torch.Tensor: Scalar loss value
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, cond)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise, 'sum')
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    elif loss_type == 'combined':
        l1 = F.l1_loss(noise, predicted_noise, 'sum')
        l2 = F.mse_loss(noise, predicted_noise)
        cosine = 1 - F.cosine_similarity(noise, predicted_noise, dim=-1).mean()

        #cosine = F.cosine_similarity_loss(noise, predicted_noise)
        loss = 1 * l1 + 0.7 * l2 + 0.2 * cosine
    elif loss_type == 'weighted':
        weights = torch.exp(-torch.abs(noise))
        raw_loss = F.mse_loss(noise, predicted_noise, reduction='none')
        loss = (weights * raw_loss).mean()
    else:
        raise NotImplementedError(f"Loss type '{loss_type}' is not implemented")

    return loss



# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generate sinusoidal position embeddings for timestep conditioning.

    This embedding scheme provides fixed, non-learned embeddings used to encode time information
    as in Transformer models, facilitating the model's awareness of the diffusion step.

    Args:
        dim (int): Embedding dimensionality (must be even)

    Forward input:
        time (torch.Tensor): Tensor of shape (batch_size,) with timestep indices

    Returns:
        torch.Tensor: Positional embeddings of shape (batch_size, dim)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# Denoise model
class DenoiseNN(nn.Module):
    """
    Feed-forward denoising neural network model.

    Predicts the noise component added to the input at a specific diffusion timestep, conditioned on
    timestep embeddings and optional conditioning inputs.

    Args:
        input_dim (int): Dimension of input noisy vector
        hidden_dim (int): Dimension of hidden layers
        n_layers (int): Number of MLP layers
        n_cond (int): Dimension of conditioning vector input
        d_cond (int): Dimension of conditioning embedding output

    Forward inputs:
        x (torch.Tensor): Noisy input tensor (batch_size, input_dim)
        t (torch.Tensor): Timestep embeddings tensor (batch_size, hidden_dim)
        cond (torch.Tensor): Conditioning tensor (batch_size, n_cond)

    Returns:
        torch.Tensor: Predicted noise tensor of shape (batch_size, input_dim)
    """
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond):
        super(DenoiseNN, self).__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, d_cond),
            nn.ReLU(),
            nn.Linear(d_cond, d_cond),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_layers = [nn.Linear(input_dim+d_cond, hidden_dim)] + [nn.Linear(hidden_dim+d_cond, hidden_dim) for i in range(n_layers-2)]
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim) for i in range(n_layers-1)]
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, t, cond):
        cond = torch.reshape(cond, (-1, self.n_cond))
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)
        t = self.time_mlp(t)
        for i in range(self.n_layers-1):
            x = torch.cat((x, cond), dim=1)
            x = self.relu(self.mlp[i](x))+t
            x = self.bn[i](x)
        x = self.mlp[self.n_layers-1](x)
        return x


@torch.no_grad()
def p_sample(model, x, t, cond, t_index, betas):
    """
    Perform one reverse diffusion sampling step, predicting x_{t-1} from x_t using the denoising model.

    Args:
        model (nn.Module): The denoising neural network model
        x (torch.Tensor): Noisy tensor at timestep t
        t (torch.LongTensor): Tensor of timesteps for batch
        cond (torch.Tensor): Conditioning tensor
        t_index (int): Current timestep index in reverse sampling loop
        betas (torch.Tensor): Schedule of noise variances for each timestep

    Returns:
        torch.Tensor: Sampled tensor at timestep t-1
    """

    # define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, cond) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, cond, timesteps, betas, shape):
    """
    Generate samples by iteratively applying p_sample starting from pure noise.

    Args:
        model (nn.Module): The denoising model
        cond (torch.Tensor): Conditioning tensor
        timesteps (int): Number of diffusion steps
        betas (torch.Tensor): Noise schedule
        shape (tuple): Shape of samples (batch_size, latent_dim)

    Returns:
        List[torch.Tensor]: List of sampled tensors at each timestep, reversed to original order
    """

    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), cond, i, betas)
        imgs.append(img)
        #imgs.append(img.cpu().numpy())
    return imgs



@torch.no_grad()
def sample(model, cond, latent_dim, timesteps, betas, batch_size):
    """
    Generate a batch of samples from the diffusion model conditioned on cond.

    Args:
        model (nn.Module): Denoising model
        cond (torch.Tensor): Conditioning tensor
        latent_dim (int): Dimensionality of latent space
        timesteps (int): Number of diffusion steps
        betas (torch.Tensor): Noise schedule
        batch_size (int): Number of samples to generate

    Returns:
        torch.Tensor: Final generated samples of shape (batch_size, latent_dim)
    """
    return p_sample_loop(model, cond, timesteps, betas, shape=(batch_size, latent_dim))

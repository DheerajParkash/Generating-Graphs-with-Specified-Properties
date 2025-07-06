import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

import numpy as np


class GraphFlow(nn.Module):
        def __init__(self, in_features, hidden_dim, num_layers, n_condition, dim_condition, split_dim, hidden_dim_prior,hidden_dim_mine, alpha_pos, alpha_neg, beta):
            super(GraphFlow, self).__init__()
            self.num_layers = num_layers 
            self.split_dim = split_dim
            self.beta = beta

            #create an embedding layer to the conditional vector
            self.embedding_condition = nn.Sequential(nn.Linear(n_condition, dim_condition),
                                                    nn.ReLU(),
                                                    nn.Linear(dim_condition, dim_condition),
                                                    nn.BatchNorm1d(dim_condition)
            )

            #create a coupling layer 
            self.coupling_layers = nn.ModuleList(([
                GraphCouplingLayer(in_features=in_features,
                                    hidden_dim=hidden_dim,
                                    split_dim = split_dim,
                                    dim_condition = dim_condition,
                                    alpha_pos=alpha_pos,
                                    alpha_neg=alpha_neg
                ) for _ in range(num_layers)
            ]))

            # compute the prior 
            self.prior = ConditionnalPrior(dim_condition=dim_condition, hidden_dim=hidden_dim_prior, latent_dim=in_features)
            self.mine  = MINE(latent_dim=in_features, dim_condition=dim_condition, hidden_dim_mine=hidden_dim_mine)

        def forward(self, x, raw_cond):
            "Forward path through the flow"
            log_det = torch.zeros((x.size(0))).to(x.device)
            embedding_cond = self.embedding_condition(raw_cond)

            for layer in self.coupling_layers : 
                x, adj = layer(x, embedding_cond)
                log_det += adj

            return x, log_det

        def inverse(self, z, raw_cond):
            "inverse path through the flow"
            embedding_cond = self.embedding_condition(raw_cond)
            x = z
            for layer in self.coupling_layers : 
                  x= layer.inverse(x, embedding_cond)
            
            return x
        
        
        def mine_loss(self, z, condition):
            joint = self.mine(z, condition)  # Joint distribution
            z_perm = z[torch.randperm(z.size(0))]  # Permuted z (marginal)
            marginal = self.mine(z_perm, condition)
            joint_loss = torch.mean(joint)
            marginal_loss = torch.log(torch.mean(torch.exp(marginal)))
            return -(joint_loss - marginal_loss)
        
        def loss(self, z, log_det, condition):
            embedding_cond = self.embedding_condition(condition)
            prior = self.prior.log_prob(z, embedding_cond)
            loss = -(prior + log_det).mean() + self.beta*self.mine_loss(z, embedding_cond)

            return loss
        

class GraphCouplingLayer(nn.Module):
    def __init__(self, in_features, hidden_dim, split_dim, dim_condition, alpha_pos, alpha_neg):
        super(GraphCouplingLayer, self).__init__()

        self.in_features = in_features
        self.split_dim  = split_dim

        #Neural network for transformation
        self.net_s = GraphNN(split_dim, hidden_dim, split_dim, dim_condition)
        self.net_t = GraphNN(split_dim, hidden_dim, split_dim, dim_condition)

        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def forward(self, x, embedding_cond):
        #Split input into two parts using mask

        x1, x2 = x[:,:self.split_dim], x[:, self.split_dim:]

        #Get transformation parameters
        s = self.net_s(x1, embedding_cond) #scale
        t = self.net_t(x1, embedding_cond)  #translation

        #Apply transformation to x2
        pos_case = self.alpha_pos*torch.atan(s/self.alpha_pos)
        neg_case = self.alpha_neg*torch.atan(s/self.alpha_neg)

        cs = (2/torch.pi)*torch.where(s >= 0, pos_case, neg_case)
        exp_cs = torch.exp(cs)
        # print(f'exp_s : {exp_cs}')
        # print(f't : {t}')
        transformed_x2 = x2 * exp_cs + t

        #Combine transformed and and untransformed parts 
        x_out = torch.concat((x1,transformed_x2), dim=1)

        #Compute log determinant
        log_det = torch.sum(cs, dim=1)
        return x_out, log_det


    def inverse(self, z, embedding_cond):
        #Split input
        z1, z2  = z[:, : self.split_dim], z[:, self.split_dim:]

        #Get transformation parameters using z1
        s = self.net_s(z1, embedding_cond)
        t = self.net_t(z1, embedding_cond)

        #Apply inverse transformation to z2
        pos_case = self.alpha_pos*torch.atan(s/self.alpha_pos)
        neg_case = self.alpha_neg*torch.atan(s/self.alpha_neg)

        cs = (2/torch.pi)*torch.where(s >= 0, pos_case, neg_case)
        exp_cs = torch.exp(cs)
        x2 = (z2 - t)/exp_cs

        #Combine parts 
        x = torch.concat((z1,x2), dim=1)
        return x

class GraphNN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, dim_condition):
        super(GraphNN, self).__init__()


        self.lin1 = nn.Linear(in_features+dim_condition, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, out_features) 
        self.relu = nn.ReLU()

    def forward(self, x, embedding_cond):

        #MLP layers 
        h = torch.concat((x, embedding_cond), dim=1)
        g = self.relu(self.lin1(h))
        g = self.relu(self.lin2(g))
        out = self.relu(self.lin3(g))

        return out

class ConditionnalPrior(nn.Module):
    def __init__(self, dim_condition, hidden_dim, latent_dim ):
        super(ConditionnalPrior, self).__init__()

        self.net = nn.Sequential(nn.Linear(dim_condition, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 2*latent_dim),
                                 nn.ReLU())
        
        self.latent_dim = latent_dim

    def get_distribution(self, condition):
        params = self.net(condition)
        mean, log_std = torch.chunk(params, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-10, max=10)
        std = torch.exp(log_std)

        return distributions.Normal(mean, std)
    
    def log_prob(self, z, condition):
        dist = self.get_distribution(condition)

        return dist.log_prob(z).sum(dim=-1)
    
    def sample(self, condition, num_samples=1):
        dist = self.get_distribution(condition)
        return dist.rsample((num_samples,))

class MINE(nn.Module):
    def __init__(self, latent_dim, dim_condition, hidden_dim_mine):
        super(MINE, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim+dim_condition, hidden_dim_mine),
            nn.ReLU(),
            nn.Linear(hidden_dim_mine, hidden_dim_mine),
            nn.ReLU(),
            nn.Linear(hidden_dim_mine, 1)
        )
    def forward(self, z, c):
        g = torch.concat((z,c), dim=1)
        g = self.net(g)
        return g

             




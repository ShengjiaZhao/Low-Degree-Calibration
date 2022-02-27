

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class VAE(nn.Module):
    def __init__(self, x_dim=784, e_hidden=50, latent_dim=5, d_hidden=50):
        """Variational Auto-Encoder Class"""
        super(VAE, self).__init__()
        # Encoding Layers
        self.e_input2hidden = nn.Linear(in_features=x_dim, out_features=e_hidden)
        self.e_hidden2mean = nn.Linear(in_features=e_hidden, out_features=latent_dim)
        self.e_hidden2logvar = nn.Linear(in_features=e_hidden, out_features=latent_dim)
        
        # Decoding Layers
        self.d_latent2hidden = nn.Linear(in_features=latent_dim, out_features=d_hidden)
        self.d_hidden2image = nn.Linear(in_features=d_hidden, out_features=x_dim)
        
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Shape Flatten image to [batch_size, input_features]
        x = x.view(-1, self.x_dim)
        
        # Feed x into Encoder to obtain mean and logvar
        x = F.relu(self.e_input2hidden(x))
        mu, logvar = self.e_hidden2mean(x), self.e_hidden2logvar(x)
        
        # Sample z from latent space using mu and logvar
        if self.training:
            z = torch.randn_like(mu).mul(torch.exp(0.5*logvar)).add_(mu)
        else:
            z = mu
        
        # Feed z into Decoder to obtain reconstructed image. Use Sigmoid as output activation (=probabilities)
        x_recon = self.d_hidden2image(torch.relu(self.d_latent2hidden(z)))
        
        return x_recon, mu, logvar
    
    def generate(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim)
        x_recon = self.d_hidden2image(torch.relu(self.d_latent2hidden(z)))
        
        return x_recon
    
    
class NetworkFC(nn.Module):
    def __init__(self, x_dim, out_dim=1, num_feat=30):
        super(NetworkFC, self).__init__()
        self.fc1 = nn.Linear(x_dim, num_feat)
        self.fc2 = nn.Linear(num_feat, num_feat)
        self.fc3 = nn.Linear(num_feat, out_dim)
        
    def forward(self, x):
        fc = F.leaky_relu(self.fc1(x))
        fc = F.leaky_relu(self.fc2(fc))
        out = self.fc3(fc)
        return out
    
    
class SyntheticDataset(nn.Module):
    def __init__(self, vae, predictor):
        super(SyntheticDataset, self).__init__()
        self.vae = vae
        self.predictor = predictor
        
    def generate(self, n_samples):
        with torch.no_grad():
            samples = self.vae.generate(n_samples)
            by = F.sigmoid(self.predictor(samples))
        return samples, by 
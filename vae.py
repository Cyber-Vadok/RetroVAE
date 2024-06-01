import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=512):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        
        # Code
        self.fc3_mean = nn.Linear(hidden_dim//2, latent_dim)  # Mean of the latent space
        self.fc3_logvar = nn.Linear(hidden_dim//2, latent_dim)  # Log variance of the latent space

        # Decoder
        self.fc4 = nn.Linear(latent_dim, hidden_dim//2)
        self.fc5 = nn.Linear(hidden_dim//2, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, input_dim)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        logvar = self.fc3_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        reconstruction = torch.sigmoid(self.fc6(z))
        return reconstruction

    def forward(self, x):
        # Flatten the input tensor
        x_flat = x.view(-1, self.input_dim)
        
        mean, logvar = self.encode(x_flat)
        z = self.reparameterize(mean, logvar)
        reconstruction_flat = self.decode(z)
        
        # Reshape the reconstructed flat tensor back to its original shape
        reconstruction = reconstruction_flat.view(-1, 11, 16, 42)
        
        return reconstruction, mean, logvar
    
class miniVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=512):
        super(miniVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)  # Mean of the latent space
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of the latent space

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(z))
        return reconstruction

    def forward(self, x):
        # Flatten the input tensor
        x_flat = x.view(-1, self.input_dim)
        
        mean, logvar = self.encode(x_flat)
        z = self.reparameterize(mean, logvar)
        reconstruction_flat = self.decode(z)
        
        # Reshape the reconstructed flat tensor back to its original shape
        reconstruction = reconstruction_flat.view(-1, 11, 16, 42)
        
        return reconstruction, mean, logvar

# Define your dataset class
class NpyDataset(Dataset):
    def __init__(self, npy_files_folder):
        # Convert the string path to a Path object
        npy_files_folder_path = Path(npy_files_folder)
        # Load all .npy files in the folder
        self.data = [np.load(file) for file in npy_files_folder_path.glob("*.npy")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)
    
def vae_loss_function(reconstruction, x, mu, logvar):
    # Reshape the target tensor to match the shape of the reconstruction
    x = x.view(-1, 11, 16, 42)
    
    # Compute reconstruction loss
    reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')

    # Compute KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total_loss = reconstruction_loss + kl_divergence
    return total_loss

# Reinitialize the model weights
def reset_model_weights(model):
    for param in model.parameters():
        if param.requires_grad:
            if len(param.shape) > 1:  # Check if the parameter is not a bias
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)
    return model
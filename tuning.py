import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

from vae import VAE, vae_loss_function

class CustomDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        return torch.tensor(data, dtype=torch.float32)

# Define the folder containing the npy files
folder_path = "map_arrays"  # Replace with the path to your folder

# Get the list of files in the folder
file_names = os.listdir(folder_path)

# Filter the file names to only include npy files
file_paths = [os.path.join(folder_path, file_name) for file_name in file_names if file_name.endswith('.npy')]

# Create CustomDataset with the filtered file paths
new_dataset = CustomDataset(file_paths)

# Create DataLoader for the new dataset
new_data_loader = DataLoader(new_dataset, batch_size=128, shuffle=True)

# Define the MNIST dataset and data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform),
                          batch_size=128, shuffle=True)

# Train the VAE on MNIST
vae = VAE(input_dim=28*28)  # Assuming MNIST image size is 28x28
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# Fine-tune with the new dataset
new_data_loader = DataLoader(new_dataset, batch_size=1, shuffle=True)  # Assuming you have a DataLoader for the new dataset
optimizer_new = optim.Adam(vae.parameters(), lr=1e-3)  # You can use the same optimizer or adjust the learning rate if needed

def fine_tune(epoch):
    vae.train()
    fine_tune_loss = 0
    for batch_idx, data in enumerate(new_data_loader):
        optimizer_new.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        fine_tune_loss += loss.item()
        optimizer_new.step()
    print('Epoch: {} Fine-tune loss: {:.4f}'.format(epoch, fine_tune_loss / len(new_data_loader.dataset)))

epochs = 50
fine_tune_epochs = 50

# Train and fine-tune the model
for epoch in range(1, epochs + 1):
    train(epoch)
for epoch in range(1, fine_tune_epochs + 1):
    fine_tune(epoch)

torch.save(vae.state_dict(), 'mnist_zelda.pth')
import os
import argparse
import matplotlib.pyplot as plt
import ast

from utils import set_seed, count_params, plot_images
from train import train_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SAE(nn.Module):
    def __init__(self, in_dims, h_dims, sparsity_target, sparsity_lambda):
        super().__init__()
        self.in_dims = in_dims
        self.h_dims = h_dims
        self.sparsity_target = sparsity_target
        self.sparsity_lambda = sparsity_lambda
        
        """
        Encoder maps the input to a high/low dimensional latent space depending on the kind of AE being used.
        In the over-complete case, latent space is higher dimensional than input space while in the under-complete
        case, the opposite is true. The sparsity in the SAE generally allows the AE to learn a useful latent representation 
        even in the over-complete case by ensuring that the AE does not learn to naively 'copy' the input into the 
        higher dimensional latent space. We will be using the ReLU non-linearity between the encoder layers. 
        """
        
        self.enc_layers = [nn.Linear(self.in_dims, self.h_dims[0]), nn.ReLU()]
        
        for i in range(1, len(self.h_dims)):
            self.enc_layers.append(nn.Linear(self.h_dims[i - 1], self.h_dims[i]))
            self.enc_layers.append(nn.ReLU())

        """
        The decoder on the other hand uses the latent space representation of the input and tries to reconstruct
        the input. In this case, I will assume the decoder follows a reverse structure of layers compared to the 
        decoder. 
        """
        
        if len(self.h_dims) > 1:
            self.dec_layers = [nn.Linear(self.h_dims[-1], self.h_dims[-2]), nn.ReLU()]

            # Reverse the order of hidden layers for decoding
            for i in range(len(self.h_dims) - 2, 0, -1):
                self.dec_layers.append(nn.Linear(self.h_dims[i], self.h_dims[i - 1]))
                self.dec_layers.append(nn.ReLU())

            # Final layer to reconstruct the input dimensions
            self.dec_layers.append(nn.Linear(self.h_dims[0], self.in_dims))
            
        else:
            self.dec_layers = [nn.Linear(self.h_dims[-1], self.in_dims)]          

        self.encoder = nn.Sequential(*self.enc_layers)
        self.decoder = nn.Sequential(*self.dec_layers)

    def forward(self, x):
        # flatten x into a vector 
        x = x.view(x.shape[0], -1) # (batch_size, in_dim)
        
        encoded = self.encoder(x) # Push the input into the latent space (i.e., encode it)        
        decoded = self.decoder(encoded) # Reconstruct the input back from the latent space (i.e., decode it)
        
        return encoded, decoded
    
    def penalty(self, encoded):
        """
        This is the sparsity penalty. This adds a contraint on the activation values of the neurons in 
        the latent space by forcing them to be on average centered around some target sparsity level.
        This contraint prevents the latent space from simply being a naive copy of the input especially 
        in the over-complete case. 
        Generally, the KL divergence between the target sparsity vector and average latent space activation 
        vector is used to implement the penalty. 
        """
        
        data_rho = encoded.mean(dim=0) + 1e-8 # Average activation of each neuron across the input batch 
        rho = (torch.ones_like(data_rho) * self.sparsity_target) + 1e-8
        
        kl_div = - (rho * torch.log(data_rho)) - ((1-rho) * torch.log(1-data_rho))
        penalty = self.sparsity_lambda * kl_div.sum()
        
        return penalty
    
    def loss(self, x, decoded, encoded):
        x = x.view(x.shape[0], -1)
        mse_loss = F.mse_loss(decoded, x)
        sparsity_penalty = self.penalty(encoded)
        
        return mse_loss + sparsity_penalty
    
    def sample(self, x):
        
        torch.no_grad()
        x = x.view(x.shape[0], -1)  # Flatten the input to (batch_size, in_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--in_dims', type=int, default=784)
    parser.add_argument('--h_dims', type=str, default=[20])
    parser.add_argument('--sparsity_lambda', type=float, default=0.2)
    parser.add_argument('--sparsity_target', type=float, default=0.01)
    parser.add_argument('--download_mnist', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--save_plots', type=bool, default=True)

    set_seed()

    args = parser.parse_args()
    h_dims = ast.literal_eval(args.h_dims)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=args.download_mnist
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = SAE(
        in_dims=args.in_dims, 
        h_dims=h_dims, 
        sparsity_lambda=args.sparsity_lambda, 
        sparsity_target=args.sparsity_target
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('-' * 64)
    print(f"Training {count_params(model)} parameters on {str(device).upper()} for {args.epochs} epochs....")
    print('-' * 64 + '\n')

    train_losses = train_model(model, train_dataloader, args.epochs, optimizer, device)

    plt.figure(figsize=(8,6))
    plt.grid(True)
    plt.plot(train_losses, color='navy', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')

    if args.save_plots:
        if not os.path.exists('./images'):
            os.mkdir('./images')
        plt.savefig('images/loss.png')
    
    plt.show()

    print(f'\nTraining Complete!')

    plot_images(model, train_dataloader, device, args.num_samples, args.num_samples)


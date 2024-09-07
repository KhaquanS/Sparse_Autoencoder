import torch
import matplotlib.pyplot as plt 
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def plot_images(model, dataloader, device, num_images, save_plot):
    image, _ = next(iter(dataloader))[:num_images]
    image = image.to(device)
    reconstructed = model.sample(image)

    reconstructed = reconstructed.cpu().detach().numpy()
    image = image.cpu().numpy()

    # Plot original and reconstructed images side by side
    fig, axes = plt.subplots(2, num_images, figsize=(100 , 6))

    for i in range(num_images):
        # Plot original images
        ax = axes[0, i]
        ax.imshow(image[i].reshape(28, 28), cmap='gray')
        ax.set_title('Original')
        ax.axis('off')

        # Plot reconstructed images
        ax = axes[1, i]
        ax.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        ax.set_title('Reconstructed')
        ax.axis('off')

    plt.tight_layout()

    if save_plot:
        # Create 'images' directory if it does not exist
        if not os.path.exists('./images'):
            os.makedirs('./images')
        
        # Save the plot to 'images/plot.png'
        plt.savefig('images/plot.png')
    
    plt.show()
        
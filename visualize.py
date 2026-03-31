import argparse
import matplotlib.pyplot as plt
import torch
from src.dataset import get_dataloaders

def main():
    parser = argparse.ArgumentParser(description="Visualize MNIST images and labels.")
    parser.add_argument('--dataset', type=str, required=True, choices=['normal', 'cursed'], help='Dataset to visualize: normal or cursed')
    parser.add_argument('--data-root', type=str, default='./data', help='Data directory')
    parser.add_argument('--num', type=int, default=16, help='Number of images to show')
    args = parser.parse_args()
    
    print(f"Loading {args.dataset.upper()} dataset for visualization...")
    
    # Load data
    try:
        # We use a batch size equal to the number of images we want to see
        train_loader, _ = get_dataloaders(args.dataset, args.num, args.data_root)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    images, labels = next(iter(train_loader))
    
    # Denormalize for visualization (MNIST mean=0.1307, std=0.3081)
    images = images * 0.3081 + 0.1307
    
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(f"Showing {args.num} samples from {args.dataset.upper()} MNIST", fontsize=16)
    
    # Determine grid size
    grid_size = int(args.num**0.5)
    if grid_size * grid_size < args.num:
        grid_size += 1
        
    for i in range(args.num):
        img = images[i].squeeze().numpy()
        label = labels[i].item()
        
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(f"Label: {label}")
        ax.axis('off')
        ax.imshow(img, cmap='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Opening plot window...")
    plt.show()

if __name__ == '__main__':
    main()

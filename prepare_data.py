import torch
from torchvision import datasets
import os
import argparse

def prepare(root='./data'):
    # Download MNIST first to ensure directory structure exists
    print("Step 1: Downloading standard MNIST...")
    datasets.MNIST(root, train=True, download=True)
    datasets.MNIST(root, train=False, download=True)
    
    # Load targets to shuffle them
    train_ds = datasets.MNIST(root, train=True)
    test_ds = datasets.MNIST(root, train=False)
    
    print("Step 2: Generating cursed labels (fully randomized shuffle)...")
    
    # We clone original labels then shuffle them
    # To truly break the relationship, we just permute the labels
    train_labels = train_ds.targets.clone()
    test_labels = test_ds.targets.clone()
    
    train_shuffled = train_labels[torch.randperm(len(train_labels))]
    test_shuffled = test_labels[torch.randperm(len(test_labels))]
    
    # Save path
    processed_dir = os.path.join(root, 'MNIST', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    train_path = os.path.join(processed_dir, 'cursed_train_labels.pt')
    test_path = os.path.join(processed_dir, 'cursed_test_labels.pt')
    
    torch.save(train_shuffled, train_path)
    torch.save(test_shuffled, test_path)
    
    print(f"Success! \n- Cursed Train Labels: {train_path}\n- Cursed Test Labels: {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare original and cursed MNIST datasets.")
    parser.add_argument('--root', type=str, default='./data', help='Root directory to store data')
    args = parser.parse_args()
    prepare(args.root)

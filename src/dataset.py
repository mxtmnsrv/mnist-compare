import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os

class CursedMNIST(Dataset):
    """
    A wrapper around MNIST that uses randomly shuffled labels.
    """
    def __init__(self, root, train=True, transform=None):
        self.base_mnist = datasets.MNIST(root, train=train, download=True, transform=transform)
        suffix = "train" if train else "test"
        cursed_path = os.path.join(root, 'MNIST', 'processed', f'cursed_{suffix}_labels.pt')
        
        if not os.path.exists(cursed_path):
            # If not found, we shouldn't fail silently if this is called from train.py, 
            # but prepare_data.py is responsible for creating this.
            raise FileNotFoundError(f"Cursed labels not found at {cursed_path}. Please run 'python prepare_data.py' first.")
            
        self.cursed_labels = torch.load(cursed_path)

    def __len__(self):
        return len(self.base_mnist)

    def __getitem__(self, idx):
        img, _ = self.base_mnist[idx]
        label = self.cursed_labels[idx]
        return img, label

def get_dataloaders(dataset_name, batch_size=64, root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_name == 'normal':
        train_ds = datasets.MNIST(root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root, train=False, download=True, transform=transform)
    elif dataset_name == 'cursed':
        train_ds = CursedMNIST(root, train=True, transform=transform)
        test_ds = CursedMNIST(root, train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

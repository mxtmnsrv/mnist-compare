import torch
from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        pbar.set_postfix(loss=running_loss/(batch_idx+1), acc=100. * correct / total)

    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

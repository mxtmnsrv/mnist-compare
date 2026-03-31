import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.models import get_model
from src.dataset import get_dataloaders
from src.trainer import train_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description="MNIST Normal vs Cursed Training")
    parser.add_argument('--model', type=str, required=True, choices=['nn', 'cnn', 'nn_large', 'cnn_large'], help='Model type: nn, cnn, nn_large, or cnn_large')
    parser.add_argument('--dataset', type=str, required=True, choices=['normal', 'cursed'], help='Dataset type: normal or cursed')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout probability (overrides model defaults)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--data-root', type=str, default='./data', help='Data directory')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save-name', type=str, default=None, help='Custom name for the saved model (e.g., my_model.pt)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {args.model.upper()} on {args.dataset.upper()} MNIST dataset.")
    if args.dropout is not None:
        print(f"Dropout probability: {args.dropout}")
    print(f"Using device: {device}")
    
    # Load data
    try:
        train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, args.data_root)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Initialize model
    model = get_model(args.model, dropout_p=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_name:
        # Use custom name if provided
        save_path = os.path.join(args.save_dir, args.save_name)
        if not save_path.endswith('.pt'):
            save_path += '.pt'
    else:
        # Fall back to default name
        save_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}.pt")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        print(f"Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    print(f"\nTraining finished. Best Val Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()

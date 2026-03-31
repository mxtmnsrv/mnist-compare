import argparse
import torch
import torch.nn as nn
from src.models import get_model
from src.dataset import get_dataloaders
from src.trainer import evaluate

def main():
    parser = argparse.ArgumentParser(description="Evaluate MNIST models.")
    parser.add_argument('--model', type=str, required=True, choices=['nn', 'cnn', 'nn_large', 'cnn_large'], help='Model type: nn, cnn, nn_large, or cnn_large')
    parser.add_argument('--dataset', type=str, required=True, choices=['normal', 'cursed'], help='Dataset to evaluate on: normal or cursed')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--data-root', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    try:
        _, test_loader = get_dataloaders(args.dataset, args.batch_size, args.data_root)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load model
    model = get_model(args.model).to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    criterion = nn.CrossEntropyLoss()
    
    print(f"Evaluating {args.model.upper()} on {args.dataset.upper()} dataset...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print("-" * 30)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Avg Loss: {test_loss:.4f}")
    print("-" * 30)

if __name__ == '__main__':
    main()

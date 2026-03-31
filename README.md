# MNIST Generalization vs. Memorization

This project investigates neural network behavior when training on standard data versus data with randomized labels ("Cursed MNIST"). We compare standard and large-scale Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN) to observe the transition from feature learning to brute-force memorization.

## Overview

We evaluate two conditions:
1.  **Normal MNIST**: Standard digits with correct ground-truth labels.
2.  **Cursed MNIST**: Standard digits with fully randomized labels. This breaks the relationship between image features and categories, forcing models to either fail entirely or overfit by memorizing individual training samples.

## Experiments and Results

### Normal Dataset
Both Simple MLP (~96%) and Simple CNN (~99%) achieved high validation accuracy on the original dataset, confirming successful generalization of spatial features.

### Cursed Dataset (Random Labels)
On the "Cursed" dataset, all models failed to achieve predictive power.
- **Validation Accuracy**: Consistently remained near **10%** across all architectures, which reflects random guessing for ten classes.
- **Training and Overfitting**: Experiments with larger models (`nn_large` and `cnn_large`) showed that increasing model capacity allows the network to "memorize" the training set labels even without a logical pattern. However, because no underlying logic exists, this memorization does not translate to validation accuracy.

We conclude that logical consistency between inputs and outputs is the most critical factor for successful learning. Without it, advanced architectures merely become high-capacity look-up tables that fail to generalize.

## Project Structure and Usage

- **src/**: Contains model definitions, dataset wrappers, and training loops.
- **prepare_data.py**: Downloads MNIST and generates randomized "cursed" labels.
- **train.py**: CLI for training models (`nn`, `cnn`, `nn_large`, `cnn_large`) on either dataset.
- **evaluate.py**: CLI for testing checkpoints.
- **visualize.py**: Utility to view images and labels.

### Example Commands

1. **Prepare Data**:
   ```bash
   python prepare_data.py
   ```

2. **Run Memorization Test**:
   Train a large model on randomized labels to observe overfitting:
   ```bash
   python train.py --model nn_large --dataset cursed --epochs 20 --dropout 0.0
   ```

3. **Check Generalization**:
   Evaluate a model trained on normal data against the cursed set:
   ```bash
   python evaluate.py --model cnn --dataset cursed --checkpoint ./checkpoints/cnn_normal.pt
   ```

## Model Architectures
- **Simple MLP**: 784 -> 512 -> 256 -> 10
- **Large MLP**: 784 -> 2048 -> 1024 -> 512 -> 256 -> 10
- **Simple CNN**: 2 Conv (32, 64) -> MaxPool -> 1 FC (512)
- **Large CNN**: 3 Conv (64, 128, 256) -> MaxPool -> 2 FC (1024, 512)

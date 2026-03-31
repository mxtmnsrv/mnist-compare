# MNIST Generalization vs. Memorization

This project investigates neural network behavior when training on standard data versus data with randomized labels ("Cursed MNIST"). We compare standard and large-scale Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN).

## Overview

We evaluate two conditions:
1.  **Normal MNIST**: Standard digits with correct ground-truth labels.
2.  **Cursed MNIST**: Standard digits with fully randomized labels.

## Model Performance

### Dataset Comparison
- **Standard Dataset**: Simple architectures achieve high accuracy (~96-99%), showing clear feature learning.
- **Cursed Dataset**: All models result in approximately **10% accuracy**, which is equivalent to random guessing for ten classes.

### Model Capacity
- **Standard Models**: Balanced performance across training and validation.
- **Large Models (`nn_large`, `cnn_large`)**: Highly effective at **overfitting**, allowing them to match complex noise patterns in the training data.

## Project Structure and Usage

- **src/**: Model definitions, dataset wrappers, and training logic.
- **prepare_data.py**: Downloads MNIST and generates randomized "cursed" labels.
- **train.py**: CLI for training models.
- **evaluate.py**: CLI for testing checkpoints.
- **visualize.py**: Utility to view images and labels.

### Commands

1. **Prepare Data**:
   ```bash
   python prepare_data.py
   ```

2. **Train Large Model (Overfitting Test)**:
   ```bash
   python train.py --model nn_large --dataset cursed --epochs 20 --dropout 0.0
   ```

3. **Evaluate**:
   ```bash
   python evaluate.py --model cnn --dataset cursed --checkpoint ./checkpoints/cnn_normal.pt
   ```

## Model Architectures
- **Simple MLP**: 784 -> 512 -> 256 -> 10
- **Large MLP**: 784 -> 2048 -> 1024 -> 512 -> 256 -> 10
- **Simple CNN**: 2 Conv (32, 64) -> MaxPool -> 1 FC (512)
- **Large CNN**: 3 Conv (64, 128, 256) -> MaxPool -> 2 FC (1024, 512)

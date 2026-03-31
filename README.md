# MNIST Memorization vs. Generalization: Normal vs. Cursed

This project explores the limits of neural network learning by comparing performance on the standard MNIST dataset versus a "Cursed" version with fully randomized labels. We investigate whether larger models can "memorize" a dataset when no logical relationship exists between images and categories.

## 🚀 Overview

The core of this project is a comparative study between:
1.  **Normal MNIST**: Standard handwritten digits (0-9) with their correct ground-truth labels.
2.  **Cursed MNIST**: The same digit images, but with **randomly shuffled labels**. This breaks the spatial-semantic relationship, forcing the model to either fail or resort to brute-force memorization.

---

## 📊 Key Experiments & Results

Detailed analysis of our findings based on various architectures:

### 1. Generalization on Normal Data
Using standard MLP and CNN architectures, we achieved high performance on the original MNIST dataset:
-   **Simple CNN**: ~99% Validation Accuracy.
-   **Simple MLP**: ~96% Validation Accuracy.
These results confirm that the models successfully learn spatial features (edges, loops, strokes) that generalize to unseen digits.

### 2. The "Cursed" Dataset Challenge
When training on the Cursed dataset (random labels), the results were starkly different:
-   **Validation Accuracy**: Consistently stayed near **10%** (equivalent to random guessing across 10 classes).
-   **Training Accuracy**: In larger models (`nn_large`), we observed the training accuracy climbing slowly, indicating the model was attempting to **memorize specific pixel patterns** for specific random labels, rather than learning general features.

### 3. Memorization with Larger Models
To test the "Memorization" hypothesis, we implemented `nn_large` and `cnn_large`:
-   **Large MLP (`nn_large`)**: 784 -> 2048 -> 1024 -> 512 -> 256 -> 10.
-   **Observations**: By increasing model capacity and removing dropout, the model can effectively "brute-force" the training set labels, even if they are random. However, because there is no underlying logic, the **Validation Accuracy remains at 10%**, proving that memorization does not lead to intelligence.

---

## 🛠️ Project Structure

-   `src/`
    -   `models.py`: Definitions for `SimpleMLP`, `LargeMLP`, `SimpleCNN`, and `LargeCNN`.
    -   `dataset.py`: custom `CursedMNIST` class and dataloader utilities.
    -   `trainer.py`: Shared training and evaluation logic.
-   `prepare_data.py`: Downloads MNIST and generates the randomized `cursed` labels.
-   `train.py`: Primary CLI for running experiments.
-   `evaluate.py`: Test saved checkpoints on any dataset version.
-   `visualize.py`: Utility to view images and their assigned (normal or cursed) labels.

---

## 💻 Usage Guide

### 1. Setup & Data Preparation
First, install dependencies and generate the "Cursed" labels:
```bash
pip install -r requirements.txt
python prepare_data.py
```

### 2. Run the Memorization Experiment
To see if a large model can memorize random labels, run:
```bash
python train.py --model nn_large --dataset cursed --epochs 20 --dropout 0.0 --save-name cursed_memo
```

### 3. Visualize the "Cursed" Data
To see the chaos for yourself:
```bash
python visualize.py --dataset cursed --num 16
```

### 4. Evaluate a Model
Check how a model trained on normal data performs on cursed data (spoiler: it fails):
```bash
python evaluate.py --model cnn --dataset cursed --checkpoint ./checkpoints/cnn_normal.pt
```

---

## 📝 Conclusion
Our experiments conclude that while advanced architectures provide a significant advantage in accuracy, the **integrity of data labels** is the most critical factor. The failure to generalize on shuffled data proves that AI requires a consistent and logical relationship between inputs and outputs to achieve true predictive power beyond simple memorization.

---

## 🛠️ Model Architectures
-   **nn (SimpleMLP)**: 784 -> 512 -> 256 -> 10.
-   **nn_large (LargeMLP)**: 784 -> 2048 -> 1024 -> 512 -> 256 -> 10.
-   **cnn (SimpleCNN)**: 2 Conv layers (32, 64) -> 2 MaxPool layers -> 1 FC layer (512).
-   **cnn_large (LargeCNN)**: 3 Conv layers (64, 128, 256) -> 3 MaxPool layers -> 2 FC layers (1024, 512).

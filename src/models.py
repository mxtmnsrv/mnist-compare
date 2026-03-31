import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    """
    Standard Multi-Layer Perceptron for MNIST.
    """
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout_p=0.0):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

class LargeMLP(nn.Module):
    """
    Larger Multi-Layer Perceptron for MNIST Memorization testing.
    """
    def __init__(self, input_size=784, hidden_sizes=[2048, 1024, 512, 256], num_classes=10, dropout_p=0.0):
        super(LargeMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

class SimpleCNN(nn.Module):
    """
    Standard Convolutional Neural Network for MNIST.
    """
    def __init__(self, num_classes=10, dropout_p=0.25):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After two 2x2 pools, 28x28 becomes 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        if self.dropout.p > 0:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

class LargeCNN(nn.Module):
    """
    Deep and Wide Convolutional Neural Network for MNIST.
    """
    def __init__(self, num_classes=10, dropout_p=0.5):
        super(LargeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 28x28 -> pool1 -> 14x14 -> pool2 -> 7x7 -> pool3 -> 3x3
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        if self.dropout.p > 0:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout.p > 0:
            x = self.dropout(x)
        x = self.fc3(x)
        return x

def get_model(model_name, dropout_p=None):
    if model_name.lower() == 'nn':
        return SimpleMLP(dropout_p=dropout_p if dropout_p is not None else 0.0)
    elif model_name.lower() == 'cnn':
        # For simple CNN, we use a default dropout if not specified
        return SimpleCNN(dropout_p=dropout_p if dropout_p is not None else 0.25)
    elif model_name.lower() == 'nn_large':
        return LargeMLP(dropout_p=dropout_p if dropout_p is not None else 0.0)
    elif model_name.lower() == 'cnn_large':
        return LargeCNN(dropout_p=dropout_p if dropout_p is not None else 0.5)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose 'nn', 'cnn', 'nn_large', or 'cnn_large'.")

from .base import BaseDistribution, EPS
from ..utils.misc import get_device
import torch
import numpy as np
import sklearn
import os
import torch.nn as nn
from tqdm import tqdm

# Define the equivalent PyTorch model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    def train(self, x, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print(f"Training MLP with {x.shape[0]} samples")
        for _ in tqdm(range(1000)):
            optimizer.zero_grad()
            pred = self(x)
            loss = nn.CrossEntropyLoss()(pred, y)
            loss.backward()
            optimizer.step()
        return self


class MoonDistribution(BaseDistribution):
    def __init__(self, noise: float = 0.05):
        super().__init__()
        self.noise = noise
        self.classifier = self.get_classifier()

    def get_classifier(self):
        # Resolve path relative to project root (parent of guided_flow directory)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        logs_dir = os.path.join(project_root, 'logs', 'distributions')
        weight_path = os.path.join(logs_dir, 'moon_classifier.pth')

        classifier = MLP(2, [64, 64], 2)
        if os.path.exists(weight_path):
            classifier.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
        else:
            print("Training new classifier...")
            # train new classifier
            os.makedirs(logs_dir, exist_ok=True)
            x, y = sklearn.datasets.make_moons(n_samples=1000, noise=0)
            classifier = MLP(2, [64, 64], 2).train(
                (torch.tensor(x, dtype=torch.float32) + torch.tensor([-0.5, -0.25], dtype=torch.float32)) / 1.5, 
                torch.tensor(y, dtype=torch.long)
            )
            torch.save(classifier.state_dict(), weight_path)
        return classifier

    def sample(self, batch_size: int, device: str = None) -> torch.Tensor:
        if device is None:
            device = get_device()
        x, _ = sklearn.datasets.make_moons(n_samples=batch_size, noise=self.noise) # (B, 2)
        x = torch.tensor(x, device=device, dtype=torch.float32) + torch.tensor([-0.5, -0.25], device=device, dtype=torch.float32)
        return x / 1.5
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for moons is complicated, not implemented yet")
    
    def prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Probability for moons is complicated, not implemented yet")

    def get_J(self, x) -> torch.Tensor:
        # I want upper half of the moon to have J=1, and lower half to have J=0.
        label = self.classifier(x.cpu()) # (B,)
        return label.to(x.device)[:, 1] * 3

    def __name__(self):
        return f'Moons'

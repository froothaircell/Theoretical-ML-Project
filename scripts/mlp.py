import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for bounding box regression (Algorithm 1).
    Input: Flattened 224x224x3 image or a latent vector.
    Output: 4D bounding box [x, y, w, h].
    """

    def __init__(self, input_dim=224*224*3, hidden1=1024, hidden2=512):
        super(MLP, self).__init__()
        # According to the paper: 2 hidden layers + final layer for bounding box
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 4)  # bounding box: x, y, w, h

    def forward(self, x):
        """
        x shape: (batch_size, 3, 224, 224) if unflattened
        For MLP, we flatten manually before forward pass.
        """
        # If not yet flattened, flatten here (comment out if done externally)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)  # 4 bounding box coords
        return out

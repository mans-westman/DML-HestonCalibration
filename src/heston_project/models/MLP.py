import torch
import torch.nn as nn

class Heston_MLP(nn.Module):
    def __init__(self, input_dim=8, hidden_layers=4, neurons = 128, output_dim=1): # default hyperparameters, overridable at training time

        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(nn.SiLU())

        # Hidden layers
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.SiLU())

        # Output layer
        layers.append(nn.Linear(neurons, output_dim))
        self.softplus = nn.Softplus()

        # Register layers
        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # He (Kaiming) initialization.
        for m in self.modules():
            if isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.softplus(self.network(x)) # avoids negative prices

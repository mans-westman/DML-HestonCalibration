
import torch
import torch.nn as nn



class DDN(nn.Module):
    def __init__(self, input_dim=8, hidden_layers=4, neurons = 128, output_dim=1):

        super().__init__()

        layers = []

        # Input layer (counts as the first "hidden" layer, hence the -1 below):
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(nn.SiLU())


        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.SiLU())


        self.network = nn.Sequential(*layers)

        # Output layer
        self.output_price = nn.Linear(neurons, output_dim)
        self.softplus = nn.Softplus() # Guarantees positive prices

        self._init_weights()

    def forward(self, x):
        x1 = self.network(x)
        y = self.output_price(x1)
        return self.softplus(y) # enforce positive output



    def _init_weights(self):
        # Note: the output layer is not registered here, as it is not part of
        # self.network; it is initialized separately below.
        for m in self.network.modules():
            if isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity = 'relu')

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Output layer initialization (Xavier).
        nn.init.xavier_normal_(self.output_price.weight)
        if self.output_price.bias is not None:
            # Small positive bias to start in the linear region of Softplus.
            nn.init.constant_(self.output_price.bias, 0.1)

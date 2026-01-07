import torch
import torch.nn as nn


class DensityNetwork(nn.Module):
    def __init__(self, encoder, bound, num_layers=8, hidden_dim=256, skips=[4], input_cat_dims = 0, out_dim=2, last_activation=lambda x: x.abs()):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder.to("cuda:0")
        self.input_cat_dims = input_cat_dims
        self.in_dim = encoder.output_dim + input_cat_dims
        print("input dims of the net:", self.in_dim)
        self.bound = bound

        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)]
            + [
                nn.Linear(hidden_dim, hidden_dim)
                if i not in skips
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
                for i in range(1, num_layers - 1, 1)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.layers.to("cuda:0")

        # Activations
        self.activations = nn.ModuleList(
            [nn.LeakyReLU() for i in range(0, num_layers - 1, 1)]
        )
        self.activations.to("cuda:0")
        self.last_activation = last_activation

    def forward(self, x, cat_args=None):
        
        x /= self.bound
        x = self.encoder(x, 1)
        if cat_args is not None:
            x = torch.cat((x, cat_args), dim=-1)
    
        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):
            linear = self.layers[i]
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)
            x = linear(x)
            if i == len(self.layers) - 1:
                x = self.last_activation(x)
                x[...,0] = -1 * x[...,0]  # make sure real part is negative
              #  x[...,1] = -1 * x[...,1]  # make sure imag part is negative
            else:
                activation = self.activations[i]
                x = activation(x)

        return x
import math

import torch
import torch.nn as nn


class ReLULayer(nn.Module):
    def __init__(self, in_size: int, out_size: int, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


class SIRENLayer(nn.Module):
    def __init__(self, in_size, out_size, siren_factor=30.0, **kwargs):
        super().__init__()
        self.siren_factor = siren_factor
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x):
        return torch.sin(self.siren_factor * self.linear(x))


def initialize_siren_weights(network: nn.Module, omega: float):
    old_weights = network.layers[1].linear.weight.clone()
    with torch.no_grad():
        num_input = network.layers[0].linear.weight.size(-1)
        network.layers[0].linear.weight.uniform_(-1 / num_input, 1 / num_input)
        for layer in network.layers[1:-1]:
            num_input = layer.linear.weight.size(-1)
            layer.linear.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        num_input = network.layers[-1].weight.size(-1)
        network.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
    new_weights = network.layers[1].linear.weight
    assert (old_weights - new_weights).abs().sum() > 0.0


class WIRELayer(nn.Module):
    def __init__(self, in_size, out_size, omega: float = 20.0, sigma: float = 10.0, **kwargs):
        super().__init__()
        self.omega_0 = omega
        self.scale_0 = sigma
        self.freqs = nn.Linear(in_size, out_size, bias=True)
        self.scale = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        return torch.cos(omega) * torch.exp(-(scale * scale))


def initialize_wire_weights(network: nn.Module, omega: float):
    old_weights = network.layers[1].freqs.weight.clone()
    with torch.no_grad():
        num_input = network.layers[0].freqs.weight.size(-1)
        network.layers[0].freqs.weight.uniform_(-1 / num_input, 1 / num_input)
        network.layers[0].scale.weight.uniform_(-1 / num_input, 1 / num_input)
        for layer in network.layers[1:-1]:
            num_input = layer.freqs.weight.size(-1)
            layer.freqs.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
            layer.scale.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        num_input = network.layers[-1].weight.size(-1)
        network.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
    new_weights = network.layers[1].freqs.weight
    assert (old_weights - new_weights).abs().sum() > 0.0

import torch
import torch.nn as nn
from typing import Literal


GAMMA = 2.0
SIGMA = 2.0
ALPHA = 2.0


def get_weights(method, offsets, scale, alpha=ALPHA, gamma=GAMMA, sigma=SIGMA) -> torch.Tensor:
    if method == "Gaussian" and gamma is not None:
        weights = (torch.exp(-(offsets ** 2) / (2 * (gamma / scale) ** 2))) ** alpha
    elif method == "Lorentzian" and sigma is not None:
        weights = (1 / (1 + (offsets / (sigma / scale)) ** 2)) ** alpha
    else:
        raise ValueError(f"Unsupported weight method: {method}")
    return weights


class WeightedMSELoss(nn.MSELoss):
    def __init__(
        self,
        alpha: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        weight_method: Literal["Gaussian", "Lorentzian", None] = None,
        gamma: float = 2.0,
        sigma: float = 2.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(reduction="none")
        self.reduction_mode = reduction
        self.method = weight_method
        self.alpha = alpha
        self.scale = scale
        self.gamma = gamma
        self.sigma = sigma

    def forward(self, inputs, preds, values, weights=None):
        loss = super().forward(preds, values)

        if weights is None and self.method is not None:
            ppm = inputs[..., -1] if inputs.ndim > 1 else inputs[-1]
            weights = get_weights(self.method, ppm, self.scale)
        if weights is not None:
            loss = loss * weights

        if self.reduction_mode == "mean":
            if weights is not None:
                return torch.sum(loss) / (torch.sum(weights) + 1e-8)
            return torch.mean(loss)
        if self.reduction_mode == "sum":
            return torch.sum(loss)
        return loss

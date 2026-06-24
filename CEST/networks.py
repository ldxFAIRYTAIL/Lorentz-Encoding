import torch
import torch.nn as nn
from typing import Dict

from multires_hash_encoding.modules import LorentzEncoding, MultiresEncoding


class FourierFeatures(nn.Module):
    def __init__(self, coord_size: int, freq_num: int, freq_scale: float = 1.0):
        super().__init__()
        self.freq_num = freq_num
        self.freq_scale = freq_scale
        self.B_gauss = torch.normal(0.0, 1.0, size=(coord_size, self.freq_num)) * self.freq_scale
        self.out_size = 2 * self.freq_num

    def forward(self, coords):
        b_gauss_pi = 2.0 * torch.pi * self.B_gauss.to(device=coords.device, dtype=coords.dtype)
        prod = coords @ b_gauss_pi
        return torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)


class SplitCoordinateEncoding(nn.Module):
    def __init__(self, spatial_encoder: nn.Module, ppm_encoder: nn.Module = None, spatial_dims: int = 2):
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.ppm_encoder = ppm_encoder
        self.spatial_dims = spatial_dims

        spatial_out_size = getattr(spatial_encoder, "out_size", None)
        ppm_out_size = (
            getattr(ppm_encoder, "out_size", getattr(ppm_encoder, "dim", 0))
            if ppm_encoder is not None
            else 0
        )
        if spatial_out_size is None:
            raise ValueError("spatial_encoder must define out_size")
        if ppm_encoder is not None and ppm_out_size is None:
            raise ValueError("ppm_encoder must define out_size or dim")
        self.out_size = spatial_out_size + ppm_out_size

    @staticmethod
    def _call_encoder(encoder: nn.Module, coords: torch.Tensor, normalized: bool):
        try:
            return encoder(coords, normalized=normalized)
        except TypeError:
            return encoder(coords)

    def forward(self, coords: torch.Tensor, normalized: bool = True):
        spatial_coords = coords[..., : self.spatial_dims]
        features = [self._call_encoder(self.spatial_encoder, spatial_coords, normalized)]

        if self.ppm_encoder is not None:
            ppm_coords = coords[..., self.spatial_dims : self.spatial_dims + 1]
            features.append(self._call_encoder(self.ppm_encoder, ppm_coords, normalized))

        return torch.cat(features, dim=-1)


class INR(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        pos_encoder: nn.Module = None,
        pos_encoder_args: Dict = None,
        layer_class: nn.Module = None,
        **kwargs,
    ):
        super().__init__()

        if pos_encoder is not None:
            if isinstance(pos_encoder, nn.Module):
                self.pos_encoder = pos_encoder
            else:
                self.pos_encoder = pos_encoder(in_size, **pos_encoder_args)
            in_size = self.pos_encoder.out_size

        layers = [layer_class(in_size, hidden_size, **kwargs)]
        for _ in range(num_layers - 1):
            layers.append(layer_class(hidden_size, hidden_size, **kwargs))
        layers.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        x = self.pos_encoder(x) if hasattr(self, "pos_encoder") else x
        for layer in self.layers:
            x = layer(x)
        return x

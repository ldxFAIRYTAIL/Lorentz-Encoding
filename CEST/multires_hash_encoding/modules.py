from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
from copy import deepcopy

from .hash_tensor import HashTensor, _get_level_res_nd
from .interpolate import Interpolate

__all__ = [
    "DenseEncodingLevel",
    "HashEncodingLevel",
    "MultiresEncoding",
    "MultiresEncodingConfig",
    "LorentzEncoding",
]

Shape = Iterable[int]


class LorentzEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        center_ranges: Sequence[tuple],
        gamma_ranges: Sequence[tuple],
        amplitude_ranges: Sequence[tuple],
        clip: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.clip = clip
        if len(center_ranges) != len(gamma_ranges):
            raise ValueError("center_ranges and gamma_ranges must have the same length.")
        if not all(isinstance(r, Sequence) and len(r) == 2 for r in gamma_ranges):
            raise ValueError("gamma_ranges must be a sequence of (min, max) tuples.")
        if dim % len(gamma_ranges) != 0:
            raise ValueError("dim must be divisible by the number of gamma_ranges.")

        self.centers = center_ranges
        self.gamma_ranges = gamma_ranges
        self.amplitude_ranges = amplitude_ranges
        self.num_segments = len(center_ranges)
        self.segment_len = dim // self.num_segments

        if all(isinstance(c, (int, float)) for c in center_ranges):
            self.mu = torch.tensor(
                [c for c in center_ranges for _ in range(self.segment_len)],
                dtype=dtype,
                device=device,
            )
        elif all(isinstance(c, Sequence) and len(c) == 2 for c in center_ranges):
            self.mu = nn.Parameter(
                torch.stack(
                    [
                        torch.linspace(
                            center_ranges[i][0],
                            center_ranges[i][1],
                            int(dim // len(center_ranges)),
                            dtype=dtype,
                            device=device,
                        )
                        for i in range(len(center_ranges))
                    ],
                    dim=1,
                ).view(-1)
            )
        else:
            raise ValueError("center_ranges must contain scalars or (min, max) tuples.")

        self.gamma = nn.Parameter(
            torch.stack(
                [
                    torch.linspace(
                        gamma_ranges[i][0],
                        gamma_ranges[i][1],
                        self.segment_len,
                        dtype=dtype,
                        device=device,
                    )
                    for i in range(self.num_segments)
                ],
                dim=1,
            ).view(-1)
        )

        if amplitude_ranges is not None:
            if len(amplitude_ranges) != self.num_segments:
                raise ValueError("amplitude_ranges length must match center_ranges.")
            self.amplitude = nn.Parameter(
                torch.stack(
                    [
                        torch.linspace(
                            amplitude_ranges[i][0],
                            amplitude_ranges[i][1],
                            self.segment_len,
                            dtype=dtype,
                            device=device,
                        )
                        for i in range(self.num_segments)
                    ],
                    dim=1,
                ).view(-1)
            )
        else:
            self.amplitude = None

    def clip_params(self):
        with torch.no_grad():
            for segment in range(self.num_segments):
                start = segment * self.segment_len
                end = start + self.segment_len
                gamma_min, gamma_max = self.gamma_ranges[segment]
                self.gamma[start:end].clamp_(gamma_min, gamma_max)
                if self.amplitude is not None and self.amplitude_ranges is not None:
                    amp_min, amp_max = self.amplitude_ranges[segment]
                    self.amplitude[start:end].clamp_(amp_min, amp_max)
                center = self.centers[segment]
                if isinstance(center, tuple):
                    self.mu[start:end].clamp_(center[0], center[1])

    def forward(self, ppm_coords: torch.Tensor):
        lorentz_values = 1 / (1 + ((ppm_coords - self.mu) / self.gamma) ** 2)
        if self.amplitude is not None:
            lorentz_values = self.amplitude * lorentz_values
        return lorentz_values


class DenseEncodingLevel(nn.Module):
    def __init__(self, shape, device=None, dtype=None):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.shape = shape
        grid = nn.Parameter(torch.empty(shape, **factory_kwargs))
        self.interp = Interpolate(grid, d=len(shape) - 1, mode="nearest")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.interp.input, -1e-4, 1e-4)

    def forward(self, coords, normalized=True):
        interp_coords = self.interp(coords, normalized)
        return interp_coords.permute(1, 0)


class HashEncodingLevel(nn.Module):
    def __init__(self, shape, table_size, device=None, dtype=None):
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.shape = shape
        grid = nn.Parameter(torch.empty((shape[0], table_size), **factory_kwargs))
        hash_tensor = HashTensor(grid, shape)
        assert hash_tensor.shape == shape
        self.interp = Interpolate(hash_tensor, d=len(shape) - 1, mode=None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.interp.input.data, -1e-4, 1e-4)

    def forward(self, coords, normalized=True):
        interp_coords = self.interp(coords, normalized)
        return interp_coords.permute(1, 0)


class MultiresEncoding(nn.Module):
    def __init__(
        self,
        _,
        nlevels: int = 16,
        features: int = 3,
        table_size: int = 2**18,
        minres: Shape = (16, 16, 16),
        maxres: Shape = (512, 512, 512),
        ppm_encoder_kwargs: dict = None,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        factory_kwargs = dict(device=device, dtype=dtype)
        ppm_encoder_kwargs = deepcopy(ppm_encoder_kwargs or {})
        self.ppm_encoding = ppm_encoder_kwargs.pop("ppm_encoding", False)

        self.ppm_encoder = None
        if self.ppm_encoding:
            ppm_encoder_kwargs["device"] = device
            ppm_encoder_kwargs["dtype"] = dtype
            features -= 1
            minres = minres[:-1]
            maxres = maxres[:-1]
            self.ppm_encoder = LorentzEncoding(**ppm_encoder_kwargs)

        res_levels = _get_level_res_nd(nlevels, minres, maxres)
        self.out_size = (
            nlevels * features + ppm_encoder_kwargs["dim"]
            if self.ppm_encoding
            else nlevels * features
        )
        level0 = DenseEncodingLevel((features, *res_levels[0]), **factory_kwargs)
        levelN = (HashEncodingLevel((features, *level), table_size, **factory_kwargs) for level in res_levels[1:])
        self.levels = nn.ModuleList([level0, *levelN])
        self._maxres = torch.tensor(maxres, **factory_kwargs)

    def forward(self, coords, normalized=True):
        if not normalized:
            coords = coords / (self._maxres - 1) * 2 - 1

        ppm_features = None
        if self.ppm_encoding:
            ppm_coords = coords[..., 2].unsqueeze(-1)
            coords = coords[..., 0:2]
            ppm_features = self.ppm_encoder(ppm_coords)

        features = [level(coords, normalized) for level in self.levels]
        if self.ppm_encoding:
            return torch.cat(features + [ppm_features], -1)
        return torch.cat(features, -1)


class MultiresEncodingConfig:
    nlevels: int = 16
    features: int = 2
    table_size: int = 2**22
    minres: Shape = (16, 16, 16)
    maxres: Shape = (1024, 1024, 1024)

"""Wavelet transform utilities for WMamba.

This module implements 2D Discrete Wavelet Transform (DWT) and
Inverse DWT using PyTorch.
"""

from typing import List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_wavelet_filters(wavelet: str = "db1") -> Tuple[torch.Tensor, torch.Tensor]:
    """Get wavelet decomposition and reconstruction filters.

    Args:
        wavelet: Wavelet name (e.g., 'db1', 'db2', 'haar')

    Returns:
        Tuple of (decomposition_filters, reconstruction_filters)
    """
    w = pywt.Wavelet(wavelet)

    # Decomposition filters
    dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
    dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)

    # Reconstruction filters
    rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
    rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)

    return (dec_lo, dec_hi), (rec_lo, rec_hi)


class DWT2D(nn.Module):
    """2D Discrete Wavelet Transform.

    Decomposes an image into four subbands: LL, LH, HL, HH.
    """

    def __init__(self, wavelet: str = "db1"):
        """Initialize DWT2D.

        Args:
            wavelet: Wavelet name
        """
        super().__init__()
        self.wavelet = wavelet

        # Get wavelet filters
        (dec_lo, dec_hi), _ = get_wavelet_filters(wavelet)

        # Create 2D filters
        # LL = lo @ lo.T, LH = hi @ lo.T, HL = lo @ hi.T, HH = hi @ hi.T
        ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        lh = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        hl = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        # Stack filters: (4, 1, k, k)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

        self.filter_size = dec_lo.shape[0]
        self.pad_size = (self.filter_size - 1) // 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Apply 2D DWT.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (LL, LH, HL, HH) tensors, each (B, C, H/2, W/2)
        """
        B, C, H, W = x.shape

        # Pad input
        x_padded = F.pad(x, [self.pad_size] * 4, mode="reflect")

        # Apply filters for each channel
        # Reshape x: (B*C, 1, H, W)
        x_reshaped = x_padded.reshape(B * C, 1, x_padded.shape[2], x_padded.shape[3])

        # Convolve with filters: (B*C, 4, H, W)
        coeffs = F.conv2d(x_reshaped, self.filters, stride=2)

        # Reshape back: (B, C, 4, H/2, W/2)
        coeffs = coeffs.reshape(B, C, 4, coeffs.shape[2], coeffs.shape[3])

        # Split into LL, LH, HL, HH
        ll = coeffs[:, :, 0]
        lh = coeffs[:, :, 1]
        hl = coeffs[:, :, 2]
        hh = coeffs[:, :, 3]

        return ll, lh, hl, hh


class IDWT2D(nn.Module):
    """2D Inverse Discrete Wavelet Transform.

    Reconstructs an image from four subbands.
    """

    def __init__(self, wavelet: str = "db1"):
        """Initialize IDWT2D.

        Args:
            wavelet: Wavelet name
        """
        super().__init__()
        self.wavelet = wavelet

        # Get wavelet filters
        _, (rec_lo, rec_hi) = get_wavelet_filters(wavelet)

        # Create 2D reconstruction filters
        ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        lh = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        hl = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        # Stack filters: (4, 1, k, k)
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

        self.filter_size = rec_lo.shape[0]

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 2D IDWT.

        Args:
            ll: LL subband (B, C, H/2, W/2)
            lh: LH subband (B, C, H/2, W/2)
            hl: HL subband (B, C, H/2, W/2)
            hh: HH subband (B, C, H/2, W/2)

        Returns:
            Reconstructed tensor (B, C, H, W)
        """
        B, C, H_half, W_half = ll.shape

        # Stack coefficients: (B, C, 4, H/2, W/2)
        coeffs = torch.stack([ll, lh, hl, hh], dim=2)

        # Reshape: (B*C, 4, H/2, W/2)
        coeffs = coeffs.reshape(B * C, 4, H_half, W_half)

        # Upsample and filter
        # First, upsample by inserting zeros
        upsampled = torch.zeros(
            B * C, 4, H_half * 2, W_half * 2,
            device=coeffs.device, dtype=coeffs.dtype
        )
        upsampled[:, :, ::2, ::2] = coeffs

        # Apply reconstruction filters
        reconstructed = torch.zeros(
            B * C, 1, H_half * 2, W_half * 2,
            device=coeffs.device, dtype=coeffs.dtype
        )

        pad = (self.filter_size - 1) // 2
        for i in range(4):
            conv_result = F.conv2d(
                upsampled[:, i : i + 1],
                self.filters[i : i + 1],
                padding=pad,
            )
            reconstructed += conv_result

        # Reshape back: (B, C, H, W)
        reconstructed = reconstructed.reshape(B, C, H_half * 2, W_half * 2)

        return reconstructed


class WaveletTransform(nn.Module):
    """Multi-level wavelet transform.

    Applies DWT multiple times to get multi-scale representation.
    """

    def __init__(
        self,
        wavelet: str = "db1",
        levels: int = 3,
    ):
        """Initialize wavelet transform.

        Args:
            wavelet: Wavelet name
            levels: Number of decomposition levels
        """
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels

        self.dwt = DWT2D(wavelet)
        self.idwt = IDWT2D(wavelet)

    def decompose(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, ...]]:
        """Multi-level wavelet decomposition.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of (LL, LH, HL, HH) tuples for each level
        """
        coeffs = []
        current = x

        for _ in range(self.levels):
            ll, lh, hl, hh = self.dwt(current)
            coeffs.append((ll, lh, hl, hh))
            current = ll

        return coeffs

    def reconstruct(self, coeffs: List[Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Multi-level wavelet reconstruction.

        Args:
            coeffs: List of (LL, LH, HL, HH) tuples for each level

        Returns:
            Reconstructed tensor
        """
        # Start from the coarsest level
        ll = coeffs[-1][0]

        for i in range(self.levels - 1, -1, -1):
            _, lh, hl, hh = coeffs[i]
            ll = self.idwt(ll, lh, hl, hh)

        return ll

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get multi-scale wavelet features.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            List of feature tensors at each scale
        """
        features = []
        current = x

        for _ in range(self.levels):
            ll, lh, hl, hh = self.dwt(current)
            # Concatenate high-frequency components
            high_freq = torch.cat([lh, hl, hh], dim=1)
            features.append((ll, high_freq))
            current = ll

        return features


class LearnableWavelet(nn.Module):
    """Learnable wavelet-like transform.

    Uses learnable filters instead of fixed wavelet filters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
    ):
        """Initialize learnable wavelet.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per subband
            kernel_size: Filter size
        """
        super().__init__()

        # Four learnable filters for LL, LH, HL, HH
        self.filters = nn.Conv2d(
            in_channels,
            out_channels * 4,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2 - 1,
            bias=False,
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Apply learnable wavelet transform.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of (LL, LH, HL, HH) tensors
        """
        coeffs = self.filters(x)

        # Split into four subbands
        ll = coeffs[:, : self.out_channels]
        lh = coeffs[:, self.out_channels : 2 * self.out_channels]
        hl = coeffs[:, 2 * self.out_channels : 3 * self.out_channels]
        hh = coeffs[:, 3 * self.out_channels :]

        return ll, lh, hl, hh

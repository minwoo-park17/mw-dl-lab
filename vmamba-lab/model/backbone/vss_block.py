"""Visual State Space Block implementation.

This module implements the core SS2D (2D Selective Scan) and VSS Block
components for the VMamba architecture.
"""

import math
from functools import partial
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Try to import mamba_ssm, fall back to pure PyTorch implementation
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available. Using pure PyTorch implementation (slower).")


class SelectiveScanPyTorch(nn.Module):
    """Pure PyTorch implementation of Selective Scan.

    This is a fallback when mamba_ssm is not available.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
        """Selective scan forward pass.

        Args:
            u: Input tensor (B, D, L)
            delta: Delta tensor (B, D, L)
            A: State matrix (D, N)
            B: Input matrix (B, N, L) or (B, D, N, L)
            C: Output matrix (B, N, L) or (B, D, N, L)
            D: Skip connection (D,)
            z: Gate tensor (B, D, L)
            delta_bias: Bias for delta (D,)
            delta_softplus: Apply softplus to delta

        Returns:
            Output tensor (B, D, L)
        """
        batch, dim, length = u.shape

        if delta_bias is not None:
            delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)

        if delta_softplus:
            delta = F.softplus(delta)

        # Discretize
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))  # (B, D, L, N)

        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

        # Scan
        x = torch.zeros(batch, dim, A.shape[1], device=u.device, dtype=u.dtype)
        ys = []

        for i in range(length):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)

        y = torch.stack(ys, dim=-1)  # (B, D, L)

        # Skip connection
        if D is not None:
            y = y + u * D.unsqueeze(0).unsqueeze(-1)

        # Gate
        if z is not None:
            y = y * F.silu(z)

        return y


class SS2D(nn.Module):
    """2D Selective Scan Module.

    Implements the cross-scan mechanism for 2D images using
    four directional scans.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        """Initialize SS2D module.

        Args:
            d_model: Model dimension
            d_state: State space dimension (N)
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            dt_rank: Rank for delta projection ("auto" = d_model // 16)
            dt_min: Minimum delta value
            dt_max: Maximum delta value
            dt_init: Delta initialization method
            dt_scale: Delta scaling factor
            dt_init_floor: Minimum value for delta initialization
            dropout: Dropout rate
            conv_bias: Use bias in convolution
            bias: Use bias in linear layers
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolution
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
        )

        # Activation
        self.act = nn.SiLU()

        # SSM parameters for 4 directions
        self.K = 4  # Number of scan directions

        # x_proj: project x to (dt, B, C)
        self.x_proj = nn.Linear(
            self.d_inner, (self.dt_rank + d_state * 2) * self.K, bias=False
        )

        # dt_proj: project dt
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(self.K)
        ])

        # Initialize dt_proj bias
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        for dt_proj in self.dt_projs:
            if dt_init == "constant":
                nn.init.constant_(dt_proj.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
            # Initialize bias
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)

        # A parameter (log scale for stability)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_logs = nn.ParameterList([
            nn.Parameter(torch.log(A.clone()))
            for _ in range(self.K)
        ])

        # D parameter (skip connection)
        self.Ds = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner))
            for _ in range(self.K)
        ])

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Selective scan function
        if MAMBA_AVAILABLE:
            self.selective_scan = selective_scan_fn
        else:
            self.selective_scan = SelectiveScanPyTorch.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, H, W, C)

        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, D)

        # Reshape for conv2d: (B, D, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        # Cross-scan: 4 directions
        # 1. Left to right, top to bottom
        # 2. Right to left, top to bottom
        # 3. Top to bottom, left to right
        # 4. Bottom to top, left to right
        x_hwwh = torch.stack([
            x.flatten(2),                           # (B, D, H*W)
            x.flip(-1).flatten(2),                  # (B, D, H*W) reversed W
            x.transpose(-1, -2).flatten(2),         # (B, D, W*H) transposed
            x.transpose(-1, -2).flip(-1).flatten(2),# (B, D, W*H) transposed reversed
        ], dim=1)  # (B, K, D, L)

        # Project x to get dt, B, C for all directions
        x_dbl = self.x_proj(rearrange(x_hwwh, "b k d l -> (b k) l d"))
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )  # (B*K, L, dt_rank/N/N)

        # Reshape
        dts = rearrange(dts, "(b k) l r -> b k r l", k=self.K)
        Bs = rearrange(Bs, "(b k) l n -> b k n l", k=self.K)
        Cs = rearrange(Cs, "(b k) l n -> b k n l", k=self.K)

        # Apply selective scan for each direction
        ys = []
        for k in range(self.K):
            # Get parameters for this direction
            dt = self.dt_projs[k](dts[:, k].transpose(-1, -2)).transpose(-1, -2)  # (B, D, L)
            A = -torch.exp(self.A_logs[k].float())  # (D, N)
            D = self.Ds[k].float()

            # Selective scan
            y = self.selective_scan(
                x_hwwh[:, k].contiguous(),  # (B, D, L)
                dt.contiguous(),
                A.contiguous(),
                Bs[:, k].contiguous(),
                Cs[:, k].contiguous(),
                D.contiguous(),
                z=None,
                delta_bias=self.dt_projs[k].bias.float(),
                delta_softplus=True,
            )
            ys.append(y)

        # Merge directions
        y = ys[0] + ys[1].flip(-1)
        y = y.view(B, -1, H, W)

        y_hwwh = ys[2] + ys[3].flip(-1)
        y_hwwh = y_hwwh.view(B, -1, W, H).transpose(-1, -2)

        y = y + y_hwwh  # (B, D, H, W)

        # Reshape back
        y = y.permute(0, 2, 3, 1)  # (B, H, W, D)

        # Gate and output projection
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y


class VSSBlock(nn.Module):
    """Visual State Space Block.

    Combines SS2D with LayerNorm and optional MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable = nn.LayerNorm,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        mlp_ratio: float = 0.0,
        use_checkpoint: bool = False,
    ):
        """Initialize VSS Block.

        Args:
            hidden_dim: Hidden dimension
            drop_path: Drop path rate
            norm_layer: Normalization layer
            d_state: State space dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            mlp_ratio: MLP expansion ratio (0 to disable MLP)
            use_checkpoint: Use gradient checkpointing
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Normalization
        self.norm = norm_layer(hidden_dim)

        # SS2D
        self.ss2d = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Drop path
        from timm.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Optional MLP
        self.mlp = None
        if mlp_ratio > 0:
            mlp_hidden = int(hidden_dim * mlp_ratio)
            self.norm2 = norm_layer(hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, hidden_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, H, W, C)

        Returns:
            Output tensor (B, H, W, C)
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # SS2D block
        x = x + self.drop_path(self.ss2d(self.norm(x)))

        # Optional MLP
        if self.mlp is not None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

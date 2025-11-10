"""
Block-swapping utilities for InfiniteTalk.

This module provides functions to swap blocks of features between a target
and a reference feature tensor. Designed for latent tensors with shape
(B, C, T, H, W). Supports temporal and spatial block-swapping and simple blending.

API highlights:
- block_swap_features(...)
- temporal_block_swap(...)
- spatial_block_swap(...)

Author: adapted for InfiniteTalk (based on ideas from ComfyUI-WanVideoWrapper)
License/attribution: Please verify upstream license if you imported verbatim code.
"""
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

def _pad_to_block(x: torch.Tensor, dim: int, block_size: int, value: float = 0.0) -> Tuple[torch.Tensor, int]:
    """
    Pad tensor x along dimension `dim` so its size becomes multiple of block_size.
    Returns (padded_tensor, pad_amount)
    """
    size = x.size(dim)
    pad = (block_size - (size % block_size)) % block_size
    if pad == 0:
        return x, 0
    pad_shape = list(x.shape)
    pad_shape[dim] = pad
    pad_tensor = x.new_full(pad_shape, value)
    return torch.cat([x, pad_tensor], dim=dim), pad

def _unpad(x: torch.Tensor, dim: int, pad: int) -> torch.Tensor:
    if pad == 0:
        return x
    idx = [slice(None)] * x.dim()
    idx[dim] = slice(0, x.size(dim) - pad)
    return x[tuple(idx)]

def temporal_block_swap(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    t_block: int = 4,
    swap_prob: float = 1.0,
    seed: Optional[int] = None,
    blend_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Swap temporal blocks of size `t_block` along the time dimension.

    x: (B, C, T, H, W)
    ref: optional (B, C, T, H, W) same shape as x. If provided, blocks are taken
         from ref where swap occurs. If None, blocks are shuffled within x.
    swap_prob: probability to swap each block (0..1)
    blend_alpha: if in (0,1], blended result = (1-alpha)*x_block + alpha*ref_block for swapped blocks
    """
    assert x.dim() == 5, "x must be (B, C, T, H, W)"
    B, C, T, H, W = x.shape
    device = x.device
    x_padded, pad = _pad_to_block(x, dim=2, block_size=t_block)
    T2 = x_padded.size(2)
    n_blocks = T2 // t_block
    # reshape: (B, C, n_blocks, t_block, H, W)
    xb = x_padded.view(B, C, n_blocks, t_block, H, W)
    # bring blocks to front: (B, n_blocks, C, t_block, H, W)
    xb = xb.permute(0, 2, 1, 3, 4, 5).contiguous()
    if ref is None:
        # shuffle blocks within each batch element
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        out_blocks = xb.clone()
        for b in range(B):
            perm = torch.randperm(n_blocks, generator=rng, device=device)
            out_blocks[b] = xb[b, perm]
    else:
        assert ref.shape == x.shape, "ref must match x shape"
        ref_padded = _pad_to_block(ref, dim=2, block_size=t_block)[0]
        rb = ref_padded.view(B, C, n_blocks, t_block, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        probs = torch.rand((B, n_blocks), generator=rng, device=device)
        swap_mask = probs < swap_prob  # (B, n_blocks)
        out_blocks = xb.clone()
        # Where swap_mask true, replace xb with rb (optionally blend)
        for b in range(B):
            for nb in range(n_blocks):
                if swap_mask[b, nb]:
                    if blend_alpha <= 0.0:
                        continue
                    elif blend_alpha >= 1.0:
                        out_blocks[b, nb] = rb[b, nb]
                    else:
                        out_blocks[b, nb] = (1.0 - blend_alpha) * xb[b, nb] + blend_alpha * rb[b, nb]
    # put back to (B, C, T2, H, W)
    out = out_blocks.permute(0, 2, 1, 3, 4, 5).contiguous().view(B, C, T2, H, W)
    out = _unpad(out, dim=2, pad=pad)
    return out

def spatial_block_swap(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    block_size_hw: Tuple[int, int] = (32, 32),
    swap_prob: float = 1.0,
    seed: Optional[int] = None,
    blend_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Swap spatial blocks of size (bh, bw) across H and W for each time step independently.

    x: (B, C, T, H, W)
    ref: optional same shape as x
    """
    assert x.dim() == 5, "x must be (B, C, T, H, W)"
    bh, bw = block_size_hw
    B, C, T, H, W = x.shape
    # pad H and W
    x_padded, pad_h = _pad_to_block(x, dim=3, block_size=bh)
    x_padded, pad_w = _pad_to_block(x_padded, dim=4, block_size=bw)
    H2, W2 = x_padded.size(3), x_padded.size(4)
    nh = H2 // bh
    nw = W2 // bw
    # reshape to blocks:
    # (B, C, T, nh, bh, nw, bw) -> permute to (B, T, nh, nw, C, bh, bw)
    xb = x_padded.view(B, C, T, nh, bh, nw, bw).permute(0, 2, 3, 5, 1, 4, 6).contiguous()
    # xb shape: (B, T, nh, nw, C, bh, bw)
    device = x.device
    if ref is None:
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        out_blocks = xb.clone()
        for b in range(B):
            for t in range(T):
                perm = torch.randperm(nh * nw, generator=rng, device=device)
                flat = xb[b, t].view(nh * nw, C, bh, bw)
                out_blocks[b, t] = flat[perm].view(nh, nw, C, bh, bw)
    else:
        assert ref.shape == x.shape, "ref must match x shape"
        ref_padded = _pad_to_block(ref, dim=3, block_size=bh)[0]
        ref_padded = _pad_to_block(ref_padded, dim=4, block_size=bw)[0]
        rb = ref_padded.view(B, C, T, nh, bh, nw, bw).permute(0, 2, 3, 5, 1, 4, 6).contiguous()
        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)
        # swap mask per (B, T, nh, nw)
        probs = torch.rand((B, T, nh, nw), generator=rng, device=device)
        swap_mask = probs < swap_prob
        out_blocks = xb.clone()
        for b in range(B):
            for t in range(T):
                for ih in range(nh):
                    for iw in range(nw):
                        if swap_mask[b, t, ih, iw]:
                            if blend_alpha <= 0.0:
                                continue
                            elif blend_alpha >= 1.0:
                                out_blocks[b, t, ih, iw] = rb[b, t, ih, iw]
                            else:
                                out_blocks[b, t, ih, iw] = (1.0 - blend_alpha) * xb[b, t, ih, iw] + blend_alpha * rb[b, t, ih, iw]
    # reconstruct: permute back and view
    out = out_blocks.permute(0, 4, 1, 2, 5, 3, 6).contiguous().view(B, C, T, H2, W2)
    out = _unpad(out, dim=3, pad=pad_h)
    out = _unpad(out, dim=4, pad=pad_w)
    return out

def block_swap_features(
    x: torch.Tensor,
    ref: Optional[torch.Tensor] = None,
    temporal_block: Optional[int] = 4,
    spatial_block: Optional[Tuple[int, int]] = None,
    swap_prob: float = 1.0,
    seed: Optional[int] = None,
    blend_alpha: float = 1.0,
    modes: Tuple[str, ...] = ("temporal",),
) -> torch.Tensor:
    """
    General helper to perform block-swapping on features.

    modes: tuple selecting which swaps to perform in sequence ('temporal' | 'spatial')
    """
    out = x
    if "temporal" in modes and temporal_block is not None and temporal_block > 0:
        out = temporal_block_swap(out, ref=ref, t_block=temporal_block, swap_prob=swap_prob, seed=seed, blend_alpha=blend_alpha)
    if "spatial" in modes and spatial_block is not None:
        out = spatial_block_swap(out, ref=ref, block_size_hw=spatial_block, swap_prob=swap_prob, seed=seed, blend_alpha=blend_alpha)
    return out

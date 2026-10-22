"""Pytest for near-plane rasterization (packed projection + rasterize) and gradient check.

Usage:
    pytest tests/test_nearplane.py -v
"""

import math

import pytest
import torch

from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)


def _rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    near_plane,
):
    C = viewmats.shape[0]
    tile_size = 16
    eps2d = 0.3
    far_plane = 1e10
    radius_clip = 0.0

    proj_results = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        packed=True,
        sparse_grad=False,
        calc_compensations=False,
        camera_model="pinhole",
        opacities=None,
    )

    (
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations,
    ) = proj_results
    opacities = opacities[gaussian_ids]
    colors = colors[gaussian_ids]
    colors = torch.cat((colors, depths[..., None]), dim=-1)

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        sort=True,
        packed=True,
        n_images=C,
        image_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )

    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=None,
        masks=None,
        packed=True,
        absgrad=False,
    )
    return render_colors


# --- Input params (shapes and scalar args); data is created in fixtures with fixed seed) ---
NEARPLANE_N = 4
NEARPLANE_C = 1
NEARPLANE_WIDTH = 32
NEARPLANE_HEIGHT = 32
NEARPLANE_NEAR = 0.1
NEARPLANE_SEED = 42

# Baseline from a known-good run (means.grad.norm() with above params and seed); update if impl changes intentionally.
BASELINE_MEANS_GRAD_NORM = 7.45689058303833


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def nearplane_inputs(device):
    """Synthetic inputs for near-plane rasterization (fixed seed for reproducibility)."""
    torch.manual_seed(NEARPLANE_SEED)
    dev = device
    N, C = NEARPLANE_N, NEARPLANE_C
    # Means in a range that projects into view
    means = torch.randn(N, 3, device=dev) * 0.3
    means[:, 2] = means[:, 2].abs() + 1.0  # in front of camera
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).expand(N, 4).clone()
    scales = torch.rand(N, 3, device=dev).abs() * 0.1 + 0.01
    opacities = torch.sigmoid(torch.randn(N, device=dev))
    colors = torch.sigmoid(torch.randn(N, 4, device=dev))

    # Single-camera view and intrinsics
    viewmats = torch.eye(4, device=dev).unsqueeze(0).expand(C, 4, 4).clone()
    Ks = torch.eye(3, device=dev).unsqueeze(0).expand(C, 3, 3).clone()
    Ks[:, 0, 0] = 300.0
    Ks[:, 1, 1] = 300.0
    Ks[:, 0, 2] = NEARPLANE_WIDTH / 2.0
    Ks[:, 1, 2] = NEARPLANE_HEIGHT / 2.0

    return {
        "means": means.requires_grad_(True),
        "quats": quats.requires_grad_(True),
        "scales": scales.requires_grad_(True),
        "opacities": opacities.requires_grad_(True),
        "colors": colors.requires_grad_(True),
        "viewmats": viewmats,
        "Ks": Ks,
        "width": NEARPLANE_WIDTH,
        "height": NEARPLANE_HEIGHT,
        "near_plane": NEARPLANE_NEAR,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nearplane_rasterization_forward(nearplane_inputs):
    """Packed projection + rasterization runs and returns expected shape."""
    inp = nearplane_inputs
    render_colors = _rasterization(
        inp["means"],
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        inp["viewmats"],
        inp["Ks"],
        inp["width"],
        inp["height"],
        inp["near_plane"],
    )
    C, H, W = NEARPLANE_C, NEARPLANE_HEIGHT, NEARPLANE_WIDTH
    # Packed rasterize returns [..., H, W, channels]; we have 4+1 channels (colors + depth)
    assert render_colors.dim() == 4
    assert render_colors.shape[0] == C
    assert render_colors.shape[1] == H
    assert render_colors.shape[2] == W
    assert render_colors.shape[3] >= 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nearplane_rasterization_backward(nearplane_inputs):
    """Backward through near-plane rasterization produces finite gradients."""
    inp = nearplane_inputs
    means = inp["means"]
    render_colors = _rasterization(
        means,
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        inp["viewmats"],
        inp["Ks"],
        inp["width"],
        inp["height"],
        inp["near_plane"],
    )
    # Dummy upstream gradient (same shape as permuted render: [C, C, H, W] for backward)
    permuted = render_colors.permute(0, 3, 1, 2)
    grad = torch.ones_like(permuted, device=permuted.device) * 0.01
    permuted.backward(grad)
    assert means.grad is not None
    assert torch.isfinite(means.grad).all()
    assert means.grad.norm().item() > 0
    computed_norm = means.grad.norm().item()
    assert computed_norm == pytest.approx(
        BASELINE_MEANS_GRAD_NORM,
        rel=1e-5,
        abs=1e-5,
    ), (
        f"means.grad.norm() {computed_norm} does not match baseline {BASELINE_MEANS_GRAD_NORM}; "
        "update BASELINE_MEANS_GRAD_NORM if the implementation changed intentionally."
    )
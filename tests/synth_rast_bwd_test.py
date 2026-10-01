"""Synthetic repro for rasterize_to_pixels_3dgs_bwd int32 overflow.

Generates fully synthetic data and runs the backward kernel.
The tile_offsets tensor is int32, which overflows when total intersections
exceed 2^31. This causes illegal memory access in the kernel.

Run with:
    pytest tests/synth_rast_bwd_test.py -v
    pytest tests/synth_rast_bwd_test.py -v -k "81 or 82"   # specific camera counts
"""
import pytest
import torch

W, H, TILE = 640, 480, 16
TILE_W, TILE_H = W // TILE, H // TILE  # 40, 30
CDIM = 16
NNZ_PER_CAM = 634_000
ISECTS_PER_GAUSS = 42
INT32_MAX = 2**31 - 1

# Overflow threshold: INT32_MAX / (634000 * 42) ≈ 80.6
DEFAULT_CAMERA_COUNTS = [60, 70, 78, 80, 81, 82, 85, 100, 132]


def _run_rasterize_bwd_kernel(C, device="cuda:0"):
    """Run rasterize_to_pixels_3dgs_bwd with synthetic data. Returns True if no crash."""
    nnz = NNZ_PER_CAM * C
    n_isects = nnz * ISECTS_PER_GAUSS

    total_tiles = C * TILE_H * TILE_W
    base = n_isects // total_tiles
    counts = torch.full((total_tiles,), base, dtype=torch.int64)
    counts[: n_isects % total_tiles] += 1
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)[:-1]])
    isect_offsets = starts.reshape(C, TILE_H, TILE_W).to(device)

    ends = torch.cat([starts[1:], torch.tensor([n_isects], dtype=torch.int64)])
    mids = ((starts + ends) // 2).to(torch.int32).reshape(C, TILE_H, TILE_W)
    last_ids = (
        mids.repeat_interleave(TILE, dim=1)
        .repeat_interleave(TILE, dim=2)[:, :H, :W]
        .to(device)
    )

    flatten_ids = torch.randint(0, nnz, (n_isects,), dtype=torch.int32, device=device)
    means2d = torch.randn(nnz, 2, device=device)
    conics = torch.randn(nnz, 3, device=device)
    colors = torch.rand(nnz, CDIM, device=device)
    opacities = torch.rand(nnz, device=device).clamp(0.01, 0.99)
    render_alphas = torch.rand(C, H, W, 1, device=device).clamp(0.01, 0.99)
    v_render_colors = torch.randn(C, H, W, CDIM, device=device) * 0.001
    v_render_alphas = torch.randn(C, H, W, 1, device=device) * 0.001

    torch.cuda.synchronize(device)
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    try:
        _make_lazy_cuda_func("rasterize_to_pixels_3dgs_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            None,  # backgrounds
            None,  # masks
            W,
            H,
            TILE,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            False,  # absgrad
        )
        torch.cuda.synchronize(device)
        return True
    except RuntimeError:
        return False
    finally:
        del (
            flatten_ids,
            means2d,
            conics,
            colors,
            opacities,
            render_alphas,
            v_render_colors,
            v_render_alphas,
            isect_offsets,
            last_ids,
        )
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("C", DEFAULT_CAMERA_COUNTS)
def test_rasterize_to_pixels_3dgs_bwd_synthetic(C):
    """rasterize_to_pixels_3dgs_bwd runs without crashing for given camera count."""
    device = "cuda:0"
    nnz = NNZ_PER_CAM * C
    n_isects = nnz * ISECTS_PER_GAUSS
    overflow = n_isects > INT32_MAX

    ok = _run_rasterize_bwd_kernel(C, device)

    assert ok, (
        f"rasterize_to_pixels_3dgs_bwd crashed for C={C}, "
        f"n_isects={n_isects:,}, overflow={overflow}"
    )

"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math
import os

import pytest
import torch
from typing_extensions import Literal, Tuple, assert_never

from gsplat._helper import load_test_data

# --- Pytest Fixture for Profiling ---
import re
@pytest.fixture(autouse=True)
def profile_test_with_torch(request):
    """
    An autouse fixture that profiles each test function using torch.profiler.

    Profiling is only enabled if the environment variable ENABLE_PROFILER is set to "1".
    The output trace is saved in a 'torch_prof/logs/tests/' directory, with a subdirectory named
    after the specific test being run.
    """
    # Check if the profiler is enabled via environment variable
    if os.getenv("ENABLE_PROFILER") != "1":
        yield # Proceed with the test without profiling
        return

    # Sanitize the test's node ID to create a valid directory name for the trace.
    # e.g., 'test_projection[fused0-pinhole-()]' -> 'test_projection_fused0_pinhole___'
    node_id = request.node.nodeid
    sanitized_node_id = re.sub(r'[:\[\]\-/]', '_', node_id.split("::")[-1])

    # Define profiler activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    log_dir = f"./torch_prof/logs/tests/{sanitized_node_id}"
    print(f"\n[Profiler] Enabled for {node_id}.")
    print(f"[Profiler] Trace will be saved to '{log_dir}'")

    # The 'with' block ensures the profiler starts and stops correctly
    with torch.profiler.profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        yield # This is where the actual test function runs

    # This code runs after the test has completed
    print(f"\n[Profiler] Results for {node_id}:")
    # Print a summary table to the console, sorted by total CUDA time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


device = torch.device("cuda:0")


def expand(data: dict, batch_dims: Tuple[int, ...]):
    # append multiple batch dimensions to the front of the tensor
    # eg. x.shape = [N, 3], batch_dims = (1, 2), return shape is [1, 2, N, 3]
    # eg. x.shape = [N, 3], batch_dims = (), return shape is [N, 3]
    ret = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and len(batch_dims) > 0:
            new_shape = batch_dims + v.shape
            ret[k] = v.expand(new_shape)
        else:
            ret[k] = v
    return ret


@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    return {
        "means": means,  # [N, 3]
        "quats": quats,  # [N, 4]
        "scales": scales,  # [N, 3]
        "opacities": opacities,  # [N]
        "viewmats": viewmats,  # [C, 4, 4]
        "Ks": Ks,  # [C, 3, 3]
        "width": width,
        "height": height,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("triu", [False, True])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_quat_scale_to_covar_preci(test_data, triu: bool, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    quats = test_data["quats"]
    scales = test_data["scales"]
    quats.requires_grad = True
    scales.requires_grad = True

    # forward
    covars, precis = quat_scale_to_covar_preci(quats, scales, triu=triu)
    _covars, _precis = _quat_scale_to_covar_preci(quats, scales, triu=triu)
    torch.testing.assert_close(covars, _covars)
    # This test is disabled because the numerical instability.
    # torch.testing.assert_close(precis, _precis, rtol=2e-2, atol=1e-2)
    # if not triu:
    #     I = torch.eye(3, device=device).expand(len(covars), 3, 3)
    #     torch.testing.assert_close(torch.bmm(covars, precis), I)
    #     torch.testing.assert_close(torch.bmm(precis, covars), I)

    # backward
    v_covars = torch.randn_like(covars)
    v_precis = torch.randn_like(precis) * 0.01
    v_quats, v_scales = torch.autograd.grad(
        (covars * v_covars + precis * v_precis).sum(),
        (quats, scales),
    )
    _v_quats, _v_scales = torch.autograd.grad(
        (_covars * v_covars + _precis * v_precis).sum(),
        (quats, scales),
    )
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e0, atol=1e-1)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e0, atol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_proj(
    test_data,
    camera_model: Literal["pinhole", "ortho", "fisheye"],
    batch_dims: Tuple[int, ...],
):
    from gsplat.cuda._torch_impl import (
        _fisheye_proj,
        _ortho_proj,
        _persp_proj,
        _world_to_cam,
    )
    from gsplat.cuda._wrapper import proj, quat_scale_to_covar_preci

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]

    covars, _ = quat_scale_to_covar_preci(test_data["quats"], test_data["scales"])
    means, covars = _world_to_cam(test_data["means"], covars, viewmats)
    means.requires_grad = True
    covars.requires_grad = True

    # forward
    means2d, covars2d = proj(means, covars, Ks, width, height, camera_model)
    if camera_model == "ortho":
        _means2d, _covars2d = _ortho_proj(means, covars, Ks, width, height)
    elif camera_model == "fisheye":
        _means2d, _covars2d = _fisheye_proj(means, covars, Ks, width, height)
    elif camera_model == "pinhole":
        _means2d, _covars2d = _persp_proj(means, covars, Ks, width, height)
    else:
        assert_never(camera_model)

    torch.testing.assert_close(means2d, _means2d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(covars2d, _covars2d, rtol=1e-1, atol=3e-2)

    # backward
    v_means2d = torch.randn_like(means2d)
    v_covars2d = torch.randn_like(covars2d)
    v_means, v_covars = torch.autograd.grad(
        (means2d * v_means2d).sum() + (covars2d * v_covars2d).sum(),
        (means, covars),
    )
    _v_means, _v_covars = torch.autograd.grad(
        (_means2d * v_means2d).sum() + (_covars2d * v_covars2d).sum(),
        (means, covars),
    )
    torch.testing.assert_close(v_means, _v_means, rtol=6e-1, atol=1e-2)
    torch.testing.assert_close(v_covars, _v_covars, rtol=1e-1, atol=1e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("z", [0.0, 1e-6, -1e-6])
def test_proj_degenerate_depth(test_data, z: float):
    """`proj()` with the pinhole camera goes straight to `persp_proj`, which does
    `rz = 1.f / z` with no near-plane guard (unlike the fused projection kernel).
    Degenerate camera-space depths must not silently leak Inf/NaN into the
    projected means / covariances.
    """
    from gsplat.cuda._torch_impl import _world_to_cam
    from gsplat.cuda._wrapper import proj, quat_scale_to_covar_preci

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]

    covars, _ = quat_scale_to_covar_preci(test_data["quats"], test_data["scales"])
    means, covars = _world_to_cam(test_data["means"], covars, viewmats)

    # override the depth of a subset of the gaussians with a degenerate value
    means = means.clone()
    means[..., ::128, 2] = z
    assert torch.isfinite(means).all() and torch.isfinite(covars).all()

    means2d, covars2d = proj(means, covars, Ks, width, height, "pinhole")

    assert torch.isfinite(means2d).all(), (
        f"means2d has {(~torch.isfinite(means2d)).sum().item()} non-finite entries "
        f"for camera-space depth z={z}"
    )
    assert torch.isfinite(covars2d).all(), (
        f"covars2d has {(~torch.isfinite(covars2d)).sum().item()} non-finite entries "
        f"for camera-space depth z={z}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("calc_compensations", [True, False])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_projection(
    test_data,
    fused: bool,
    calc_compensations: bool,
    camera_model: Literal["pinhole", "ortho", "fisheye"],
    batch_dims: Tuple[int, ...],
):
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.cuda._wrapper import fully_fused_projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]

    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    if fused:
        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [..., N, 6]
        radii, means2d, depths, conics, compensations = fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    _covars, _ = quat_scale_to_covar_preci(quats, scales, triu=False)  # [..., N, 3, 3]
    _radii, _means2d, _depths, _conics, _compensations = _fully_fused_projection(
        means,
        _covars,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
    )

    # radii is integer so we allow for 1 unit difference
    valid = (radii > 0).all(dim=-1) & (_radii > 0).all(dim=-1)
    torch.testing.assert_close(radii, _radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], _conics[valid], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(
            compensations[valid], _compensations[valid], rtol=1e-4, atol=1e-3
        )

    # backward
    v_means2d = torch.randn_like(means2d) * valid[..., None]
    v_depths = torch.randn_like(depths) * valid
    v_conics = torch.randn_like(conics) * valid[..., None]
    if calc_compensations:
        v_compensations = torch.randn_like(compensations) * valid
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (conics * v_conics).sum()
        + ((compensations * v_compensations).sum() if calc_compensations else 0),
        (viewmats, quats, scales, means),
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum()
        + ((_compensations * v_compensations).sum() if calc_compensations else 0),
        (viewmats, quats, scales, means),
    )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=2e-1, atol=2e-2)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-1, atol=2e-1)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("sparse_grad", [False])
@pytest.mark.parametrize("calc_compensations", [False, True])
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho", "fisheye"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_fully_fused_projection_packed(
    test_data,
    fused: bool,
    sparse_grad: bool,
    calc_compensations: bool,
    camera_model: Literal["pinhole", "ortho", "fisheye"],
    batch_dims: Tuple[int, ...],
):
    from gsplat.cuda._wrapper import fully_fused_projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]

    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    # forward
    if fused:
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            packed=True,
            sparse_grad=sparse_grad,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
        _radii, _means2d, _depths, _conics, _compensations = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            packed=False,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [..., N, 6]
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            packed=True,
            sparse_grad=sparse_grad,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )
        _radii, _means2d, _depths, _conics, _compensations = fully_fused_projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            packed=False,
            calc_compensations=calc_compensations,
            camera_model=camera_model,
        )

    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]

    # recover packed tensors to full matrices for testing
    __radii = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), radii, (B, C, N, 2)
    ).to_dense()
    __radii = __radii.reshape(batch_dims + (C, N, 2))
    __means2d = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), means2d, (B, C, N, 2)
    ).to_dense()
    __means2d = __means2d.reshape(batch_dims + (C, N, 2))
    __depths = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), depths, (B, C, N)
    ).to_dense()
    __depths = __depths.reshape(batch_dims + (C, N))
    __conics = torch.sparse_coo_tensor(
        torch.stack([batch_ids, camera_ids, gaussian_ids]), conics, (B, C, N, 3)
    ).to_dense()
    __conics = __conics.reshape(batch_dims + (C, N, 3))
    if calc_compensations:
        __compensations = torch.sparse_coo_tensor(
            torch.stack([batch_ids, camera_ids, gaussian_ids]),
            compensations,
            (B, C, N),
        ).to_dense()
        __compensations = __compensations.reshape(batch_dims + (C, N))
    sel = (__radii > 0).all(dim=-1) & (_radii > 0).all(dim=-1)
    torch.testing.assert_close(__radii[sel], _radii[sel], rtol=0, atol=1)
    torch.testing.assert_close(__means2d[sel], _means2d[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__depths[sel], _depths[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__conics[sel], _conics[sel], rtol=1e-4, atol=1e-4)
    if calc_compensations:
        torch.testing.assert_close(
            __compensations[sel], _compensations[sel], rtol=1e-4, atol=1e-3
        )

    # backward
    v_means2d = torch.randn_like(_means2d) * sel[..., None]
    v_depths = torch.randn_like(_depths) * sel
    v_conics = torch.randn_like(_conics) * sel[..., None]
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d[(__radii > 0).all(dim=-1)]).sum()
        + (depths * v_depths[(__radii > 0).all(dim=-1)]).sum()
        + (conics * v_conics[(__radii > 0).all(dim=-1)]).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    if sparse_grad:
        v_quats = v_quats.to_dense()
        v_scales = v_scales.to_dense()
        v_means = v_means.to_dense()

    # print(f"Comparing v_viewmats: {v_viewmats}")
    # print(f"Against _v_viewmats: {_v_viewmats}")

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_scales, _v_scales, rtol=5e-2, atol=5e-2)
    # print(f"Comparing v_means: {v_means}")
    # print(f"Against _v_means: {_v_means}")
    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_isect(test_data, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    torch.manual_seed(42)

    B = math.prod(batch_dims)
    C, N = 3, 1000
    I = B * C
    width, height = 40, 60

    test_data = {
        "means2d": torch.randn(C, N, 2, device=device) * width,
        "radii": torch.randint(0, width, (C, N, 2), device=device, dtype=torch.int32),
        "depths": torch.rand(C, N, device=device),
    }
    test_data = expand(test_data, batch_dims)
    means2d = test_data["means2d"]
    radii = test_data["radii"]
    depths = test_data["depths"]

    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)

    _tiles_per_gauss, _isect_ids, _gauss_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    _isect_offsets = _isect_offset_encode(_isect_ids, I, tile_width, tile_height)

    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(flatten_ids, _gauss_ids)
    torch.testing.assert_close(isect_offsets, _isect_offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("n_isects", [0, 1])
def test_isect_offset_encode_tiny(n_isects: int):
    """Boundary case of `isect_offset_encode`: 0 or 1 total intersections.

    With n_isects == 0 the kernel is not launched at all and the output must be
    zero-filled; with n_isects == 1 the single thread takes the `idx == 0` and
    the `idx == n_isects - 1` branches simultaneously, so it alone has to fill
    both the leading and the trailing part of the (uninitialized) offsets array.
    """
    from gsplat.cuda._torch_impl import _isect_offset_encode
    from gsplat.cuda._wrapper import isect_offset_encode

    I = 2
    tile_width, tile_height = 5, 3
    n_tiles = tile_width * tile_height
    tile_n_bits = n_tiles.bit_length()

    if n_isects == 0:
        isect_ids = torch.zeros(0, device=device, dtype=torch.int64)
        # every entry of the [I, tile_height, tile_width] output must be zero
        expected = torch.zeros(
            (I, tile_height, tile_width), device=device, dtype=torch.int32
        )
    else:
        # a single intersection on image 1, tile (x=2, y=1), i.e. not the very
        # first nor the very last tile, so that both fill loops do real work.
        image_id, tile_x, tile_y = 1, 2, 1
        tile_id = tile_y * tile_width + tile_x
        depth_bits = 12345  # lower 32 bits carry the depth, they must be ignored
        isect_ids = torch.tensor(
            [(((image_id << tile_n_bits) | tile_id) << 32) | depth_bits],
            device=device,
            dtype=torch.int64,
        )
        # offsets[j] == number of intersections stored before tile j, so it is
        # 0 up to and including the occupied tile and 1 for every tile after it.
        flat_id = image_id * n_tiles + tile_id
        expected = torch.arange(I * n_tiles, device=device, dtype=torch.int32)
        expected = (expected > flat_id).to(torch.int32)
        expected = expected.reshape(I, tile_height, tile_width)

    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    _isect_offsets = _isect_offset_encode(isect_ids, I, tile_width, tile_height)

    assert isect_offsets.shape == (I, tile_height, tile_width)
    assert (isect_offsets >= 0).all(), isect_offsets
    torch.testing.assert_close(isect_offsets, expected)
    torch.testing.assert_close(isect_offsets, _isect_offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3, 32, 128])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_rasterize_to_pixels(test_data, channels: int, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl import _rasterize_to_pixels
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        quat_scale_to_covar_preci,
        rasterize_to_pixels,
    )

    torch.manual_seed(42)

    N = test_data["means"].shape[-2]
    C = test_data["viewmats"].shape[-3]
    I = math.prod(batch_dims) * C
    test_data.update(
        {
            "colors": torch.rand(C, N, channels, device=device),
            "backgrounds": torch.rand((C, channels), device=device),
        }
    )
    test_data = expand(test_data, batch_dims)
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    colors = test_data["colors"]
    backgrounds = test_data["backgrounds"]

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    # Project Gaussians to 2D
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )
    opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))

    # Identify intersecting tiles
    # tile_size = 16 if channels <= 32 else 4
    # Changing to 8 tile size as 4 is leading to hang issues on AMD GPUs
    tile_size = 16 if channels <= 32 else 8
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    # forward
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
        backgrounds=backgrounds,
    )
    _render_colors, _render_alphas = _rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )
    torch.testing.assert_close(render_colors, _render_colors)
    torch.testing.assert_close(render_alphas, _render_alphas)

    # backward
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

    v_means2d, v_conics, v_colors, v_opacities, v_backgrounds = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )
    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_opacities,
        _v_backgrounds,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum()
        + (_render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(v_conics, _v_conics, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=8e-3, atol=6e-3)
    torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_sh(test_data, sh_degree: int, batch_dims: Tuple[int, ...]):
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.cuda._wrapper import spherical_harmonics

    torch.manual_seed(42)

    N = 1000
    test_data = {
        "coeffs": torch.randn(N, (4 + 1) ** 2, 3, device=device),
        "dirs": torch.randn(N, 3, device=device),
    }
    test_data = expand(test_data, batch_dims)
    coeffs = test_data["coeffs"]
    dirs = test_data["dirs"]
    coeffs.requires_grad = True
    dirs.requires_grad = True

    colors = spherical_harmonics(sh_degree, dirs, coeffs)
    _colors = _spherical_harmonics(sh_degree, dirs, coeffs)
    torch.testing.assert_close(colors, _colors, rtol=1e-4, atol=1e-4)

    v_colors = torch.randn_like(colors)

    v_coeffs, v_dirs = torch.autograd.grad(
        (colors * v_colors).sum(), (coeffs, dirs), retain_graph=True, allow_unused=True
    )
    _v_coeffs, _v_dirs = torch.autograd.grad(
        (_colors * v_colors).sum(), (coeffs, dirs), retain_graph=True, allow_unused=True
    )
    torch.testing.assert_close(v_coeffs, _v_coeffs, rtol=1e-4, atol=1e-4)
    if sh_degree > 0:
        torch.testing.assert_close(v_dirs, _v_dirs, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", [1, 2, 3, 4])
def test_sh_zero_direction(sh_degree: int):
    """A zero-magnitude direction must not poison the SH gradients.

    ``sh_coeffs_to_color_fast_vjp`` normalizes ``dir`` with an unguarded
    ``rsqrtf(dir.x^2 + dir.y^2 + dir.z^2)``. For ``dir == (0, 0, 0)`` that
    reciprocal is ``+inf``, so ``x = dir.x * inorm`` is ``NaN`` and the NaN
    propagates into every higher-band ``v_coeffs`` entry and into ``v_dir``.
    Such a direction occurs whenever a Gaussian mean coincides with the camera
    center, and it silently corrupts the whole training step.
    """
    from gsplat.cuda._wrapper import spherical_harmonics

    torch.manual_seed(42)

    N = 8
    coeffs = torch.randn(N, (4 + 1) ** 2, 3, device=device)
    dirs = torch.randn(N, 3, device=device)
    # One degenerate row mixed in with otherwise well-behaved directions.
    zero_idx = 3
    dirs[zero_idx] = 0.0
    coeffs.requires_grad = True
    dirs.requires_grad = True

    colors = spherical_harmonics(sh_degree, dirs, coeffs)
    v_colors = torch.randn_like(colors)
    v_coeffs, v_dirs = torch.autograd.grad(
        (colors * v_colors).sum(), (coeffs, dirs), allow_unused=True
    )

    assert torch.isfinite(v_coeffs[zero_idx]).all(), (
        "v_coeffs for the zero-magnitude direction is not finite: "
        f"{v_coeffs[zero_idx]}"
    )
    assert torch.isfinite(v_dirs[zero_idx]).all(), (
        "v_dirs for the zero-magnitude direction is not finite: "
        f"{v_dirs[zero_idx]}"
    )

    # The degenerate row must not be "fixed" by zeroing everything out: the
    # remaining rows still have to agree with the reference implementation.
    from gsplat.cuda._torch_impl import _spherical_harmonics

    _colors = _spherical_harmonics(sh_degree, dirs, coeffs)
    _v_coeffs, _v_dirs = torch.autograd.grad(
        (_colors * v_colors).sum(), (coeffs, dirs), allow_unused=True
    )
    keep = torch.ones(N, dtype=torch.bool, device=device)
    keep[zero_idx] = False
    torch.testing.assert_close(
        v_coeffs[keep], _v_coeffs[keep], rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(v_dirs[keep], _v_dirs[keep], rtol=1e-4, atol=1e-4)

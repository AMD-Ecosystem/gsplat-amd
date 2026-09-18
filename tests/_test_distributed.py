import pytest
import torch
import torch.distributed.nn.functional as distF

from gsplat.distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
    cli,
)


def _main_all_gather_int32(local_rank: int, world_rank: int, world_size: int, _):
    device = torch.device("cuda", local_rank)

    value = world_rank
    collected = all_gather_int32(world_size, value, device=device)
    for i in range(world_size):
        assert collected[i] == i

    value = torch.tensor(world_rank, device=device, dtype=torch.int)
    collected = all_gather_int32(world_size, value, device=device)
    for i in range(world_size):
        assert collected[i] == torch.tensor(i, device=device, dtype=torch.int)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_all_gather_int32():
    cli(_main_all_gather_int32, None, verbose=True)


def _main_all_to_all_int32(local_rank: int, world_rank: int, world_size: int, _):
    device = torch.device("cuda", local_rank)

    values = list(range(world_size))
    collected = all_to_all_int32(world_size, values, device=device)
    for i in range(world_size):
        assert collected[i] == world_rank

    values = torch.arange(world_size, device=device, dtype=torch.int)
    collected = all_to_all_int32(world_size, values, device=device)
    for i in range(world_size):
        assert collected[i] == torch.tensor(world_rank, device=device, dtype=torch.int)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_all_to_all_int32():
    cli(_main_all_to_all_int32, None, verbose=True)


def _main_all_gather_tensor_list(local_rank: int, world_rank: int, world_size: int, _):
    device = torch.device("cuda", local_rank)
    N = 10

    tensor_list = [
        torch.full((N, 2), world_rank, device=device),
        torch.full((N, 3, 3), world_rank, device=device),
    ]

    target_list = [
        torch.cat([torch.full((N, 2), i, device=device) for i in range(world_size)]),
        torch.cat([torch.full((N, 3, 3), i, device=device) for i in range(world_size)]),
    ]

    collected = all_gather_tensor_list(world_size, tensor_list)
    for tensor, target in zip(collected, target_list):
        assert torch.equal(tensor, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_all_gather_tensor_list():
    cli(_main_all_gather_tensor_list, None, verbose=True)


def test_all_gather_tensor_list_world_size_one(monkeypatch):
    """With world_size == 1 the input list is handed back as-is, no collective."""

    def _fail(*args, **kwargs):
        raise AssertionError("no collective op should be issued when world_size == 1")

    monkeypatch.setattr(torch.distributed, "all_gather", _fail)
    monkeypatch.setattr(distF, "all_gather", _fail)

    N = 10
    tensor_list = [
        torch.full((N, 2), 7.0),
        torch.full((N, 3, 3), 7.0, requires_grad=True),
    ]

    collected = all_gather_tensor_list(1, tensor_list)

    # the very same list object, holding the very same tensor objects
    assert collected is tensor_list
    assert len(collected) == len(tensor_list)
    for out, tensor in zip(collected, tensor_list):
        assert out is tensor
        assert out.shape == tensor.shape


def _main_all_to_all_tensor_list(local_rank: int, world_rank: int, world_size: int, _):
    device = torch.device("cuda", local_rank)
    splits = torch.arange(0, world_size, device=device)
    N = splits.sum().item()

    tensor_list = [
        torch.full((N, 2), world_rank, device=device),
        torch.full((N, 3, 3), world_rank, device=device),
    ]

    target_list = [
        torch.cat(
            [
                torch.full((splits[world_rank], 2), i, device=device)
                for i in range(world_size)
            ]
        ),
        torch.cat(
            [
                torch.full((splits[world_rank], 3, 3), i, device=device)
                for i in range(world_size)
            ]
        ),
    ]

    collected = all_to_all_tensor_list(world_size, tensor_list, splits)
    for tensor, target in zip(collected, target_list):
        assert torch.equal(tensor, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_all_to_all_tensor_list():
    cli(_main_all_to_all_tensor_list, None, verbose=True)


if __name__ == "__main__":
    test_all_gather_int32()
    test_all_to_all_int32()
    test_all_gather_tensor_list()
    test_all_to_all_tensor_list()

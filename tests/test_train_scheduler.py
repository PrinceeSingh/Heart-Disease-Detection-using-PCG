import torch

from src.train import build_scheduler


def test_build_scheduler_works_with_current_torch_version():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(1))], lr=0.01)

    scheduler = build_scheduler(optimizer)

    assert scheduler is not None
    assert scheduler.__class__.__name__ == "ReduceLROnPlateau"

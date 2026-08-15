"""Tests for the training orchestrator (model.training.train)."""

from model.training.train import SystemInfo


def test_mt_optimal_batch_size_matches_cls_gpu_tier():
    system = SystemInfo(has_gpu=True, gpu_memory_gb=11.6)
    assert system.optimal_batch_size("mt") == 16
    assert system.optimal_batch_size("mt") == system.optimal_batch_size("cls")


def test_mt_optimal_batch_size_falls_back_without_gpu():
    system = SystemInfo(has_gpu=False, gpu_memory_gb=0.0)
    assert system.optimal_batch_size("mt") == 4

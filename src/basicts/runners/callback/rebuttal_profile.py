"""Runtime measurements used by the NeurIPS rebuttal experiments."""

from __future__ import annotations

import copy
import json
import os
import time
from statistics import mean

import torch
import torch.nn.functional as F

from .callback import BasicTSCallback


class RebuttalProfiler(BasicTSCallback):
    def __init__(self, warmup: int = 20, iterations: int = 100):
        self.warmup = warmup
        self.iterations = iterations
        self.epoch_start = None
        self.epoch_times = []
        self.train_peak_bytes = 0
        self.best_value = None
        self.best_epoch = None

    @staticmethod
    def _sync(runner):
        if runner.cfg.gpus is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_start(self, runner, **kwargs):
        if runner.cfg.gpus is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_start(self, runner, **kwargs):
        self._sync(runner)
        self.epoch_start = time.perf_counter()
        if runner.cfg.gpus is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(self, runner, **kwargs):
        self._sync(runner)
        if self.epoch_start is not None:
            self.epoch_times.append(time.perf_counter() - self.epoch_start)
        if runner.cfg.gpus is not None and torch.cuda.is_available():
            self.train_peak_bytes = max(self.train_peak_bytes, torch.cuda.max_memory_allocated())

    def on_validate_end(self, runner, train_epoch=None, **kwargs):
        value = runner.meter_pool.get_value(f"val/{runner.target_metric}")
        if value is None:
            return
        better = self.best_value is None or (
            value < self.best_value if runner.metrics_best == "min" else value > self.best_value
        )
        if better:
            self.best_value = float(value)
            self.best_epoch = train_epoch

    @staticmethod
    def _basis_cosine(model):
        values = []
        for module in model.modules():
            table = getattr(module, "mode_table", None)
            if table is None or table.ndim != 2 or table.shape[0] < 2:
                continue
            normalized = F.normalize(table.detach().float(), dim=-1)
            gram = normalized @ normalized.T
            mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
            values.append(gram[mask].abs().mean().item())
        return mean(values) if values else None

    @torch.no_grad()
    def _profile_inference(self, runner):
        batch = next(iter(runner.test_data_loader))
        batch = runner.taskflow.preprocess(runner, batch)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = runner.to_running_device(value)
        inputs = batch["inputs"]
        kwargs = {key: batch[key] for key in runner.forward_params if key in batch and key != "targets"}

        runner.model.eval()
        for _ in range(self.warmup):
            runner.model(inputs, **kwargs)
        self._sync(runner)
        if runner.cfg.gpus is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(self.iterations):
            runner.model(inputs, **kwargs)
        self._sync(runner)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() if runner.cfg.gpus is not None and torch.cuda.is_available() else 0
        return elapsed * 1000.0 / self.iterations, peak, inputs

    def on_test_end(self, runner, **kwargs):
        latency_ms, inference_peak, inputs = self._profile_inference(runner)
        profile = {
            "params": sum(parameter.numel() for parameter in runner.model.parameters()),
            "trainable_params": sum(
                parameter.numel() for parameter in runner.model.parameters() if parameter.requires_grad
            ),
            "train_seconds_per_epoch_mean": mean(self.epoch_times) if self.epoch_times else None,
            "train_seconds_per_epoch_all": self.epoch_times,
            "best_validation_epoch": self.best_epoch,
            "train_peak_gb": self.train_peak_bytes / 1024**3,
            "inference_ms_per_batch": latency_ms,
            "inference_peak_gb": inference_peak / 1024**3,
            "profile_batch_shape": list(inputs.shape),
            "mean_off_diagonal_basis_cosine": self._basis_cosine(runner.model),
        }
        try:
            from thop import profile as thop_profile

            # THOP registers bookkeeping buffers on every visited module but only
            # removes them from modules with a counting hook. Profiling a copy
            # prevents those buffers from contaminating BasicTS's subsequent
            # strict reload of the best-validation checkpoint.
            profiling_model = copy.deepcopy(runner.model)
            macs, _ = thop_profile(profiling_model, inputs=(inputs,), verbose=False)
            profile["gmacs"] = float(macs) / 1e9
            del profiling_model
        except Exception as exc:  # profiling should not invalidate a completed run
            profile["gmacs"] = None
            profile["gmacs_error"] = str(exc)

        path = os.path.join(runner.ckpt_save_dir, "rebuttal_profile.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2)
        runner.logger.info(f"Rebuttal profile saved to {path}.")

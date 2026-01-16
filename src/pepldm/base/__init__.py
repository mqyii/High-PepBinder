import os
import math

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import lightning as L
from torch.optim import AdamW
from torchmetrics import Metric
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizerBase
from lightning.pytorch.callbacks import Callback
from loguru import logger


@dataclass
class MetricCollection:
    metrics: Dict[str, Metric] = field(default_factory=dict)

    @classmethod
    def from_metrics(cls, metrics: Dict[str, Metric]):
        return cls(metrics=metrics)

    def add_metric(self, name: str, metric: Metric):
        if name in self.metrics:
            raise KeyError(f"Metric '{name}' already exists.")
        self.metrics[name] = metric

    def update(self, name: str, value: float, labels=None, **kwargs):
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found in collection.")
        if labels is not None:
            self.metrics[name].update(value, labels, **kwargs)
        else:
            self.metrics[name].update(value, **kwargs)

    def compute(self, name: str):
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found in collection.")
        return self.metrics[name].compute()

    def compute_all(self, prefix: Optional[str] = None):
        results = {}
        for name, metric in self.metrics.items():
            if prefix is None or name.startswith(prefix):
                results[name] = metric.compute()
        return results

    def reset(self, name: str):
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found in collection.")
        self.metrics[name].reset()

    def reset_all(self, prefix: Optional[str] = None):
        for name, metric in self.metrics.items():
            if prefix is None or name.startswith(prefix):
                metric.reset()

    def to(self, device: torch.device, prefix: Optional[str] = None):
        for name, metric in self.metrics.items():
            if prefix is None or name.startswith(prefix):
                metric.to(device)


def log_step(
    pl_module: L.LightningModule,
    name: str,
    value: float,
    prog_bar: bool = True,
    sync_dist: bool = True,
    on_step: bool = True,
    on_epoch: bool = False,
):
    """记录单步指标"""
    pl_module.log(
        name,
        value,
        prog_bar=prog_bar,
        sync_dist=sync_dist,
        on_step=on_step,
        on_epoch=on_epoch,
    )


def log_epoch(
    pl_module: L.LightningModule,
    name: str,
    value: float,
    prog_bar: bool = True,
    sync_dist: bool = True,
    on_epoch: bool = True,
):
    """记录 epoch 指标"""
    pl_module.log(
        name, value, prog_bar=prog_bar, sync_dist=sync_dist, on_epoch=on_epoch
    )


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        warmup_ratio: float = 0.05,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.warmup_ratio = warmup_ratio
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def _init_metrics(self):
        raise NotImplementedError

    def on_train_start(self):
        self.metrics.to(self.device, prefix="train_")

    def on_validation_start(self):
        self.metrics.to(self.device, prefix="val_")

    def on_test_start(self):
        self.metrics.to(self.device, prefix="test_")
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.warmup_steps and self.total_steps:
            warmup_steps = self.warmup_steps
            total_steps = self.total_steps
        else:
            steps_per_epoch = math.ceil(
                len(self.trainer.datamodule.train_dataloader())
                / self.trainer.accumulate_grad_batches
            )
            total_steps = steps_per_epoch * self.trainer.max_epochs
            warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def log_epoch(self, name: str, value: float):
        log_epoch(self, name, value)

    def log_step(self, name: str, value: float):
        log_step(self, name, value)

    def load_weights_from_lightning(
        self, checkpoint_path: str, map_location: str = "cpu"
    ):
        """
        Load model weights from a PyTorch Lightning checkpoint.

        Args:
            checkpoint_path (str): Path to the .ckpt file.
            map_location (str): Device to load checkpoint onto (default: "cpu").
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )

        if "state_dict" not in checkpoint:
            raise ValueError(f"No state_dict found in checkpoint {checkpoint_path}")
        original_state_dict = checkpoint["state_dict"]

        # 当前模型参数
        model_state_dict = self.state_dict()

        # 过滤出匹配的参数
        filtered_state_dict = {
            k: v
            for k, v in original_state_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }

        # 加载
        missing_keys, unexpected_keys = self.load_state_dict(
            filtered_state_dict, strict=False
        )

        # 统计信息
        total_params = len(model_state_dict)
        loaded_params = len(filtered_state_dict)
        ratio = loaded_params / total_params * 100

        logger.info(
            f"Loaded {loaded_params}/{total_params} ({ratio:.2f}%) weights "
            f"from checkpoint: {checkpoint_path}"
        )
        if missing_keys:
            logger.warning(
                f"Missing keys (not loaded): {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}"
            )
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys (ignored): {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        # 清理内存
        del checkpoint, original_state_dict
        torch.cuda.empty_cache()


class HFModelCheckpoint(Callback):

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        subdir: str = "hf_model",
    ):
        """
        Args:
            monitor: 要监控的 metric 名称 (比如 'val_loss' 或 'val_acc')
            mode: 'min' 表示越小越好, 'max' 表示越大越好
            subdir: 保存 HuggingFace 模型与 tokenizer 的子目录
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.subdir = subdir
        self.best_score = None

    def _is_better(self, current):
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current < self.best_score
        elif self.mode == "max":
            return current > self.best_score
        else:
            raise ValueError(f"Invalid mode {self.mode}, must be 'min' or 'max'")

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            print(f"[SaveHFOnlyCallback] Metric {self.monitor} not found in metrics")
            return

        current_score = metrics[self.monitor].item()
        if self._is_better(current_score):
            self.best_score = current_score
            log_dir = (
                trainer.logger.log_dir if trainer.logger else trainer.default_root_dir
            )
            save_dir = os.path.join(log_dir, self.subdir)
            os.makedirs(save_dir, exist_ok=True)

            # 保存 HF 模型
            pl_module.model.save_pretrained(save_dir)
            print(
                f"[SaveHFOnlyCallback] Best {self.monitor}={current_score:.4f} -> saved model at {save_dir}"
            )

            # 保存 tokenizer（如果有）
            if hasattr(pl_module, "tokenizer") and pl_module.tokenizer is not None:
                pl_module.tokenizer.save_pretrained(save_dir)
                print(f"[SaveHFOnlyCallback] Tokenizer saved -> {save_dir}")


def check_data_module(
    data_module: L.LightningDataModule,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    stage: str = "fit",
    exit_after_check: bool = True,
):
    import torch

    data_module.setup(stage)

    # 打印一个 batch 的结构和样本
    dataloader = (
        data_module.train_dataloader()
        if stage == "fit"
        else data_module.predict_dataloader()
    )
    for batch in dataloader:
        logger.info("🔎 Checking a sample batch...")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                logger.debug(f"{k:<15} shape={tuple(v.shape)} dtype={v.dtype}")
                logger.debug(
                    f"{k:<15} sample={v[0].tolist() if v.ndim > 1 else v[0].item()}"
                )
                if k == "input_ids":
                    logger.debug(f"{k:<15} sample={tokenizer.decode(v[0])}")
            elif isinstance(v, (list, tuple)):
                logger.debug(f"{k:<15} len={len(v)} type={type(v[0])}")
                logger.debug(f"{k:<15} sample={v[0]}")
            else:
                logger.debug(f"{k:<15} value={v}")
        break

    # 打印 tokenizer 相关信息
    if tokenizer is not None:
        logger.info("🔎 Checking tokenizer properties...")

        attributes = [
            "all_token_ids",
            "special_token_ids",
            "special_tokens_map",
            "all_special_tokens",
            "all_special_ids",
            "vocab_size",
        ]

        for attr in attributes:
            value = getattr(tokenizer, attr, None)
            if value is not None:
                logger.debug(f"{attr:<20} = {value}")
            else:
                logger.debug(f"{attr:<20} not found")

    if exit_after_check:
        import sys

        logger.info("✅ Data check finished. Exiting...")
        sys.exit(0)


def check_graph_data_module(
    data_module: L.LightningDataModule,
    stage: str = "fit",
    exit_after_check: bool = True,
):
    import torch

    data_module.setup(stage)

    # 打印一个 batch 的结构和样本
    dataloader = (
        data_module.train_dataloader()
        if stage == "fit"
        else data_module.predict_dataloader()
    )
    for batch in dataloader:
        logger.info("🔎 Checking a sample batch...")
        logger.info(batch)
        break

    if exit_after_check:
        import sys

        logger.info("✅ Data check finished. Exiting...")
        sys.exit(0)

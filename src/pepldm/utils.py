import types
import pandas as pd

from abc import abstractmethod
from datasets import Dataset
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss module for classification tasks with class imbalance.

    This implementation supports three explicit task types:
    - "binary": Binary classification using sigmoid + BCE focal loss.
    - "multiclass": Single-label multi-class classification using softmax + CE focal loss.
    - "multilabel": Multi-label classification using sigmoid + BCE focal loss.

    Args:
        num_classes (int, optional):
            Number of classes. Required for "multiclass" and "multilabel".
        task (str):
            One of {"binary", "multiclass", "multilabel"}.
        alpha (float, optional):
            Balancing factor for positive vs. negative samples. Default is 0.25.
        gamma (float, optional):
            Focusing parameter that down-weights easy examples. Default is 2.
        reduction (str, optional):
            One of {"none", "mean", "sum"}. Specifies the reduction applied to the output.
            Default is "mean".
        ignore_index (int, optional):
            Specifies a target value that is ignored and not included in the loss.
            Default is -1.

    Shape:
        - logits: (N, C) or (N,) for binary classification
        - target:
            * (N,) for "binary" or "multiclass"
            * (N, C) for "multilabel" (multi-hot format)

    Returns:
        torch.Tensor:
            The computed focal loss. Shape depends on ``reduction``:
            - "none" → same shape as input loss
            - "mean" or "sum" → scalar tensor
    """

    def __init__(
        self,
        num_classes=None,
        task="multiclass",
        alpha=0.25,
        gamma=2,
        reduction="mean",
        ignore_index=-1,
    ):
        super().__init__()
        self.task = task
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.num_classes = num_classes

        if task == "multiclass" and (num_classes is None or num_classes < 2):
            raise ValueError("num_classes must be >=2 for multiclass")
        if task == "binary":
            self.sigmoid = True
        elif task == "multilabel":
            self.sigmoid = True
        else:
            self.sigmoid = False

    def forward(self, logits, target):
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            if not valid.any():
                return torch.tensor(0.0, device=logits.device)
            logits = logits[valid]
            target = target[valid]

        if self.task == "binary":
            logits = logits.view(-1)
            target = target.float().view(-1)
            prob = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            pt = prob * target + (1 - prob) * (1 - target)
            loss = ce * ((1 - pt) ** self.gamma)
            if self.alpha is not None:
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                loss = alpha_t * loss

        elif self.task == "multilabel":
            prob = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(
                logits, target.float(), reduction="none"
            )
            pt = prob * target + (1 - prob) * (1 - target)
            loss = ce * ((1 - pt) ** self.gamma)
            if self.alpha is not None:
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                loss = alpha_t * loss

        else:  # multiclass
            ce = F.cross_entropy(logits, target, reduction="none")
            prob = torch.softmax(logits, dim=1)
            pt = prob[torch.arange(len(target)), target]
            loss = ce * ((1 - pt) ** self.gamma)
            if self.alpha is not None:
                loss = loss * self.alpha

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class Pipeline:
    def __init__(
        self,
        model,
        tokenizer=None,
        feature_extractor=None,
        processor=None,
        device="cuda:0",
        batch_size=1,
        preprocess_params={},
        forward_params={},
        postprocess_params={},
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.processor = processor
        self.device = device
        self.batch_size = batch_size

        self.model.to(self.device)

        self._preprocess_params = preprocess_params
        self._forward_params = forward_params
        self._postprocess_params = postprocess_params

        self.kwargs = kwargs

    def __call__(self, inputs):

        preprocess_params, forward_params, postprocess_params = (
            self._sanitize_parameters(**self.kwargs)
        )

        # Fuse __init__ params and __call__ params without modifying the __init__ ones.
        preprocess_params = {**self._preprocess_params, **preprocess_params}
        forward_params = {**self._forward_params, **forward_params}
        postprocess_params = {**self._postprocess_params, **postprocess_params}

        is_dataset = Dataset is not None and isinstance(inputs, Dataset)
        is_generator = isinstance(inputs, types.GeneratorType)
        is_list = isinstance(inputs, list)

        if is_dataset or is_generator or is_list:
            return self.run_multi(
                inputs, preprocess_params, forward_params, postprocess_params
            )
        else:
            return self.run_single(
                inputs, preprocess_params, forward_params, postprocess_params
            )

    @abstractmethod
    def preprocess(self, input_, **preprocess_parameters):
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def _forward(self, input_tensors, **forward_parameters):
        raise NotImplementedError("_forward not implemented")

    @abstractmethod
    def postprocess(self, model_outputs, **postprocess_parameters):
        raise NotImplementedError("postprocess not implemented")

    @abstractmethod
    def _sanitize_parameters(self, **kwargs):
        raise NotImplementedError

    def run_multi(self, inputs, preprocess_params, forward_params, postprocess_params):
        return [
            self.run_single(item, preprocess_params, forward_params, postprocess_params)
            for item in inputs
        ]

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self._forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def iterate(self, inputs, preprocess_params, forward_params, postprocess_params):
        for input_ in inputs:
            yield self.run_single(
                input_, preprocess_params, forward_params, postprocess_params
            )

class PepLDMPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {"num_samples": kwargs.get("num_samples", 100)}
        forward_kwargs = {
            "num_steps": kwargs.get("num_steps", 1000),
            "min_seqs": kwargs.get("min_seqs", 1),
            "max_tries": kwargs.get("max_tries", 20),
        }
        postprocess_kwargs = {}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, target_seqs: str, num_samples: int = 100, **kwargs):
        logger.info(f"Preprocessing: {target_seqs}")
        if isinstance(target_seqs, str):
            target_seqs = [target_seqs] * num_samples

        prot_tokenized = self.tokenizer(
            target_seqs,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        return {
            "prot_input_ids": prot_tokenized["input_ids"].to(self.device),
            "prot_attention_mask": prot_tokenized["attention_mask"].to(self.device),
        }

    def _forward(self, model_inputs, num_steps=1000, **kwargs):
        min_seqs = kwargs.get("min_seqs", 1)
        max_tries = kwargs.get("max_tries", 20)
        outputs = []
        i_try = 0
        seen_seqs = set()
        while len(seen_seqs) < min_seqs:
            i_try += 1
            if i_try >= max_tries:
                break
            logger.debug(f"Try {i_try} of {max_tries}")
            output = self.model.generate(
                **model_inputs,
                num_steps=num_steps,  # 采样步数
                cond_scale=1.0,  # guidance 强度
            )
            df_output = pd.DataFrame(output)
            len_original = len(df_output)

            len_current = len(df_output)

            seen_seqs.update(df_output["seq_gen"].tolist())
            logger.debug(
                f"Original: {len_original}, Current: {len_current}, Accumulated Unique: {len(seen_seqs)}"
            )
            if len_current == 0:
                continue
            outputs.append(df_output)

        return outputs

    def postprocess(self, model_outputs, **kwargs):
        return model_outputs
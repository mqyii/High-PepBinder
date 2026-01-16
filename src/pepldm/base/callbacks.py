import torch
import pandas as pd
import lightning as L
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger


class CSVWriterCallback(L.Callback):
    def __init__(
        self,
        save_dir: str = "./outputs",
        filename: str = "predictions.csv",
        output_keys: List[str] | None = None,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.output_keys = output_keys
        self._records: List[Dict[str, Any]] = []

    def _collect_batch_outputs(self, outputs, batch):
        # Lightning 有时会把 outputs 包一层 list
        if outputs is None:
            return
        if isinstance(outputs, list):
            if len(outputs) == 0 or outputs[0] is None:
                return
            outputs = outputs[0]

        # 必须有 probs（这是你伪标签真正要用的）
        if "probs" not in outputs:
            logger.warning("Missing `probs` in predict outputs, skip batch.")
            return

        probs = outputs["probs"]
        logits = outputs.get("logits", None)

        if probs is None:
            return

        probs = probs.detach().cpu().tolist()
        if logits is not None:
            logits = logits.detach().cpu().tolist()

        batch_size = len(probs)

        # 不记录模型输入
        ignore_keys = {"input_ids", "attention_mask"}
        other_keys = [k for k in batch.keys() if k not in ignore_keys]

        for i in range(batch_size):
            rec = {}

            for k in other_keys:
                v = batch[k][i]
                if torch.is_tensor(v):
                    v = v.item() if v.numel() == 1 else v.detach().cpu().numpy()
                rec[k] = v

            rec["prob"] = probs[i]
            if logits is not None:
                rec["logit"] = logits[i]

            self._records.append(rec)

    def _write_to_csv(self):
        if not self._records:
            logger.warning("No prediction records to write.")
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / self.filename

        df = pd.DataFrame(self._records)
        if path.suffix == ".csv":
            df.to_csv(path, index=False)
        elif path.suffix == ".parquet":
            df.to_parquet(path)
        elif path.suffix == ".pkl":
            df.to_pickle(path)
        else:
            raise ValueError(f"Unsupported file extension: {path}")

        logger.success(f"Saved {len(df)} rows to {path}")

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._collect_batch_outputs(outputs, batch)

    def on_predict_epoch_end(self, trainer, pl_module):
        self._write_to_csv()
        self._records.clear()

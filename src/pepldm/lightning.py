from typing import Optional
import torch
from torch.nn import functional as F
from torchmetrics import (
    MetricCollection,
    MeanMetric,
    Accuracy,
    AUROC,
    BinaryAveragePrecision,
    F1Score,
)
from loguru import logger
from .base import BaseLightningModule
from .esmc import ESMCModel, ESMCConfig


class DiffusionGenerationMixin:

    @torch.inference_mode()
    def generate(
        self,
        prot_input_ids: torch.Tensor,
        prot_attention_mask: torch.Tensor,
        num_steps: int = 1000,
        cond_scale: float = 1.0,
        max_length: int = 40,
    ):

        device = prot_input_ids.device

        # 1. 编码蛋白质 embedding
        prot_embeds = self.prot_encoder(
            prot_input_ids, prot_attention_mask
        ).last_hidden_state
        prot_embeds_pooled = self.prot_pooler(prot_embeds, prot_attention_mask)

        # 2. 初始化随机噪声
        B = prot_embeds_pooled.size(0)
        x_t = torch.randn(B, max_length, self.model.config.input_dim, device=device)

        # 3. 从 T -> 0 逐步反推
        T = self.model.betas.size(0)
        timesteps = torch.linspace(T - 1, 0, num_steps, device=device).long()

        for t in tqdm(timesteps):
            t_batch = torch.full((B,), t, device=device).long()

            # 预测噪声 / x0
            out = self.model(
                inputs=x_t,
                t=t_batch,
                cond=prot_embeds_pooled,
                use_cond_dropout=False,
                return_loss=False,
            )
            noise_pred = out["logits"]

            # 还原 x0
            x0_pred = self.model.predict_start_from_noise(x_t, t_batch, noise_pred)

            # classifier-free guidance
            if cond_scale != 1.0:
                uncond_out = self.model(
                    inputs=x_t,
                    t=t_batch,
                    cond=None,
                    use_cond_dropout=False,
                    return_loss=False,
                )
                noise_uncond = uncond_out["logits"]
                noise_pred = noise_uncond + (noise_pred - noise_uncond) * cond_scale
                x0_pred = self.model.predict_start_from_noise(x_t, t_batch, noise_pred)

            # 计算 posterior
            mean, var, log_var = self.model.q_posterior(x0_pred, x_t, t_batch)

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.exp(0.5 * log_var) * noise
            else:
                x_t = mean

        # 4. 解码到氨基酸序列
        logits = self.pep_decoder(x_t)  # [B, L, vocab_size]
        pred_ids = torch.argmax(logits, dim=-1)

        filtered = self.filter_and_pad_sequences(
            pred_ids,
        )

        # ✅ 若过滤结果为空，直接输出空结果
        if (
            "input_ids" not in filtered
            or filtered["input_ids"].numel() == 0
            or filtered["input_ids"].size(1) == 0
        ):
            logger.warning(
                "All peptide sequences were filtered out, returning empty output."
            )
            # 返回与 format_outputs 结构一致的空格式
            return []
        else:
            logger.debug(f"Filtered input_ids: {filtered['input_ids']}")
            logger.debug(f"Filtered attention_mask: {filtered['attention_mask']}")

        pep_embeds = self.pep_encoder(
            filtered["input_ids"], filtered["attention_mask"]
        ).last_hidden_state
        # pep_embeds_pooled = self.pep_pooler(pep_embeds)

        # 对齐 prot 的 batch size
        prot_embeds = prot_embeds[:1].expand(pep_embeds.size(0), -1, -1)
        prot_attention_mask = prot_attention_mask[:1].expand(pep_embeds.size(0), -1)

        affinity_probs = (
            torch.sigmoid(
                self.affinity_head(
                    prot_embeds,
                    pep_embeds,
                    prot_attention_mask,
                    filtered["attention_mask"],
                )
            )
            .detach()
            .squeeze(-1)
            .cpu()
            .numpy()
        )

        return self.format_outputs(filtered["input_ids"], affinity_probs)

    def format_outputs(self, pred_ids, affinity_probs):
        # 把 token ids 转成字符串序列
        pred_ids = pred_ids[:, 1:]  # drop cls token
        seqs = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # 返回 list[dict] 的形式，每个样本一个字典s
        results = []
        for seq, aff in zip(seqs, affinity_probs):
            results.append(
                {
                    "seq_gen": seq.replace(" ", ""),
                    "affinity_pred": aff,
                }
            )

        return results

    def filter_and_pad_sequences(self, pred_ids):
        b, l = pred_ids.shape
        mask = torch.ones(b, dtype=torch.bool, device=pred_ids.device)

        filtered_seqs = []
        MIN_LEN, MAX_LEN = 5 + 2, 40 + 2

        n_success = 0
        n_failed_cls = 0
        n_failed_eos = 0
        n_failed_middle = 0
        n_failed_length = 0
        for i in range(b):
            seq = pred_ids[i]

            # 1. 必须以 cls_id 开头
            if seq[0].item() != 0:
                # mask[i] = False
                # continue
                n_failed_cls += 1

            # 2. 找到第一个 eos
            eos_pos = (seq == 2).nonzero(as_tuple=True)[0]
            if len(eos_pos) == 0:
                mask[i] = False
                n_failed_eos += 1
                continue

            eos_pos = eos_pos[0].item()
            # 截断序列（包含 eos）
            subseq = seq[: eos_pos + 1]

            # 3. 检查中间是否都合法
            middle = subseq[1:-1]  # 去掉 cls_id 和 eos
            if not all(token.item() in VALID_IDS for token in middle):
                mask[i] = False
                n_failed_middle += 1
                continue

            # 4. 长度约束
            if subseq.size(0) < MIN_LEN or subseq.size(0) > MAX_LEN:
                mask[i] = False
                n_failed_length += 1
                continue

            # 通过检查的序列
            filtered_seqs.append(subseq)
            n_success += 1

        logger.info(f"Filtered {n_success} sequences out of {b}.")
        logger.info(
            f"Failed due to cls: {n_failed_cls}, eos: {n_failed_eos}, middle: {n_failed_middle}, length: {n_failed_length}"
        )

        # 没有通过的，直接返回空
        if len(filtered_seqs) == 0:
            return {
                "input_ids": torch.empty(
                    0, 0, dtype=torch.long, device=pred_ids.device
                ),
                "attention_mask": torch.empty(
                    0, 0, dtype=torch.long, device=pred_ids.device
                ),
            }

        # 5. 找到最大长度并 pad（限制在 MAX_LEN）
        max_len = min(max(seq.size(0) for seq in filtered_seqs), MAX_LEN)
        input_ids = torch.full(
            (len(filtered_seqs), max_len),
            1,  # pad_id，最好换成 self.tokenizer.pad_id
            dtype=torch.long,
            device=pred_ids.device,
        )
        attention_mask = torch.zeros(
            (len(filtered_seqs), max_len),
            dtype=torch.long,
            device=pred_ids.device,
        )

        for i, seq in enumerate(filtered_seqs):
            trunc_len = min(seq.size(0), max_len)
            input_ids[i, :trunc_len] = seq[:trunc_len]
            attention_mask[i, :trunc_len] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class PepLDM2(BaseLightningModule, DiffusionGenerationMixin):
    def __init__(
        self,
        model_args: dict = {},
        esm_args: dict = {},
        tokenizer: Optional[ESMSeqTokenizer] = None,
        load_from_esmc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model_args = model_args
        self.esm_args = esm_args

        self.prot_encoder = ESMCModel(ESMCConfig(**esm_args))
        self.prot_pooler = Pooler(pooling_types=["cls", "mean"])
        self.hidden_size = ESMCConfig(**esm_args).hidden_size
        self.pep_encoder = ESMCModel(ESMCConfig(**esm_args))
        self.pep_decoder = RegressionHead(self.hidden_size, self.pep_encoder.vocab_size)
        self.affinity_head = CrossAttentionHead(ESMCConfig(**esm_args).hidden_size)
        self.pep_pooler = Pooler(pooling_types=["cls", "mean"])
        self.prot_encoder.to(self.device)
        self.pep_encoder.to(self.device)
        # self.prot_encoder.eval()

        for param in self.prot_encoder.parameters():
            param.requires_grad = False

        model_config = PepLDMConfig(**self.model_args)
        self.model: PepLDMModel = PepLDMModel(config=model_config)

        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.ignore_index = (
            self.tokenizer.pad_token_id if self.tokenizer is not None else -100
        )
        logger.info(f"Ignore index: {self.ignore_index}")

        if load_from_esmc:
            load_weights_from_esm(self.prot_encoder)
            load_weights_from_esm(self.pep_encoder)

        # for param in self.pep_encoder.parameters():
        #     param.requires_grad = True
        # for param in self.pep_decoder.parameters():
        #     param.requires_grad = True
        # for param in self.model.parameters():
        #     param.requires_grad = True
        # for param in self.affinity_head.parameters():
        #     param.requires_grad = True

        self._init_metrics()
        self.criterion = FocalLoss(task="binary")

    def _init_metrics(self):
        self.metrics: MetricCollection = MetricCollection.from_metrics(
            {
                "train_loss": MeanMetric(sync_on_compute=True),
                "val_loss": MeanMetric(sync_on_compute=True),
                "val_ce_loss": MeanMetric(sync_on_compute=True),
                "val_mse_loss": MeanMetric(sync_on_compute=True),
                "val_affinity_loss": MeanMetric(sync_on_compute=True),
                "val_acc": Accuracy(sync_on_compute=True, task="binary"),
                "val_auc": AUROC(sync_on_compute=True, task="binary"),
                "val_prauc": BinaryAveragePrecision(sync_on_compute=True),
                "val_f1": F1Score(sync_on_compute=True, task="binary"),
            }
        )

    def _load_weights_from_lightning(
        self, gen_net: str, aff_net: str, map_location: str = "cpu"
    ):
        gen_checkpoint = torch.load(
            gen_net, map_location=map_location, weights_only=False
        )
        aff_checkpoint = torch.load(
            aff_net, map_location=map_location, weights_only=False
        )

        if "state_dict" not in gen_checkpoint:
            raise ValueError(f"No state_dict found in checkpoint {gen_net}")
        if "state_dict" not in aff_checkpoint:
            raise ValueError(f"No state_dict found in checkpoint {aff_net}")

        gen_state_dict = gen_checkpoint["state_dict"]
        aff_state_dict = aff_checkpoint["state_dict"]

        aff_state_dict_filtered = {
            k: v for k, v in aff_state_dict.items() if "affinity_head" in k
        }

        self.load_state_dict(gen_state_dict, strict=False)
        self.load_state_dict(aff_state_dict_filtered, strict=False)

        # 统计信息
        total_params = len(gen_state_dict)
        loaded_params = len(gen_state_dict) + len(aff_state_dict_filtered)
        ratio = loaded_params / total_params * 100

        logger.info(
            f"Loaded {loaded_params}/{total_params} ({ratio:.2f}%) weights "
            f"from checkpoint: {gen_net} and {aff_net}"
        )

        # 清理内存
        del gen_checkpoint, aff_checkpoint
        torch.cuda.empty_cache()

    def forward(
        self,
        prot_input_ids,
        prot_attention_mask,
        pep_input_ids,
        pep_attention_mask,
        affinity,
    ):

        prot_embeds = self.prot_encoder(
            prot_input_ids, prot_attention_mask
        ).last_hidden_state
        prot_embeds_pooled = self.prot_pooler(prot_embeds, prot_attention_mask)

        pep_embeds = self.pep_encoder(
            pep_input_ids, pep_attention_mask
        ).last_hidden_state
        # pep_embeds_pooled_ori = self.pep_pooler(pep_embeds, pep_attention_mask)

        affinity_logits_ori = self.affinity_head(
            prot_embeds, pep_embeds, prot_attention_mask, pep_attention_mask
        )
        affinity_loss_ori = self.criterion(affinity_logits_ori, affinity)

        B = pep_embeds.size(0)
        T = self.model.betas.size(0)
        t = torch.randint(0, T, (B,), device=pep_embeds.device)

        noise = torch.randn_like(pep_embeds)
        x_t = self.model.q_sample(pep_embeds, t, noise)

        x_self_cond = None
        if self.model.config.use_self_conditioning and torch.rand(1) < 0.5:
            with torch.no_grad():
                x_self_cond = self.model(
                    inputs=x_t,
                    t=t,
                    cond=prot_embeds_pooled,
                    x_self_cond=x_self_cond,
                    return_loss=False,
                )["logits"].detach()

        if self.model.objective == "epsilon":
            labels = noise
        elif self.model.objective == "sample":
            labels = pep_embeds
        elif self.model.objective == "v_prediction":
            labels = self.model.predict_v(pep_embeds, t, noise)
        else:
            raise ValueError(f"Unknown objective {self.model.objective}")

        diff_output = self.model(
            inputs=x_t,
            labels=labels,
            t=t,
            cond=prot_embeds_pooled,
            x_self_cond=x_self_cond,
            cfg_dropout_prob=0.1,  # use 10% dropout for conditional dropout
        )
        mse_loss = diff_output["loss"]
        noise_pred = diff_output["logits"]

        x0_pred = self.model.predict_start_from_noise(x_t, t, noise_pred)
        logits_pred = self.pep_decoder(x0_pred)

        ce_loss_pred = F.cross_entropy(
            logits_pred.view(-1, logits_pred.size(-1)),
            pep_input_ids.view(-1),
            ignore_index=self.ignore_index,
        )

        logits_ori = self.pep_decoder(pep_embeds)

        ce_loss_orig = F.cross_entropy(
            logits_ori.view(-1, logits_ori.size(-1)),
            pep_input_ids.view(-1),
            ignore_index=self.ignore_index,
        )
        ce_loss = ce_loss_pred + ce_loss_orig

        # pep_embeds_pooled_pred = self.pep_pooler(x0_pred, pep_attention_mask)
        # affinity_logits_pred = self.affinity_head(
        #     torch.cat([prot_embeds_pooled, pep_embeds_pooled_pred], dim=-1)
        # )
        affinity_logits_pred = self.affinity_head(
            prot_embeds, x0_pred, prot_attention_mask, pep_attention_mask
        )

        affinity_loss_pred = self.criterion(affinity_logits_pred, affinity)

        affinity_loss = affinity_loss_pred + affinity_loss_ori

        loss = mse_loss + ce_loss + affinity_loss

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "ce_loss": ce_loss,
            "affinity_loss": affinity_loss,
            "preds": affinity_logits_ori,
            "labels": affinity,
        }

    def training_step(self, batch, batch_idx):
        outputs = self(
            prot_input_ids=batch["prot_input_ids"],
            prot_attention_mask=batch["prot_attention_mask"],
            pep_input_ids=batch["pep_input_ids"],
            pep_attention_mask=batch["pep_attention_mask"],
            affinity=batch["affinity"],
        )
        self.log_step(
            "train_loss_step",
            outputs["loss"],
        )
        self.metrics.update("train_loss", outputs["loss"])
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self(
            prot_input_ids=batch["prot_input_ids"],
            prot_attention_mask=batch["prot_attention_mask"],
            pep_input_ids=batch["pep_input_ids"],
            pep_attention_mask=batch["pep_attention_mask"],
            affinity=batch["affinity"],
        )

        self.metrics.update(
            "val_acc", outputs["preds"].detach(), outputs["labels"].detach()
        )
        self.metrics.update(
            "val_auc", outputs["preds"].detach(), outputs["labels"].detach()
        )
        self.metrics.update(
            "val_prauc", outputs["preds"].detach(), outputs["labels"].detach()
        )
        self.metrics.update(
            "val_f1", outputs["preds"].detach(), outputs["labels"].detach()
        )
        self.metrics.update("val_loss", outputs["loss"].detach())
        self.metrics.update("val_ce_loss", outputs["ce_loss"].detach())
        self.metrics.update("val_mse_loss", outputs["mse_loss"].detach())
        self.metrics.update("val_affinity_loss", outputs["affinity_loss"].detach())

    def on_train_epoch_end(self):
        self.log_epoch(
            "train_loss",
            self.metrics.compute("train_loss"),
        )
        self.metrics.reset_all(prefix="train")

    def on_validation_epoch_end(self):
        self.log_epoch(
            "val_loss",
            self.metrics.compute("val_loss"),
        )
        self.log_epoch(
            "val_ce_loss",
            self.metrics.compute("val_ce_loss"),
        )
        self.log_epoch(
            "val_mse_loss",
            self.metrics.compute("val_mse_loss"),
        )
        self.log_epoch(
            "val_affinity_loss",
            self.metrics.compute("val_affinity_loss"),
        )
        self.log_epoch(
            "val_acc",
            self.metrics.compute("val_acc"),
        )
        self.log_epoch(
            "val_auc",
            self.metrics.compute("val_auc"),
        )
        self.log_epoch(
            "val_prauc",
            self.metrics.compute("val_prauc"),
        )
        self.log_epoch(
            "val_f1",
            self.metrics.compute("val_f1"),
        )
        self.metrics.reset_all(prefix="val")

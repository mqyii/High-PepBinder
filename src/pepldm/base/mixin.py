import torch
from abc import ABC, abstractmethod
from typing import Any, List
import torch.nn.functional as F

from astra.models.pepclip.configuration_pepclip import PepCLIPConfig
from astra.models.pepclip.modeling_pepclip import ESMCForCLIP
from astra.tokenizers.esm import ESMSeqTokenizer


class GenMixin(ABC):
    @abstractmethod
    def forward(self, inputs: Any, **kwargs) -> Any:
        """模型前向计算，子类必须实现"""
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """把 token id 序列转成字符串，子类必须实现"""
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """把文本转成 token id 序列，子类必须实现"""
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_length: int = 20,
        do_sample: bool = True,
        top_k: int = 50,
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        """核心生成逻辑"""
        input_ids = self.encode(prompt)
        generated = list(input_ids)

        for _ in range(max_length):
            logits = self.forward(generated, **kwargs)

            # 取最后一个 token 的分布
            next_token_logits = logits[-1]

            if do_sample:
                # Top-k 采样
                next_token = self._sample_top_k(
                    next_token_logits, top_k=top_k, temperature=temperature
                )
            else:
                # 贪心解码
                next_token = int(next_token_logits.argmax())

            generated.append(next_token)

            # 停止条件：假设 token=0 是 <eos>
            if next_token == 0:
                break

        return self.decode(generated)

    def _sample_top_k(self, logits, top_k=50, temperature=1.0):
        logits = torch.tensor(logits) / temperature
        values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
        probs = F.softmax(values, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        return indices[idx].item()


class CLIPMixin:
    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        candidates: list[str],
        topk: int = 10,
        retrieval_direction: str = "pep",  # "pep" = prot->pep, "prot" = pep->prot
        batch_size: int = 64,
    ) -> list[tuple[str, float]]:
        assert retrieval_direction in ["pep", "prot"], "Invalid retrieval direction"

        queries = [query] * len(candidates)
        if retrieval_direction == "prot":
            queries, candidates = candidates, queries

        sims = self.cal_similarity(
            prot_seq=queries,
            pep_seq=candidates,
            retrieval_direction=retrieval_direction,
            batch_size=batch_size,
        )

        sims = sims[0]  # (N,)
        topk_vals, topk_indices = sims.topk(k=topk)
        return [
            (candidates[i], topk_vals[j].item()) for j, i in enumerate(topk_indices)
        ]

    @classmethod
    def from_huggingface(cls, model_dir: str, device: str = "cuda"):
        model_config = PepCLIPConfig.from_pretrained(model_dir)
        tokenizer = ESMSeqTokenizer()
        hf_model = ESMCForCLIP.from_pretrained(model_dir)
        model = cls(
            model_args=model_config.to_dict(),
            tokenizer=tokenizer,
            compile_model=False,
        )
        model.model = hf_model.to(device).eval()
        model.tokenizer = tokenizer
        return model

    @classmethod
    def from_lightning(
        cls, checkpoint_path: str, device: str = "cuda", strict: bool = True
    ):
        tokenizer = ESMSeqTokenizer()
        model = cls.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            strict=strict,
            compile_model=False,
            load_from_esmc=False,
            tokenizer=tokenizer,
        )
        model.to(device).eval()
        model.tokenizer = tokenizer
        return model

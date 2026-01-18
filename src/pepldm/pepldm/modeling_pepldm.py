from dataclasses import dataclass
import torch
import math
from typing import Optional, Any, Tuple
from einops import rearrange, repeat
import collections
import itertools
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from functools import partial
from transformers.modeling_utils import PreTrainedModel
from torch.nn.functional import mse_loss

xformers_installed = True
try:
    from xformers.ops import memory_efficient_attention
    from xformers.components.attention import ScaledDotProduct
except ImportError:
    xformers_installed = False

flash_installed = True
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    flash_installed = False

from .configuration_pepldm import PepLDMConfig


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(itertools.repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class InputProj(nn.Module):
    def __init__(self, input_dim, hidden_size, bias=True):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_size, bias=bias)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        return self.norm(self.proj(x))


class SinusoidalTimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class GaussianFourierProjection(nn.Module):
    """
    https://arxiv.org/abs/2006.10739
    https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
    """

    def __init__(self, embed_dim: int, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        w = torch.randn(embed_dim // 2) * scale
        self.register_buffer("W", w)

    def forward(self, t: torch.Tensor):
        # t: (batch_size,)
        # w: (embed_dim // 2,)
        t = t.to(self.W.dtype)
        t_proj = 2.0 * torch.pi * t[:, None] @ self.W[None, :]
        embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        return embed


class FourierTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.fourier_projection = GaussianFourierProjection(
            embed_dim=frequency_embedding_size
        )
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def timestep_embedding(self, t, *args, **kwargs):
        return self.fourier_projection(t)

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class SinePositionalEmbedding(nn.Module):
    # https://github.com/facebookresearch/xformers/blob/main/xformers/components/positional_embedding/sine.py#L21
    def __init__(self, dim_model: int, *args, **kwargs):
        super().__init__()
        self.dim_model = dim_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        pos = (
            torch.arange(0, seq_len, device=x.device, dtype=torch.float32)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(0, self.dim_model, device=x.device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / self.dim_model))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        output = x.unsqueeze(-1) if x.ndim == 2 else x

        return output + pos.unsqueeze(0)


@dataclass
class DenoiserKwargs:
    x_t: torch.Tensor
    t: torch.Tensor
    cond: Optional[torch.Tensor] = None
    x_self_cond: Optional[torch.Tensor] = None


class BaseDenoiser(nn.Module):
    def __init__(
        self,
        input_dim=32,
        hidden_size=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        use_self_conditioning=True,
        timestep_embedding_strategy: str = "fourier",
        conditional=False,
        cond_dim=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_self_conditioning = use_self_conditioning
        self.conditional = conditional

        # cast input dimension to hidden dimension
        self.x_proj = InputProj(
            input_dim=self.input_dim, hidden_size=self.hidden_size, bias=True
        )

        self.final_layer = FinalLayer(
            hidden_size=self.hidden_size, out_channels=self.input_dim
        )

        # self conditioning
        if use_self_conditioning:
            self.self_conditioning_mlp = Mlp(
                in_features=self.input_dim * 2,
                out_features=self.input_dim,
                norm_layer=nn.LayerNorm,
            )
        else:
            self.self_conditioning_mlp = None
        assert timestep_embedding_strategy in ["fourier", "sinusoidal", "default", None]
        if timestep_embedding_strategy == "fourier":
            self.t_embedder = FourierTimestepEmbedder(
                hidden_size=hidden_size, frequency_embedding_size=256
            )
        else:
            # default: sinusoidal transform
            self.t_embedder = SinusoidalTimestepEmbedder(
                hidden_size=hidden_size, frequency_embedding_size=256
            )

        self.pos_embed = SinePositionalEmbedding(hidden_size)

        # cond embedder
        self.uncond_embed = nn.Parameter(torch.zeros(hidden_size))
        if conditional:
            self.cond_proj = torch.nn.Linear(cond_dim, hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize uncond embedding:
        nn.init.normal_(self.uncond_embed, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def make_cfg_embedding(
        self, cond: Optional[torch.Tensor], cfg_dropout_prob: float, B
    ):
        # B = cond.shape[0] if cond is not None else None
        device = self.uncond_embed.device
        dtype = self.uncond_embed.dtype

        if not self.conditional or cond is None:  # unconditional
            return self.uncond_embed.unsqueeze(0).expand(B, -1)

        cond = cond.squeeze(-1).to(dtype=self.cond_proj.weight.dtype)
        cond_emb = self.cond_proj(cond)

        keep_mask = torch.rand(B, 1, device=device) > cfg_dropout_prob
        uncond = self.uncond_embed.unsqueeze(0).expand(B, -1)  # (B, hidden_dim)
        cond_emb = cond_emb * keep_mask.to(dtype) + uncond * (~keep_mask).to(dtype)

        return cond_emb

    def forward_with_cond_drop(
        self, denoiser_kwargs: DenoiserKwargs, cfg_dropout_prob: float, mask=None
    ):
        """Forward pass for diffusion training, with label dropout."""

        x = denoiser_kwargs.x_t
        t = denoiser_kwargs.t
        cond = denoiser_kwargs.cond  # None or [B, cond_dim]
        x_self_cond = denoiser_kwargs.x_self_cond

        if x_self_cond is not None:
            x = self.self_conditioning_mlp(torch.cat([x, x_self_cond], dim=-1))

        x = self.x_proj(x)
        x = self.pos_embed(x)
        t = self.t_embedder(t)  # (N, D)

        B = x.shape[0]
        cond_emb = self.make_cfg_embedding(cond, cfg_dropout_prob, B)

        c = t + cond_emb  # [B, hidden_dim]

        # must be defined in subclasses!
        # mask = torch.ones(x.shape[:2], device=x.device).bool()
        x = self.blocks_forward_pass(x, c, mask)

        return self.final_layer(x, c)  # (N, L, out_channels)

    def forward_with_cond_scale(
        self,
        denoiser_kwargs: DenoiserKwargs,
        cond_scale: float = 1.0,
        rescaled_phi: float = 0.0,
    ):
        """Forward pass for sampling model predictions, with a conditioning scale.
        Adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py#L355
        """

        # force conditioning: no label drop
        logits = self.forward_with_cond_drop(denoiser_kwargs, 0.0)

        if cond_scale == 1.0:
            return logits

        # force unconditional: always no label drop
        null_logits = self.forward_with_cond_drop(denoiser_kwargs, 1.0)

        # apply cond scaling factor
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        # use rescaling technique proposed in https://arxiv.org/abs/2305.08891
        if rescaled_phi == 0.0:
            return scaled_logits

        std_fn = partial(
            torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True
        )
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)

    def forward(
        self,
        denoiser_kwargs: DenoiserKwargs,
        use_cond_dropout: bool = False,
        **kwargs: Any,
    ):
        if use_cond_dropout:
            return self.forward_with_cond_drop(denoiser_kwargs, **kwargs)
        else:
            return self.forward_with_cond_scale(denoiser_kwargs, **kwargs)


def exists(val):
    return val is not None


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        qkv_bias=False,
        dropout=0.0,
        attention_mode="standard",
    ):
        super().__init__()
        assert attention_mode in [
            "standard",
            "xformers_scaled_dot_product",
            "xformers_memory_efficient",
            "flash",
        ]
        self.attention_mode = attention_mode
        self.use_flash = flash_installed and attention_mode == "flash"

        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        if attention_mode == "xformers_scaled_dot_product":
            self.xformers_scaled_dot_product_fn = ScaledDotProduct()

    def xformers_scaled_dot_product_attention(self, x, mask=None):
        if not xformers_installed:
            raise ImportError("xformers is not installed, cannot use xformer attention")

        b, l, _, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        # xformers scaled dot product attention fn applies the scaling by dim_head ** -0.5 here:
        # https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/core.py#L207

        if mask.ndim == 2:
            # mask = repeat(mask, "b l -> b h l l_prime", h=h, l_prime=l)
            mask = repeat(mask, "b l -> b l l_prime", l_prime=l)

        out = self.xformers_scaled_dot_product_fn(q, k, v, att_mask=mask)
        # out = rearrange(out, "b h l d -> b l (h d)", h=h)
        return self.to_out(out)

    def xformers_memory_efficient_attention(self, x, mask=None):
        if not xformers_installed:
            raise ImportError("xformers is not installed, cannot use xformer attention")

        dtype, device = x.dtype, x.device
        b, l, _, h = *x.shape, self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # xformers memory efficient attention implementation automatically applies the scaling by dim_heads ** -0.5
        # https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py#L219

        # expects query/key/value tensors to have shape [B, L, H, D]
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b l h d", h=h), (q, k, v))

        if mask is not None:
            # 保证 attn_bias 跟 q 是同样的 float dtype
            attn_bias = mask.to(dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
            attn_bias = rearrange(attn_bias, "b l -> b () () l")
            attn_bias = attn_bias.expand(-1, q.shape[2], q.shape[1], -1)  # B, H, L, L
        else:
            attn_bias = None

        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        out = rearrange(out, "b l h d -> b l (h d)")
        return self.to_out(out)

    def standard_attention(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, "b j -> b () () j")
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b l (h d)", h=h)
        return self.to_out(out)

    def flash_attention_padded(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Flash Attention 2 does not implement padding and attention mask in the kernel operation,
        but does offer utilities to make use of `flash_attn_varlen_qkvpacked_func` from (B, L) padding mask.

        Inspired by https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py
        """

        h, d = self.heads, self.dim_head
        b, l, _ = x.size()
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        # Transform the data into the format required by flash attention
        qkv = torch.stack([q, k, v], dim=2)
        qkv = qkv.transpose(1, 3)  # shape: [b, l, 3, num_heads, head_dim]
        key_padding_mask = mask  # shape: [b, l]

        if key_padding_mask is None:
            qkv = qkv.reshape(-1, 3, h, d)
            cu_q_lens = torch.arange(
                0, (b + 1) * l, step=l, dtype=torch.int32, device=qkv.device
            )
            max_s = l
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=self.scale, causal=False
            )
            output = output.view(b, l, -1)
        else:
            # hidden_states: (batch, seqlen, ...)
            # attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
            # qkv = qkv.reshape(b, l, -1)
            qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)

            # If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
            # (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            qkv = qkv.view(-1, 3, h, d)
            output_unpad = flash_attn_varlen_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=self.scale, causal=False
            )
            output_unpad = output_unpad.reshape(-1, h, d)
            output = pad_input(output_unpad, indices, b, l)  # shape: [b, l, h, d]
            output = rearrange(output, "b l h d -> b l (h d)", h=h)

        return output

    def forward(self, x, mask=None):
        if self.attention_mode == "xformers_scaled_dot_product":
            return self.xformers_scaled_dot_product_attention(x, mask)
        elif self.attention_mode == "xformers_memory_efficient":
            return self.xformers_memory_efficient_attention(x, mask)
        elif self.attention_mode == "flash":
            return self.flash_attention_padded(x, mask)
        else:
            return self.standard_attention(x, mask)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    Block using Flash Attention components, optionally with:
    - ada-LN conditioning (https://arxiv.org/abs/2212.09748)
    - skip connections (https://arxiv.org/abs/2209.12152)
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_skip_connect=True,
        attention_mode="xformers_scaled_dot_product",
        **block_kwargs,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        assert attention_mode in [
            "standard",
            "xformers_scaled_dot_product",
            "xformers_memory_efficient",
            "flash",
        ]
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=False,
            attention_mode=attention_mode,
            dropout=0.0,
            **block_kwargs,
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.skip_linear = (
            Mlp(
                in_features=hidden_size * 2,
                hidden_features=hidden_size,
                out_features=hidden_size,
                norm_layer=nn.LayerNorm,
                act_layer=approx_gelu,
            )
            if use_skip_connect
            else None
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, mask, skip=None):
        """Apply multi-head attention (with mask) and adaLN conditioning (mask agnostic)."""
        if skip is not None and self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DitModel(BaseDenoiser):
    def __init__(
        self,
        config: PepLDMConfig,
    ):
        self.config = config
        self.attention_mode = config.attention_mode
        self.use_skip_connect = config.use_skip_connect
        super().__init__(
            input_dim=config.input_dim,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            use_self_conditioning=config.use_self_conditioning,
            timestep_embedding_strategy=config.timestep_embedding_strategy,
            conditional=config.conditional,
            cond_dim=config.cond_dim,
        )
        if self.use_skip_connect:
            self._make_denoising_blocks_with_skip()
        else:
            self._make_denoising_blocks()
        self.initialize_weights()
        self.initialize_adaln_weights()

    def blocks_forward_pass(self, x, c, mask):
        if self.use_skip_connect:
            assert (
                hasattr(self, "in_blocks")
                and hasattr(self, "mid_block")
                and hasattr(self, "out_blocks")
            )
            skips = []

            for block in self.in_blocks:
                x = block(x, c, mask, skip=None)
                skips.append(x)

            x = self.mid_block(x, c, mask, skip=None)

            for block in self.out_blocks:
                x = block(x, c, mask, skip=skips.pop())

        else:
            for block in self.blocks:
                x = block(x, c, mask)

        return x

    def _make_denoising_blocks(self):
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_skip_connect=False,
                    attention_mode=self.attention_mode,
                )
                for _ in range(self.depth)
            ]
        )

    def _make_denoising_blocks_with_skip(self):
        in_blocks = [
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_skip_connect=self.use_skip_connect,
                attention_mode=self.attention_mode,
            )
            for _ in range(self.depth // 2)
        ]
        mid_block = DiTBlock(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            use_skip_connect=self.use_skip_connect,
            attention_mode=self.attention_mode,
        )
        out_blocks = [
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_skip_connect=self.use_skip_connect,
                attention_mode=self.attention_mode,
            )
            for _ in range(self.depth // 2)
        ]

        # these will be used in the forward pass & weight initialization
        self.in_blocks = nn.ModuleList(in_blocks)
        self.mid_block = mid_block
        self.out_blocks = nn.ModuleList(out_blocks)

    def initialize_adaln_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        if self.use_skip_connect:
            for block in self.in_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.mid_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.mid_block.adaLN_modulation[-1].bias, 0)
            for block in self.out_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


class PepLDMModel(PreTrainedModel):
    base_model_prefix = "pepldm"
    supports_gradient_checkpointing = True
    config_class = PepLDMConfig

    def __init__(self, config: PepLDMConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.objective = config.objective
        self.config = config
        self.model = DitModel(config=config)

        assert self.objective in {
            "epsilon",
            "sample",
            "v_prediction",
        }, f"Unsupported objective: {self.objective}"

        betas = torch.linspace(0.0001, 0.02, 1000, dtype=torch.float32).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def regbuf(name, value):
            self.register_buffer(name, value)

        regbuf("betas", betas)
        regbuf("alphas_cumprod", alphas_cumprod)
        regbuf("alphas_cumprod_prev", alphas_cumprod_prev)
        regbuf("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        regbuf("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        regbuf("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        regbuf("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        regbuf("posterior_variance", posterior_variance)
        regbuf(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        regbuf(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        regbuf(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        snr = alphas_cumprod / (1.0 - alphas_cumprod)
        regbuf("snr", snr)

        min_snr_gamma = 5.0
        clipped_snr = snr.clamp(max=min_snr_gamma)
        if self.objective == "epsilon":
            loss_weight = clipped_snr / snr
        elif self.objective == "sample":
            loss_weight = clipped_snr
        elif self.objective == "v_prediction":
            loss_weight = clipped_snr / (snr + 1.0)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")
        regbuf("loss_weight", loss_weight)

    def extract(
        self, a: torch.Tensor, t: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        """Helper function to gather and reshape coefficients"""
        out = a[t]
        while len(out.shape) < len(shape):
            out = out.unsqueeze(-1)
        return out.expand(shape)

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        noise = noise if noise is not None else torch.randn_like(x_start)
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * x_start
        )

    def predict_start_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = self.extract(self.posterior_variance, t, x_t.shape)
        log_var = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def get_loss_weight(self, t: torch.Tensor, shape: torch.Size):
        return self.extract(self.loss_weight, t, shape)

    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        use_cond_dropout: Optional[bool] = False,
        return_loss: bool = True,
        **kwargs,
    ):
        denoiser_kwargs = DenoiserKwargs(
            x_t=inputs,
            t=t,
            cond=cond,
            x_self_cond=x_self_cond,
        )
        logits = self.model.forward(
            denoiser_kwargs=denoiser_kwargs,
            use_cond_dropout=use_cond_dropout,
        )
        loss = None
        if return_loss:
            loss = self.get_loss_weight(t, logits.shape) * F.mse_loss(
                logits, labels, reduction="none"
            )
            loss = loss.mean()
        return {
            "loss": loss,
            "logits": logits,
            "denoiser_kwargs": denoiser_kwargs,
        }

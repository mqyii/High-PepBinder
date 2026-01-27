import click
import lightning as L
import pandas as pd
from loguru import logger
from pathlib import Path
import torch

from .utils import PepLDMPipeline
from .tokenizer import ESMSeqTokenizer
from .lightning import PepLDM2

from astra.tokenizers.esm.tokenization import ESMSeqTokenizer as AstraTokenizer
import tokenizers

@click.group()
def cli():
    pass


@cli.command()
@click.option("--input_path", type=str, required=True)
@click.option("--output_path", type=str, required=True)
@click.option("--model_path", type=str, required=True)
@click.option("--seed", type=int, default=42)
@click.option("--device", type=str, default="cuda:0")
def generate(input_path, output_path, seed, device, model_path):
    L.seed_everything(seed)
    input_path = Path(input_path)
    assert input_path.exists(), f"Input path {input_path} does not exist"

    output_path = Path(output_path)
    assert output_path.parent.exists(), f"Output path {output_path} does not exist"

    model_path = Path(model_path)
    assert model_path.exists(), f"Model path {model_path} does not exist"

    tokenizer = ESMSeqTokenizer()

    torch.serialization.add_safe_globals([AstraTokenizer, tokenizers.Tokenizer, tokenizers.models.Model])
    pepldm = PepLDM2.load_from_checkpoint(
        checkpoint_path=model_path,
        tokenizer=tokenizer,
        strict=False,
    ).to(device)

    pepldm.eval()

    pipeline = PepLDMPipeline(
        model=pepldm, # pepldm
        tokenizer=tokenizer,
        device=device,
        num_steps=500, # 1000
        num_samples=100, # 1000
        min_seqs=50, # 6000
        max_tries=3, # 30
    )
    df = pd.read_csv(input_path)
    target_list = df["target_sequence"].tolist()
    assert len(df) > 0 and len(target_list) > 0, "No target sequences found"
    # 跑 pipeline 得到多条肽链结果
    result = pipeline(target_list)  # result 应该是 list，每个元素是 df_peptide

    # 把结果展开成一个总表
    df_list = []
    for target, df_list_per_target in zip(target_list, result):
        for df_peptide in df_list_per_target:  # ✅ 每个 target 可能生成多个 df
            df_peptide = df_peptide.copy()
            df_peptide["target_sequence"] = target  # 保持键一致
            df_list.append(df_peptide)

    df_peptide_all = pd.concat(df_list, ignore_index=True)

    df_merged = df.merge(df_peptide_all, on="target_sequence", how="left")
    df_merged.to_csv(output_path, index=False)


if __name__ == "__main__":
    cli()


# import re
# import sys
# import types
# from pathlib import Path
# import pickle

# import click
# import lightning as L
# import pandas as pd
# import torch
# from loguru import logger

# from .lightning import PepLDM2
# from .tokenizer import ESMSeqTokenizer
# from .utils import PepLDMPipeline

# # ----------------------------
# # Safe loader for "foreign" Lightning checkpoints
# # (ckpt downloaded from GitHub that pickled custom classes like `astra.*`
# #  and HuggingFace `tokenizers.*`)
# # ----------------------------

# _UNSUPPORTED_GLOBAL_RE = re.compile(r"Unsupported global:\s+GLOBAL\s+([A-Za-z0-9_\.]+)")


# def _ensure_module(mod_name: str) -> types.ModuleType:
#     """Create (or return) a module object and register into sys.modules."""
#     if mod_name in sys.modules:
#         return sys.modules[mod_name]
#     m = types.ModuleType(mod_name)
#     sys.modules[mod_name] = m
#     return m


# def _stub_qualname(qualname: str):
#     """
#     Create a stub module + class for qualname like 'astra.tokenizers.esm.tokenization.ESMSeqTokenizer'.
#     Returns the created class object.
#     """
#     parts = qualname.split(".")
#     if len(parts) < 2:
#         raise ValueError(f"Cannot stub qualname: {qualname}")
#     cls_name = parts[-1]
#     mod_name = ".".join(parts[:-1])

#     # Ensure parent packages exist
#     for i in range(1, len(parts)):
#         _ensure_module(".".join(parts[:i]))

#     mod = _ensure_module(mod_name)

#     # Create stub class
#     Stub = type(cls_name, (), {})
#     Stub.__module__ = mod_name
#     setattr(mod, cls_name, Stub)
#     return Stub


# def _resolve_qualname(qualname: str):
#     """
#     Try to import and resolve the class/function referenced by qualname.
#     If import fails, create a stub class instead.
#     """
#     parts = qualname.split(".")
#     if len(parts) < 2:
#         raise ValueError(f"Cannot resolve qualname: {qualname}")
#     mod_name = ".".join(parts[:-1])
#     attr_name = parts[-1]
#     try:
#         module = __import__(mod_name, fromlist=[attr_name])
#         obj = getattr(module, attr_name)
#         return obj
#     except Exception:
#         # Fall back to stub
#         return _stub_qualname(qualname)


# def load_foreign_lightning_ckpt_weights_only(
#     ckpt_path: Path,
#     map_location: str = "cpu",
#     max_rounds: int = 64,
# ):
#     """
#     Load a Lightning checkpoint dict using torch.load(..., weights_only=True),
#     automatically allowlisting / stubbing any "Unsupported global" classes it complains about.

#     This is specifically to handle checkpoints that pickle tokenizers / custom classes.
#     """
#     # Proactively ensure common astra module path exists (if checkpoint references it)
#     # We do not know all classes; the loop will stub any missing qualnames.
#     _ensure_module("astra")

#     allowlisted = []

#     for round_idx in range(1, max_rounds + 1):
#         try:
#             if allowlisted:
#                 torch.serialization.add_safe_globals(allowlisted)
#             ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=True)
#             return ckpt
#         except pickle.UnpicklingError as e:
#             msg = str(e)
#             m = _UNSUPPORTED_GLOBAL_RE.search(msg)
#             if not m:
#                 raise
#             qualname = m.group(1)

#             obj = _resolve_qualname(qualname)
#             allowlisted.append(obj)

#             logger.warning(
#                 f"[weights_only load] allowlisted/stubbed unsupported global "
#                 f"({round_idx}/{max_rounds}): {qualname}"
#             )
#         except Exception:
#             raise

#     raise RuntimeError(
#         f"Failed to load checkpoint with weights_only=True after {max_rounds} allowlist rounds. "
#         f"Last allowlisted: {allowlisted[-1] if allowlisted else None}"
#     )


# @click.group()
# def cli():
#     pass


# @cli.command()
# @click.option("--input_path", type=str, required=True)
# @click.option("--output_path", type=str, required=True)
# @click.option("--model_path", type=str, required=True)
# @click.option("--seed", type=int, default=42)
# @click.option("--device", type=str, default="cuda:0")
# def generate(input_path, output_path, seed, device, model_path):
#     """
#     One-step generation:
#     - Loads a GitHub-downloaded Lightning ckpt safely with weights_only=True
#       (auto allowlist/stub foreign classes referenced by pickle)
#     - Reconstructs PepLDM2 from hyper_parameters (model_args/esm_args)
#     - Loads state_dict and runs pipeline generation
#     """
#     L.seed_everything(seed)

#     input_path = Path(input_path)
#     if not input_path.exists():
#         raise FileNotFoundError(f"Input path {input_path} does not exist")

#     output_path = Path(output_path)
#     if not output_path.parent.exists():
#         raise FileNotFoundError(f"Output dir {output_path.parent} does not exist")

#     model_path = Path(model_path)
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model path {model_path} does not exist")

#     tokenizer = ESMSeqTokenizer()

#     suffix = model_path.suffix.lower()

#     if suffix in [".ckpt", ".pt", ".pth"]:
#         # Preferred path: if ckpt -> load dict (hyper_parameters + state_dict) via safe weights-only loader
#         if suffix == ".ckpt":
#             raw = load_foreign_lightning_ckpt_weights_only(model_path, map_location="cpu")
#             if not isinstance(raw, dict):
#                 raise ValueError("Loaded checkpoint is not a dict; cannot proceed.")

#             hp = raw.get("hyper_parameters", {}) or {}
#             model_args = hp.get("model_args", None)
#             esm_args = hp.get("esm_args", None)

#             if model_args is None or esm_args is None:
#                 raise ValueError(
#                     "Checkpoint missing hyper_parameters.model_args or hyper_parameters.esm_args. "
#                     "Cannot reconstruct PepLDM2 architecture from this ckpt."
#                 )

#             model = PepLDM2(
#                 model_args=model_args,
#                 esm_args=esm_args,
#                 tokenizer=tokenizer,
#                 load_from_esmc=False,
#             ).to(device)

#             sd = raw.get("state_dict", None)
#             if sd is None:
#                 raise ValueError("Checkpoint dict has no 'state_dict' key.")
#             model.load_state_dict(sd, strict=False)
#             model.eval()

#         else:
#             # If user points to a pure state_dict file, you must also provide model_args/esm_args somehow.
#             # We refuse here rather than silently creating a wrong architecture.
#             raise ValueError(
#                 f"model_path={model_path} looks like a pure weight file ({suffix}). "
#                 "This one-step mode requires a Lightning .ckpt containing hyper_parameters "
#                 "(model_args/esm_args). Please pass the original .ckpt."
#             )
#     else:
#         raise ValueError(f"Unsupported model file extension: {suffix}")

#     pipeline = PepLDMPipeline(
#         model=model,
#         tokenizer=tokenizer,
#         device=device,
#         num_steps=1000,
#         num_samples=1000,
#         min_seqs=6000,
#         max_tries=30,
#     )

#     df = pd.read_csv(input_path)
#     if "target_sequence" not in df.columns:
#         raise ValueError("Input CSV must contain a 'target_sequence' column.")
#     target_list = df["target_sequence"].tolist()
#     if len(target_list) == 0:
#         raise ValueError("No target sequences found in input CSV.")

#     result = pipeline(target_list)  # list; each element corresponds to one target

#     df_list = []
#     for target, df_list_per_target in zip(target_list, result):
#         for df_peptide in df_list_per_target:
#             df_peptide = df_peptide.copy()
#             df_peptide["target_sequence"] = target
#             df_list.append(df_peptide)

#     if not df_list:
#         raise ValueError("Pipeline returned no generated peptides.")

#     df_peptide_all = pd.concat(df_list, ignore_index=True)
#     df_merged = df.merge(df_peptide_all, on="target_sequence", how="left")
#     df_merged.to_csv(output_path, index=False)
#     logger.info(f"Saved generated peptides to: {output_path}")


# if __name__ == "__main__":
#     cli()

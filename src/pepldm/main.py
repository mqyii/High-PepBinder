import click
import lightning as L
import pandas as pd
from loguru import logger
from pathlib import Path

from .utils import PepLDMPipeline
from .tokenizer import ESMSeqTokenizer
from .lightning import PepLDM2


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
    pepldm = PepLDM2.load_from_checkpoint(
        checkpoint_path=model_path,
        tokenizer=tokenizer,
        strict=False,
    ).to(device)

    pepldm.eval()

    pipeline = PepLDMPipeline(
        model=pepldm,
        tokenizer=tokenizer,
        device=device,
        num_steps=1000,
        num_samples=1000,
        min_seqs=6000,
        max_tries=30,
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

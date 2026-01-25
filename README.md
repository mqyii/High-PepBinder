# High-PepBinder

This is the official implementation for the paper titled 'High-PepBinder: A pLM-Guided Latent Diffusion Framework for Affinity-Aware Target-Specific Peptide Design'.

![](./F1.png)

## Setups

```bash
git clone git@github.com:mqyii/High-PepBinder.git
cd High-PepBinder
conda create -n High-PepBinder python=3.12 -y
conda activate High-PepBinder
pip install -e .
```

## Peptide Binder De Novo Design

- The prediction code is tested on GPU A800 with `python==3.12` and `torch==2.7.1+cu126`.
- The trained model ckpt can be found at [zenodo](https://zenodo.org/records/18282817)

```bash
pepldm generate --input_path target_sequence.csv \
    --output_path peptide_generated.csv \
    --model_path pepldm_step=882-val_loss=0.51.ckpt \
    --seed 42 \
    --device cuda:0
```

## Datasets

- The raw PepPBA dataset can be found at `datasets` dir.

## Training

- see `configs` for details

```bash
# pretrain the LDM on FRGDB
pepldm train frag

# finetuneing on the peptide real 
pepldm train pep

# trainning the affinity head on XX

pepldm train affinity

# train concistencly 
pepldm train2
```

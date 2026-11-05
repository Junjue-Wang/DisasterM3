# Training DisasterM3 with LLaMA-Factory

This folder is the official training code for the LoRA fine-tuning of **Qwen2.5-VL-7B** and **InternVL3-8B** from Appendix B.3 of the [DisasterM3 paper](https://arxiv.org/abs/2505.21089) (NeurIPS 2025) with LLaMA-Factory. The merged models then run on the benchmark through `pyscripts/run_vllm.py`.

> Run all commands from this `training/` folder unless a step says otherwise. The paths in the configs are relative to it.

## Contents

1. [Files](#1-files)
2. [Environment](#2-environment)
3. [Preparing the training data](#3-preparing-the-training-data)
4. [Training](#4-training)
5. [Merging the LoRA weights](#5-merging-the-lora-weights)
6. [Inference on the benchmark](#6-inference-on-the-benchmark)
7. [Comparison with the paper's settings](#7-comparison-with-the-papers-settings)

## 1. Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Builds the training data `data/DisasterM3.json` from the DisasterM3 instruct set |
| `prepare_bench.py` | Converts the benchmark set to LLaMA-Factory format (only used by the fallback in 6.3) |
| `data/dataset_info.json` | Registers the datasets: `disaster_m3` (training) and `disaster_m3_bench_<subset>` (inference) |
| `configs/qwen25_vl_7b_lora_sft.yaml`<br>`configs/internvl3_8b_lora_sft.yaml` | Training configs (paper settings) |
| `configs/qwen25_vl_7b_merge.yaml`<br>`configs/internvl3_8b_merge.yaml` | Merge the LoRA weights |
| `configs/ds_z2_config.json` | DeepSpeed ZeRO-2 config |

## 2. Environment

**Training** uses LLaMA-Factory pinned to upstream commit `e2b1594d` (version 0.9.4.dev0). This version needs `numpy<2`, so use Python 3.9 to 3.12. Clone it outside this repository:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && git checkout e2b1594d
pip install -e ".[torch,metrics,deepspeed]"
pip install "pyarrow<21"                        # this LLaMA-Factory version pins numpy<2, while newer pyarrow needs numpy>=2
pip install flash-attn --no-build-isolation     # the configs set flash_attn: fa2; if this fails to install, change fa2 to sdpa
```

**Inference** is best kept in a separate environment with Python 3.9 to 3.13, or 3.9 to 3.12 if you need the fallback in 6.3. vLLM supports HF-format InternVL3 from 0.10.2 on, and 0.10.2 is also the highest version this LLaMA-Factory commit allows. `transformers==4.56.2` is the highest version it allows too, so the merged files are written and read by the same version.

```bash
pip install vllm==0.10.2 transformers==4.56.2 qwen_vl_utils
pip install -e /path/to/LLaMA-Factory           # only needed for the fallback in 6.3
pip install "pydantic>=2.11.7"                  # only after the line above: LLaMA-Factory pins pydantic<=2.10.6, while vLLM 0.10.2 needs >=2.11.7
```

## 3. Preparing the training data

Download the data from [Hugging Face](https://huggingface.co/datasets/Kingdrone-Junjue/DisasterM3), or request it through the form linked in the [main README](../README.md). The **instruct set** is used for training and the **benchmark set** for inference.

`hf` is the command-line tool of `huggingface_hub`. The commands below keep you in `training/`:

```bash
DATA=/path/to/DisasterM3
hf download Kingdrone-Junjue/DisasterM3 --repo-type dataset --local-dir $DATA --include "DisasterM3_Instruct/*"

unzip -q $DATA/DisasterM3_Instruct/train_images.zip     -d $DATA/DisasterM3_Instruct
unzip -q $DATA/DisasterM3_Instruct/box_train_images.zip -d $DATA/DisasterM3_Instruct/train_images
unzip -q $DATA/DisasterM3_Instruct/masks.zip            -d $DATA/DisasterM3_Instruct   # optional, not used here
```

`box_train_images.zip` must be extracted into `train_images/`, as above. `train_images.zip` is a ZIP64 archive, so use Info-ZIP `unzip` 6.0 or later, or Python's `zipfile`. All three archives take about 33 GB once extracted.

The folder then looks like this:

```
DisasterM3_Instruct/                 # pass this folder as --data_root
├── train_release.json
├── train_images/                    # pre-disaster and post-disaster images
│   └── box_train_images/            # images of the relational reasoning task
└── masks/                           # optional, referring expression segmentation only
```

Build the training data:

```bash
python prepare_data.py --data_root $DATA/DisasterM3_Instruct
```

This writes `data/DisasterM3.json` and prints the number of samples per task. The script first checks that the annotation file and all images exist, and lists anything it cannot find. Fix the layout, or rerun with `--drop_missing` to skip those samples. The JSON stores **absolute image paths**, so do not move the data folder afterwards.

The output follows the paper's training data:

- There are 8 tasks: BBR (`bearing_body`), BDC (`building_damage_counting`), DTR (`disaster_type`), DSR (`landuse`), DRE (`road_damage_counting`), ORR (`relational_reasoning_qa`), `caption` and `recovery`. Referring expression segmentation is not included, because the paper trains that task with LISA and PSALM, not LLaMA-Factory.
- The samples of `building_damage_counting` and `road_damage_counting` are **repeated twice** (2x oversampling). The paper text does not state this. To turn it off, set the values in `TASK_REPEATS` to 1.
- The pair `socal_fire_00001247` is excluded from the tasks that use both images. The single-image `landuse` sample of its pre-disaster image stays in, as in the paper's training data.

| Task | Samples | Repeat | Total |
|---|---:|---:|---:|
| `bearing_body` | 7,765 | 1 | 7,765 |
| `building_damage_counting` | 14,531 | 2 | 29,062 |
| `disaster_type` | 1,627 | 1 | 1,627 |
| `landuse` | 7,090 | 1 | 7,090 |
| `road_damage_counting` | 7,336 | 2 | 14,672 |
| `relational_reasoning_qa` | 1,882 | 1 | 1,882 |
| `caption` | 7,765 | 1 | 7,765 |
| `recovery` | 7,764 | 1 | 7,764 |
| **All** | | | **77,627** |

## 4. Training

```bash
llamafactory-cli train configs/qwen25_vl_7b_lora_sft.yaml
llamafactory-cli train configs/internvl3_8b_lora_sft.yaml
```

The LoRA weights are saved to `saves/<model>_disaster_m3/lora/sft`.

The configs follow the paper's **4×H100 (96GB)** setup with a **global batch of 256**:

| Config | `per_device_train_batch_size` | `gradient_accumulation_steps` | GPUs |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 8 | 8 | 4 |
| InternVL3-8B | 64 | 1 | 4 |

- With a different GPU count or memory size, change these two values and keep `batch per GPU × gradient accumulation × GPUs = 256`.
- `llamafactory-cli` uses all visible GPUs. Select them with `CUDA_VISIBLE_DEVICES`.
- With a single GPU, add `FORCE_TORCHRUN=1` to use DeepSpeed.

## 5. Merging the LoRA weights

```bash
llamafactory-cli export configs/qwen25_vl_7b_merge.yaml     # -> output/Qwen2.5-VL-7B-DisasterM3-merged
llamafactory-cli export configs/internvl3_8b_merge.yaml     # -> output/InternVL3-8B-hf-DisasterM3-merged
```

> **Keep these folder names.** `run_vllm.py` picks the model type from keywords in the path: Qwen needs `qwen`, `vl` and `7b`, and InternVL needs `intern`, `vl` and `-8b`. Other names stop with `Unknown model_id` or `NotImplementedError`. The match covers the whole path, so parent folders should not contain the name of another model, such as `llava`, or `qwen` for the InternVL model. The last two path components also name the results folder.

After merging, check that `tokenizer_config.json` and `preprocessor_config.json` are in the merged folder. Inference needs them, and the export only logs a warning if saving them fails.

## 6. Inference on the benchmark

### 6.1 Using run_vllm.py

`run_vllm.py` uses fixed paths. It reads from `data/` next to the repository folder and writes to `results/` next to it:

```
workspace/
├── disaster_m3/        # this repository
│   └── training/       # run the commands from here
├── data/               # written by split_benchmark.py
│   ├── <subset>.json
│   └── images -> /path/to/DisasterM3/DisasterM3_Bench/test_images
└── results/            # written by run_vllm.py
```

Download the benchmark set and split it into one file per subset. `$DATA` is the folder from section 3:

```bash
hf download Kingdrone-Junjue/DisasterM3 --repo-type dataset --local-dir $DATA --include "DisasterM3_Bench/*" --exclude "DisasterM3_Bench/masks/*"

python ../pyscripts/split_benchmark.py --bench_root $DATA/DisasterM3_Bench
```

`split_benchmark.py` checks that all images are in `test_images/`, links `data/images` to that folder and writes the eight subsets in release order. Referring expression segmentation is skipped. If an image is missing or the link cannot be created, the script stops before writing the subsets and says what to fix, then rerun it. If `data/images` is already a real folder, move it away first.

| Subset | Rows |
|---|---:|
| `bearing_body` | 2,363 |
| `building_damage_counting` | 4,982 |
| `disaster_type` | 420 |
| `road_damage_counting` | 2,178 |
| `landuse` | 2,007 |
| `relational_reasoning_qa` | 1,018 |
| `caption` | 2,363 |
| `recovery` | 2,363 |
| **All** | **17,694** |

Then run a subset:

```bash
PYTHONPATH=$PWD/.. python ../pyscripts/run_vllm.py \
    --model_id $PWD/output/Qwen2.5-VL-7B-DisasterM3-merged \
    --subset bearing_body
```

- `PYTHONPATH` must point to the repository root, otherwise `from models import ...` cannot find the module.
- `--subset` is one of `bearing_body`, `building_damage_counting`, `disaster_type`, `road_damage_counting`, `landuse`, `relational_reasoning_qa`, `caption` and `recovery`.
- Results go to `../../results/<subset>/output--Qwen2.5-VL-7B-DisasterM3-merged/finished.json`. Each entry is `{"id": "<subset>_<index>", "response": ...}`.
- A rerun resumes from the finished samples. Add `--overwrite` to start over, which you also need after merging a new model into the same folder.
- Qwen was trained on images of at most 768×768 pixels, while `run_vllm.py` uses the native resolution of the model by default. Add `--image_size 768` to match training.
- `run_vllm.py` runs inference only and does not compute scores.

### 6.2 InternVL3-8B: start with a small run

LLaMA-Factory trains the HF-format `InternVL3-8B-hf`. `run_vllm.py` recognises such checkpoints from their tokenizer and then uses the same setup as training: the `<IMG_CONTEXT>` image token, the system prompt of the `intern_vl` template, and one 448×448 tile per image. Native `OpenGVLab/InternVL3-*` models are not affected.

**This InternVL path has been checked on CPU only and has not been run on a GPU yet.** Run one subset first and read the first lines of `finished.jsonl`:

```bash
PYTHONPATH=$PWD/.. python ../pyscripts/run_vllm.py \
    --model_id $PWD/output/InternVL3-8B-hf-DisasterM3-merged \
    --subset disaster_type \
    --batch_size 2 \
    --temperature 0.01
```

- If the lines show normal option letters, continue with the other subsets.
- If you see an error or unusual output, stop with Ctrl-C, delete the `../../results/<subset>/...` folder and use 6.3.

`--temperature 0.01` is close to greedy decoding, and it is what `run_vllm.py` uses for Qwen. Pass it for this model: its generation config sets no temperature, so vLLM would otherwise sample at its default of 1.0.

### 6.3 Fallback: LLaMA-Factory's vllm_infer.py

This route is only needed if `run_vllm.py` cannot run a model. It needs the split from 6.1:

```bash
# 1) Convert the benchmark to LLaMA-Factory format, with the same prompts as run_vllm.py. Writes data/bench/<subset>.json
python prepare_bench.py --bench_root ../../data

# 2) Run each subset
mkdir -p results/internvl3_8b
python /path/to/LLaMA-Factory/scripts/vllm_infer.py \
    --model_name_or_path output/InternVL3-8B-hf-DisasterM3-merged \
    --template intern_vl \
    --dataset disaster_m3_bench_disaster_type \
    --dataset_dir data \
    --cutoff_len 8192 \
    --max_new_tokens 2048 \
    --temperature 0.01 \
    --vllm_config '{"mm_processor_kwargs": {"crop_to_patches": False}}' \
    --save_name results/internvl3_8b/disaster_type.jsonl
```

- Each output line is `{"prompt", "predict", "label"}`, and **line i is the id `<subset>_<i>`**. `data/bench/<subset>.json` also stores the `id` field for reference.
- `label` comes from the `ground_truth` field of the benchmark set.
- `--temperature 0.01` is close to greedy decoding, as in 6.2.
- `--vllm_config` keeps one tile per image, as in training. Keep `False` capitalised, because the script reads this value as a Python literal.
- This route expands the image tokens differently from training, so prefer `run_vllm.py` where it works.

## 7. Comparison with the paper's settings

| Item | Paper (Appendix B.3) | This folder |
|---|---|---|
| Models | Qwen2.5-VL-7B, InternVL3-8B | `Qwen/Qwen2.5-VL-7B-Instruct`, `OpenGVLab/InternVL3-8B-hf` |
| Method | LoRA, vision encoder frozen, only the LLM is trained | `lora_target: all`. LLaMA-Factory freezes the vision encoder and the projector by default, so LoRA only attaches to the 196 linear layers of the LLM |
| LoRA | rank 64, alpha 16, dropout 0.05 | Same |
| Optimizer | AdamW, β1=0.9, β2=0.95 | Same |
| Learning rate | 2e-4, cosine | Same, plus `warmup_ratio` 0.03 (not stated in the paper) |
| Global batch | 256 | Same (4 GPUs) |
| Epochs | 1 | Same |
| Hardware | 4×H100 96GB | See section 4 |
| Image resolution | Not stated | Qwen: at most 768×768 pixels (LLaMA-Factory's default at the time, set explicitly in the config)<br>InternVL: each image is resized to one 448×448 tile (`crop_to_patches: false` in the `-hf` preprocessor config) |
| Training data | Instruct set | See section 3, including the 2x oversampling of the counting tasks |
| Other | Not stated | bf16, `cutoff_len` 8192, DeepSpeed ZeRO-2, FlashAttention-2 |

> **Validation.** The config parsing, the modules LoRA attaches to, the data preparation and the prompt building were checked at CPU level only. `prepare_data.py` gives the same tasks, prompts and sample order as the original training setup. Full training, merging and inference have **not** been run through this folder on a GPU, so run a few dozen steps first and confirm that the loss goes down.

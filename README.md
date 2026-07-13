<div align="center">

# Med-OPD: Improving Medical Vision-Language Models via Evidence-Aware On-Policy Distillation

*Evidence-aware post-training for medical vision-language models.*

</div>

<div align="center">
  <a href="#overview"><b>Overview</b></a> &bull;
  <a href="#getting-started"><b>Getting Started</b></a> &bull;
  <a href="#training"><b>Training</b></a> &bull;
  <a href="#implementation-notes"><b>Implementation Notes</b></a>
</div>

---

## Overview

This directory contains the source code for **Med-OPD**, which adapts on-policy distillation (OPD) to medical vision-language models (Med-VLMs). Standard OPD distills every token in a student-generated rollout equally, although only a small subset of a medical response usually describes diagnosis-critical visual evidence. Med-OPD addresses this imbalance with **Medical Evidence Advantage (MEA)**, which measures how strongly each generated token depends on fine-grained evidence in the medical image.

For each student rollout, the teacher evaluates the token sequence under both the original image and an evidence-degraded image. The difference in teacher confidence defines MEA. Med-OPD then uses MEA at two levels:

- **Trajectory-level weighting:** gives more training weight to rollouts with stronger overall dependence on medical visual evidence.
- **Token-level grouped distillation:** separately aggregates reverse-KL losses for high- and low-MEA token groups, preventing diagnosis-critical tokens from being diluted by long clinical narratives.

The training code also supports an answer-aware teacher hint (`INJECT_MODE=gt_contrastive`) that directs teacher scoring toward visual findings relevant to the target diagnosis. The student does not receive this hint at inference time.

## Repository Structure

```text
Med-OPD/
|-- vaopd_CT.sh                    # Med-OPD training on the CT subset
|-- vaopd_mri.sh                   # Med-OPD training on the MRI subset
|-- vaopd_Disease_Diagnosis.sh     # Med-OPD training on Disease Diagnosis
|-- vaopd_Lesion_Grading.sh        # Med-OPD training on Lesion Grading
|-- opd_standard__*.sh             # Standard OPD baselines
|-- merge_ckpts.sh                 # Merge FSDP checkpoints into HF weights
|-- eval_ckpts.py                  # Evaluate saved checkpoints
|-- scripts/                       # Inference, evaluation, and verification utilities
`-- verl/                          # Customized verl implementation
```

> This is the Med-OPD implementation. A few script filenames and configuration keys retain internal names from the original development code; these names do not denote a different method.

## Getting Started

### Environment Setup

The implementation is based on [verl](https://github.com/verl-project/verl) (v0.7.0), following the same OPD environment used by the reference implementation.

```bash
conda create -n verl python==3.12
conda activate verl

cd Med-OPD/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
cd ..
```

The provided scripts target AMD GPUs with ROCm. Before launching training, adjust the machine-specific paths and GPU settings in the selected script:

```bash
# In the selected vaopd_*.sh file
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCM_HOME=/opt/rocm-7.0.2
export PYTHONPATH="/path/to/Med-OPD/verl:$PYTHONPATH"

export TRAIN_DATASET=/path/to/train.parquet
export TEST_DATASET='["/path/to/test.parquet"]'
export ACTOR_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct
export REWARD_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
```

## Training

Each script launches a Ray head and then runs Med-OPD training through `verl.trainer.main_ppo`. Choose the script that matches the intended OmniMedVQA subset:

```bash
cd Med-OPD

# CT
bash vaopd_CT.sh

# MRI
bash vaopd_mri.sh

# Disease Diagnosis
bash vaopd_Disease_Diagnosis.sh

# Lesion Grading
bash vaopd_Lesion_Grading.sh
```

The corresponding standard OPD baselines can be launched with `opd_standard__CTComputed_Tomography.sh`, `opd_standard__MRI.sh`, `opd_standard__Disease_Diagnosis.sh`, and `opd_standard__Lesion_Grading.sh`.

## Implementation Notes

| Parameter | Default | Description |
|---|---:|---|
| `ADV_ESTIMATOR` | `VAOPD` | Internal estimator identifier used by the Med-OPD source code. |
| `N_RESPONSES` | `8` | Number of student rollouts sampled per training example. |
| `LOG_PROB_TOP_K` | `16` | Student top-k candidate tokens retained for distillation. |
| `TOP_K_STRATEGY` | `only_stu` | Builds the token candidate set from the student distribution. |
| `VAOPD_PV` | `0.2` | Fraction of tokens assigned to the high-MEA group. |
| `VAOPD_LAMBDA` | `0.5` | Relative weight of the high-MEA group loss. |
| `VAOPD_TAU` | `1.0` | Temperature for trajectory-level MEA weighting. |
| `data.vaopd_degrade_area_ratio` | `0.1` | Evidence degradation ratio used for MEA computation. |
| `INJECT_MODE` | task-dependent | `gt_contrastive` enables the answer-aware teacher hint; `none` disables it. |

The default experiment configuration uses Qwen3-VL-2B-Instruct as the student and Qwen3-VL-4B-Instruct as the teacher. Training scripts save FSDP checkpoints under `checkpoint/`; use `merge_ckpts.sh` to create Hugging Face-compatible model weights for inference or evaluation.

```bash
bash merge_ckpts.sh <CKPT_DIR> [HIP_ID]
```

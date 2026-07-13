#!/bin/bash
set -x

# ==========================================================================
# VA-OPD: Visual-Advantage On-Policy Distillation (Liu et al., 2026)
# 适配环境：AMD MI308X + ROCm 7.0.2 + torch 2.10.0+rocm7.1 + vllm 0.11.0+rocm702
# 与 token_reward_direct_plus_grpo.sh 的区别（见文件内 >>> VA-OPD <<< 标注）：
#   1. ADV_ESTIMATOR=VAOPD
#   2. data.vaopd_degrade_image=True  (dataset 预处理生成降质图)
#   3. VAOPD_PV / VAOPD_LAMBDA / VAOPD_TAU  超参
# ==========================================================================

# ==========================================
# 1. 日志配置
# ==========================================
LOG_DIR="/mnt/yunhang/VA-OPD/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=========================================="
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo "=========================================="
ulimit -n 1048576
echo "fd limit: $(ulimit -n)"

# ==========================================
# 2. ROCm / AMD 必备环境变量
# ==========================================
# 清理之前的 ray
# ray stop --force 2>/dev/null
# sleep 3

# --- GPU 可见性（AMD 必须用 HIP，不是 CUDA；同时也设 CUDA 保兜底）---
# === GPU 可见性 ===
# 只用 HIP_VISIBLE_DEVICES 限定 ray head 能看到几张卡
# 不要设 CUDA_VISIBLE_DEVICES（让 Ray 自己管 worker 分配）
# 不要设 ROCR_VISIBLE_DEVICES（verl 不允许跟 HIP 同时存在）
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=2,3,4,7
#export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
# --- flash-attn ROCm Triton 后端必须 ---
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# --- vLLM ROCm 必须 ---
export VLLM_USE_TRITON_FLASH_ATTN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # ROCm 不能 fork

# --- ROCm 路径（保险）---
export ROCM_HOME=/opt/rocm-7.0.2
export ROCM_PATH=/opt/rocm-7.0.2
export PATH=/opt/rocm-7.0.2/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.0.2/lib:$LD_LIBRARY_PATH

# --- Cache 全部搬出 overlay（根分区只剩 4.6G，Triton JIT cache 会炸）---
export TRITON_CACHE_DIR=/mnt/yunhang/.cache/triton
export OUTLINES_CACHE_DIR=/mnt/yunhang/.cache/outlines/vaopd_$$
export HF_HOME=/mnt/yunhang/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=/mnt/yunhang/.cache/pip
mkdir -p "$TRITON_CACHE_DIR" "$OUTLINES_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR"

# --- 分布式 / NCCL（ROCm 上 NCCL 实际是 RCCL）---
export RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1
export PROJECT_NAME='VAOPD_COT'
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/mnt/yunhang/VA-OPD/verl:$PYTHONPATH"
export SWANLAB_MODE=cloud

# >>> 本任务专属 Ray 地址（端口 6376）<<<
export RAY_ADDRESS="127.0.0.1:6376"

# 打印关键 env，方便排错
echo "=========================================="
echo "ENV CHECK:"
echo "  HIP_VISIBLE_DEVICES = $HIP_VISIBLE_DEVICES"
echo "  RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES = $RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"
echo "  FLASH_ATTENTION_TRITON_AMD_ENABLE = $FLASH_ATTENTION_TRITON_AMD_ENABLE"
echo "  VLLM_USE_TRITON_FLASH_ATTN = $VLLM_USE_TRITON_FLASH_ATTN"
echo "  VLLM_WORKER_MULTIPROC_METHOD = $VLLM_WORKER_MULTIPROC_METHOD"
echo "  TRITON_CACHE_DIR = $TRITON_CACHE_DIR"
echo "=========================================="

# ==========================================
# 3. 实验与超参数（不动）
# ==========================================
# >>> VA-OPD <<< 用 VAOPD estimator（分组 reverse-KL loss）
export ADV_ESTIMATOR=VAOPD
export GRPO_OUTCOME_WEIGHT=1.0   # VAOPD 路径下不使用，仅占位

# >>> VA-OPD <<< 三个超参（与论文一致）
export VAOPD_PV=0.2       # p_v：高-VA 组比例
export VAOPD_LAMBDA=0.5   # lambda：高-VA 组权重
export VAOPD_TAU=1.0      # tau：rollout softmax 温度

# >>> VA-OPD <<< heavy-tail 直方图开关（论文 Fig.2a）
#   1 = 统计 batch 的 per-token VA 分布, 画两张图(线性+log-y),
#       上传 swanlab 并本地保存 PNG + 原始 .npy + jsonl 日志
#   0 = 关闭（默认）
export VAOPD_PLOT_HEAVYTAIL=${VAOPD_PLOT_HEAVYTAIL:-1}
#   画图频率: 每多少步生成一次这两张图(默认 10; step 1 总是画一次)
export VAOPD_PLOT_EVERY=${VAOPD_PLOT_EVERY:-10}
#   图片/日志本地保存目录, 留空则默认 <default_local_dir>/va_heavytail
export VAOPD_PLOT_DIR=${VAOPD_PLOT_DIR:-/mnt/yunhang/VA-OPD/va_heavytail}

# 长度
export MAX_PROMPT_LENGTH=8192
export MAX_RESP_LENGTH=256
export MAX_VAL_RESP_LENGTH=256
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))

# 采样与批次
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export N_RESPONSES=8
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}        # VAOPD 依赖 student top-k 子集
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-"only_stu"} # VAOPD 必须 only_stu
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"student_p"}

# CoT 开关:True 给 prompt 追加 <think>...</think><answer>X</answer> 思考链指令;
# False 则只要求 <answer>X</answer> 直答。学生 rollout 与教师打分共用同一 prompt,
# 因此该开关同时控制教师模型与学生模型是否使用 CoT。
export USE_COT=${USE_COT:-True}

# 损失与其他
export USE_KL=${USE_KL:-False}
# >>> VA-OPD 格式奖励 loss（加法形式，OPD-main 风格）<<<
#   ENABLE_FORMAT_REWARD: 主开关。True -> naive reward manager 按 <answer>../\boxed
#       产出 per-sample 格式信号，ray_trainer 据此算出 group-normalized 的 format_advantages。
#   FORMAT_LOSS_WEIGHT (lambda): 格式奖励 loss 的权重（dp_actor 用 os.getenv 读取）。总损失为
#       L = L_vaopd + FORMAT_LOSS_WEIGHT * L_format
#       其中 L_format 是对 format_advantages 的裁剪 PG loss。0.0 表示关闭（默认）。
#   注意：要启用格式奖励，两者都要设：ENABLE_FORMAT_REWARD=True 且 FORMAT_LOSS_WEIGHT>0。
export ENABLE_FORMAT_REWARD=${ENABLE_FORMAT_REWARD:-True}
export FORMAT_LOSS_WEIGHT=${FORMAT_LOSS_WEIGHT:-1}

# >>> Teacher GT 注入（缓解过度蒸馏 / 分布错位）<<<
#   INJECT_MODE: none -> 不注入(baseline); gt_contrastive -> 注入 GT + 对比推理提示
#   GT_INJECT_SCHEDULE: 仅在 gt_contrastive 时生效（fsdp_workers 用 os.getenv 读取）
#       constant    -> 固定比例注入，比例 = GT_INJECT_RATIO（如 0.4=每步随机 40% 样本看到 GT）
#       anneal_down -> 从 GT_INJECT_START 线性退火到 GT_INJECT_END（默认 1.0->0.0）
#       anneal_up   -> 从 GT_INJECT_START 线性上升到 GT_INJECT_END（默认 0.0->1.0）；均 GT_INJECT_ANNEAL_STEPS 步内完成
#   默认 gt_contrastive + constant + ratio=1.0（全量注入，等价原行为）。
export INJECT_MODE=${INJECT_MODE:-gt_contrastive}
export GT_INJECT_SCHEDULE=${GT_INJECT_SCHEDULE:-constant}
export GT_INJECT_RATIO=${GT_INJECT_RATIO:-1.0}
export GT_INJECT_START=${GT_INJECT_START:-1.0}
export GT_INJECT_END=${GT_INJECT_END:-0.0}
export GT_INJECT_ANNEAL_STEPS=${GT_INJECT_ANNEAL_STEPS:-84}

export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export IS_PLOT=${IS_PLOT:-True}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"}

# ==========================================
# 4. 数据集与模型路径
# ==========================================
export TRAIN_DATASET=/mnt/yunhang/Omnimedvqa_1k/Cross-modality_generalization/train/OmniMedVQA_CTComputed_Tomography_train.parquet
export TRAIN_DATASET_NAME=CTComputed_Tomography
export TEST_DATASET='["/mnt/yunhang/Omnimedvqa_1k/Cross-modality_generalization/test/OmniMedVQA_CTComputed_Tomography_test.parquet"]'

export ACTOR_MODEL_PATH=/mnt/yunhang/model/Qwen3-VL-2B-Instruct
export ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")

export REWARD_MODEL_PATH=/mnt/yunhang/model/Qwen3-VL-4B-Instruct
export REWARD_MODEL_NAME=$(basename "$REWARD_MODEL_PATH")

export PROJECT_PATH=checkpoint
export PARALLEL_SIZE=1
# 注入信息 tag：便于区分不同 GT 注入策略的 run
if [ "$INJECT_MODE" = "gt_contrastive" ]; then
    case "$GT_INJECT_SCHEDULE" in
        anneal_down|anneal_up)
            export INJECT_TAG="-inj_${GT_INJECT_SCHEDULE}_${GT_INJECT_START}to${GT_INJECT_END}_${GT_INJECT_ANNEAL_STEPS}" ;;
        *)
            export INJECT_TAG="-inj_const_${GT_INJECT_RATIO}" ;;
    esac
else
    export INJECT_TAG="-inj_none"
fi
export RUN_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
export RUN_NAME=${ADV_ESTIMATOR}_${TRAIN_DATASET_NAME}_${ACTOR_MODEL_NAME}_${REWARD_MODEL_NAME}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-pv_${VAOPD_PV}-lambda_${VAOPD_LAMBDA}-cot_${USE_COT}${INJECT_TAG}-${RUN_TIMESTAMP}
export CKPT_PATH=${PROJECT_PATH}/${RUN_NAME}
export SWANLAB_LOG_DIR=${PROJECT_PATH}/swanlab_log
export EXPERIMENT_NAME=${RUN_NAME}
# ==========================================
# 4. 继续训练
# ==========================================
# export RESUME_FROM="${RESUME_FROM:-/mnt/yunhang/VA-OPD/checkpoint/VAOPD_geometry3k_Qwen3-VL-2B-Instruct_Qwen3-VL-4B-Instruct_2048-T_1.0-Tch_1.0-n_8-mbs_128-topk_16-pv_0.2-lambda_0.5-cot_False-2026-06-29_23-42-17}"
# if [ -n "$RESUME_FROM" ]; then
#     export CKPT_PATH="$RESUME_FROM"
#     export EXPERIMENT_NAME=$(basename "$RESUME_FROM")
#     echo "[续训] default_local_dir = $CKPT_PATH"
#     echo "[续训] resume_mode=auto 将从 latest_checkpointed_iteration.txt 的最新 step 继续"
# fi

# ==========================================
# 5. 动态参数
# ==========================================
KL_ARGS=""
if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi

LR_SCHEDULER=${LR_SCHEDULER:-constant}
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS="actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_RATIO:-0.03} \
    actor_rollout_ref.actor.optim.min_lr_ratio=${MIN_LR_RATIO:-0.1} \
    actor_rollout_ref.actor.optim.num_cycles=0.5"
else
    # constant_with_warmup
    LR_ARGS="actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_RATIO:-0.03}"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=12888
VLLM_MAX_NUM_BATCHED_TOKENS=$MAX_MODEL_LEN
echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
echo "VLLM_MAX_NUM_BATCHED_TOKENS: $VLLM_MAX_NUM_BATCHED_TOKENS"

# ==========================================
# 6. 启动 Ray + 训练
# ==========================================
# 显式指定 GPU 数量，避免 Ray 在 ROCm 上探测不到
# ray stop --force
# sleep 5
ray start --head --num-gpus=4 --port=6376 --temp-dir=/tmp/ray_ai
sleep 5

# Ray status 自检（只看本端口）
ray status --address="$RAY_ADDRESS"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=True \
    data.seed=42 \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.image_key=images \
    +data.vaopd_degrade_image=True \
    +data.vaopd_degrade_area_ratio=0.1 \
    +data.use_cot=$USE_COT \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    $LR_ARGS \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    $KL_ARGS \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    +reward_model.inject_mode=$INJECT_MODE \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=4 \
    custom_reward_function.path="verl/verl/utils/reward_score/medical_vqa/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    trainer.log_val_generations=2 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=validation_log/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT


# ==========================================
# 7. 收尾：只停本任务(6376)的 Ray，不碰 Vision-OPD(6380)
# ==========================================
ray stop --address="127.0.0.1:6376" 2>/dev/null || true

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
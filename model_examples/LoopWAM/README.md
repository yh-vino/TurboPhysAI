# LoopWAM 应用说明

本文说明如何基于官方 LoopWAM 仓库应用 TurboPhysAI 的 HCU 训练优化。
TurboPhysAI 的安装方式参见[安装指南](../../docs/zh/get_started/installation.md)。
下文命令在已安装 TurboPhysAI 和 LoopWAM HCU 依赖的环境中执行；模型源码、数据和权重需另行准备。

## 1. 模型简介

[LoopWAM](https://github.com/loop-wam/LoopWAM) 是面向机器人任务的世界模型与动作联合模型，
以 Wan2.2-TI2V-5B 为视频基座，通过 Mixture-of-Transformers 结构融合视频专家与动作专家，
联合学习未来视频和动作序列。本文使用 RoboTwin 2.0 专家数据及 LoopWAM、π0.5、X-VLA rollout 的混合训练配置。

## 2. 优化接入基线

TurboPhysAI 的 LoopWAM 优化基于官方仓库 commit
`0bdcad41fcf274bffadad6e9af4f2aadfc4cde61` 接入。
优化接入基线和模型验证状态见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备模型源码

```bash
cd /workspace
mkdir -p model
git clone https://github.com/loop-wam/LoopWAM.git model/LoopWAM
cd model/LoopWAM
git checkout --detach 0bdcad41fcf274bffadad6e9af4f2aadfc4cde61
mkdir -p data checkpoints runs
```

也可以将准备好的官方 LoopWAM 仓库放入宿主机工作目录的 `model/LoopWAM`。
下文模型准备和训练命令均在 `/workspace/model/LoopWAM` 执行。

## 4. 安装模型依赖与准备运行配置

在已准备好的 HCU Python 环境中安装模型包：

```bash
cd /workspace/model/LoopWAM
python -m pip install -e . --no-deps
```

接入环境使用 DTK 26.04、Torch 2.10、Accelerate 1.12.0，以及适配该 Torch 的 torchcodec/FFmpeg。
通信优化支持 DeepSpeed **0.18.4 或 0.19.2**，注意力优化需要 DTK 厂商版
`flash_attn_with_mask_func`。上游 `pyproject.toml` 固定的 CUDA Torch 和 DeepSpeed 0.18.5
不适用于本示例，安装模型包时保留 HCU 环境中已准备好的依赖。

将本示例的 Accelerate 和 DeepSpeed 配置复制到工作目录。
以下复制命令从 TurboPhysAI 源码或发行包目录执行：

```bash
mkdir -p /workspace/configs/loopwam
cp model_examples/LoopWAM/accelerate.yaml \
  model_examples/LoopWAM/deepspeed_zero1.json \
  /workspace/configs/loopwam/
```

容器部署时，将这些文件放入宿主机工作目录的 `configs/loopwam/`，
使其在容器内对应 `/workspace/configs/loopwam/`。
TurboPhysAI 的 OptimizationConfig 和 RuntimeConfig 随组件安装，后续通过 `--model loopwam`
自动选择，无需指定 Python 包的安装目录。

## 5. 准备 RoboTwin 数据

按照[官方数据准备说明](https://github.com/loop-wam/LoopWAM/blob/0bdcad41fcf274bffadad6e9af4f2aadfc4cde61/README.md#dataset)
准备 RoboTwin 2.0 专家演示，并转换成训练使用的 stitched LeRobot 格式。
三类 rollout 数据可通过官方脚本下载并解压到 `data/`：

```bash
cd /workspace/model/LoopWAM
bash scripts/prepare_data.sh robotwin
```

训练配置使用以下目录，数据文件、视频和元数据需要完整：

```text
data/
├── robotwin2_0_stitched/
│   ├── meta/
│   ├── data/
│   ├── videos/
│   └── dataset_stats.json
└── robotwin2_0-ours/
    └── lerobot_format_new/
        ├── loopwam_processed_stitched/
        ├── pi05_processed_stitched/
        └── xvla_processed_stitched/
```

`dataset_stats.json` 使用专家数据的归一化统计。每个 rollout 目录也应包含
`meta/`、`data/` 和 `videos/`。TurboPhysAI 不随组件分发这些数据。

## 6. 准备预训练权重与文本缓存

训练需要 Wan2.2-TI2V-5B 的视频 DiT、VAE，以及插值得到的 ActionDiT 骨干。
按[官方预训练权重说明](https://github.com/loop-wam/LoopWAM/blob/0bdcad41fcf274bffadad6e9af4f2aadfc4cde61/README.md#pretrained-weights)
下载并生成权重：

```bash
cd /workspace/model/LoopWAM
bash scripts/download_pretrained.sh
ls -lh checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

该脚本将 Wan 权重放在 `checkpoints/` 下，并生成上述 ActionDiT 文件。
数据和权重准备完成后，为训练任务预计算文本 embedding：

```bash
cd /workspace/model/LoopWAM
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

turbo-physai run --model loopwam \
  python scripts/precompute_text_embeds.py \
    task=robotwin_quality_score_3low_1cam_stitched_384_1e-4_action_weighted
```

已准备好的缓存也可直接放到模型配置指定的位置：

```text
data/robotwin2_0-ours/text_embeds_cache/
├── robotwin2_0_quality_score/
├── loopwam_processed_new_quality_score/
├── pi05_processed_quality_score/
└── xvla_processed_quality_score/
```

训练配置不加载文本编码器，启动训练前必须准备完整的文本缓存。

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的
[OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和
[RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，在训练进程启动时应用 LoopWAM 优化。
RuntimeConfig 包含 MIOpen、硬件队列和 NUMA 设置。

### 7.1 单机八卡训练

激活已准备好的 LoopWAM HCU 环境后执行：

```bash
cd /workspace/model/LoopWAM
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_DATASETS_CACHE=/workspace/cache/loopwam-arrow

turbo-physai run \
  --model loopwam \
  accelerate launch \
    --config_file /workspace/configs/loopwam/accelerate.yaml \
    --deepspeed_config_file /workspace/configs/loopwam/deepspeed_zero1.json \
    --num_processes 8 \
    scripts/train.py \
      task=robotwin_quality_score_3low_1cam_stitched_384_1e-4_action_weighted \
      model.video_cross_attend_proprio=false \
      data.train.num_frames=65 data.val.num_frames=65 \
      data.train.action_video_freq_ratio=8 data.val.action_video_freq_ratio=8 \
      batch_size=8 num_workers=2 \
      output_dir=./runs/turbo_physai_8card
```

`--num_processes` 指定训练进程总数。示例使用每卡 batch 8、bf16 和 DeepSpeed ZeRO-1，
训练周期、学习率、评估和保存频率沿用上游 task 配置。
首次检查训练链路时，可在命令末尾追加 `max_steps=300 eval_every=0 save_every=0`；
训练结束时仍会保存 checkpoint。

### 7.2 多机运行要求

多机时在每个节点使用相同任务配置启动，并设置 Accelerate 的 `--num_machines`、
`--machine_rank`、`--main_process_ip` 和 `--main_process_port`；
`--num_processes` 填写所有节点的进程总数。
跨节点 RDMA 插件、拓扑文件和网卡需按部署环境配置。
初始化屏障另使用 `MASTER_PORT + 1`，可通过 `FASTWAM_PRE_ACCELERATOR_STORE_PORT` 覆盖。

随包配置按每 4 节点分组；超过 4 节点时节点数须能被 4 整除。
节点分组 checkpoint 的恢复要求总卡数和分组布局一致。
需要自定义优化或运行环境时，通过 `--optimization-config` 和 `--runtime-config` 指定文件。

优化应用状态的确认方法见
[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
已执行的双卡算子、通信和合成数据训练检查见[接入验证记录](VALIDATION.md)；
完整 LoopWAM 的精度、性能、显存和长稳仍需在目标训练配置上验收。

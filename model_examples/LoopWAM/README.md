# LoopWAM HCU 接入

本目录提供 LoopWAM 训练优化的启动脚本和部署配置。优化实现位于
[`turbo_physai/optimizations/models/loopwam/`](../../turbo_physai/optimizations/models/loopwam/)。

本次已执行的检查和未覆盖范围见[接入验证记录](VALIDATION.md)。

## 基线与环境

- 模型仓库：[loop-wam/LoopWAM](https://github.com/loop-wam/LoopWAM)，
  接入基线 `0bdcad41fcf274bffadad6e9af4f2aadfc4cde61`。
- 优化参考：基于上述基线的 HCU 交付实现
  `0a610c7093e78059720e9088294481b40aa5d085`。
- 目标环境：BW1000、DTK 26.04、Torch 2.10、Accelerate 1.12.0。
- ZeRO-1 通信优化接受 **DeepSpeed 0.18.4 或 0.19.2 的确切版本**，并校验所依赖方法的源码。
  上游依赖中的 DeepSpeed 0.18.5 不适用于这套通信优化。
- 带 mask 的注意力依赖 DTK 厂商版 `flash_attn_with_mask_func`。
  不支持该接口时直接报错。
- 通信仅覆盖纯数据并行 ZeRO-1，无 CPU offload、MoE、ZenFlow 或序列并行。

接入验证采用算子与小模型测试。完整 LoopWAM 的精度、稳态性能、显存和长稳结果
需在目标训练配置上重新验证；交付版已有性能结果不代表本次 TurboPhysAI 接入后的实测结果。

## 优化选择

| Group ID | 优化内容 | 报告 OPT |
| --- | --- | --- |
| `loopwam.dataset_columns` | HF Column 读取适配，去掉未使用的时间戳列物化 | 数据适配 |
| `loopwam.dataset_cache` | 加锁创建共享 Arrow 缓存，其他进程直接映射文件 | 数据适配 |
| `loopwam.initialization_barrier` | 数据加载后、Accelerator 初始化前同步各 rank | 启动适配 |
| `loopwam.vae_channels_last` | VAE CausalConv3d 使用 channels_last_3d | OPT-03 |
| `loopwam.masked_flash_attention` | MoT 布尔 mask 注意力使用厂商融合实现 | OPT-04 |
| `loopwam.rmsnorm` | fp32 归一化后按原顺序转换 dtype、乘权重 | OPT-05 |
| `loopwam.zero1_deferred_gather` | 参数收集与下一步前向重叠；读取前等待 | OPT-08、OPT-13 |
| `loopwam.zero1_grad_overlap` | 梯度规约与反向重叠，保护双缓冲复用 | OPT-09 |
| `loopwam.zero1_node_sharding` | 按节点组分片优化器状态，保持全局梯度平均 | OPT-11 |
| `loopwam.fused_metrics` | 合并每步标量汇总，保留训练调度和保存时机 | OPT-12 |

这些 Group 均属于 LoopWAM 模型专用优化。NUMA、MIOpen 算法搜索与布局建议、硬件队列数
由 RuntimeConfig 管理（OPT-02、OPT-06、OPT-07）。SHCA 插件、拓扑文件和网卡配置（OPT-01）
由集群启动环境提供。OPT-10 对应恢复上游行为，干净基线已具备该行为。

Recipe 默认每 4 节点一组；超过 4 节点时节点数须能被 4 整除。
修改 `configs/recipe.yaml` 的 `shard_nodes` 后重新生成配置。
`skip_bit16_rebind` 是延迟参数收集 Group 的选项，默认开启。
梯度重叠可单独启用，但交付参考中多机收益依赖与参数收集重叠同时启用。

## 准备源码、数据和权重

在已安装 TurboPhysAI 和模型依赖的 HCU 环境中准备干净模型源码：

```bash
git clone https://github.com/loop-wam/LoopWAM.git /workspace/LoopWAM
git -C /workspace/LoopWAM checkout --detach \
  0bdcad41fcf274bffadad6e9af4f2aadfc4cde61
```

数据和权重按模型仓库的说明准备。启动脚本使用 RoboTwin 专家数据及三类 rollout 数据，
路径结构见 [`configs/data/robotwin_mixed_quality_score_3low_stitched_ssd.yaml`](configs/data/robotwin_mixed_quality_score_3low_stitched_ssd.yaml)。
该配置通过 Hydra 的附加搜索目录加载，不写入模型仓库。共享 Arrow 缓存要求 Linux `flock`
和所有 rank 可访问的 `HF_DATASETS_CACHE`。更换数据内容后应使用新的缓存目录。

## 启动训练

以下命令从 TurboPhysAI 源码目录执行。运行前激活目标 HCU 环境，安装模型所需的
torchcodec/FFmpeg、datasets、flash-attn 和对应版本的 DeepSpeed。避免按上游 CUDA 依赖
覆盖已准备好的 HCU 环境。

```bash
export LOOPWAM_REPO=/workspace/LoopWAM
export LOOPWAM_DATA_ROOT=/datasets/loopwam
export LOOPWAM_TEXT_CACHE_ROOT=/datasets/loopwam/text_embeds_cache
export HF_DATASETS_CACHE=/workspace/cache/loopwam-arrow
export DIFFSYNTH_MODEL_BASE_PATH=/models/loopwam
export OUTPUT_DIR=/workspace/runs/loopwam-smoke
bash model_examples/LoopWAM/train_hcu.sh
```

默认每节点 8 卡、每卡 batch 8、300 步、关闭评估与周期性保存，结束时仍按上游逻辑保存
checkpoint。指标汇总 Group 默认跳过前 50 步统计；需采集纯训练性能时，应同时记录各 rank
步时、分位数和显存。通过脚本末尾的 Hydra 参数调整训练配置，例如：

```bash
bash model_examples/LoopWAM/train_hcu.sh \
  max_steps=5000 eval_every=500 save_every=2500
```

多机时，每个节点启动一次脚本，并提供相同 `NNODES`、`MASTER_ADDR`、`MASTER_PORT`、
共享输出目录，以及各自的 `MACHINE_RANK`。作业调度器需为每节点提供足够的 CPU 和 8 张 HCU。
SCNet 等环境还需在启动前加载 RDMA 插件，设置部署环境对应的 `NCCL_TOPO_FILE`、
`NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME`。初始化屏障使用 `MASTER_PORT + 1`，可通过
`FASTWAM_PRE_ACCELERATOR_STORE_PORT` 改为其他未占用端口。

启用 `--model loopwam` 时，Runner 自动选择模型 OptimizationConfig 和 RuntimeConfig。
优化是否应用以启动日志中的 OptimizationReport 为准；安装成功之后仍可能出现训练期错误。
旧的 `LOOPWAM_*` 优化开关不参与 Group 选择，避免在已打交付补丁的模型仓库上重复安装。

## 生成与检查配置

从 TurboPhysAI 源码目录执行，模型仓库须干净且 HEAD 与 `--commit` 一致：

```bash
python -m turbo_physai.cli optimization generate \
  --recipe turbo_physai/optimizations/models/loopwam/configs/recipe.yaml \
  --repo /workspace/LoopWAM \
  --commit 0bdcad41fcf274bffadad6e9af4f2aadfc4cde61 \
  --output turbo_physai/optimizations/models/loopwam/configs/optimization.yaml \
  --force
python -m turbo_physai.cli optimization validate \
  turbo_physai/optimizations/models/loopwam/configs/optimization.yaml
python -m turbo_physai.cli optimization check \
  turbo_physai/optimizations/models/loopwam/configs/optimization.yaml \
  --repo /workspace/LoopWAM
```

生成配置包括模型目标和 Accelerate `prepare` 的源码证据；调整依赖版本后须重新验证并生成。
运行时不通过强制启用跳过不匹配证据。

## 验证

CPU 契约测试：

```bash
python -m pytest -q test/optimizations/test_loopwam.py
```

在 HCU 环境中从干净的 LoopWAM 基线目录运行双卡小模型测试。将下面的
`TPA_SOURCE` 改为 TurboPhysAI 的源码目录：

```bash
export TPA_SOURCE=/workspace/TurboPhysAI
export PYTHONPATH="$TPA_SOURCE:$LOOPWAM_REPO/src:${PYTHONPATH:-}"
export GPU_MAX_HW_QUEUES=8
cd "$LOOPWAM_REPO"
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$TPA_SOURCE/test/optimizations/loopwam_hcu_smoke.py" \
  --output /workspace/runs/loopwam-hcu-smoke
```

小模型测试保存各 rank 的 OptimizationReport，并检查真实厂商算子前向/梯度、通信补丁、
优化器更新、跨副本权重一致性和固定布局 checkpoint 恢复。测试使用单机双卡模拟节点组，
不能替代真实跨节点通信和完整模型训练验证。节点分组 checkpoint 恢复要求总卡数和
分组布局一致，转换为合并 FP32 权重的工具链尚需单独验证。

还可以使用真实 `Wan22Trainer` 搭配合成小模型，检查初始化屏障、
`Accelerator.prepare`、训练循环、周期保存和断点恢复的衔接：

```bash
accelerate launch \
  --config_file "$TPA_SOURCE/model_examples/LoopWAM/accelerate.yaml" \
  --deepspeed_config_file "$TPA_SOURCE/model_examples/LoopWAM/deepspeed_zero1.json" \
  --num_processes 2 --num_machines 1 --machine_rank 0 \
  "$TPA_SOURCE/test/optimizations/loopwam_trainer_smoke.py" \
  --output /workspace/runs/loopwam-trainer-smoke
```

该脚本直接调用 `apply()`，因此启动时不再包裹 `turbo-physai run`。
合成数据测试不读取 RoboTwin 数据或完整 LoopWAM 权重。

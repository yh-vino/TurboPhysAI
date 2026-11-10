# LoopWAM 优化开发说明

本目录维护 LoopWAM 模型专用的优化实现、Catalog、Recipe 和生成后的配置。
训练用户参见 [LoopWAM 应用说明](../../../../model_examples/LoopWAM/README.md)，
已执行检查和验证边界参见[接入验证记录](../../../../model_examples/LoopWAM/VALIDATION.md)。

优化参考为 HCU 交付实现 `0a610c7093e78059720e9088294481b40aa5d085`，
模型基线为 `0bdcad41fcf274bffadad6e9af4f2aadfc4cde61`。
以下开发命令从 TurboPhysAI 源码目录执行。

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
修改本目录的 `configs/recipe.yaml` 的 `shard_nodes` 后重新生成配置。
`skip_bit16_rebind` 是延迟参数收集 Group 的选项，默认开启。
梯度重叠可单独启用，但交付参考中多机收益依赖与参数收集重叠同时启用。

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
export LOOPWAM_REPO=/workspace/model/LoopWAM
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

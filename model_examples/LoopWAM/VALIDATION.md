# LoopWAM 接入验证记录

验证日期：2026-11-10。目标环境为 BW1000、DTK 26.04、Torch 2.10，
使用单机双卡分别检查 DeepSpeed 0.18.4 和 0.19.2。

模型基线：`0bdcad41fcf274bffadad6e9af4f2aadfc4cde61`。
优化参考：`0a610c7093e78059720e9088294481b40aa5d085`。

## 已执行检查

| 检查 | 结果 |
| --- | --- |
| 干净上游基线上的配置生成与 `optimization check` | 通过 |
| LoopWAM 契约测试 | 31 项通过 |
| 框架纯 Python 回归 | 137 项通过，1 项跳过 |
| Accelerate/Hydra 启动配置解析 | 通过，未加载真实数据和模型权重 |
| DeepSpeed 0.18.4 双卡算子与通信 | 通过 |
| DeepSpeed 0.19.2 双卡算子与通信 | 通过 |
| DeepSpeed 0.18.4 真实 Trainer + 合成小模型 | 通过，完成 5 次更新并验证恢复 |
| DeepSpeed 0.19.2 真实 Trainer + 合成小模型 | 通过，完成 5 次更新并验证恢复 |
| 文档链接、脚本语法、许可证头与差异空白检查 | 通过 |

契约测试命令：

```bash
python -m pytest -q test/optimizations/test_loopwam.py
```

框架回归命令（排除需要本地 Torch 的 BEVFusion 测试文件）：

```bash
python -m pytest -q test/engine --ignore=test/engine/test_bevfusion.py
```

HCU 测试入口为
[`loopwam_hcu_smoke.py`](../../test/optimizations/loopwam_hcu_smoke.py) 和
[`loopwam_trainer_smoke.py`](../../test/optimizations/loopwam_trainer_smoke.py)，
启动命令见[接入说明](README.md#验证)。

## 验证内容

- 两个版本中，10 个 Group 均报告为 `applied`，未使用强制启用。
- 在真实 HCU 上检查 VAE 卷积、RMSNorm 和带 mask 的厂商 FlashAttention。
  RMSNorm 与注意力同时检查所需梯度；VAE 卷积在模型中为冻结模块，
  其梯度契约由 CPU 单测补充覆盖。
- 双卡小模型比较原生 ZeRO-1 与延迟参数收集、梯度重叠、节点组分片组合。
  两个版本的最大权重绝对差均为 0，跨副本权重完全一致；
  保存后加载并重做下一次更新，权重完全一致。
- 使用真实上游 `Wan22Trainer` 训练合成小模型，覆盖初始化屏障、
  `Accelerator.prepare`、通信安装、指标汇总、周期保存、最终保存及恢复。
  从第 4 步恢复后重做第 5 步，两个版本的权重均与恢复前完全一致。
- 模型基线保持干净；优化实现、配置和测试均独立于模型源码维护。

## 未覆盖的验收范围

双卡测试中的节点组通过逻辑拓扑模拟，不代表真实跨节点网络验证。
本次未加载完整 LoopWAM 权重、RoboTwin 视频或文本 embedding，
未执行完整模型精度对齐、稳态性能对比、显存峰值采集和长时间稳定性测试。

原交付报告中的性能数据不可直接作为本次 TurboPhysAI 接入的性能结论。
完整模型及目标集群上的验收需使用相同数据、超参和环境另行执行。

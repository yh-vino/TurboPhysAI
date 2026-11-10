# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Run with torchrun --nproc-per-node=2 on HCU; no model weights/data needed.

Tests real vendor kernels and ZeRO-1 arithmetic/checkpointing. Single-rank shard
replicas emulate two nodes within the allocation; this is not a multi-node
network, full-model precision, stability, or performance acceptance test.
"""

import argparse
import gc
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F


def kernels():
    from fastwam.models.wan22.wan_video_dit import RMSNorm
    from fastwam.models.wan22.wan_video_vae import CausalConv3d
    from fastwam.models.wan22.mot import MoT
    torch.manual_seed(91)
    x = torch.randn(2, 16, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    norm = RMSNorm(256).cuda()
    out = norm(x)
    ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)).to(x.dtype) * norm.weight
    torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
    a = torch.autograd.grad(out.sum(), (x, norm.weight), retain_graph=True)
    b = torch.autograd.grad(ref.sum(), (x, norm.weight))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0.03, atol=0.02)
    conv = CausalConv3d(2, 3, 3, padding=1).cuda().bfloat16()
    video = torch.randn(2, 2, 3, 8, 8, device='cuda', dtype=torch.bfloat16)
    ref = F.conv3d(F.pad(video, conv._padding), conv.weight, conv.bias)
    torch.testing.assert_close(conv(video), ref, rtol=0.03, atol=0.02)
    mot = MoT.__new__(MoT)
    torch.nn.Module.__init__(mot)
    mot.num_heads = 2
    mot.mot_checkpoint_mixed_attn = True
    q, k, v = [torch.randn(2, 16, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    mask = torch.ones(16, 16, device='cuda', dtype=torch.bool).tril()
    out = mot._mixed_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(
        q.reshape(2, 16, 2, 128).transpose(1, 2), k.reshape(2, 16, 2, 128).transpose(1, 2),
        v.reshape(2, 16, 2, 128).transpose(1, 2), attn_mask=mask).transpose(1, 2).reshape_as(q)
    torch.testing.assert_close(out, ref, rtol=0.03, atol=0.03)
    a = torch.autograd.grad(out.float().square().mean(), (q, k, v), retain_graph=True)
    b = torch.autograd.grad(ref.float().square().mean(), (q, k, v))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0.05, atol=1e-4)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(6)])
        self.mot = SimpleNamespace(expert_order=['tiny'],
            mixtures={'tiny': SimpleNamespace(blocks=self.blocks)}, num_layers=len(self.blocks))

    def training_loss(self, x):
        manager = getattr(self.mot, '_turbophysai_deferred_gather', None)
        for i, block in enumerate(self.blocks):
            if manager is not None:
                manager.wait(i + 1)
            x = torch.tanh(block(x))
        return x.float().square().mean()

    def forward(self, x):
        return self.training_loss(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    import turbo_physai
    from turbo_physai.engine.contracts import to_primitive
    report = turbo_physai.apply(model='loopwam')
    (args.output / f'report_rank{os.environ["RANK"]}.json').write_text(json.dumps(to_primitive(report), indent=2))
    assert len(report.execution) == 10
    assert all(g.status.value == 'applied' for g in report.execution), [(g.group_id, g.status.value) for g in report.execution]
    from fastwam.trainer import Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    assert hasattr(accelerator, "_turbophysai_initialization_store")
    kernels()
    import deepspeed
    from turbo_physai.optimizations.models.loopwam import deferred_gather, grad_overlap, zero1_sharding
    deepspeed.init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    os.environ['FASTWAM_GRAD_OVERLAP_CHECK_STEPS'] = '2,5'
    os.environ['FASTWAM_DEFERRED_GATHER_CHECK_STEPS'] = '2,5'
    config = {'train_micro_batch_size_per_gpu': 2, 'gradient_accumulation_steps': 1,
              'train_batch_size': 2 * world, 'bf16': {'enabled': True},
              'communication_data_type': 'fp32', 'gradient_clipping': 0.8,
              'zero_allow_untested_optimizer': True, 'steps_per_print': 1000,
              'zero_optimization': {'stage': 1, 'overlap_comm': True,
                  'contiguous_gradients': True, 'reduce_bucket_size': 512, 'allgather_bucket_size': 512}}

    def initialize(sharded):
        torch.manual_seed(101)
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.95), weight_decay=0.01)
        # Exercise replica math with one rank per logical node, on two HCU cards.
        with patch.object(zero1_sharding, '_node_groups', return_value=[[r] for r in range(world)]):
            with zero1_sharding.prepare_context(1 if sharded else 0):
                engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=copy.deepcopy(config))
        zero1_sharding.install_checkpointing(engine)
        if sharded:
            deferred_gather.install(engine, model, skip_rebind=True)
            grad_overlap.install(engine)
        return engine

    def step(engine, n):
        gen = torch.Generator(device='cuda').manual_seed(9000 + 100 * rank + n)
        x = torch.randn(2, 16, generator=gen, device='cuda', dtype=torch.bfloat16)
        loss = engine(x)
        engine.backward(loss)
        engine.step()
        manager = getattr(engine.module.mot, '_turbophysai_deferred_gather', None)
        if manager is not None:
            manager.finish()
        weights = torch.cat([p.detach().float().flatten() for p in engine.module.parameters()])
        peers = [torch.empty_like(weights) for _ in range(world)]
        dist.all_gather(peers, weights)
        assert all(torch.equal(weights, other) for other in peers)
        return weights.cpu(), float(engine.get_global_grad_norm())

    baseline = initialize(False)
    reference = [step(baseline, n) for n in range(1, 6)]
    baseline.destroy()
    del baseline
    dist.barrier()
    engine = initialize(True)
    errors = []
    for n in range(1, 6):
        weights, norm = step(engine, n)
        ref_weights, ref_norm = reference[n - 1]
        errors.append((weights - ref_weights).abs().max().item())
        assert errors[-1] < 0.0021, errors
        assert abs(norm - ref_norm) < 0.01, (norm, ref_norm)
        if n == 3:
            engine.save_checkpoint(str(args.output / 'checkpoint'), tag='step3')
        if n == 4:
            path, _ = engine.load_checkpoint(str(args.output / 'checkpoint'), tag='step3')
            assert path
            repeated, _ = step(engine, 4)
            assert torch.equal(weights, repeated)
    if rank == 0:
        result = {'deepspeed': deepspeed.__version__, 'world': world,
                  'max_weight_difference': max(errors), 'replicas_exact': True,
                  'save_resume_repeat_exact': True, 'vendor_kernels': 'passed',
                  'scope': 'two HCU cards, synthetic node layout, tiny model'}
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        print('LOOPWAM_HCU_SMOKE_OK', json.dumps(result), flush=True)
    dist.barrier()
    engine.destroy()
    del engine
    gc.collect()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

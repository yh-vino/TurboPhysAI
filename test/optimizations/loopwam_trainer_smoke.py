# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the real upstream trainer with a tiny model and synthetic data.

Launch with accelerate and the LoopWAM DeepSpeed config, on two HCU cards.
No LoopWAM weights, videos, text embeddings, or real dataset are loaded.
"""

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import turbo_physai
    from turbo_physai.engine.contracts import to_primitive
    report = turbo_physai.apply(model='loopwam')
    assert len(report.execution) == 10
    assert all(g.status.value == 'applied' for g in report.execution)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f'trainer_report_rank{os.environ["RANK"]}.json').write_text(
        json.dumps(to_primitive(report), indent=2))
    from fastwam.trainer import Wan22Trainer
    from omegaconf import OmegaConf

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = torch.nn.ModuleList([torch.nn.Linear(16, 16) for _ in range(4)])
            self.mot = SimpleNamespace(expert_order=['tiny'],
                mixtures={'tiny': SimpleNamespace(blocks=self.dit)}, num_layers=len(self.dit))

        def training_loss(self, sample):
            manager = getattr(self.mot, '_turbophysai_deferred_gather', None)
            x = sample.to(device=self.dit[0].weight.device, dtype=self.dit[0].weight.dtype)
            for i, block in enumerate(self.dit):
                if manager is not None:
                    manager.wait(i + 1)
                x = torch.tanh(block(x))
            loss = x.float().square().mean()
            return loss, {'tiny': loss.detach()}

        def save_checkpoint(self, path, optimizer=None, step=None):
            torch.save({'state_dict': self.state_dict(), 'step': step}, path)

    cfg = OmegaConf.create({
        'output_dir': str(args.output), 'learning_rate': 0.002, 'weight_decay': 0.01,
        'batch_size': 2, 'num_workers': 0, 'num_epochs': 1, 'max_steps': 4,
        'log_every': 1, 'save_every': 2, 'eval_every': 0, 'eval_num_inference_steps': 1,
        'gradient_accumulation_steps': 1, 'max_grad_norm': 1., 'seed': 37,
        'resume': None, 'mixed_precision': 'bf16', 'lr_scheduler_type': 'constant',
        'wandb': {'enabled': False},
    })
    torch.manual_seed(101)
    dataset = [torch.ones(16) for _ in range(32)]
    trainer = Wan22Trainer(Model(), dataset, cfg=cfg)
    assert hasattr(trainer.accelerator, '_turbophysai_initialization_store')
    trainer.train()
    assert trainer.global_step == 4
    model = trainer.accelerator.unwrap_model(trainer.model)
    assert model.mot._turbophysai_deferred_gather.step_count == 4
    trainer.max_steps = 5
    trainer.train()
    expected = [p.detach().clone() for p in model.parameters()]
    trainer.load_training_state(str(args.output / 'checkpoints/state/step_000004'))
    assert trainer.global_step == 4
    trainer.train()
    assert trainer.global_step == 5
    assert all(torch.equal(a, b) for a, b in zip(expected, model.parameters()))
    if trainer.accelerator.is_main_process:
        result = {'steps': 5, 'trainer': 'upstream Wan22Trainer', 'synthetic_model': True,
                  'resume_next_update_exact': True, 'groups_applied': len(report.execution)}
        (args.output / 'trainer_result.json').write_text(json.dumps(result, indent=2))
        print('LOOPWAM_TRAINER_SMOKE_OK', json.dumps(result), flush=True)
    trainer.accelerator.wait_for_everyone()
    accelerator = trainer.accelerator
    accelerator.free_memory()
    del trainer, model
    gc.collect()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

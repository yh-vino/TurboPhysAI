# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Training lifecycle adapters; wrapper construction has no runtime side effects."""

import functools
import inspect
import logging
import time

logger = logging.getLogger(__name__)


def accelerator_barrier_wrapper(original, options):
    timeout = options.get("timeout_seconds", 3600)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout_seconds must be a positive integer")

    @functools.wraps(original)
    def initialize(*args, **kwargs):
        import os
        from datetime import timedelta

        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world > 1:
            import torch.distributed as dist

            rank = int(os.environ["RANK"])
            port = int(os.environ.get("FASTWAM_PRE_ACCELERATOR_STORE_PORT",
                                      str(int(os.environ.get("MASTER_PORT", "29500")) + 1)))
            if not 1 <= port <= 65535:
                raise ValueError("Invalid LoopWAM pre-Accelerator TCPStore port")
            store = dist.TCPStore(os.environ["MASTER_ADDR"], port, world, rank == 0,
                                  timeout=timedelta(seconds=timeout))
            store.set(f"ready/{rank}", "1")
            store.wait([f"ready/{i}" for i in range(world)])
            # Every client observes rank 0's release before the server is
            # destroyed. Keeping the store on the Accelerator also keeps it
            # alive while slow peers leave this barrier.
            store.set(f"left/{rank}", "1")
            store.wait([f"left/{i}" for i in range(world)])
        accelerator = original(*args, **kwargs)
        if world > 1:
            accelerator._turbophysai_initialization_store = store
        return accelerator

    return initialize


def _model(trainer):
    return trainer.accelerator.unwrap_model(trainer.model)


def _finish(trainer):
    manager = getattr(_model(trainer).mot, "_turbophysai_deferred_gather", None)
    if manager is not None:
        manager.finish()


def deferred_train_mode_wrapper(original, options):
    skip_rebind = options.get("skip_bit16_rebind", False)
    if not isinstance(skip_rebind, bool):
        raise ValueError("skip_bit16_rebind must be a boolean")

    @functools.wraps(original)
    def train_mode(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = _model(self)
        if not hasattr(model.mot, "_turbophysai_deferred_gather"):
            from .deferred_gather import install

            install(self.model, model, skip_rebind=skip_rebind)
        return result

    return train_mode


def layer_wait_wrapper(original, _options):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def attention_io(self, *args, **kwargs):
        manager = getattr(self, "_turbophysai_deferred_gather", None)
        if manager is not None:
            # Waiting before modulation covers direct Parameter reads as well
            # as QKV projections; a block.forward hook would miss these reads.
            block = signature.bind(self, *args, **kwargs).arguments["block"]
            manager.wait(manager.block_stages[id(block)])
        return original(self, *args, **kwargs)

    return attention_io


def finish_gather_wrapper(original, _options):
    @functools.wraps(original)
    def finish_before_read(self, *args, **kwargs):
        _finish(self)
        return original(self, *args, **kwargs)

    return finish_before_read


def gradient_overlap_wrapper(original, _options):
    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        from .grad_overlap import install

        install(self.model)

    return initialize


def sharded_prepare_wrapper(original, options):
    nodes = options.get("shard_nodes", 4)
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
        raise ValueError("LoopWAM shard_nodes must be a positive integer")

    @functools.wraps(original)
    def prepare(self, *args, **kwargs):
        from .zero1_sharding import install_checkpointing, prepare_context

        # This config is specific to the single-model LoopWAM trainer.
        if not args or not hasattr(args[0], "mot"):
            raise RuntimeError("LoopWAM sharding expects accelerator.prepare(model, optimizer, loader, scheduler)")
        with prepare_context(nodes):
            prepared = original(self, *args, **kwargs)
        install_checkpointing(prepared[0])
        return prepared

    return prepare


def gather_metrics(accelerator, loss, grad_norm, loss_dict):
    """One collective and one host transfer, preserving rank-mean semantics."""
    import torch

    keys = list(loss_dict)
    values = [loss.detach().float().reshape(1)]
    values.append(torch.as_tensor(grad_norm, device=loss.device, dtype=torch.float32).reshape(1))
    values.extend(torch.as_tensor(loss_dict[key], device=loss.device, dtype=torch.float32).reshape(1) for key in keys)
    local = torch.cat(values)
    means = accelerator.gather(local).reshape(-1, local.numel()).mean(dim=0).tolist()
    return means[0], means[1], dict(zip(keys, means[2:]))


def fused_metrics_wrapper(original, options):
    warmup = options.get("benchmark_warmup_steps", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("benchmark_warmup_steps must be a nonnegative integer")

    @functools.wraps(original)
    def train(self):
        self._set_dit_only_train_mode()
        if self.max_steps is None:
            raise ValueError("max_steps is required for LoopWAM training")
        try:
            return _train(self, warmup)
        finally:
            _finish(self)

    return train


def _train(trainer, warmup):
    """Keep upstream scheduling, accumulation, evaluation and checkpoint cadence."""
    accelerator = trainer.accelerator
    trainer.run_start_step = trainer.global_step
    trainer.run_start_time = time.perf_counter()
    measured_start = None
    iterator = iter(trainer.train_loader)
    while trainer.global_step < trainer.max_steps:
        try:
            sample = next(iterator)
            trainer.batch_in_epoch += 1
        except StopIteration:
            trainer.epoch += 1
            trainer.batch_in_epoch = 0
            trainer.train_sampler.clear_resume_batch_offset()
            iterator = iter(trainer.train_loader)
            continue

        if measured_start is None and trainer.global_step - trainer.run_start_step >= warmup:
            measured_start = time.perf_counter()
            measured_step = trainer.global_step
        with accelerator.accumulate(trainer.model):
            model = trainer.model if hasattr(trainer.model, "training_loss") else _model(trainer)
            with accelerator.autocast():
                loss, loss_dict = model.training_loss(sample)
            accelerator.backward(loss)
            if not accelerator.sync_gradients:
                continue
            norm = accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
            trainer.optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                trainer.scheduler.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.global_step += 1
            loss_value, norm_value, details = gather_metrics(accelerator, loss, norm, loss_dict)
            learning_rate = float(trainer.optimizer.param_groups[0]["lr"])

            if trainer.log_every > 0 and trainer.global_step % trainer.log_every == 0 and accelerator.is_main_process:
                eta, speed = trainer._estimate_eta()
                logger.info("[train] epoch=%d step=%d/%d loss=%.4f grad_norm=%.4f lr=%.2e "
                            "speed=%.2f step/s eta=%s metrics=%s", trainer.epoch,
                            trainer.global_step, trainer.max_steps, loss_value,
                            norm_value, learning_rate, speed, eta, details)
                payload = {"train/loss": loss_value, "train/grad_norm": norm_value,
                           "train/lr": learning_rate, "performance/steps_per_sec": speed,
                           "performance/samples_per_sec": speed * trainer.batch_size * accelerator.num_processes}
                payload.update({f"train/{key}": value for key, value in details.items()})
                trainer._wandb_log(payload)

            if trainer.eval_every > 0 and trainer.val_dataset is not None and trainer.global_step % trainer.eval_every == 0:
                metrics = trainer.evaluate()
                accelerator.wait_for_everyone()
                if metrics is not None and accelerator.is_main_process:
                    payload = {"eval/val_loss": float(metrics["val_loss"])}
                    payload.update({f"eval/{key}": float(metrics[key]) for key in (
                        "psnr_rg", "ssim_rg", "psnr_rd", "ssim_rd", "psnr_dg", "ssim_dg",
                        "action_l2", "action_l1",
                    ) if key in metrics})
                    trainer._wandb_log(payload)
                    logger.info("[eval] step=%d metrics=%s", trainer.global_step, metrics)
            if trainer.save_every > 0 and trainer.global_step % trainer.save_every == 0:
                trainer.save_checkpoint()

    if measured_start is not None and accelerator.is_main_process:
        duration = time.perf_counter() - measured_start
        count = trainer.global_step - measured_step
        logger.info("LOOPWAM_BENCHMARK measured_steps=%d step_time_seconds=%.6f samples_per_sec=%.3f",
                    count, duration / count, count * trainer.batch_size * accelerator.num_processes / duration)
    trainer.save_checkpoint()

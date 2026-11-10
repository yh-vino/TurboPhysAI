# Copyright (c) Microsoft Corporation.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team. Modified by Hygon: LoopWAM ZeRO-1 scheduling adapters.

"""Overlap ZeRO-1 gradient reduction with backward in the ``loopwam.zero1_grad_overlap`` Group.

DeepSpeed's ``overlap_comm`` reduces each full gradient bucket on
``reduction_stream`` while backward fills the other of two bucket buffers, but
in DeepSpeed 0.18.4 and 0.19.2 it does not overlap anything:

- ``average_tensor`` makes the compute stream wait for the reduction stream
  right after queueing every bucket's reduction, so backward stalls per bucket.
- in 0.18.4 the bucket swap after ``reduce_ipg_grads`` also waits on the
  reduction stream; and ``reduce_ipg_grads`` ends with ``IPGBucket.clear()``,
  which resets ``bucket.index`` to 0, so ``1 - bucket.index`` sends every bucket
  after the first into buffer 1. Without those waits backward would overwrite
  buffer 1 while it is still being reduced (an earlier patch that dropped the
  waits produced wrong gradients this way). 0.19.2 no longer resets the index
  and drops the swap wait, but still relies on the wait in ``average_tensor``.

This module keeps DeepSpeed's reduction code and replaces only the waits:

- ``average_tensor`` no longer makes compute wait; the bucket buffer slice is
  recorded on the reduction stream instead.
- the bucket swap tracks the buffer index itself, records an event after each
  bucket's reduction (which includes ``copy_grads_in_partition``), and makes
  compute wait for that event only before refilling the same buffer.
- the backward epilogue still reduces the last bucket and synchronizes the
  device before the optimizer reads ``averaged_gradients``.

``FASTWAM_GRAD_OVERLAP_CHECK_STEPS=5,20`` captures every rank's local gradient
at those steps, recomputes the average with a separate fp32 AllReduce, and
compares it with this rank's ``averaged_gradients`` partition.

Requires a DeepSpeed config with ``overlap_comm`` and ``contiguous_gradients``
enabled (stage 1, no CPU offload, no MoE, no coalesced gradient reduction). The
DeepSpeed version is identified by hashing the sources of the methods this
module relies on; only DeepSpeed 0.18.4 and 0.19.2 sources are accepted.
"""

import hashlib
import inspect
import os
import textwrap
import types

import torch
import torch.distributed as dist

from logging import getLogger as get_logger
from .zero1_compat import require_supported_version

logger = get_logger(__name__)

# sha256 prefixes of the dedented DeepSpeed sources (as ``inspect.getsource`` returns them)
# this module relies on, per supported DeepSpeed version.
_EXPECTED_SOURCES = {
    "0.18.4": {
        "average_tensor": "5f318f8e84d55a24",
        "reduce_independent_p_g_buckets_and_remove_grads": "97c4bf2cacb33844",
        "independent_gradient_partition_epilogue": "f4b2457a79f6af46",
        "reduce_ipg_grads": "58fdaae0722d0569",
        "setup_buckets": "095880e3f6e183f7",
    },
    "0.19.2": {
        "average_tensor": "5f318f8e84d55a24",
        "reduce_independent_p_g_buckets_and_remove_grads": "b80eac0169efcd92",
        "independent_gradient_partition_epilogue": "177bf3c28362c2ea",
        "reduce_ipg_grads": "c5f6908919419cd1",
        "setup_buckets": "095880e3f6e183f7",
    },
}

# Indented as in the dedented method source (method body at 4 spaces).
_AVERAGE_TENSOR_WAIT = """\
    if self.overlap_comm:
        stream = self.reduction_stream
        if not get_accelerator().resolves_data_dependency():
            stream.wait_stream(get_accelerator().current_stream())
            get_accelerator().current_stream().wait_stream(stream)
"""
_AVERAGE_TENSOR_NO_WAIT = """\
    if self.overlap_comm:
        stream = self.reduction_stream
        if not get_accelerator().resolves_data_dependency():
            stream.wait_stream(get_accelerator().current_stream())
            tensor.record_stream(stream)
"""


def _source(cls, name):
    return textwrap.dedent(inspect.getsource(getattr(cls, name)))


def _source_version(cls) -> str:
    """The supported DeepSpeed version whose method sources match ``cls``, or raise."""
    names = next(iter(_EXPECTED_SOURCES.values())).keys()
    digests = {name: hashlib.sha256(_source(cls, name).encode()).hexdigest()[:16] for name in names}
    for version, expected in _EXPECTED_SOURCES.items():
        if digests == expected:
            return version
    raise RuntimeError(
        f"Gradient overlap supports DeepSpeed {sorted(_EXPECTED_SOURCES)}; {cls.__name__} sources differ: {digests}"
    )


def _average_tensor_without_compute_wait(optimizer):
    cls = type(optimizer)
    src = _source(cls, "average_tensor")
    if src.count(_AVERAGE_TENSOR_WAIT) != 1:
        raise RuntimeError("Unexpected DeepSpeed average_tensor source")
    src = src.replace(_AVERAGE_TENSOR_WAIT, _AVERAGE_TENSOR_NO_WAIT)
    if hasattr(optimizer, "_fastwam_gradient_group"):
        from .zero1_sharding import gradient_average_source
        src = gradient_average_source(src)
    namespace = {}
    function = cls.average_tensor
    exec(compile(src, "<fastwam_grad_overlap:average_tensor>", "exec"), function.__globals__, namespace)
    return types.MethodType(namespace["average_tensor"], optimizer)


class GradOverlap:
    def __init__(self, optimizer, version):
        self.optimizer = optimizer
        self.version = version
        self.index = {}  # comm dtype -> buffer being filled
        self.events = {}  # (comm dtype, buffer index) -> event after that buffer's last reduction
        self.check_steps = {
            int(s) for s in os.environ.get("FASTWAM_GRAD_OVERLAP_CHECK_STEPS", "").split(",") if s.strip()
        }
        self.step_count = 0  # backward passes seen, 1-based while one is running
        self.reference = None  # list of CPU flat tensors per group while checking

    # ---- bucket swap -------------------------------------------------------
    def reduce_and_swap(self, opt, bucket, comm_dtype):
        current = self.index.get(comm_dtype, 0)
        bucket.index = current
        # Queues on reduction_stream; in 0.18.4 the trailing clear() resets bucket.index.
        if self.version == "0.18.4":
            opt.reduce_ipg_grads()
        else:
            opt.reduce_ipg_grads(comm_dtype=comm_dtype)
        event = torch.cuda.Event()
        event.record(opt.reduction_stream)  # after the reduction and copy_grads_in_partition
        self.events[(comm_dtype, current)] = event
        other = 1 - current
        previous = self.events.get((comm_dtype, other))
        if previous is not None:
            torch.cuda.current_stream().wait_event(previous)
        self.index[comm_dtype] = other
        bucket.index = other

    def reduce_independent_p_g_buckets_and_remove_grads(self, opt, param, i):
        # DeepSpeed's logic for contiguous_gradients + overlap_comm (0.18.4 or 0.19.2), with the
        # swap above. The versions differ only in where a not-yet-ready gradient is skipped.
        grad_reduc = opt.get_gradient_for_reduction(param)
        comm_dtype = opt.get_param_comm_dtype(param)
        bucket = opt.ipg_buckets[comm_dtype]
        bucket.index = self.index.get(comm_dtype, 0)
        if bucket.elements + param.numel() > opt.reduce_bucket_size:
            self.reduce_and_swap(opt, bucket, comm_dtype)

        ready = getattr(param, "ds_grad_is_ready", True)
        if self.version == "0.19.2" and not ready:
            return

        param_id = opt.get_param_id(param)
        assert not opt.params_already_reduced[param_id], f"Gradient computed twice for parameter {param_id}"

        if self.reference is not None:
            self.capture(param, i, grad_reduc)

        if param.numel() > opt.reduce_bucket_size:
            opt.extra_large_param_to_reduce[comm_dtype] = param
        else:
            new_grad_tensor = bucket.buffer[bucket.index].narrow(0, bucket.elements, param.numel())
            new_grad_tensor.copy_(grad_reduc.view(-1))
            grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)

        bucket.elements += param.numel()
        assert grad_reduc is not None, f"Invalid to reduce parameter {param_id} with None gradient"
        if ready:
            bucket.grads.append(grad_reduc)
            bucket.params.append((i, param.param_idx_in_group, param_id))

    def epilogue(self, opt, original):
        for comm_dtype, bucket in opt.ipg_buckets.items():
            bucket.index = self.index.get(comm_dtype, 0)
        result = original()  # reduces the last bucket, synchronizes, builds averaged_gradients
        self.index.clear()
        self.events.clear()
        if self.reference is not None:
            self.check(opt)
        return result

    # ---- correctness check --------------------------------------------------
    def begin_backward(self, opt):
        if getattr(opt, "_coalesce_grad_reduction", False):
            raise RuntimeError("Gradient overlap does not support DeepSpeed coalesce_grad_reduction")
        self.step_count += 1
        if self.step_count in self.check_steps and self.reference is None:
            self.reference = [torch.zeros(flat.numel(), dtype=torch.float32) for flat in opt.bit16_groups_flat]

    def capture(self, param, i, grad):
        flat = self.optimizer.bit16_groups_flat[i]
        start = (param.data_ptr() - flat.data_ptr()) // flat.element_size()
        self.reference[i][start:start + param.numel()].copy_(grad.detach().reshape(-1).float().cpu())

    def check(self, opt):
        device = opt.bit16_groups_flat[0].device
        worst = 0.0
        for i, cpu_flat in enumerate(self.reference):
            group = opt.real_dp_process_group[i]
            rank = dist.get_rank(group=group)
            gradient_group = getattr(opt, "_fastwam_gradient_group", group)
            world = dist.get_world_size(group=gradient_group)
            mine = torch.cat([g.reshape(-1).float() for g in opt.averaged_gradients[i]])
            partition_size = int(opt.partition_size[i])  # DeepSpeed may store it as a float
            part_start = rank * partition_size
            length = min(mine.numel(), partition_size)
            reference = torch.empty(length, dtype=torch.float32, device=device)
            chunk = 1 << 27
            for lo in range(0, cpu_flat.numel(), chunk):
                hi = min(lo + chunk, cpu_flat.numel())
                block = cpu_flat[lo:hi].to(device)
                dist.all_reduce(block, group=gradient_group)
                block.div_(world)
                a, b = max(lo, part_start), min(hi, part_start + length)
                if a < b:
                    reference[a - part_start:b - part_start].copy_(block[a - lo:b - lo])
                del block
            diff = (mine[:length] - reference).norm().item()
            scale = reference.norm().item()
            worst = max(worst, diff / max(scale, 1e-12))
        flag = torch.tensor([worst], device=device, dtype=torch.float32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        worst_all = float(flag.item())
        logger.info(
            "GRAD_OVERLAP_CHECK step=%d max_relative_error=%.3e ok=%s", self.step_count, worst_all, worst_all < 1e-2
        )
        self.reference = None
        if worst_all >= 1e-2:
            raise RuntimeError(f"Overlapped gradient reduction differs from the reference: {worst_all:.3e}")


def install(engine):
    """Patch ``engine.optimizer`` (DeepSpeed ZeRO-1 with overlap_comm) to overlap reduction with backward."""
    installed_version = require_supported_version()
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    optimizer = engine.optimizer
    if type(optimizer) is not DeepSpeedZeroOptimizer:
        raise RuntimeError(f"Gradient overlap needs DeepSpeedZeroOptimizer, got {type(optimizer).__name__}")
    if optimizer.partition_gradients:
        raise RuntimeError("Gradient overlap is implemented for ZeRO stage 1")
    if not (optimizer.overlap_comm and optimizer.contiguous_gradients):
        raise RuntimeError("Gradient overlap needs overlap_comm and contiguous_gradients in the DeepSpeed config")
    if optimizer.cpu_offload or optimizer.zenflow or optimizer.reduction_stream is None:
        raise RuntimeError("Gradient overlap does not support CPU offload, ZenFlow or synchronized devices")
    if optimizer.has_moe_layers:
        raise RuntimeError("Gradient overlap does not support MoE parameter groups")
    version = _source_version(type(optimizer))
    if version != installed_version:
        raise RuntimeError(
            f"DeepSpeed package version {installed_version} does not match optimizer sources {version}"
        )

    overlap = GradOverlap(optimizer, version)
    optimizer.average_tensor = _average_tensor_without_compute_wait(optimizer)
    optimizer.reduce_independent_p_g_buckets_and_remove_grads = types.MethodType(
        lambda self, param, i: overlap.reduce_independent_p_g_buckets_and_remove_grads(self, param, i), optimizer
    )
    original_epilogue = optimizer.independent_gradient_partition_epilogue
    optimizer.independent_gradient_partition_epilogue = types.MethodType(
        lambda self: overlap.epilogue(self, original_epilogue), optimizer
    )
    original_setup = optimizer.setup_buckets

    def setup_buckets(self):
        if not self.ready_for_gradients:  # first gradient hook of a backward pass
            overlap.begin_backward(self)
        return original_setup()

    optimizer.setup_buckets = types.MethodType(setup_buckets, optimizer)
    logger.info(
        "GRAD_OVERLAP installed: deepspeed_sources=%s reduce_bucket_size=%d check_steps=%s",
        version,
        optimizer.reduce_bucket_size,
        sorted(overlap.check_steps),
    )
    return overlap

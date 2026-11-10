# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Overlap DeepSpeed ZeRO-1's step-end parameter AllGather with the next forward.

DeepSpeed ends every optimizer step with one AllGather of the updated bit16
partitions (``all_gather_dp_groups``), and the next forward cannot start until
all of it has arrived. With the ``loopwam.zero1_deferred_gather`` Group that AllGather is
replaced, inside a per-instance copy of ``optimizer.step`` (DeepSpeed's module and
its other callers keep the native AllGather), by asynchronous
broadcasts of the same bytes, split into pieces ordered by first use in the
next forward: parameters outside the MoT layers (stage 0), then MoT layer 0,
1, ... Each piece lies inside one owner's partition, so it is one broadcast
from that owner. Nothing waits at the step end. ``training_loss`` waits for
stage 0 before it runs, and the MoT loop waits for layer ``i`` right before
layer ``i``, so layer ``i``'s parameters arrive while earlier layers compute.
``finish`` waits for everything; it runs at the end of the MoT forward and
before any path that may read parameters outside it (evaluation, checkpoint
saving).

All ranks check at install that they built the same piece plan. The broadcasts
use a dedicated process group with the same ranks, so the small
collectives the trainer issues after each step (loss gathers) neither queue
behind them nor make the training stream wait for them.

Approach follows the deferred gather in HYGON-AI/TurboPhysAI PR #27 (OpenWAM),
adapted to ZeRO-1 and the LoopWAM MoT.
"""

import functools
import hashlib
import inspect
import os
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist

from logging import getLogger as get_logger
from .zero1_compat import require_supported_version

logger = get_logger(__name__)



@dataclass
class Piece:
    group: int
    start: int
    length: int
    stage: int
    owner: int


def _offset(param: torch.Tensor, flat: torch.Tensor):
    start = (param.data_ptr() - flat.data_ptr()) // flat.element_size()
    if param.untyped_storage().data_ptr() != flat.untyped_storage().data_ptr() or not (
        0 <= start and start + param.numel() <= flat.numel()
    ):
        return None
    return start


def _stage_of_params(mot) -> dict:
    """MoT layer ``i`` of every expert is stage ``i + 1``; parameters not listed are stage 0."""
    stages = {}
    for name in mot.expert_order:
        for layer_idx, block in enumerate(mot.mixtures[name].blocks):
            for param in block.parameters():
                stages[id(param)] = layer_idx + 1
    return stages


def plan(optimizer, mot, min_piece: int):
    """Pieces in waiting order, covering every element of every flat buffer once."""
    stages = _stage_of_params(mot)
    tail = mot.num_layers + 1
    pieces = []
    for g, flat in enumerate(optimizer.bit16_groups_flat):
        # Element ranges of each stage, from the parameters' views into the flat buffer.
        bounds = []
        for param in optimizer.bit16_groups[g]:
            start = _offset(param, flat)
            if start is None:
                raise RuntimeError("A ZeRO bit16 parameter is not a view of its flat buffer")
            bounds.append((start, start + param.numel(), stages.get(id(param), 0)))
        bounds.sort()
        ranges, cursor = [], 0
        for start, end, stage in bounds:
            if start > cursor:  # alignment padding between parameters
                ranges.append((cursor, start, tail))
            ranges.append((start, end, stage))
            cursor = end
        if cursor < flat.numel():
            ranges.append((cursor, flat.numel(), tail))

        # Owner boundaries from the partition views.
        parts = optimizer.parallel_partitioned_bit16_groups[g]
        owner_starts = [(p.data_ptr() - flat.data_ptr()) // flat.element_size() for p in parts]
        owner_ends = [s + p.numel() for s, p in zip(owner_starts, parts)]

        # Split stage ranges at owner boundaries. A run joins the previous run of the same owner
        # when it has the same stage, or when it is short and needed no earlier than that run
        # (arriving early is harmless; arriving late would stall its layer).
        runs = []  # (start, end, stage, owner), contiguous
        for start, end, stage in ranges:
            for owner, (owner_start, owner_end) in enumerate(zip(owner_starts, owner_ends)):
                lo, hi = max(start, owner_start), min(end, owner_end)
                if lo >= hi:
                    continue
                if runs and runs[-1][3] == owner and runs[-1][1] == lo and (
                    runs[-1][2] == stage or (hi - lo < min_piece and runs[-1][2] <= stage)
                ):
                    runs[-1] = (runs[-1][0], hi, runs[-1][2], owner)
                else:
                    runs.append((lo, hi, stage, owner))
        pieces += [Piece(g, s, e - s, stage, owner) for s, e, stage, owner in runs]
    pieces.sort(key=lambda p: (p.stage, p.group, p.start))
    return pieces, tail


class DeferredGather:
    def __init__(self, optimizer, mot):
        self.optimizer = optimizer
        self.block_stages = {id(block): i + 1 for name in mot.expert_order
                             for i, block in enumerate(mot.mixtures[name].blocks)}
        self.dp_groups = optimizer.real_dp_process_group
        min_piece = int(os.environ.get("FASTWAM_DEFERRED_GATHER_MIN_PIECE", str(1 << 20)))
        self.pieces, self.tail = plan(optimizer, mot, min_piece)
        self.flats = list(optimizer.bit16_groups_flat)
        self.views = [self.flats[p.group].narrow(0, p.start, p.length) for p in self.pieces]
        # One broadcast group per DP group, same ranks; created collectively on all ranks.
        self.groups = getattr(optimizer, "_fastwam_parameter_gather_groups", None)
        if self.groups is None:
            self.groups = [dist.new_group(ranks=dist.get_process_group_ranks(g)) for g in self.dp_groups]
        if len(self.groups) != len(self.dp_groups) or any(
            dist.get_process_group_ranks(a) != dist.get_process_group_ranks(b)
            for a, b in zip(self.groups, self.dp_groups)
        ):
            raise RuntimeError("Deferred gather groups do not match the optimizer shard groups")
        self.sources = [
            [dist.get_global_rank(g, owner) for owner in range(dist.get_world_size(g))] for g in self.dp_groups
        ]
        self._check_plan()
        self.works = []
        self.next = 0
        self.check_steps = {
            int(s) for s in os.environ.get("FASTWAM_DEFERRED_GATHER_CHECK_STEPS", "").split(",") if s.strip()
        }
        self.step_count = 0
        total = sum(p.length for p in self.pieces)
        layers = sum(p.length for p in self.pieces if 0 < p.stage < self.tail)
        logger.info(
            "DEFERRED_GATHER pieces=%d elements=%d in_layers=%.3f stage0=%d tail=%d check_steps=%s",
            len(self.pieces),
            total,
            layers / max(total, 1),
            sum(p.length for p in self.pieces if p.stage == 0),
            sum(p.length for p in self.pieces if p.stage == self.tail),
            sorted(self.check_steps),
        )

    def _check_plan(self):
        """Every rank must broadcast the same pieces in the same order, or the broadcasts mismatch."""
        text = repr([(p.stage, p.group, p.start, p.length, p.owner) for p in self.pieces])
        digest = int.from_bytes(hashlib.sha256(text.encode()).digest()[:7], "little")
        values = torch.tensor([digest, -digest], dtype=torch.int64, device=self.flats[0].device)
        for group in self.groups:
            dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
        if values.tolist() != [digest, -digest]:
            raise RuntimeError("Deferred gather plans differ across ranks")

    def launch(self):
        """Queue every piece and return; the training stream does not wait here."""
        self.finish()
        self.works = [
            dist.broadcast(
                view,
                src=self.sources[piece.group][piece.owner],
                group=self.groups[piece.group],
                async_op=True,
            )
            for piece, view in zip(self.pieces, self.views)
        ]
        self.next = 0

    def wait(self, stage: int):
        """Make the current stream wait for every piece needed up to ``stage``."""
        works, pieces = self.works, self.pieces
        while self.next < len(works) and pieces[self.next].stage <= stage:
            works[self.next].wait()
            self.next += 1

    def finish(self):
        self.wait(self.tail)
        self.works = []
        self.next = 0

    def check(self):
        """Compare the deferred result with DeepSpeed's own AllGather into a scratch buffer."""
        self.finish()
        ok = True
        for g, flat in enumerate(self.flats):
            parts = self.optimizer.parallel_partitioned_bit16_groups[g]
            rank = dist.get_rank(group=self.dp_groups[g])
            reference = torch.empty_like(flat)
            ref_parts = reference.narrow(0, 0, parts[0].numel() * len(parts))
            dist.all_gather_into_tensor(ref_parts, parts[rank], group=self.dp_groups[g])
            covered = parts[0].numel() * len(parts)
            ok &= bool(torch.equal(ref_parts, flat.narrow(0, 0, covered)))
            del reference
        flag = torch.tensor([1 if ok else 0], device=self.flats[0].device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        logger.info("DEFERRED_GATHER_CHECK step=%d all_ranks_equal=%s", self.step_count, bool(flag.item()))
        if not flag.item():
            raise RuntimeError("Deferred gather result differs from DeepSpeed AllGather")


_GATHER_PARAMETERS = (
    "groups_flat",
    "partitioned_param_groups",
    "dp_process_group",
    "start_alignment_factor",
    "allgather_bucket_size",
)


def _step_with_gather(optimizer, gather):
    """A bound copy of ``optimizer.step`` whose step-end ``all_gather_dp_groups`` is ``gather``.

    Only this instance's step sees the replacement; DeepSpeed's module globals and its
    other callers (checkpoint loading, ``update_lp_params``) keep the native AllGather.
    """
    function = optimizer.step.__func__
    if "all_gather_dp_groups" not in function.__code__.co_names:
        raise RuntimeError("Unsupported DeepSpeed step: no direct all_gather_dp_groups call")
    native = function.__globals__["all_gather_dp_groups"]
    if tuple(inspect.signature(native).parameters) != _GATHER_PARAMETERS:
        raise RuntimeError("Unsupported DeepSpeed all_gather_dp_groups interface")
    namespace = dict(function.__globals__)
    namespace["all_gather_dp_groups"] = gather
    replacement = types.FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    replacement.__kwdefaults__ = function.__kwdefaults__
    functools.update_wrapper(replacement, function)
    return types.MethodType(replacement, optimizer)


def _bit16_views_intact(optimizer) -> bool:
    """Whether every bit16 parameter already is the view ``_update_model_bit16_weights`` would assign."""
    for i, flat in enumerate(optimizer.bit16_groups_flat):
        expected = optimizer.unflatten(flat, optimizer.round_robin_bit16_meta[i])
        for param, view in zip(optimizer.round_robin_bit16_groups[i], expected):
            if param.data_ptr() != view.data_ptr() or param.shape != view.shape or param.stride() != view.stride():
                return False
    return True


def install(engine, model, *, skip_rebind=False):
    """Route the AllGather inside ``engine.optimizer.step`` through a DeferredGather."""
    require_supported_version()
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    optimizer = engine.optimizer
    if hasattr(model.mot, "_turbophysai_deferred_gather"):
        raise RuntimeError("LoopWAM deferred gather is already installed on this model")
    if type(optimizer) is not DeepSpeedZeroOptimizer:
        raise RuntimeError(f"Deferred gather needs DeepSpeedZeroOptimizer, got {type(optimizer).__name__}")
    if optimizer.partition_gradients:
        raise RuntimeError("Deferred gather is implemented for ZeRO stage 1 only")
    if optimizer.cpu_offload:
        raise RuntimeError("Deferred gather does not support optimizer CPU offload")
    deferred = DeferredGather(optimizer, model.mot)

    def gather(groups_flat, partitioned_param_groups, dp_process_group, start_alignment_factor, allgather_bucket_size):
        if len(groups_flat) != len(deferred.flats) or not all(a is b for a, b in zip(groups_flat, deferred.flats)):
            raise RuntimeError("ZeRO flat buffers changed after the deferred gather plan was built")
        deferred.launch()
        if deferred.step_count in deferred.check_steps:
            deferred.check()

    deferred_step = _step_with_gather(optimizer, gather)
    # DeepSpeed rebinds every bit16 parameter to its flat-buffer view at each step end
    # ("TODO: we probably don't need this? just to be safe"). The flat buffers stay fixed in
    # training and the bit16 weights are updated in place, so the rebind only costs CPU time.
    # Other callers (checkpoint loading, device moves) keep the native rebind.
    if skip_rebind and not _bit16_views_intact(optimizer):
        raise RuntimeError("bit16 parameters are not views of the ZeRO flat buffers; cannot skip the rebind")

    def step(self, *args, **kwargs):
        deferred.step_count += 1
        if not skip_rebind:
            return deferred_step(*args, **kwargs)
        if deferred.step_count in deferred.check_steps and not _bit16_views_intact(self):
            raise RuntimeError("bit16 parameters stopped being views of the ZeRO flat buffers")
        saved = self.__dict__.get("_update_model_bit16_weights")
        had_override = "_update_model_bit16_weights" in self.__dict__
        self._update_model_bit16_weights = lambda group_index: None  # this step only
        try:
            return deferred_step(*args, **kwargs)
        finally:
            if had_override:
                self._update_model_bit16_weights = saved
            else:
                del self._update_model_bit16_weights

    optimizer.step = types.MethodType(step, optimizer)

    original_training_loss = model.training_loss

    def training_loss(*args, **kwargs):
        deferred.wait(0)
        try:
            return original_training_loss(*args, **kwargs)
        finally:
            deferred.finish()

    model.training_loss = training_loss
    model.mot._turbophysai_deferred_gather = deferred
    logger.info("DEFERRED_GATHER skip_bit16_rebind=%s", skip_rebind)
    return deferred

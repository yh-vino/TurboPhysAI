# Copyright (c) Microsoft Corporation.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team. Modified by Hygon: LoopWAM ZeRO-1 scheduling adapters.

"""Bound ZeRO-1 state/parameter sharding by the configured ``shard_nodes`` value.

0 (default) preserves global sharding. A positive value groups that many
consecutive nodes per optimizer replica, or one global group when the total
node count does not exceed that value. Gradients are still averaged over
the original global data-parallel group; only ownership, norms and parameter
collection use the smaller group. Requires equal, complete node groups.
"""

from contextlib import contextmanager
import functools
import hashlib
import inspect
import os
import socket
import textwrap
import types

import torch.distributed as dist

from logging import getLogger as get_logger
from .zero1_compat import require_supported_version

logger = get_logger(__name__)
_CHECKPOINT_KEY = "fastwam_zero1_sharding"
_COMMON_SOURCES = {
    "average_tensor": "5f318f8e84d55a24",
    "gradient_reduction_w_predivide": "60dade97a7e8419f",
    "has_overflow": "53a5fc9ff81a8853",
    "load_state_dict": "251b84273661dd6d",
}
_VERSION_SOURCES = {
    "0.18.4": {
        "__init__": "41c8a700baa35fee",
        "allreduce_and_copy_with_multiple_ranks": "93fb3d24574335cb",
        "state_dict": "0d4dd9a87ac19a6c",
    },
    "0.19.2": {
        "__init__": "f2a6a3803add60e9",
        "allreduce_and_copy_with_multiple_ranks": "c2c29c0d6167925d",
        "state_dict": "ad1a336a89f3926e",
    },
}


def _source(cls, name):
    return textwrap.dedent(inspect.getsource(getattr(cls, name)))


def _check_sources(cls, version):
    expected = {**_COMMON_SOURCES, **_VERSION_SOURCES[version]}
    actual = {name: hashlib.sha256(_source(cls, name).encode()).hexdigest()[:16] for name in expected}
    if actual != expected:
        raise RuntimeError(f"Unsupported DeepSpeed ZeRO-1 sharding sources: {actual}")


def gradient_average_source(source):
    """Keep global averaging when grad_overlap rebuilds average_tensor too."""
    old = "tensor.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))"
    if source.count(old) != 1:
        raise RuntimeError("Unexpected DeepSpeed gradient averaging divisor")
    return source.replace(old, old.replace("self.dp_process_group", "self._fastwam_gradient_group"))


def _bind_source(optimizer, name, replacements):
    function = getattr(type(optimizer), name)
    source = _source(type(optimizer), name)
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Unexpected DeepSpeed {name} source fragment: {old!r}")
        source = source.replace(old, new)
    namespace = {}
    exec(compile(source, f"<fastwam_zero1_sharding:{name}>", "exec"), function.__globals__, namespace)
    return types.MethodType(namespace[name], optimizer)


def _install(optimizer, gradient_group, gather_group, shard_ranks, nodes, world_size):
    shard_group = optimizer.dp_process_group
    optimizer._fastwam_gradient_group = gradient_group
    optimizer._fastwam_parameter_gather_groups = [gather_group] * len(optimizer.real_dp_process_group)
    optimizer.average_tensor = _bind_source(optimizer, "average_tensor", [(
        "tensor.div_(dist.get_world_size(group=self.dp_process_group) / float(self.sequence_parallel_size))",
        "tensor.div_(dist.get_world_size(group=self._fastwam_gradient_group) / float(self.sequence_parallel_size))",
    )])
    # Bucket ownership remains shard-local, but its AllReduce includes ALL ranks.
    optimizer.allreduce_and_copy_with_multiple_ranks = _bind_source(
        optimizer, "allreduce_and_copy_with_multiple_ranks",
        [("process_group=process_group)", "process_group=self._fastwam_gradient_group)")],
    )
    for name in ("gradient_reduction_w_predivide", "has_overflow"):
        # Preserve native arithmetic; change only the gradient/overflow domain.
        function = getattr(type(optimizer), name)
        source = _source(type(optimizer), name).replace("self.dp_process_group", "self._fastwam_gradient_group")
        namespace = {}
        exec(compile(source, f"<fastwam_zero1_sharding:{name}>", "exec"), function.__globals__, namespace)
        setattr(optimizer, name, types.MethodType(namespace[name], optimizer))

    metadata = {"version": 1, "shard_nodes": nodes, "world_size": world_size, "shard_ranks": shard_ranks}
    optimizer._fastwam_sharding = metadata
    state_dict = optimizer.state_dict
    load_state_dict = optimizer.load_state_dict

    def save_state(self):
        # The engine uses global ranks for checkpoint filenames, while native
        # optimizer serialization must still see the local partition layout.
        current_group = self.dp_process_group
        self.dp_process_group = shard_group
        try:
            state = state_dict()
        finally:
            self.dp_process_group = current_group
        state[_CHECKPOINT_KEY] = metadata.copy()
        return state

    def load_state(self, state_dict_list, *args, **kwargs):
        if kwargs.get("checkpoint_folder"):
            raise RuntimeError("Node-group sharding requires a native, fixed-layout ZeRO checkpoint")
        if state_dict_list is None or len(state_dict_list) != world_size:
            raise RuntimeError("Resume requires the same world size and optimizer sharding layout")
        selected = [state_dict_list[rank] for rank in shard_ranks]
        # DeepSpeed's rigid checkpoint loader reads only this global rank's
        # file and leaves None placeholders for the other partitions.
        if selected[dist.get_rank(group=shard_group)] is None or any(
            state is not None and state.get(_CHECKPOINT_KEY) != metadata for state in selected
        ):
            raise RuntimeError("Checkpoint optimizer sharding differs; use matching groups or load weights only")
        current_group = self.dp_process_group
        self.dp_process_group = shard_group
        try:
            return load_state_dict(selected, *args, **kwargs)
        finally:
            self.dp_process_group = current_group

    optimizer.state_dict = types.MethodType(save_state, optimizer)
    optimizer.load_state_dict = types.MethodType(load_state, optimizer)


def install_checkpointing(engine):
    """Keep replica files distinct when the engine chooses checkpoint ranks."""
    optimizer = getattr(engine, "optimizer", None)
    if not hasattr(optimizer, "_fastwam_sharding"):
        return
    if getattr(engine, "_fastwam_shard_checkpointing", False):
        return
    # Engine initialization suppresses writers whose shard-local rank differs
    # from their global DP rank. Every replica needs its own global-rank files.
    engine.save_zero_checkpoint = True
    for name in ("save_checkpoint", "load_checkpoint"):
        native = getattr(engine, name)

        @functools.wraps(native)
        def checkpoint(self, *args, _native=native, **kwargs):
            group = optimizer.dp_process_group
            optimizer.dp_process_group = optimizer._fastwam_gradient_group
            try:
                return _native(*args, **kwargs)
            finally:
                optimizer.dp_process_group = group

        setattr(engine, name, types.MethodType(checkpoint, engine))
    engine._fastwam_shard_checkpointing = True


def _node_groups(nodes):
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    entries = [None] * world
    dist.all_gather_object(entries, (socket.gethostname(), local_rank, local_size))
    hosts = {}
    for rank, (host, index, size) in enumerate(entries):
        hosts.setdefault(host, []).append((rank, index, size))
    for host, members in hosts.items():
        if len(members) != local_size or {i for _, i, _ in members} != set(range(local_size)):
            raise RuntimeError(f"Optimizer sharding needs complete nodes; invalid ranks on {host}")
        if any(size != local_size for _, _, size in members):
            raise RuntimeError("Optimizer sharding requires equal ranks per node")
    if len(hosts) <= nodes:
        return [sorted(rank for members in hosts.values() for rank, _, _ in members)]
    if len(hosts) % nodes:
        raise ValueError(f"shard_nodes={nodes} must divide node count {len(hosts)}")
    host_list = list(hosts.values())  # stable order of first global rank on each host
    return [sorted(rank for members in host_list[i:i + nodes] for rank, _, _ in members)
            for i in range(0, len(host_list), nodes)]


@contextmanager
def prepare_context(nodes):
    """Use around accelerator.prepare, before ZeRO allocates optimizer shards."""
    nodes = int(nodes)
    if nodes < 0:
        raise ValueError("shard_nodes must be nonnegative")
    if nodes == 0:
        yield
        return
    version = require_supported_version()
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    if not dist.is_initialized():
        raise RuntimeError("Initialize the global data-parallel group before optimizer sharding")
    groups = _node_groups(nodes)
    if len(groups) == 1:
        logger.info("ZERO1_SHARD nodes=%d covers all ranks; preserving global sharding", nodes)
        yield
        return
    cls = DeepSpeedZeroOptimizer
    _check_sources(cls, version)
    gradient_group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    for ranks in groups:
        group = dist.new_group(ranks=ranks)
        gather_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            shard_group, shard_gather_group, shard_ranks = group, gather_group, ranks

    native_init = cls.__init__
    signature = inspect.signature(native_init)
    installed = []

    @functools.wraps(native_init)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if values["partition_grads"] or values["has_moe_layers"] or values["elastic_checkpoint"]:
            raise RuntimeError("Node-group optimizer sharding supports fixed-layout, non-MoE ZeRO-1 only")
        if not values["use_multi_rank_bucket_allreduce"] or values["zenflow_config"] is not None:
            raise RuntimeError("Node-group sharding requires multi-rank bucket AllReduce without ZenFlow")
        offload = values["offload_optimizer_config"]
        if offload is not None and offload.device != "none":
            raise RuntimeError("Node-group sharding does not support optimizer offload")
        if values["mpu"] is not None or values["expert_parallel_group"] is not None:
            raise RuntimeError("Node-group sharding supports plain data parallelism only")
        original_group = values["dp_process_group"]
        if dist.get_process_group_ranks(original_group or gradient_group) != list(range(world)):
            raise RuntimeError("Node-group sharding requires the original DP group to cover all ranks")
        values["dp_process_group"] = shard_group
        native_init(*bound.args, **bound.kwargs)
        if self.sequence_parallel_size != 1:
            raise RuntimeError("Node-group sharding does not support sequence parallelism")
        _install(self, gradient_group, shard_gather_group, shard_ranks, nodes, world)
        installed.append(self)
        logger.info("ZERO1_SHARD nodes=%d ranks=%s replicas=%d gradient_world=%d",
                    nodes, shard_ranks, len(groups), world)

    cls.__init__ = initialize
    try:
        yield
        if len(installed) != 1:
            raise RuntimeError(f"Expected one ZeRO-1 optimizer, patched {len(installed)}")
    finally:
        cls.__init__ = native_init

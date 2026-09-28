# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Public operator APIs, loaded on demand with their backend dependencies."""

from importlib import import_module
from typing import Any


_OPERATORS = {
    "ModulatedDeformConv2dFunction": (
        "turbo_physai.operators.modulated_deform_conv",
        "ModulatedDeformConv2dFunction",
    ),
    "modulated_deform_conv2d": (
        "turbo_physai.operators.modulated_deform_conv",
        "modulated_deform_conv2d",
    ),
    "grid_sample": ("turbo_physai.operators._grid_sample", "grid_sample"),
    "interpolate": ("turbo_physai.operators.upsample_bilinear_2d", "interpolate"),
    "deformable_aggregation_function": (
        "turbo_physai.operators.deformable_aggregation",
        "deformable_aggregation_function",
    ),
    "DeformableAggregationFunction": (
        "turbo_physai.operators.deformable_aggregation",
        "DeformableAggregationFunction",
    ),
    "ms_deform_attn_forward": (
        "turbo_physai.operators.multi_scale_deformable_attention",
        "ms_deform_attn_forward",
    ),
    "ms_deform_attn_backward": (
        "turbo_physai.operators.multi_scale_deformable_attention",
        "ms_deform_attn_backward",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _OPERATORS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = sorted(_OPERATORS)

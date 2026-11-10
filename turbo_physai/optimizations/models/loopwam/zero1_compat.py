# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact DeepSpeed package versions accepted by the communication patches."""

SUPPORTED_DEEPSPEED_VERSIONS = ("0.18.4", "0.19.2")


def require_supported_version():
    import deepspeed

    version = deepspeed.__version__
    if version not in SUPPORTED_DEEPSPEED_VERSIONS:
        raise RuntimeError(
            f"Communication overlap requires exact DeepSpeed versions "
            f"{SUPPORTED_DEEPSPEED_VERSIONS}; got {version!r}"
        )
    return version

# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Public imports and native-extension wiring without HCU dependencies."""

import subprocess
import sys
import textwrap
import unittest


class OperatorApiTest(unittest.TestCase):
    def run_script(self, script):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_namespace_and_cli_do_not_load_backends(self):
        self.run_script(
            """
            import importlib.abc
            import sys

            class BlockBackends(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'torch', 'lightop', 'hipdnn', 'mmcv', 'mmdet3d'} or fullname in {'turbo_physai.ops', 'turbo_physai._C'}:
                        raise AssertionError(fullname)

            sys.meta_path.insert(0, BlockBackends())
            import turbo_physai
            from turbo_physai import operators
            import turbo_physai.cli
            from turbo_physai.optimizations.common.mmdet3d import bev_pool, voxelization, sparse_conv

            assert callable(turbo_physai.apply)
            assert operators is turbo_physai.operators
            for name in operators.__all__:
                assert name in dir(operators)
                assert not hasattr(turbo_physai, name), name
            assert not hasattr(operators, 'unknown_operator')
            assert 'torch' not in sys.modules
            """
        )

    def test_public_exports_resolve_with_only_backend_stubs(self):
        self.run_script(
            """
            import sys
            import types
            import torch

            extension = types.ModuleType('turbo_physai._C')
            def unused(*args, **kwargs):
                raise AssertionError('unexpected operator execution')
            for name in ('grid_sample_forward', 'grid_sample_backward',
                         'upsample_bilinear_2d_forward', 'upsample_bilinear_2d_backward',
                         'deformable_aggregation_forward', 'deformable_aggregation_backward'):
                setattr(extension, name, unused)
            sys.modules[extension.__name__] = extension
            sys.modules['hipdnn'] = types.ModuleType('hipdnn')
            lightop = types.ModuleType('lightop')
            lightop.op = types.SimpleNamespace(ms_deform_attn_forward=unused, ms_deform_attn_backward=unused)
            sys.modules['lightop'] = lightop

            from turbo_physai import operators
            for name in operators.__all__:
                exported = getattr(operators, name)
                assert callable(exported), name
                assert getattr(operators, name) is exported
            assert 'turbo_physai.ops' not in sys.modules
            """
        )

    def test_grid_sample_import_order_and_native_forward_backward(self):
        self.run_script(
            """
            import importlib
            import sys
            import types
            import warnings
            import torch

            calls = []
            extension = types.ModuleType('turbo_physai._C')
            def forward(x, grid, mode, padding, align_corners):
                calls.append(('forward', mode, padding, align_corners))
                return torch.full_like(x, 2.)
            def backward(grad, x, grid, mode, padding, align_corners, mask):
                calls.append(('backward', mode, padding, align_corners, mask))
                return torch.full_like(x, 3.), torch.full_like(grid, 4.)
            extension.grid_sample_forward = forward
            extension.grid_sample_backward = backward
            sys.modules[extension.__name__] = extension

            implementation = importlib.import_module('turbo_physai.operators._grid_sample')
            from turbo_physai.operators import grid_sample
            assert grid_sample is implementation.grid_sample
            x = torch.ones(1, 1, 2, 2, requires_grad=True)
            grid = torch.zeros(1, 2, 2, 2, requires_grad=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                output = grid_sample(x, grid)
            assert caught and 'align_corners=False' in str(caught[0].message)
            output.sum().backward()
            torch.testing.assert_close(output, torch.full_like(x, 2.))
            torch.testing.assert_close(x.grad, torch.full_like(x, 3.))
            torch.testing.assert_close(grid.grad, torch.full_like(grid, 4.))
            assert calls == [('forward', 0, 0, False), ('backward', 0, 0, False, [True, True])]
            """
        )

    def test_framework_implementations_load_renamed_extension(self):
        self.run_script(
            """
            import sys
            import types
            extension = types.ModuleType('turbo_physai._C')
            sys.modules[extension.__name__] = extension
            from turbo_physai.optimizations.common.mmdet3d import bev_pool, voxelization, sparse_conv
            for implementation in (bev_pool, voxelization, sparse_conv):
                assert implementation._ops() is extension
            assert 'turbo_physai.ops' not in sys.modules
            """
        )

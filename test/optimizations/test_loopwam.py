# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contracts for LoopWAM adapters; vendor kernels have a separate HCU test."""

import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import contextmanager

import pytest

from turbo_physai.optimizations.models.loopwam import training


@contextmanager
def fake_module(name, value):
    missing = object()
    previous = sys.modules.get(name, missing)
    sys.modules[name] = value
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def test_catalog_is_import_light():
    result = subprocess.run([sys.executable, '-c',
        'import sys; import turbo_physai.optimizations.models.loopwam.catalog; '
        'assert not any(n in sys.modules for n in ("torch", "fastwam", "deepspeed", "accelerate"))'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_group_targets_do_not_conflict():
    from turbo_physai.optimizations.models.loopwam import catalog  # noqa: F401
    from turbo_physai.engine.definitions.registry import default_registry
    from turbo_physai.engine.checking.conflicts import find_group_conflicts
    groups = [g for g in default_registry.groups if g.startswith('loopwam.')]
    assert len(groups) == 10
    assert not find_group_conflicts(default_registry, groups)


def test_wrappers_have_no_construction_side_effects():
    env = dict(os.environ)
    original = Mock()
    for factory in (training.deferred_train_mode_wrapper, training.layer_wait_wrapper,
                    training.finish_gather_wrapper, training.gradient_overlap_wrapper,
                    training.sharded_prepare_wrapper, training.fused_metrics_wrapper):
        factory(original, {})
    original.assert_not_called()
    assert dict(os.environ) == env


@pytest.mark.parametrize('factory,options', [
    (training.sharded_prepare_wrapper, {'shard_nodes': 0}),
    (training.sharded_prepare_wrapper, {'shard_nodes': True}),
    (training.sharded_prepare_wrapper, {'shard_nodes': '4'}),
    (training.fused_metrics_wrapper, {'benchmark_warmup_steps': -1}),
    (training.deferred_train_mode_wrapper, {'skip_bit16_rebind': 'yes'}),
])
def test_invalid_options_fail_at_check(factory, options):
    with pytest.raises(ValueError):
        factory(Mock(), options)


def test_layer_wait_precedes_direct_parameter_reads():
    events = []
    block = object()
    manager = SimpleNamespace(block_stages={id(block): 3}, wait=lambda s: events.append(('wait', s)))
    mot = SimpleNamespace(_turbophysai_deferred_gather=manager)
    def original(self, expert, block, x):
        events.append(('read', block))
        return x
    wrapped = training.layer_wait_wrapper(original, {})
    assert wrapped(mot, None, block=block, x=42) == 42
    assert events == [('wait', 3), ('read', block)]


def test_checkpoint_finishes_pending_collectives_first():
    events = []
    model = SimpleNamespace(mot=SimpleNamespace(_turbophysai_deferred_gather=SimpleNamespace(
        finish=lambda: events.append('finish'))))
    trainer = SimpleNamespace(model=model, accelerator=SimpleNamespace(unwrap_model=lambda m: m))
    wrapped = training.finish_gather_wrapper(lambda self: events.append('save'), {})
    wrapped(trainer)
    assert events == ['finish', 'save']


def test_deferred_install_is_once_per_model():
    torch = pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam import deferred_gather
    model = SimpleNamespace(mot=SimpleNamespace())
    trainer = SimpleNamespace(model=object(), accelerator=SimpleNamespace(unwrap_model=lambda m: model))
    def install(engine, raw, **kwargs):
        assert engine is trainer.model and kwargs == {'skip_rebind': True}
        raw.mot._turbophysai_deferred_gather = object()
    with patch.object(deferred_gather, 'install', side_effect=install) as mock:
        wrapped = training.deferred_train_mode_wrapper(lambda self: 7, {'skip_bit16_rebind': True})
        assert wrapped(trainer) == 7
        assert wrapped(trainer) == 7
        mock.assert_called_once()


def test_rmsnorm_preserves_forward_and_gradients():
    torch = pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam.kernels import rmsnorm_forward
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(2, 5, 32, dtype=dtype, requires_grad=True)
        weight = torch.randn(32, requires_grad=True)
        owner = SimpleNamespace(weight=weight, eps=1e-6)
        ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + owner.eps)).to(dtype) * weight
        out = rmsnorm_forward(owner, x)
        assert out.dtype == ref.dtype
        torch.testing.assert_close(out, ref, rtol=0.015 if dtype == torch.bfloat16 else 1e-5, atol=1e-5)
        actual = torch.autograd.grad(out.sum(), (x, weight), retain_graph=True)
        expected = torch.autograd.grad(ref.sum(), (x, weight))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0.03 if dtype == torch.bfloat16 else 2e-5, atol=0.016 if dtype == torch.bfloat16 else 2e-5)


@pytest.mark.parametrize('cached', [False, True])
def test_vae_layout_forward_and_gradients(cached):
    torch = pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam.kernels import causal_conv3d_forward
    torch.manual_seed(17)
    conv = torch.nn.Conv3d(2, 3, 3, padding=0)
    conv._padding = (1, 1, 1, 1, 2, 0)
    reference = copy.deepcopy(conv)
    x = torch.randn(2, 2, 3, 5, 5, requires_grad=True)
    cache = torch.randn(2, 2, 1, 5, 5, requires_grad=True) if cached else None
    padding = list(conv._padding)
    value = x
    if cache is not None:
        value = torch.cat((cache, x), dim=2)
        padding[4] -= cache.shape[2]
    expected = reference(torch.nn.functional.pad(value, padding))
    parameter = conv.weight
    actual = causal_conv3d_forward(conv, x, cache)
    assert conv.weight is parameter and conv.weight.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(actual, expected)
    inputs = (x, cache) if cached else (x,)
    a = torch.autograd.grad(actual.square().sum(), (*inputs, conv.weight, conv.bias), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), (*inputs, reference.weight, reference.bias))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
    conv.weight = torch.nn.Parameter(conv.weight.detach().contiguous())
    torch.testing.assert_close(causal_conv3d_forward(conv, x, cache), expected)


@pytest.mark.parametrize('dimensions', [2, 3, 4])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_mask_adapter_and_checkpoint_gradients(dimensions, checkpointed):
    torch = pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam.kernels import mixed_attention
    torch.manual_seed(18)
    q, k, v = [torch.randn(2, 5, 16, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    if dimensions == 3:
        mask = torch.stack((mask, torch.ones_like(mask)))
    shaped = mask if dimensions in (2, 3) else mask[None, None]
    def vendor(q, k, v, mask, **kwargs):
        if 'causal' in kwargs:
            kwargs['is_causal'] = kwargs.pop('causal')
        # Use one high-precision mathematical reference here so CPU SDPA's
        # mask-dependent BF16 backend choice does not affect adapter checks.
        return torch.nn.functional.scaled_dot_product_attention(
            q.double().transpose(1, 2), k.double().transpose(1, 2), v.double().transpose(1, 2), attn_mask=mask,
            **kwargs).transpose(1, 2).contiguous().to(q.dtype)
    owner = SimpleNamespace(num_heads=2, mot_checkpoint_mixed_attn=checkpointed, training=True)
    # This tests the adapter/checkpoint contract with a real differentiable SDPA,
    # not the vendor kernel. The actual vendor function is covered on HCU.
    with fake_module('flash_attn', SimpleNamespace(flash_attn_with_mask_func=vendor)):
        out = mixed_attention(owner, q, k, v, shaped)
        ref = vendor(q.reshape(2, 5, 2, 8), k.reshape(2, 5, 2, 8), v.reshape(2, 5, 2, 8), mask,
                     dropout_p=0.0, is_causal=False).reshape(2, 5, 16)
        torch.testing.assert_close(out, ref)
        actual = torch.autograd.grad(out.float().sum(), (q, k, v), retain_graph=True)
        expected = torch.autograd.grad(ref.float().sum(), (q, k, v))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)


def test_vendor_failure_is_not_silently_retried():
    torch = pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam.kernels import masked_attention
    q = torch.randn(1, 3, 8, dtype=torch.bfloat16)
    bad = Mock(side_effect=RuntimeError('kernel failed'))
    with fake_module('flash_attn', SimpleNamespace(flash_attn_with_mask_func=bad)):
        with pytest.raises(RuntimeError, match='kernel failed'):
            masked_attention(q, q, q, torch.ones(3, 3, dtype=torch.bool), 2)
        bad.assert_called_once()


def test_fused_metrics_one_collective_and_rank_mean():
    torch = pytest.importorskip('torch')
    accelerator = SimpleNamespace(gather=Mock(side_effect=lambda x: torch.cat((x, x + 2))))
    loss = torch.tensor(2., requires_grad=True)
    assert training.gather_metrics(accelerator, loss, torch.tensor(4.), {'a': 6., 'b': torch.tensor(8.)}) == (3., 5., {'a': 7., 'b': 9.})
    accelerator.gather.assert_called_once()


def test_deferred_plan_covers_padding_and_owner_boundaries():
    torch = pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam.deferred_gather import plan
    flat = torch.arange(24.)
    outside = torch.nn.Parameter(flat[:4])
    layer0 = torch.nn.Module(); layer0.p = torch.nn.Parameter(flat[4:14])
    layer1 = torch.nn.Module(); layer1.p = torch.nn.Parameter(flat[14:22])
    mot = SimpleNamespace(expert_order=['video'], mixtures={'video': SimpleNamespace(blocks=[layer0, layer1])}, num_layers=2)
    opt = SimpleNamespace(bit16_groups_flat=[flat], bit16_groups=[[outside, layer0.p, layer1.p]],
                          parallel_partitioned_bit16_groups=[[flat[:12], flat[12:]]])
    pieces, tail = plan(opt, mot, min_piece=0)
    covered = []
    for piece in pieces:
        covered.extend(range(piece.start, piece.start + piece.length))
        assert piece.owner == piece.start // 12
        assert (piece.start + piece.length - 1) // 12 == piece.owner
    assert sorted(covered) == list(range(24))
    assert len(covered) == len(set(covered))
    assert tail == 3
    assert [p.stage for p in pieces] == sorted(p.stage for p in pieces)


def test_unsupported_deepspeed_version_is_rejected():
    from turbo_physai.optimizations.models.loopwam.zero1_compat import require_supported_version
    with fake_module('deepspeed', SimpleNamespace(__version__='0.18.5')):
        with pytest.raises(RuntimeError, match='exact DeepSpeed versions'):
            require_supported_version()


def test_dataset_queries_accept_iterable_columns_without_changing_selection():
    torch = pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam import data
    class Column:
        def __init__(self, values): self.values = values
        def __iter__(self): return iter(self.values)
    selections = []
    class Dataset:
        def select(self, indices):
            selections.append(indices)
            return {key: Column([torch.tensor(float(i)) for i in indices]) for key in ('action', 'images', 'timestamp')}
    owner = SimpleNamespace(hf_dataset=Dataset(), meta=SimpleNamespace(video_keys=['video']), during_training=True)
    result = data._query_hf_dataset_fast(owner, {'action': [1, 3], 'images': [1, 3], 'video': [1, 3]})
    assert len(selections) == 1
    torch.testing.assert_close(result['action'], torch.tensor([1., 3.]))
    owner.during_training = False
    assert set(data._query_hf_dataset_fast(owner, {'action': [1], 'images': [1]})) == {'action'}
    assert data._get_query_timestamps(owner, 0.5, {'video': [2, 4]}) == {'video': [2., 4.]}


def test_fused_training_preserves_accumulation_scheduler_eval_and_saves():
    from contextlib import nullcontext
    torch = pytest.importorskip('torch')
    calls = []
    class Accelerator:
        num_processes = 2
        is_main_process = True
        optimizer_step_was_skipped = False
        sync_gradients = False
        def __init__(self): self.micro_step = 0
        def accumulate(self, model):
            self.micro_step += 1
            self.sync_gradients = self.micro_step % 2 == 0
            return nullcontext()
        def autocast(self): return nullcontext()
        def backward(self, loss): calls.append('backward')
        def clip_grad_norm_(self, *a): return torch.tensor(3.)
        def gather(self, x): calls.append('gather'); return x
        def wait_for_everyone(self): pass
        def unwrap_model(self, model): return model
    model = SimpleNamespace(training_loss=lambda x: (torch.tensor(float(x)), {'action': 1.}),
                            parameters=lambda: [], mot=SimpleNamespace())
    trainer = SimpleNamespace(
        accelerator=Accelerator(), model=model, max_steps=2, global_step=0,
        epoch=0, batch_in_epoch=0, batch_size=8, log_every=1, save_every=2,
        eval_every=1, val_dataset=object(), max_grad_norm=1., train_loader=[1, 2, 3, 4],
        optimizer=SimpleNamespace(step=lambda: calls.append('step'),
                                  zero_grad=lambda **k: calls.append('zero'), param_groups=[{'lr': 1e-4}]),
        scheduler=SimpleNamespace(step=lambda: calls.append('scheduler')),
        _set_dit_only_train_mode=lambda: calls.append('mode'),
        _estimate_eta=lambda: ('1s', 2.), _wandb_log=lambda p: calls.append(('log', p)),
        evaluate=lambda: {'val_loss': 1., 'psnr_rd': 30., 'video_path': '/fixture/eval.mp4'},
        save_checkpoint=lambda: calls.append('save'))
    training.fused_metrics_wrapper(lambda self: None, {})(trainer)
    assert trainer.global_step == 2 and trainer.batch_in_epoch == 4
    assert calls.count('backward') == 4
    assert calls.count('gather') == calls.count('scheduler') == calls.count('step') == 2
    assert calls.count('save') == 2  # periodic and final, matching upstream
    assert len([c for c in calls if isinstance(c, tuple) and 'eval/val_loss' in c[1]]) == 2


def test_training_exception_still_drains_pending_parameter_gathers():
    trainer = SimpleNamespace(_set_dit_only_train_mode=lambda: None, max_steps=1)
    with patch.object(training, '_train', side_effect=RuntimeError('backward failed')), patch.object(training, '_finish') as finish:
        with pytest.raises(RuntimeError, match='backward failed'):
            training.fused_metrics_wrapper(lambda self: None, {})(trainer)
        finish.assert_called_once_with(trainer)


def test_arrow_cache_reuses_manifest_and_rebuilds_missing_files(tmp_path, monkeypatch):
    from turbo_physai.optimizations.models.loopwam import data
    arrow = tmp_path / 'part.arrow'
    builder = SimpleNamespace(cache_files=[{'filename': str(arrow)}], set_transform=Mock())
    def build(*args, **kwargs):
        arrow.write_bytes(b'arrow fixture')
        return builder
    load = Mock(side_effect=build)
    reader = SimpleNamespace(set_transform=Mock())
    from_file = Mock(return_value=reader)
    datasets = SimpleNamespace(Dataset=SimpleNamespace(from_file=from_file),
                               load_dataset=load, concatenate_datasets=Mock())
    upstream = SimpleNamespace(hf_transform_to_torch=object())
    monkeypatch.setenv('HF_DATASETS_CACHE', str(tmp_path / 'cache'))
    owner = SimpleNamespace(root=tmp_path / 'dataset', episodes=None)
    owner.root.mkdir()
    # Simulate packages only at the two dependency entry points used here.
    with fake_module('datasets', datasets), fake_module(
            'fastwam.datasets.lerobot.lerobot.lerobot_dataset', upstream):
        assert data.load_hf_dataset(owner) is builder
        assert data.load_hf_dataset(owner) is reader
        assert load.call_count == 1
        from_file.assert_called_once_with(str(arrow))
        arrow.unlink()
        assert data.load_hf_dataset(owner) is builder
        assert load.call_count == 2


def test_dataset_init_does_not_read_unused_timestamp_columns(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam import data
    class UnusedColumns:
        def __getitem__(self, key):
            raise AssertionError('disabled validation must not materialize columns')
    class Dataset(torch.utils.data.Dataset):
        def get_episodes_file_paths(self): return ['episode.parquet']
        def load_hf_dataset(self): return UnusedColumns()
    (tmp_path / 'episode.parquet').touch()
    upstream = SimpleNamespace(
        HF_LEROBOT_HOME=tmp_path, CODEBASE_VERSION='v2.1',
        LeRobotDatasetMetadata=lambda *a, **k: SimpleNamespace(episodes={}),
        get_safe_default_codec=lambda: 'torchcodec', aggregate_stats=Mock(),
        get_episode_data_index=lambda *a: {'from': torch.tensor([0]), 'to': torch.tensor([1])},
        check_timestamps_sync=Mock(), get_delta_indices=Mock())
    monkeypatch.delenv('FASTWAM_VALIDATE_TIMESTAMPS', raising=False)
    with fake_module('fastwam.datasets.lerobot.lerobot.lerobot_dataset', upstream):
        owner = Dataset()
        data.dataset_init(owner, 'fixture', root=tmp_path)
        assert isinstance(owner.hf_dataset, UnusedColumns)
        assert owner.delta_indices is None
        upstream.check_timestamps_sync.assert_not_called()


def test_gradient_overlap_is_installed_after_trainer_resume():
    pytest.importorskip('torch')
    from turbo_physai.optimizations.models.loopwam import grad_overlap
    owner = SimpleNamespace()
    events = []
    def initialize(self, marker):
        self.model = marker
        events.append('resume-complete')
    with patch.object(grad_overlap, 'install', side_effect=lambda engine: events.append(engine)):
        training.gradient_overlap_wrapper(initialize, {})(owner, 'prepared-engine')
    assert events == ['resume-complete', 'prepared-engine']


def test_launch_script_preserves_rank_counts_paths_and_hydra_arguments(tmp_path):
    import json
    script = Path(__file__).resolve().parents[2] / 'model_examples/LoopWAM/train_hcu.sh'
    model = tmp_path / 'model'; model.mkdir()
    binary = tmp_path / 'bin'; binary.mkdir()
    output = tmp_path / 'argv.json'
    fake_python = binary / 'python'
    fake_python.write_text('#!/bin/sh\nexec ' + sys.executable +
        ' -c \'import json,os,sys;open(os.environ["ARGV_OUTPUT"],"w").write(json.dumps(sys.argv[1:]))\' "$@"\n')
    fake_python.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               ARGV_OUTPUT=str(output), LOOPWAM_REPO=str(model),
               LOOPWAM_DATA_ROOT='/datasets', LOOPWAM_TEXT_CACHE_ROOT='/embeddings',
               NNODES='2', NPROC_PER_NODE='8', MACHINE_RANK='1', MASTER_ADDR='head',
               OUTPUT_DIR='/output/run with spaces')
    result = subprocess.run(['bash', str(script), 'max_steps=120', 'wandb.name=trial with spaces'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = json.loads(output.read_text())
    assert argv[argv.index('--num_processes') + 1] == '16'
    assert argv[argv.index('--machine_rank') + 1] == '1'
    assert 'output_dir=/output/run with spaces' in argv
    assert argv[-2:] == ['max_steps=120', 'wandb.name=trial with spaces']

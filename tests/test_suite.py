import json
from types import SimpleNamespace

import pytest

from vdd.suite import METHODS, aggregate, stage, verified


def test_stage_resume_and_corruption(tmp_path, monkeypatch):
    local, remote = tmp_path / 'local', tmp_path / 'drive'
    calls = []
    def command(out):
        calls.append(out)
        out.mkdir()
        (out / 'metrics.json').write_text('{"ok": true}')
        return ['fake']
    monkeypatch.setattr('vdd.suite.subprocess.Popen', lambda *a, **k: SimpleNamespace(stdout=['done\n'], wait=lambda: 0))
    first = stage('test', 'bottle', local, remote, command, ['metrics.json'], tmp_path)
    second = stage('test', 'bottle', local, remote, command, ['metrics.json'], tmp_path)
    assert first == second and len(calls) == 1
    receipt = remote / 'bottle/test/complete.json'
    saved = verified(receipt)
    (saved / 'metrics.json').write_text('corrupted')
    with pytest.raises(ValueError, match='changed saved stage'):
        verified(receipt)


def test_failed_stage_has_no_completion_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr('vdd.suite.subprocess.Popen', lambda *a, **k: SimpleNamespace(stdout=['failure\n'], wait=lambda: 1))
    with pytest.raises(RuntimeError, match='exit 1'):
        stage('test', 'bottle', tmp_path / 'local', tmp_path / 'remote', lambda out: ['fake'], ['model.pt'], tmp_path)
    assert not (tmp_path / 'remote/bottle/test/complete.json').exists()


def test_macro_mean_does_not_zero_fill_missing_categories(tmp_path):
    for category, auroc in [('bottle', 1.), ('cable', .8)]:
        directory = tmp_path / category
        directory.mkdir()
        (directory / 'comparison.json').write_text(json.dumps({method: {'auroc': auroc} for method in METHODS}))
    result = aggregate(tmp_path)
    assert result['metrics']['patchcore']['categories'] == 2
    assert result['metrics']['patchcore']['auroc'] == pytest.approx(.9)


def test_fresh_bottle_run_without_legacy_results(tmp_path, monkeypatch):
    from vdd import suite

    remote, local = tmp_path / 'saved', tmp_path / 'work'
    monkeypatch.setattr('sys.argv', ['suite', '--root', str(tmp_path / 'data'),
                                    '--remote', str(remote), '--local', str(local),
                                    '--categories', 'bottle', '--device', 'cpu'])
    def manifest(root, category, path, **kwargs):
        path.write_text('{}')
    monkeypatch.setattr('vdd.data.make_manifest', manifest)
    monkeypatch.setattr(suite.subprocess, 'check_output', lambda *a, **k: 'test environment')
    calls = []
    def fake_stage(name, category, local, remote, command, required, project):
        target = local / category / name
        target.mkdir(exist_ok=True)
        cmd = command(target)
        assert cmd[cmd.index('--device') + 1] == 'cpu'
        calls.append((category, name))
        metrics = {'auroc': .9}
        if name == 'baseline_evaluate':
            metrics = {method: metrics for method in METHODS[:3]}
        (target / 'metrics.json').write_text(json.dumps({'metrics': metrics}))
        return target
    monkeypatch.setattr(suite, 'stage', fake_stage)
    suite.main()
    assert len(calls) == 5
    assert all(category == 'bottle' for category, _ in calls)
    result = json.loads((remote / 'summary.json').read_text())
    assert result['complete_categories'] == result['expected_categories'] == ['bottle']
    assert set(result['metrics']) == set(METHODS)
    assert not (remote / 'bottle/reused_from.json').exists()
    # A changed protocol must not silently overwrite a previous experiment.
    protocol = json.loads((remote / 'protocol.json').read_text())
    protocol['size'] = 128
    (remote / 'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='Protocol or code changed'):
        suite.main()


@pytest.mark.parametrize('nested', [False, True])
def test_suite_rejects_overlapping_work_and_saved_paths(tmp_path, monkeypatch, nested):
    from vdd.suite import main

    local = tmp_path / 'run'
    remote = local / 'saved' if nested else local
    monkeypatch.setattr('sys.argv', ['suite', '--root', str(tmp_path / 'data'),
                                    '--remote', str(remote), '--local', str(local)])
    with pytest.raises(SystemExit):
        main()
    assert not local.exists()


def test_analysis_reads_fresh_and_legacy_bottle_layouts(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'notebooks/analyze_suite.py'
    spec = importlib.util.spec_from_file_location('analyze_suite', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage_dir = tmp_path / 'bottle/patchcore'
    stage_dir.mkdir(parents=True)
    (stage_dir / 'complete.json').write_text('{"attempt": "attempt_1"}')
    assert module.stage_path(tmp_path, 'bottle', 'patchcore') == stage_dir / 'attempt_1'
    old = tmp_path / 'old_bottle'
    (tmp_path / 'bottle/reused_from.json').write_text(json.dumps({'path': str(old)}))
    assert module.stage_path(tmp_path, 'bottle', 'patchcore') == old / 'patchcore'
    assert module.stage_path(tmp_path, 'bottle', 'baseline_evaluate') == old / 'baseline/report'

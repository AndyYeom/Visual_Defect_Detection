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

"""Resumable, stage-level MVTec comparison. Completion requires verified Drive copies."""
 # stage callbacks execute synchronously; closures never escape the iteration.
# ruff: noqa: B023
import argparse
import csv
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
METHODS = ['autoencoder', 'cnn', 'ensemble', 'patchcore', 'anomalydino', 'ae_anomalydino']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def verified(receipt):
    receipt = Path(receipt)
    if not receipt.is_file():
        return None
    data = json.loads(receipt.read_text())
    folder = receipt.parent / data['attempt']
    for name, digest in data['files'].items():
        file = folder / name
        if not file.is_file() or sha(file) != digest:
            raise ValueError(f'Incomplete or changed saved stage: {file}')
    return folder


def stage(name, category, local, remote, command, required, project):
    saved = remote / category / name
    cached = verified(saved / 'complete.json')
    if cached:
        target = local / category / name / cached.name
        if not target.exists():
            shutil.copytree(cached, target)
        else:
            receipt = json.loads((saved / 'complete.json').read_text())
            for f, digest in receipt['files'].items():
                if not (target / f).is_file() or sha(target / f) != digest:
                    shutil.copy2(cached / f, target / f)
        print(f'SKIP verified {category}/{name}', flush=True)
        return target
    attempt = datetime.datetime.now(datetime.UTC).strftime('attempt_%Y%m%d_%H%M%S_%f')
    target = local / category / name / attempt
    target.parent.mkdir(parents=True, exist_ok=True)
    log_path = target.parent / (attempt + '.log')
    args = command(target)
    print(f'START {category}/{name}', flush=True)
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', *map(str, args)], cwd=project,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            # Avoid rendering hundreds of coreset progress bars in Colab.
            if 'Selecting Coreset Indices' not in line:
                print(line, end='', flush=True)
        code = process.wait()
    saved.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log_path, saved / log_path.name)
    if target.exists():
        shutil.copytree(target, saved / attempt)
    if code:
        raise RuntimeError(f'{category}/{name} failed: exit {code}; log saved to Drive')
    for file in required:
        if not (target / file).is_file():
            raise RuntimeError(f'Missing required artifact: {target / file}')
    # Hash every artifact, not just the presence of a metrics file.
    files = {str(p.relative_to(target)): sha(p) for p in target.rglob('*') if p.is_file()}
    for artifact, digest in files.items():
        if sha(saved / attempt / artifact) != digest:
            raise RuntimeError(f'Copy verification failed: {artifact}')
    write_json(saved / 'complete.json', {'attempt': attempt, 'files': files})
    print(f'SAVED AND VERIFIED {category}/{name}', flush=True)
    return target


def aggregate(remote, categories=None):
    categories = CATEGORIES if categories is None else categories
    rows, complete = [], []
    for category in categories:
        file = remote / category / 'comparison.json'
        if not file.is_file():
            continue
        data = json.loads(file.read_text())
        if set(data) != set(METHODS):
            raise ValueError(f'Incomplete method comparison for {category}')
        complete.append(category)
        for method, values in data.items():
            rows.append({'category': category, 'method': method, **values})
    fields = ['category', 'method'] + sorted({key for r in rows for key in r} - {'category', 'method'})
    with (remote / 'category_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        summary[method] = {'categories': len(selected)}
        for metric in ['auroc', 'f1', 'precision', 'recall', 'false_positive_rate', 'pixel_auroc', 'pixel_ap']:
            values = [r[metric] for r in selected if r.get(metric) is not None]
            if values:
                summary[method][metric] = sum(values) / len(values)
    result = {'complete_categories': complete, 'expected_categories': categories,
              'aggregation': 'Unweighted category macro mean; missing categories are never zero-filled',
              'metrics': summary}
    write_json(remote / 'summary.json', result)
    print(f'COMPLETED CATEGORIES: {len(complete)}/{len(categories)}', flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--remote', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--reuse-bottle', help='Optional compatible legacy bottle run; omitted for fresh runs')
    p.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cuda')
    args = p.parse_args()
    root, remote, local = map(Path, [args.root, args.remote, args.local])
    if local.resolve() == remote.resolve() or local.resolve().is_relative_to(remote.resolve()) or remote.resolve().is_relative_to(local.resolve()):
        p.error('--local and --remote must be separate, non-nested directories')
    categories = list(dict.fromkeys(args.categories))
    if args.reuse_bottle and 'bottle' not in categories:
        p.error('--reuse-bottle requires bottle in --categories')
    project = Path(__file__).resolve().parents[2]
    remote.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    protocol = {'size': 224, 'seed': 42, 'epochs': 10, 'batch_size': 16, 'quantile': .95,
                'weight': .5, 'validation_fraction': .2, 'anomalib': '2.7.0',
                'categories': categories, 'methods': METHODS, 'device': args.device,
                'reused_bottle': str(Path(args.reuse_bottle).resolve()) if args.reuse_bottle else None,
                'code': {name: sha(Path(__file__).parent / name) for name in
                         ['benchmark.py', 'pipeline.py', 'models.py', 'data.py', 'scoring.py', 'device.py', 'suite.py']}}
    protocol_path = remote / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol or code changed: choose a new experiment folder')
    write_json(protocol_path, protocol)
    (remote / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    print('EXPERIMENT:', remote, flush=True)
    if args.reuse_bottle:
        # Existing bottle was evaluated at the exact configuration before the other categories.
        old = Path(args.reuse_bottle)
        bottle = remote / 'bottle'
        bottle.mkdir(exist_ok=True)
        baseline_meta = json.loads((old / 'baseline/model/training.json').read_text())
        manifest_hash = sha(old / 'split.json')
        if baseline_meta['manifest_sha256'] != manifest_hash or any(baseline_meta[k] != protocol[k] for k in ['size', 'seed', 'epochs', 'batch_size']):
            raise ValueError('Bottle baseline configuration mismatch')
        for method in ['patchcore', 'anomalydino']:
            meta = json.loads((old / method / 'training.json').read_text())
            from .benchmark import SETTINGS
            if meta['size'] != 224 or meta['seed'] != 42 or meta['manifest_sha256'] != manifest_hash or meta['settings'] != SETTINGS[method]:
                raise ValueError('Bottle benchmark configuration mismatch')
        shutil.copy2(old / 'comparison.json', bottle / 'comparison.json')
        write_json(bottle / 'reused_from.json', {'path': str(old), 'manifest_sha256': manifest_hash,
                                              'note': 'Existing complete bottle run; original model artifacts remain here'})
    failures = {}
    aggregate(remote, categories)
    for category in categories:
        if category == 'bottle' and args.reuse_bottle:
            continue
        try:
            folder = local / category
            folder.mkdir(exist_ok=True)
            manifest = folder / 'split.json'
            saved_manifest = remote / category / 'split.json'
            if saved_manifest.exists():
                shutil.copy2(saved_manifest, manifest)
            else:
                from .data import make_manifest
                make_manifest(root, category, manifest, seed=42, validation_fraction=.2)
                saved_manifest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(manifest, saved_manifest)
            common = ['--root', root, '--manifest', manifest]
            train = stage('baseline_train', category, local, remote, lambda out: [
                'vdd.cli', 'train', *common, '--output', out, '--epochs', 10, '--size', 224,
                '--batch-size', 16, '--seed', 42, '--weight', .5, '--quantile', .95, '--device', args.device],
                ['model.pt', 'training.json'], project)
            report = stage('baseline_evaluate', category, local, remote, lambda out: [
                'vdd.cli', 'evaluate', *common, '--checkpoint', train / 'model.pt', '--output', out, '--device', args.device],
                ['metrics.json', 'predictions.csv'], project)
            outputs = {}
            for method in ['patchcore', 'anomalydino']:
                outputs[method] = stage(method, category, local, remote, lambda out: [
                    'vdd.benchmark', *common, '--output', out, '--method', method, '--size', 224, '--device', args.device],
                    ['model.pt', 'training.json', 'metrics.json', 'predictions.csv', 'validation.csv', 'anomaly_maps.npz'], project)
            outputs['ae_anomalydino'] = stage('ae_anomalydino', category, local, remote, lambda out: [
                'vdd.benchmark', *common, '--output', out, '--method', 'ae_anomalydino', '--device', args.device,
                '--checkpoint', train / 'model.pt', '--dino-dir', outputs['anomalydino']],
                ['metrics.json', 'predictions.csv'], project)
            comparison = json.loads((report / 'metrics.json').read_text())['metrics']
            for method, directory in outputs.items():
                comparison[method] = json.loads((directory / 'metrics.json').read_text())['metrics']
            write_json(remote / category / 'comparison.json', comparison)
            aggregate(remote, categories)
        except Exception as exc:  # noqa: BLE001 - preserve progress and continue independent categories
            failures[category] = str(exc)
            print('CATEGORY FAILED:', category, str(exc), flush=True)
        write_json(remote / 'status.json', {'failures': failures, 'last_category': category,
                                           'updated_utc': datetime.datetime.now(datetime.UTC).isoformat()})
    result = aggregate(remote, categories)
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise RuntimeError(f'Incomplete categories: {failures}; rerun unchanged to retry failed stages')
    print(f'ALL {len(categories)} CATEGORIES COMPLETE AND SAVED', flush=True)


if __name__ == '__main__':
    main()

"""Run after the GPU baseline notebook has mounted Drive and extracted MVTec."""
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path('/content/drive/MyDrive/PRS_Group30')
PROJECT = Path('/content/Visual_Defect_Detection_GPU')
ROOT = Path('/content/mvtec_data')
ORIGINAL = BASE / 'results/20261030_134620_449309/bottle/split.json'
RUN_ID = 'benchmark224_' + datetime.datetime.now(datetime.UTC).strftime('%Y%m%d_%H%M%S')
LOCAL = Path('/content/vdd_runs') / RUN_ID
REMOTE = BASE / 'results' / RUN_ID
LOCAL.mkdir(parents=True)
REMOTE.mkdir(parents=True)
shutil.copy2(ORIGINAL, LOCAL / 'split.json')
shutil.copy2(BASE / 'Visual_Defect_Detection_GPU.zip', REMOTE / 'source.zip')
(REMOTE / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
MANIFEST = LOCAL / 'split.json'
print('RESULTS:', REMOTE, flush=True)


def run(name, module, *args):
    cmd = [sys.executable, '-u', '-m', module, *map(str, args)]
    print('\nSTART', name, flush=True)
    with (LOCAL / (name + '.log')).open('w') as log:
        process = subprocess.Popen(cmd, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    shutil.copytree(LOCAL, REMOTE, dirs_exist_ok=True)
    if code:
        raise RuntimeError(f'{name} failed ({code}); saved logs at {REMOTE}')


run('tests', 'pytest', '-q')
run('baseline_train', 'vdd.cli', 'train', '--root', ROOT, '--manifest', MANIFEST,
    '--output', LOCAL / 'baseline/model', '--epochs', 10, '--size', 224,
    '--batch-size', 16, '--seed', 42, '--weight', .5, '--quantile', .95, '--device', 'cuda')
run('baseline_evaluate', 'vdd.cli', 'evaluate', '--root', ROOT, '--manifest', MANIFEST,
    '--checkpoint', LOCAL / 'baseline/model/model.pt', '--output', LOCAL / 'baseline/report', '--device', 'cuda')
for method in ['patchcore', 'anomalydino']:
    run(method, 'vdd.benchmark', '--root', ROOT, '--manifest', MANIFEST,
        '--output', LOCAL / method, '--method', method, '--size', 224, '--device', 'cuda')
run('ae_anomalydino', 'vdd.benchmark', '--root', ROOT, '--manifest', MANIFEST,
    '--output', LOCAL / 'ae_anomalydino', '--method', 'ae_anomalydino', '--device', 'cuda',
    '--checkpoint', LOCAL / 'baseline/model/model.pt', '--dino-dir', LOCAL / 'anomalydino')
comparison = json.loads((LOCAL / 'baseline/report/metrics.json').read_text())['metrics']
for method in ['patchcore', 'anomalydino', 'ae_anomalydino']:
    comparison[method] = json.loads((LOCAL / method / 'metrics.json').read_text())['metrics']
(LOCAL / 'comparison.json').write_text(json.dumps(comparison, indent=2))
shutil.copytree(LOCAL, REMOTE, dirs_exist_ok=True)
print('\nCOMPLETE', REMOTE, json.dumps(comparison, indent=2), flush=True)

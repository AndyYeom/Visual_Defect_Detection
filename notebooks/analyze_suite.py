"""Summarize fixed-protocol experiments; never changes models or thresholds."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vdd.benchmark import mask_for
from vdd.suite import CATEGORIES, METHODS


def stage_path(run, category, stage):
    if category == 'bottle':
        old = Path(json.loads((run / category / 'reused_from.json').read_text())['path'])
        return old / ('baseline/report' if stage == 'baseline_evaluate' else stage)
    folder = run / category / stage
    return folder / json.loads((folder / 'complete.json').read_text())['attempt']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    run, root = Path(a.run), Path(a.root)
    summary = json.loads((run / 'summary.json').read_text())
    if len(summary['complete_categories']) != 15:
        raise ValueError('Analysis requires all 15 completed categories')
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    data = {c: json.loads((run / c / 'comparison.json').read_text()) for c in CATEGORIES}
    grid = np.array([[data[c][m]['auroc'] for m in METHODS] for c in CATEGORIES])
    names = ['AE', 'CNN', 'AE+CNN', 'PatchCore', 'AnomalyDINO', 'AE+DINO']
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    im = ax.imshow(grid, vmin=.5, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(6), names); ax.set_yticks(range(15), CATEGORIES)
    for i in range(15):
        for j in range(6):
            ax.text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center', color='white' if grid[i,j] > .86 else 'black')
    ax.set_title('MVTec AD: image AUROC by category\nNormal-only training; fixed 224px protocol, seed 42')
    fig.colorbar(im, ax=ax, label='Image AUROC'); fig.savefig(out/'image_auroc.png', dpi=160); plt.close(fig)
    failures = []
    for category in CATEGORIES:
        for method in METHODS:
            folder = stage_path(run, category, 'baseline_evaluate' if method in METHODS[:3] else method)
            with (folder / 'predictions.csv').open() as f:
                rows = list(csv.DictReader(f))
            threshold = data[category][method]['threshold']
            for row in rows:
                score = float(row[method] if method in METHODS[:3] else row['score'])
                label = int(row['label']); prediction = int(score > threshold)
                if label != prediction:
                    failures.append({'category':category, 'method':method, 'path':row['path'],
                                     'label':label, 'prediction':prediction, 'score':score,
                                     'threshold':threshold, 'error':'missed_defect' if label else 'false_alarm'})
    with (out/'errors.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['category','method','path','label','prediction','score','threshold','error'])
        w.writeheader(); w.writerows(failures)
    # Lowest-AUROC category of each new model, lowest-scoring defect (explicit failure analysis).
    for method in ['patchcore','anomalydino']:
        category = min(CATEGORIES, key=lambda c:data[c][method]['auroc'])
        folder = stage_path(run,category,method)
        with (folder/'predictions.csv').open() as f: rows=list(csv.DictReader(f))
        defects = sorted((r for r in rows if int(r['label'])==1),key=lambda r:float(r['score']))[:3]
        maps = np.load(folder/'anomaly_maps.npz')
        lookup = {path:i for i,path in enumerate(maps['paths'])}
        fig, axs=plt.subplots(len(defects),3,figsize=(9,3*len(defects)),squeeze=False,constrained_layout=True)
        for i,row in enumerate(defects):
            image=Image.open(root/row['path']).convert('RGB').resize((224,224))
            mask=mask_for(root,{'path':row['path'],'label':1},224)
            heatmap=maps['maps'][lookup[row['path']]]
            axs[i,0].imshow(image); axs[i,1].imshow(mask,cmap='gray',vmin=0,vmax=1)
            axs[i,2].imshow(image); axs[i,2].imshow(heatmap,cmap='inferno',alpha=.55)
            axs[i,0].set_title(row['path'].split('/test/')[-1])
            axs[i,1].set_title('Ground truth')
            axs[i,2].set_title(f"Score {float(row['score']):.3f} / threshold {data[category][method]['threshold']:.3f}")
            for ax in axs[i]: ax.axis('off')
        fig.suptitle(f'{method}: {category} — three lowest-scoring defects\nHeatmaps normalized per image for display only')
        fig.savefig(out/f'{method}_hard_cases.png',dpi=150); plt.close(fig)
    print(json.dumps({'analysis':str(out),'error_rows':len(failures),'macro_metrics':summary['metrics']},indent=2))


if __name__=='__main__': main()

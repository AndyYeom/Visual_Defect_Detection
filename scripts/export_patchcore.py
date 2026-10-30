"""Export existing evaluated weights without changing scores or calibration."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--local-root', type=Path)
    args = p.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run/'summary.json').read_text())
    assert len(summary['complete_categories']) == 15
    shared = None
    manifest = {'schema_version':1,'method':'patchcore','categories':{},'files':{}}
    for category in summary['complete_categories']:
        reused = category == 'bottle' and (run/category/'reused_from.json').is_file()
        if reused:
            old=Path(json.loads((run/category/'reused_from.json').read_text())['path'])
            folder=old/'patchcore'
        else:
            stage=run/category/'patchcore'
            folder=stage/json.loads((stage/'complete.json').read_text())['attempt']
        checkpoint = folder/'model.pt'
        if not checkpoint.exists() and args.local_root and not reused:
            checkpoint = args.local_root/category/'patchcore'/folder.name/'model.pt'
            receipt = json.loads((stage/'complete.json').read_text())
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == receipt['files']['model.pt']
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        backbone={k:v for k,v in state['state_dict'].items() if k!='memory_bank'}
        if shared is None:
            shared=backbone
            torch.save(shared,out/'backbone.pt')
        else:
            assert set(shared)==set(backbone)
            assert all(torch.equal(shared[k],backbone[k]) for k in shared), 'Backbones differ'
        torch.save({'memory_bank':state['state_dict']['memory_bank'],'metadata':state['metadata']},out/f'{category}.pt')
        manifest['categories'][category]={'file':f'{category}.pt','threshold':state['metadata']['calibration']['threshold']}
        print('Exported',category,flush=True)
    for file in out.glob('*.pt'):
        manifest['files'][file.name]={'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(run/'summary.json',out/'experiment_summary.json')
    shutil.copy2(run/'category_metrics.csv',out/'category_metrics.csv')
    shutil.copytree(run/'analysis',out/'analysis',dirs_exist_ok=True)
    archive=shutil.make_archive(str(out.parent/'patchcore-models-v1'), 'zip',out)
    print('PACKAGE',archive,Path(archive).stat().st_size,flush=True)


if __name__=='__main__': main()

"""Install the versioned model package with SHA-256 verification (no login needed)."""
import argparse
import hashlib
import json
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,help='Use an already downloaded ZIP instead of the network')
    p.add_argument('--destination',type=Path,default=ROOT/'artifacts/patchcore')
    args=p.parse_args()
    spec=json.loads((ROOT/'configs/model_release.json').read_text())
    with tempfile.TemporaryDirectory() as temporary:
        temporary=Path(temporary)
        archive=args.archive
        if archive is None:
            archive=temporary/'models.zip'
            print('Downloading trained models. This may take a few minutes.',flush=True)
            request=urllib.request.Request(spec['url'],headers={'User-Agent':'Group30-Visual-Defect-Detection'})
            context = ssl.create_default_context()
            context.load_verify_locations(cafile=certifi.where())
            with urllib.request.urlopen(request,timeout=180,context=context) as response,archive.open('wb') as target:
                shutil.copyfileobj(response,target)
        if digest(archive)!=spec['sha256']:
            raise ValueError('Archive checksum mismatch. Download the release asset again.')
        extracted=temporary/'extracted'; extracted.mkdir()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (extracted/member.filename).resolve().is_relative_to(extracted.resolve()):
                    raise ValueError('Unsafe archive path')
            z.extractall(extracted)
        manifest=json.loads((extracted/'manifest.json').read_text())
        for filename,expected in manifest['files'].items():
            if Path(filename).name!=filename or digest(extracted/filename)!=expected['sha256']:
                raise ValueError(f'Invalid model file: {filename}')
        args.destination.mkdir(parents=True,exist_ok=True)
        shutil.copytree(extracted,args.destination,dirs_exist_ok=True)
    print(f'Installed {len(manifest["categories"])} category models at {args.destination}')
    print('Start: python -m streamlit run app/streamlit_app.py')


if __name__=='__main__': main()

"""Offline inference from the verified shared-backbone deployment package."""
import hashlib
import json
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from PIL import Image

from .benchmark import predict_one, preprocessing


def verify_file(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Model file missing: {path}. Run python scripts/download_models.py')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != expected['sha256'] or path.stat().st_size != expected['bytes']:
        raise ValueError(f'Model integrity check failed: {path.name}. Download the package again.')


class PatchCoreDetector:
    def __init__(self, model_dir, category, device='auto'):
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
        self.directory = Path(model_dir)
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        if manifest['schema_version'] != 1 or manifest['method'] != 'patchcore':
            raise ValueError('Unsupported deployment package')
        if category not in manifest['categories']:
            raise ValueError('Category is not available in this model package')
        filename = manifest['categories'][category]['file']
        for name in ['backbone.pt', filename]:
            if Path(name).name != name:
                raise ValueError('Invalid model filename')
            verify_file(self.directory / name, manifest['files'][name])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        if self.device.type not in {'cpu', 'cuda'}:
            raise ValueError('The deployment supports CPU and CUDA')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; select CPU')
        torch.set_num_threads(2)
        state = torch.load(self.directory / filename, map_location='cpu', weights_only=True)
        self.metadata = state['metadata']
        if self.metadata['category'] != category or self.metadata['method'] != 'patchcore':
            raise ValueError('Model/category mismatch')
        shared = torch.load(self.directory / 'backbone.pt', map_location='cpu', weights_only=True)
        shared['memory_bank'] = state['memory_bank']
        settings = self.metadata['settings']
        self.model = PatchcoreModel(layers=settings['layers'], backbone=settings['backbone'],
                                   pre_trained=False, num_neighbors=settings['num_neighbors'])
        self.model.load_state_dict(shared)
        self.model.to(self.device).eval().requires_grad_(False)
        self.transform = preprocessing(self.metadata['size'], 'patchcore')
        self.lock = Lock()

    def predict(self, image):
        with self.lock:
            score, heatmap, elapsed = predict_one(self.model, image, self.transform, self.device)
        threshold = self.metadata['calibration']['threshold']
        return {'category': self.metadata['category'], 'method': 'PatchCore', 'score': score,
                'threshold': threshold, 'anomalous': bool(score > threshold),
                'inference_ms': elapsed, 'device': str(self.device), 'heatmap': heatmap}


def overlay_png(image, heatmap):
    values = np.asarray(heatmap, dtype=np.float32)
    values = (values - values.min()) / max(float(np.ptp(values)), 1e-8)
    colors = np.stack([values, np.zeros_like(values), 1-values], -1)
    heat = Image.fromarray((colors * 255).astype('uint8')).resize(image.size)
    overlay = Image.blend(image.convert('RGB'), heat, .42)
    output = BytesIO()
    overlay.save(output, format='PNG')
    return output.getvalue()

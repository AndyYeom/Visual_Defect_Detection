"""Group 30 final PatchCore inspection demo. No training or threshold tuning."""
import hashlib
import json
import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from vdd.patchcore_inference import PatchCoreDetector, overlay_png

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get('VDD_MODEL_DIR', str(ROOT / 'artifacts/patchcore')))
DATA_ROOT = os.environ.get('VDD_DATA_ROOT', '')
st.set_page_config(page_title='InspectIQ · Group 30', page_icon='🔎', layout='wide')
st.markdown('''<style>
.block-container {max-width:1200px;padding-top:4rem;}
[data-testid="stMetric"] {background:#eef5f7;border-radius:12px;padding:16px;}
h1 {letter-spacing:-1px;}
</style>''', unsafe_allow_html=True)
st.caption('NUS-ISS · PATTERN RECOGNITION SYSTEMS · GROUP 30')
st.title('InspectIQ — Visual Defect Detection')
st.write('Inspect a product image and locate regions that differ from normal appearance.')
with st.sidebar:
    st.header('Inspection setup')
    st.caption('PatchCore · 15 product categories')
if not (MODEL_DIR / 'manifest.json').is_file():
    st.info('One-time setup: download the trained models, then refresh this page.')
    st.code('python scripts/download_models.py')
    st.caption('See README for manual download and offline setup. No training is required.')
    st.stop()
try:
    manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
except (OSError, ValueError):
    st.error('The model manifest could not be read. Download the models again.')
    st.stop()
category = st.sidebar.selectbox('Product category', sorted(manifest['categories']), format_func=lambda s:s.replace('_',' ').title())
device = st.sidebar.selectbox('Compute device', ['auto', 'cpu', 'cuda'], format_func=lambda s:{'auto':'Automatic','cpu':'CPU','cuda':'NVIDIA GPU'}[s])
st.sidebar.info('Select the product type before inspection. A model cannot identify an unknown product category.')
st.sidebar.caption('224 × 224 model input · normal-only training · fixed validation threshold')
upload = st.file_uploader('Upload a product image', type=['png','jpg','jpeg'])
raw = upload.getvalue() if upload else None
source_name = upload.name if upload else None
if DATA_ROOT and (Path(DATA_ROOT)/category/'test').is_dir():
    with st.expander('Try an existing MVTec test example'):
        examples=sorted((Path(DATA_ROOT)/category/'test').glob('*/*.png'))
        choice=st.selectbox('Example image',examples,format_func=lambda p:str(p.relative_to(Path(DATA_ROOT)/category)))
        if st.button('Use this example'):
            st.session_state['example']=(category,str(choice),choice.read_bytes())
    example=st.session_state.get('example')
    if raw is None and example and example[0]==category:
        _,source_name,raw=example
if raw is None:
    st.markdown('### Ready for inspection')
    st.write('1. Choose the product category.\n2. Upload an image of that product.\n3. Select **Run inspection** to see the result and heatmap.')
    st.stop()
try:
    if len(raw)>20*1024*1024:
        raise ValueError('Please upload an image smaller than 20 MB.')
    with Image.open(BytesIO(raw)) as opened:
        if opened.width*opened.height > 25_000_000:
            raise ValueError('Please use an image with at most 25 million pixels.')
        image=ImageOps.exif_transpose(opened).convert('RGB')
except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
    st.error(f'Cannot read image: {error}')
    st.stop()
key=(category,device,hashlib.sha256(raw).hexdigest())


@st.cache_resource(max_entries=2)
def detector(category, device, manifest_stamp):
    return PatchCoreDetector(MODEL_DIR, category, device)


if st.button('Run inspection', type='primary'):
    try:
        with st.spinner('Loading model and inspecting image…'):
            model=detector(category,device,(MODEL_DIR/'manifest.json').stat().st_mtime_ns)
            result=model.predict(image)
            overlay=overlay_png(image,result.pop('heatmap'))
            result.update({'timestamp_utc':datetime.now(UTC).isoformat(),'image_sha256':key[2],
                           'filename':Path(source_name).name,'protocol':'224px; normal validation 95th percentile'})
            st.session_state['inspection']=(key,result,overlay)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        st.error(f'Inspection failed: {error}')
record=st.session_state.get('inspection')
if not record or record[0]!=key:
    st.image(image,caption='Ready to inspect',width=480)
    st.stop()
_,result,overlay=record
if result['anomalous']:
    st.error('Potential defect detected — review the highlighted regions.')
else:
    st.success('No anomaly detected at the current threshold.')
a,b,c=st.columns(3)
a.metric('Anomaly score',f"{result['score']:.3f}")
b.metric('Decision threshold',f"{result['threshold']:.3f}")
c.metric('Inference time',f"{result['inference_ms']:.0f} ms")
st.caption(f"{category.replace('_',' ').title()} · PatchCore · {result['device'].upper()} · Score above threshold = potential defect. Scores are not probabilities.")
left,right=st.columns(2)
left.image(image,caption='Original image',width='stretch')
right.image(overlay,caption='Suspected defect regions',width='stretch')
st.caption('Red indicates relatively higher anomaly scores. Colors are scaled per image for visualization; they are not a calibrated defect mask. Human review is required.')
a,b=st.columns(2)
a.download_button('Download inspection report',json.dumps(result,indent=2),file_name=f'{category}_inspection.json',mime='application/json')
b.download_button('Download heatmap overlay',overlay,file_name=f'{category}_overlay.png',mime='image/png')
with st.expander('About this model and evaluation'):
    st.write('Selected after comparing six methods on all 15 MVTec AD categories. PatchCore achieved macro image AUROC 0.9840 and F1 0.9514 under our fixed protocol. These are experimental averages, not a guarantee for a new image.')
    st.write('Training used normal images only. Thresholds were calibrated on held-out normal validation images. Category-average test false-positive rate was 15.2%; real deployment needs representative validation and human review.')

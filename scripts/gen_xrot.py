"""final_XRot.ipynb (worker-compatible): cross-domain protocol with 90-degree-rotated copies of the TN19 training images added.

Reason: 96% of the TN19 boxes have a vertical long side, 88-100% of the boxes of TT, BIM2 and TNDL a horizontal one.
Selection uses the TN19 validation split only; test = BIM2 + TNDL + TT of the cross-domain split, positives only, scored per farm.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA = '''import glob, os, shutil, zipfile
import cv2, numpy as np
BASE = '/content/drive/MyDrive/Colab Workspaces/solar_uav/01_uav_pv_benchmark'
DATA = '/content/data'
zipfile.ZipFile(f'{BASE}/dataset/solar_uav_obb_X.zip').extractall('/content/ex')
root = '/content/ex/solar_uav_obb_X'
for sub in ('images', 'labels'):
    for sp in ('train', 'val', 'test'):
        os.makedirs(f'{DATA}/{sub}/{sp}', exist_ok=True)
        for f in os.listdir(f'{root}/{sub}/{sp}'):
            shutil.move(f'{root}/{sub}/{sp}/{f}', f'{DATA}/{sub}/{sp}/{f}')
n_rot = 0
for f in sorted(glob.glob(f'{DATA}/images/train/*')):
    im = cv2.imread(f)
    stem = os.path.splitext(os.path.basename(f))[0]
    cv2.imwrite(f'{DATA}/images/train/{stem}__rot90.jpg', cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), [cv2.IMWRITE_JPEG_QUALITY, 95])
    out = []
    for line in open(f'{DATA}/labels/train/{stem}.txt'):
        v = line.split()
        if len(v) == 9:
            xy = np.array(v[1:], float).reshape(4, 2)
            new = np.stack([1 - xy[:, 1], xy[:, 0]], 1).reshape(-1)
            out.append(v[0] + ' ' + ' '.join('%.6f' % t for t in new))
    open(f'{DATA}/labels/train/{stem}__rot90.txt', 'w').write(chr(10).join(out) + (chr(10) if out else ''))
    n_rot += 1
def farm(f):
    return os.path.basename(f).split('_')[0]
lists = {sp: sorted(glob.glob(f'{DATA}/images/{sp}/*')) for sp in ('train', 'val', 'test')}
def w(name, fs):
    open(f'{DATA}/{name}.txt', 'w').write(chr(10).join(fs) + chr(10))
w('train_l', lists['train']); w('val_l', lists['val']); w('test_l', lists['test'])
for fm in ('BIM2', 'TNDL', 'TT'):
    w(f'test_{fm}', [f for f in lists['test'] if farm(f) == fm])
names = 'nc: 4' + chr(10) + 'names:' + chr(10) + '  0: SingleHotSpot' + chr(10) + '  1: MultiHotSpot' + chr(10) + '  2: Bypass' + chr(10) + '  3: ModuleOpenCircuit' + chr(10)
def yaml(name, test):
    open(f'{DATA}/{name}.yaml', 'w').write(f'path: {DATA}' + chr(10) + 'train: train_l.txt' + chr(10) + 'val: val_l.txt' + chr(10) + f'test: {test}.txt' + chr(10) + names)
yaml('all', 'test_l')
for fm in ('BIM2', 'TNDL', 'TT'):
    yaml(fm, f'test_{fm}')
print('train', len(lists['train']) + n_rot, 'images (', n_rot, 'rotated ), val', len(lists['val']), ', test', {fm: len([f for f in lists['test'] if farm(f) == fm]) for fm in ('BIM2', 'TNDL', 'TT')})
'''

RUN = '''import json, gc, time
from ultralytics import YOLO
TAG = 'XRot'
SEEDS = [0]
RES = f'{BASE}/results/final/{TAG}'
def ap_by_class(m, model):
    return {model.names[int(c)]: float(v) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}
def summ(m, model):
    return {'precision': float(m.box.mp), 'recall': float(m.box.mr), 'mAP50': float(m.box.map50), 'mAP50_95': float(m.box.map), 'per_class_ap50_95': ap_by_class(m, model)}
for seed in SEEDS:
    t0 = time.time()
    name = f'{TAG}_s{seed}'
    try:
        YOLO('yolo11n-obb.pt').train(data=f'{DATA}/all.yaml', epochs=100, imgsz=640, batch=16, seed=seed, deterministic=True, patience=30,
                                     project='/content/runs', name=name, workers=2, plots=False, exist_ok=True)
        best = YOLO(f'/content/runs/{name}/weights/best.pt')
        out = {'seed': seed, 'epochs_run': sum(1 for _ in open(f'/content/runs/{name}/results.csv')) - 1}
        for tag, yaml_, sp in (('val_tn19', 'all.yaml', 'val'), ('test_all', 'all.yaml', 'test'), ('test_BIM2', 'BIM2.yaml', 'test'), ('test_TNDL', 'TNDL.yaml', 'test'), ('test_TT', 'TT.yaml', 'test')):
            m = best.val(data=f'{DATA}/{yaml_}', split=sp, imgsz=640, conf=0.001, iou=0.6, plots=False, name=f'{name}_{tag}', exist_ok=True)
            out[tag] = summ(m, best)
        out['minutes'] = (time.time() - t0) / 60
        os.makedirs(f'{RES}/s{seed}', exist_ok=True)
        json.dump(out, open(f'{RES}/s{seed}/metrics.json', 'w'), indent=2)
        print('seed', seed, 'done', json.dumps({k: out[k]['mAP50_95'] for k in out if isinstance(out[k], dict)}))
    except Exception as e:
        print('seed', seed, 'FAILED', repr(e))
    gc.collect(); torch.cuda.empty_cache()
print('ALL DONE')
'''


def code(cells, s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})


def nb(cells):
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


cells = []
code(cells, "!pip -q install ultralytics\nimport ultralytics, torch\nprint(ultralytics.__version__, torch.__version__, torch.cuda.get_device_name(0))")
code(cells, "from google.colab import drive\ndrive.mount('/content/drive')")
code(cells, DATA)
code(cells, RUN)
json.dump(nb(cells), open(os.path.join(HERE, "final_XRot.ipynb"), "w", encoding="utf-8"), indent=1)
for c in cells[2:]:
    compile("".join(c["source"]), "xrot", "exec")
print("ok")

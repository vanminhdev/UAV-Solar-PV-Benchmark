# Baseline controls (Colab) and local scoring

Controls that the Technical Validation needs on top of the baseline of `train_baseline_obb.py`:
test-time rotation control, random image-level split control, stored weights, bootstrap over test frames, score against the A2/A3/consensus labels.

## Files

| file | where it runs | what it does |
|---|---|---|
| `collect_baseline_per_seed.py` | local | writes `dataset/metadata/splits/baseline_per_seed_results.csv` (per protocol and seed) and `baseline_same_class_means.csv` (same three classes for every protocol) from the stored `metrics.json` files |
| `colab_controls.py` | Colab GPU | jobs `X2` (cross-domain, plus rotation control), `C2` (main split, weights kept), `RS` (random split); three seeds each; writes `metrics.json`, `best.pt`, `results.csv`, `args.yaml` per run |
| `score_baseline_local.py` | local CPU | per-frame statistics of the stored weights, mAP@0.5:0.95, percentile bootstrap over frames (1,000 resamples, seed 20261026); also used to score against other label sets |

## Run on Colab

1. Open a Colab notebook with a GPU runtime (L4 or T4) and mount Drive (`drive.mount('/content/drive')`).
2. The dataset zips must be in `<BASE>/dataset/` (`solar_uav_obb.zip`, `solar_uav_obb_X.zip`, `solar_uav_obb_C2_neg.zip`, `solar_uav_obb_X2_neg.zip`; a zip may be stored as `parts/<zip>.partNNN` plus `parts/<zip>.parts_done`). Nothing needs to be uploaded if they are already there.
3. Copy `colab_controls.py` to the runtime and run, with `BASE = /content/drive/MyDrive/Colab Workspaces/solar_uav/01_uav_pv_benchmark`:

```
pip install -q ultralytics
python colab_controls.py --dataset-dir "$BASE/dataset" --results "$BASE/results/controls" --work /content/work --tags X2 C2 RS --par 3
```

`--par 3` runs three trainings at once on one GPU (about 23 s per epoch each on an L4, about 45 min per run). A finished run is skipped when
`results/controls/<TAG>/s<seed>/metrics.json` exists, so the command can be restarted. Keep the browser tab open (Colab stops background jobs when the session is closed).

What the jobs do:

* `X2`: trains on TN19 (protocol cross-domain), scores the usual metrics, then scores the same weights on the test frames of BIM2, TNDL and TT
  rotated by 90 degrees clockwise (labels rotated with the images), rotated by 180 degrees (null control, orientation unchanged) and on the TN19
  validation frames rotated by 90 degrees. All per farm.
* `C2`: protocol C again, only to keep `best.pt` (the first round did not store weights); the metrics are compared with the stored first-round
  values (`.local/colab_upload/results_final_v2/C2`) to check reproducibility.
* `RS`: pools the frames of protocol C (positives and background frames; same four classes) and re-assigns them at random to train, validation and
  test with the sizes of protocol C (positives and background separately). Split seed 1000 + training seed. `rs_split.csv` lists the assignment.

## Local scoring

```
python -m venv --system-site-packages venv && venv/Scripts/pip install -U ultralytics   # >= 8.3
python -X utf8 score_baseline_local.py bootstrap --tag C2 --weights <controls>/C2 --images <view>/images/test --work <tmp> --out <out>
```

`--list` restricts the frames, `--labels` selects another label folder (used for the A2/A3/consensus scoring on the agreement frames that lie in the test split).
The interval of the three-seed mean uses the same frame resample for the three seeds.

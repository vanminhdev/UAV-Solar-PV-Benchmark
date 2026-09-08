# UAV Solar PV Codebase & Benchmarking Pipeline

Source code and reproducible experiment pipelines accompanying the scientific paper:
*"A Multi-Scale, Multimodal UAV Photogrammetry and Radiometric Thermal Dataset for Persistent PV Module-Level Spatial Benchmarking"*

## Structure

- `configs/`: Pipeline settings, CRS, I/O paths, baseline hyperparameters (`pipeline_config.yaml`).
- `src/`: Modular Python library:
  - `geo_utils.py`: Coordinate transformations, camera ray-casting, georeferencing.
  - `gpkg_builder.py`: Building Master GeoPackage with `aoi_boundary` and `modules`.
  - `spatial_split.py`: Footprint-disjoint spatial strip split with buffer isolation.
  - `mapper.py`: Spatial registration and module association algorithms.
  - `metrics.py`: Top-1 accuracy, RMSE 2D, mIoU, Cohen\'s Kappa, mAP.
- `notebooks/`: Step-by-step reproducible workflows (01 to 05).
- `scripts/`: Independent validation scripts (e.g., `compute_qc_metrics.py`).
- `models/`: Weights and configs for baseline models.

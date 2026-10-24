# UAV Solar PV Benchmark - code

Code accompanying the dataset *"A multimodal multi-altitude UAV dataset with persistent module identifiers for photovoltaic inspection in Vietnam"*.
The dataset itself is distributed separately (Zenodo); this repository only holds processing code.

## Layout

| Path | Content |
|---|---|
| `qgis/` | QGIS Python-console scripts used by the annotators: `draw_grid.py` (draw a module grid along a reference line), `module_id.py` (sweep-line `String_ID`/`module_id` assignment), `build_master_gpkg.py` (AOI + master GeoPackage), `editcolor_export.py` (styling). Original, unmodified copies. |
| `scripts/build_image_inventory.py` | EXIF/DJI-XMP inventory of all UAV JPGs -> `image_inventory.csv`, `flight_summary.csv`. |
| `scripts/inspect_geo.py` | GeoTIFF header audit (CRS, GSD, bounds) and GeoPackage audit (module count, unique ids, module size). |
| `scripts/drive_tools.py` | Crawl and resumably download public Google Drive folders. |
| `configs/pipeline_config.yaml` | Paths, CRS and split settings. |

Training, evaluation, spatial-split and mapping code will be added under `src/` and `notebooks/`;
model weights go under `models/` (git-ignored).

## Setup

```
pip install -r requirements.txt
```

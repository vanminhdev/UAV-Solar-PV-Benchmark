# UAV Solar PV Benchmark - code

Code accompanying the dataset *"A multimodal multi-altitude UAV dataset with persistent module identifiers for photovoltaic inspection in Vietnam"*.
The dataset itself is distributed separately (Zenodo; see `dataset/README.md` and `dataset/DATA_DICTIONARY.md`); this repository holds the code that built the release tables and the code behind the Technical Validation.

All paths in the scripts are relative to the dataset root unless a `--dataset`, `--out` or similar argument says otherwise. Every script that writes a table says so in its docstring (`python scripts/<name>.py --help` or read the first lines).

## Setup

```
pip install -r requirements.txt
```

Python 3.10 or newer. The QGIS scripts run inside the QGIS Python console, not in this environment. Scripts that need the DJI Thermal SDK (`decode_rjpeg.py`) expect the SDK libraries to be installed separately; the SDK is not redistributed. The baseline script needs `torch` and `ultralytics` (version used for the paper: ultralytics 8.4.198) and a GPU for reasonable run times.

## Layout

| Path | Content |
|---|---|
| `configs/pipeline_config.yaml` | Early pipeline settings (paths, CRS). The split settings in it are superseded by the split scripts below and the tables in `metadata/splits/`. |
| `qgis/` | QGIS console scripts used for the module reference grid (original, unmodified copies): `draw_grid.py`, `module_id.py`, `build_master_gpkg.py`, `editcolor_export.py`. |
| `scripts/` | Processing, split, leakage-check, label and baseline scripts, grouped below. |
| `tools/label_tool/` | Single-file browser tool used to draw and check oriented thermal boxes (`label_tool.html`), and its portable build script (`build_portable.py`). No server is needed; open the HTML file in a browser. |
| `requirements.txt` | Python dependencies of the scripts. |

## Scripts by step

### 1. Organising the raw data and building the inventory
| Script | What it does |
|---|---|
| `drive_tools.py` | Crawl and resumably download public Google Drive folders. |
| `organise_dataset.py`, `organise_maps.py`, `organise_reports.py` | Move raw images, orthomosaics, DSMs, camera files and processing reports into the dataset layout using the `flight_id` naming rule. |
| `build_image_inventory.py` | EXIF and DJI XMP inventory of every UAV JPG -> `image_inventory.csv`, `flight_summary.csv`. |
| `build_flight_manifest.py` | `flight_manifest.csv` and `thermal_speed.csv`. |
| `xmp_speed_and_flags.py`, `apply_xmp_speed_primary.py` | Cross-check of ground speed from XMP and per-image quality flags; makes the XMP flight-controller speed the primary standard. |
| `build_metadata_field_coverage.py` | Which metadata fields each camera system records, and for how many images. |
| `build_pairing.py` | RGB (W) to thermal (T) pairing table. |
| `register_rgb_thermal.py` | Transfer-error study of the RGB-thermal registration (DJI H20T shots). |
| `inspect_geo.py` | Read-only audit of GeoTIFF headers and GeoPackage contents. |
| `parse_metashape_reports.py` | Summarise the Metashape PDF reports into one table. |
| `make_checksums.py` | SHA-256 checksums of the dataset. |

### 2. Thermal decoding and environment context
| Script | What it does |
|---|---|
| `decode_rjpeg.py` | Decode radiometric R-JPEG frames into temperature matrices with the DJI Thermal SDK (libdirp) through ctypes. |
| `check_thermal_alignment.py` | Pixel alignment of the palette rendering against the decoded temperature matrix. |
| `fetch_environment_context.py` | Hourly reanalysis weather and irradiance per farm and flight (model-based context, not in-situ measurements). |

### 3. Geometry: footprints, ortho checks, reference grid
| Script | What it does |
|---|---|
| `build_footprints.py` | Ground footprint of every image (`image_footprints.gpkg`), overlap report and the first split-leakage audit. |
| `projection_error.py` | Consistency of the frame-to-map projection of single thermal frames (EPSG:32649). |
| `calibrate_with_rtk_points.py` | Evaluate a base orthomosaic against RTK points and propose the correction. |
| `ortho_control_points.py` (+ `orthoreg_common.py`, `orthoreg_match.py`, `orthoreg_register.py`, `orthoreg_figs.py`) | Control points and fitted transform between thermal-flight orthomosaics and the base orthomosaic. |
| `audit_grid.py` | How the reference grid `TN19_modules_master.gpkg` relates to the real modules (image-based, heuristic). |
| `associate_boxes_to_grid.py` | Link TN19 thermal boxes to reference-grid cells. |
| `plot_coverage.py`, `plot_label_coverage.py` | Coverage figures of flights and of labelled frames. |

### 4. Thermal labels
| Script | What it does |
|---|---|
| `merge_labels.py` | Merge the per-flight ortho-level box files into the dataset annotations. |
| `integrate_frame_labels.py` | Integrate the frame-level labels (YOLO-OBB, one text file per 640 x 512 thermal frame) into the dataset. |
| `frame_vs_ortho_agreement.py` | Consistency between ortho-level and frame-level labels. |
| `box_temperature.py` | Per-box apparent-temperature statistics for `thermal_frame_boxes.csv`. |
| `build_review_package.py`, `review_common.py`, `apply_review_decisions.py` | Second-pass review of the labelled frames: build the review package, apply the decisions as a change report and proposed tables (originals are never overwritten). They work on review files that are not part of the release. |
| `make_agreement_sample.py`, `agreement_metrics.py` | Design of the annotator-agreement sample (blind packages) and the agreement metrics for YOLO-OBB labels (box matching by rotated IoU, class agreement, counts). Both scripts need the private sample key and the blind package files (not released); they are given for transparency. Their outputs are released in `dataset/annotation_agreement/` (labels of A1, A2, A3 and result tables). |
| `build_inspection_cell_layer.py` | Cell-level layer of TN19 (`TN19_inspection_cells.gpkg`, `cell_repeat_flights.csv`) from the box-to-cell association. |

### 5. Splits and leakage checks
| Script | What it does |
|---|---|
| `make_splits.py` | Pair-level split protocols A-D (deterministic, fixed seed): `metadata/splits/main_spatial_buf*.csv`, `xalt_*`, `xfarm_*`, `tndl_xalt_*`. |
| `make_frame_label_split.py` | Frame-level, leakage-aware split of the 3,000 labelled frames: `frame_label_split*.csv`. Overlap graph of footprints enlarged by a margin (2.3 m, 5 m for BIM2), whole components for the small farms, six ground strips for TN19. |
| `leakage_common.py` | Shared helpers: units (whole frames and crop windows) of a split table and their ground footprints. |
| `leakage_mc.py` | Monte Carlo of the residual leakage under the measured footprint errors. |
| `leakage_images.py` | Image-based check of the cross-split pairs (independent of the footprints). |
| `leakage_margin_table.py`, `leakage_report.py` | Margin-sensitivity table and the consolidated leakage report. |

### 6. Baseline detector (Technical Validation)
| Script | What it does |
|---|---|
| `make_yolo_dataset_view.py` | Build an ultralytics-style view (`images/`, `labels/`, split lists, `data.yaml`) of the frame-level labels from a split table. `--mode symlink|hardlink|copy|strip` (`strip` writes JPEG copies without the application segments), `--split-table` selects the protocol, `--no-negatives` skips reviewed negative frames. |
| `train_baseline_obb.py` | Baseline oriented-box detector (YOLO11n-obb): 640 px, batch 16, at most 100 epochs, patience 30, one run per seed; the best-validation checkpoint is scored once on the test split. Writes `metrics.json`, `results.csv` and `args.yaml` per seed. |

## Reproducing the baseline tables

1. Build a view for the protocol with `make_yolo_dataset_view.py`:
   - protocol C (main frame-level split): `--split-table metadata/splits/frame_label_split.csv`;
   - protocol A (strip split, 10 m buffer): the table in `metadata/splits/_strip_buffer_strict/` with `--mode strip`;
   - cross-domain (train and validation TN19, test the other farms): `metadata/splits/frame_label_split_xdomain_tn19_to_others.csv`.
2. Train and score with `train_baseline_obb.py --data <view> --out <results> --seeds 0 1 2`. Pass the stems of the background frames with `--neg-names` so that false boxes per background frame are also reported.
3. Frames marked `buffer` or `crop` in a split table are not used as whole frames (see `dataset/README.md`, section 8.1).

The cross-domain diagnosis in the paper (rotated copies of the TN19 training images, training with a share of the target-farm frames) was run on Colab; its notebook generators are not part of this repository.

Background frames: frames without a released box (not confirmed negatives; the main annotator recorded no anomaly), chosen at random with a fixed seed so that none is closer than the split margin to a frame of another part, with a cap per split (train 10 % of the positive training images, val at most 300, test at most 1,000). Their lists are in `dataset/metadata/splits/baseline_background_frames.csv`.

## Notes

- The thermal images in the YOLO views are the 8-bit preview inside the R-JPEG, not temperatures; labels are normalised to the 640 x 512 frame.
- Split and leakage statements in the paper refer to nominal footprints (flat terrain) with the stated margin; they do not prove that no pixel is shared (see Technical Validation).
- Scripts are provided as used for the paper. Several scripts carry hard-coded folder names of the authors' working copy near the top of the file; adjust them to your dataset root before running.
- Several scripts use default folders under `.local/` (a private working folder that is not part of this repository), for example the agreement study inputs or scratch outputs. Pass the corresponding command-line arguments, or point them to the files of the released dataset (annotation_agreement/, metadata/...), to run them on the data downloaded from Zenodo.

## License

The code is released under the MIT License (see `LICENSE`). The dataset has its own license (CC BY 4.0), stated in the dataset record on Zenodo. The DJI Thermal SDK is not part of this repository and is subject to its own terms.

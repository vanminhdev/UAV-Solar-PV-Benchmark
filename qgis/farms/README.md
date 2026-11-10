# Per-farm QGIS scripts for the reference grid

Run inside the QGIS Python console. Order per farm: `draw_grid_<farm>.py` (module cells along drawn string lines),
`module_id_<farm>.py` (module_id = block / string / sequence number), `aoi_boundary_<farm>.py` (padded outline of all
modules, written as layer `aoi_boundary`). The module cell size (`PANEL_WIDTH`, `PANEL_HEIGHT`, metres) differs by farm.

| farm | PANEL_WIDTH | PANEL_HEIGHT |
|---|---|---|
| TN19 | 1.904 | 0.96 |
| TNDL | 1.049 | 2.05 |
| TT | 1.040 | 2.05 |
| BIM2 | 1.029 | 1.982 |

The grids were drawn by hand on the orthomosaics (string lines) and are released as `module_reference/<farm>_modules_master.gpkg`.

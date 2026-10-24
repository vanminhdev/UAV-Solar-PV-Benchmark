"""Summarise the Agisoft Metashape processing reports (PDF) of dataset/ into one csv.

    python parse_metashape_reports.py <dataset_root> <out_csv>

One row per report found under dataset/<farm>/photogrammetry/<flight_id>/*_report*.pdf. Values are read from the
report text (survey data table, camera location error table, processing-parameters pages); a field that is not
found stays empty. Reports of mapping flights may be missing some fields.
"""
import csv
import re
import sys
from pathlib import Path

import fitz

FIELDS = [
    "farm_dir", "flight_id", "report_file", "report_date", "n_images", "aligned_images", "flying_altitude_m",
    "ground_resolution_mm_px", "coverage_km2", "tie_points", "projections", "reprojection_error_px",
    "accuracy_cameras_prior_m", "accuracy_tiepoints_px", "n_ground_control_points", "n_cameras_xml",
    "cam_err_x_m", "cam_err_y_m", "cam_err_z_m", "cam_err_xy_m", "cam_err_total_m",
    "alignment_accuracy", "key_point_limit", "tie_point_limit", "depth_map_quality", "depth_filtering",
    "dem_resolution_cm_px", "ortho_resolution_cm_px", "report_coordinate_system", "software", "software_version",
    "os", "cpu", "gpu", "ram_gb", "cameras",
]


def g(pat, t, grp=1, flags=0):
    m = re.search(pat, t, flags)
    return m.group(grp).strip() if m else ""


def parse(pdf):
    t = "\n".join(p.get_text() for p in fitz.open(pdf))
    one = re.sub(r"\s*\n\s*", " | ", t)
    row = dict(
        report_date=g(r"Processing Report\s*\n\s*([0-9]+ \w+ [0-9]{4})", t),
        n_images=g(r"Number of images:\s*\n?\s*([\d,]+)", t),
        aligned_images=g(r"Aligned images\s*\|\s*([\d,]+)", one),
        flying_altitude_m=g(r"Flying altitude:\s*\n?\s*([\d.]+) m", t),
        coverage_km2=g(r"Coverage area:\s*\n?\s*([\d.]+) km", t),
        tie_points=g(r"Tie points:\s*\n?\s*([\d,]+)", t),
        projections=g(r"Projections:\s*\n?\s*([\d,]+)", t),
        reprojection_error_px=g(r"Reprojection error:\s*\n?\s*([\d.]+) pix", t),
        alignment_accuracy=g(r"Alignment parameters \| Accuracy \| (\w+)", one),
        key_point_limit=g(r"Key point limit \| ([\d,]+)", one),
        tie_point_limit=g(r"Tie point limit \| ([\d,]+)", one),
        depth_map_quality=g(r"Depth maps generation parameters \| Quality \| (\w+)", one),
        depth_filtering=g(r"Filtering mode \| (\w+)", one),
        report_coordinate_system=g(r"Coordinate system \| ([^|]+?) \|", one),
        software=g(r"Software name \| ([^|]+?) \|", one),
        software_version=g(r"Software version \| (\d[^|]*?build \d+)", one),
        os=g(r"OS \| ([^|]+?) \|", one),
        cpu=g(r"CPU \| ([^|]+?) \|", one),
        gpu=g(r"GPU\(s\) \| ([^|]+?)( \||$)", one),
        ram_gb=g(r"RAM \| ([\d.]+) GB", one),
    )
    gr = g(r"Ground resolution:\s*\n?\s*([\d.]+) (mm|cm)/pix", t, 0)
    m = re.search(r"Ground resolution:\s*\n?\s*([\d.]+) (mm|cm)/pix", t)
    if m:
        row["ground_resolution_mm_px"] = round(float(m.group(1)) * (10 if m.group(2) == "cm" else 1), 3)
    m = re.search(r"DEM \| Size \| [\d, x]+ \| Resolution \| ([\d.]+) (mm|cm)/pix", one)
    if m:
        row["dem_resolution_cm_px"] = round(float(m.group(1)) / (10 if m.group(2) == "mm" else 1), 3)
    m = re.search(r"Orthomosaic \| Size \| [\d, x]+ \| Resolution \| ([\d.]+) (mm|cm)/pix", one)
    if m:
        row["ortho_resolution_cm_px"] = round(float(m.group(1)) / (10 if m.group(2) == "mm" else 1), 3)
    m = re.search(r"X error \(m\) \| Y error \(m\) \| Z error \(m\) \| XY error \(m\) \| Total error \(m\) \| ([\d.e-]+) \| ([\d.e-]+) \| ([\d.e-]+) \| ([\d.e-]+) \| ([\d.e-]+)", one)
    if m:
        for k, v in zip(("cam_err_x_m", "cam_err_y_m", "cam_err_z_m", "cam_err_xy_m", "cam_err_total_m"), m.groups()):
            row[k] = v
    cams = re.findall(r"\| (Z[A-Z0-9]+ \([\d.]+mm\)|[A-Z0-9-]+ \([^)|]+\)) \| (\d+ x \d+) \| ([\d.]+) mm", one)
    row["cameras"] = "; ".join(f"{a} {b} f={c}mm" for a, b, c in dict.fromkeys(cams))
    return row


def parse_xml(xml):
    """Reference settings stored in the Metashape cameras.xml export (camera accuracy prior, GCP count)."""
    if not xml.exists():
        return {}
    s = xml.read_text(encoding="utf-8", errors="ignore")
    return dict(
        accuracy_cameras_prior_m=g(r'accuracy_cameras" value="([\d.]+)"', s),
        accuracy_tiepoints_px=g(r'accuracy_tiepoints" value="([\d.]+)"', s),
        n_ground_control_points=len(re.findall(r"<marker ", s)),
        n_cameras_xml=len(re.findall(r"<camera ", s)),
    )


def main():
    ds, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for pdf in sorted(ds.glob("farm_0*/photogrammetry/*/*_report*.pdf")):
        r = parse(pdf)
        xml = next(pdf.parent.glob(pdf.name.replace("_report", "_cameras").replace(".pdf", ".xml")), Path("none"))
        r.update(parse_xml(xml))
        r.update(farm_dir=pdf.parts[-4], flight_id=pdf.parts[-2], report_file=pdf.name)
        rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "reports ->", out)


if __name__ == "__main__":
    main()

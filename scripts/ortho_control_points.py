#!/usr/bin/env python
"""Control points between the thermal-flight orthomosaics and the base (mapping) orthomosaic, the fitted
ortho -> base transforms, and a runnable projection chain  frame pixel -> map -> base ortho -> reference cell.

    python -X utf8 code/scripts/ortho_control_points.py match     # automatic window matching (about 3 min) -> cache
    python -X utf8 code/scripts/ortho_control_points.py select    # choose spread points, auto description, paired-crop figures
    python -X utf8 code/scripts/ortho_control_points.py review    # write control_point_review.csv (visual-review decisions, see REVIEW)
    python -X utf8 code/scripts/ortho_control_points.py build     # write the CSVs of dataset/metadata/ortho_registration/
    python -X utf8 code/scripts/ortho_control_points.py test      # consistency test of apply_chain on the labelled flights

Library use (reads only released tables, see README.md of the output folder):
    from ortho_control_points import apply_chain
    apply_chain('TN19_20260411_31M_T3', 'DJI_20260411092212_0281_T.JPG', 55.2, 288.0)

Inputs are read only. The cache lives in <project>/.local/tmp/orthoreg (git-ignored).
Helper modules (same folder): orthoreg_common.py, orthoreg_match.py, orthoreg_register.py, orthoreg_figs.py.
"""
from __future__ import annotations
import argparse, pickle, re, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
DS = ROOT / 'dataset'
OUT = DS / 'metadata' / 'ortho_registration'
CACHE = ROOT / '.local' / 'tmp' / 'orthoreg'
EPSG = 32649

MAP1 = {'TN19': 'TN19_20260411_87M_MAP1', 'TNDL': 'TNDL_20260411_87M_MAP1', 'TT': 'TT_20260412_87M_MAP1'}
BIM2_BASES = ['BIM2_20260410_100M_MAP3', 'BIM2_20260410_110M_MAP1', 'BIM2_20260411_100M_MAP2_part1', 'BIM2_20260411_100M_MAP2_part2']

# Visual review decisions (paired crops, figures/cp_<flight>.jpg). verified_visually = a landmark / vegetation / row-end pattern
# coincides at the cross in both crops within about 0.5 m (crop scale 0.16 m/px); auto_only = not judged or only periodic
# lattice / a straight road in the window; rejected = visibly wrong or unreliable (reason in note).
# Flights not listed: all points verified_visually. Point numbers are the P<nn> suffix of point_id.
REVIEW = {
    'TN19_20260411_35M_T5': dict(auto_only={7}, note={7: 'bare ground; road coincides but ground cover differs between flights'}),
    'TN19_20260411_42M_T1': dict(auto_only={13}, note={13: 'interior window, low correlation (0.4), no landmark judged'}),
    'TT_20260412_39M_T1': dict(verified={1, 2, 3, 5, 6, 10, 11, 13, 14}),
    'TT_20260412_39M_T2': dict(verified={1, 2, 3, 4, 6, 7, 11, 13, 14}),
    'TT_20260412_39M_T3': dict(verified={4}),
    'TT_20260412_39M_T4': dict(verified={1, 2, 7, 8, 13}),
    'TT_20260412_39M_T5': dict(verified=set(), rejected={2, 5, 6},
                               note={2: 'base ortho (TT MAP1) is ghosted/warped at this spot',
                                     5: 'open water / shoreline, changes between flights', 6: 'open water / shoreline, changes between flights'}),
}
FAIL_REASON = {'TT_20260412_39M_T6': 'only 1.6 % of the valid ortho area is covered by valid pixels of the TT 87 m base ortho '
                                      '(the thermal-flight ortho extends north of the base ortho)'}


def jobs():
    from orthoreg_common import ortho_paths
    J = []
    for k in ortho_paths():
        if not re.fullmatch(r'T\d+', k.split('_')[-1]): continue
        pre = k.split('_')[0]
        J += [(k, MAP1[pre])] if pre in MAP1 else [(k, b) for b in BIM2_BASES]
    J.append(('TN19_20260411_87M_MAP2', MAP1['TN19']))      # extra: second TN19 mapping ortho registered to MAP1
    return J


# ------------------------------------------------------------------------------------------------ match / select
def cmd_match():
    import time, rasterio
    from orthoreg_register import load_pair, search_pair_robust
    from orthoreg_common import ortho_paths
    P = ortho_paths(); CACHE.mkdir(parents=True, exist_ok=True); rows = []
    for f, b in jobs():
        t = time.time(); print(f, '->', b, flush=True)
        D = load_pair(P[f], P[b]); R = search_pair_robust(D)
        with rasterio.open(P[f]) as r1, rasterio.open(P[b]) as r2:
            a, c = r1.bounds, r2.bounds
        ov = max(0, min(a.right, c.right) - max(a.left, c.left)) * max(0, min(a.top, c.top) - max(a.bottom, c.bottom)) / ((a.right - a.left) * (a.top - a.bottom))
        if R is None:
            rows.append(dict(flight_id=f, base_ortho_flight_id=b, bbox_overlap_frac=round(ov, 2), n_candidate_windows=0, n_inlier_windows=0))
            pickle.dump(None, open(CACHE / f'match_{f}__{b}.pkl', 'wb')); continue
        F = R['F']; inl = [m for m in R['cand'] if m.get('inlier')]
        rows.append(dict(flight_id=f, base_ortho_flight_id=b, bbox_overlap_frac=round(ov, 2), n_candidate_windows=len(R['cand']), n_inlier_windows=len(inl),
                         median_zncc=round(float(np.median([m['zncc'] for m in inl])), 3), scale=round(float(np.sqrt(abs(np.linalg.det(F['m'])))), 4),
                         shift_E_m=round(float(F['t'][0]), 2), shift_N_m=round(float(F['t'][1]), 2)))
        pickle.dump(dict(cand=R['cand'], F=F), open(CACHE / f'match_{f}__{b}.pkl', 'wb'))
        print('   %d inlier windows, %.0fs' % (len(inl), time.time() - t), flush=True)
    df = pd.DataFrame(rows)
    # base choice: farms with one mapping ortho -> that one; BIM2 -> bbox overlap >= 0.8 and most inlier windows
    df['chosen_as_base'] = False
    for f, g in df.groupby('flight_id'):
        if f.startswith('BIM2'):
            g = g[g.bbox_overlap_frac >= 0.8]
            if len(g): df.loc[g.n_inlier_windows.idxmax(), 'chosen_as_base'] = True
        else:
            df.loc[g.index, 'chosen_as_base'] = True
    df.to_csv(CACHE / 'base_selection.csv', index=False)
    print(df.to_string())


def primary_pairs():
    df = pd.read_csv(CACHE / 'base_selection.csv')
    return [(r.flight_id, r.base_ortho_flight_id) for r in df[df.chosen_as_base].itertuples()]


def cmd_select(K=14):
    from orthoreg_figs import describe, montage
    from orthoreg_register import load_pair
    from orthoreg_common import ortho_paths
    P = ortho_paths(); (OUT / 'figures').mkdir(parents=True, exist_ok=True); allsel = {}
    for f, b in primary_pairs():
        d = pickle.load(open(CACHE / f'match_{f}__{b}.pkl', 'rb'))
        if d is None: continue
        D = load_pair(P[f], P[b]); cand = d['cand']
        for m in cand: m['desc'], m['pfrac'] = describe(D, m['E'], m['N'])
        c = [m for m in cand if m.get('inlier') and m['ratio'] <= 0.92 and not m['edge']]
        if not c: continue
        q = np.array([m['zncc'] * (1 - 0.5 * m['ratio']) + (0.3 if any(k in m['desc'] for k in ('end/gap', 'road', 'edge')) else 0) for m in c])
        xy = np.array([[m['E'], m['N']] for m in c]); sel = [int(np.argmax(q))]
        while len(sel) < min(K, len(c)):
            dd = np.min(np.linalg.norm(xy[:, None] - xy[sel][None], axis=2), axis=1); sc = dd * (0.5 + q); sc[sel] = -1
            j = int(np.argmax(sc))
            if sc[j] <= 0: break
            sel.append(j)
        S = [c[i] for i in sel]; S.sort(key=lambda m: (-round(m['N'] / 40), m['E']))
        for i, m in enumerate(S, 1): m['point_id'] = f'{f}_P{i:02d}'
        allsel[f] = dict(base=b, pts=S, F=d['F'])
        montage(D, S, f'{f} vs {b}', str(OUT / 'figures' / f'cp_{f}.jpg'))
        print(f, len(S), flush=True)
    pickle.dump(allsel, open(CACHE / 'selection.pkl', 'wb'))


def cmd_review():
    sel = pickle.load(open(CACHE / 'selection.pkl', 'rb')); rows = []
    for f, a in sel.items():
        rv = REVIEW.get(f, {})
        for m in a['pts']:
            n = int(m['point_id'][-2:]); note = rv.get('note', {}).get(n, '')
            if n in rv.get('rejected', set()): st = 'rejected'
            elif 'verified' in rv: st = 'verified_visually' if n in rv['verified'] else 'auto_only'
            elif n in rv.get('auto_only', set()): st = 'auto_only'
            else: st = 'verified_visually'
            rows.append(dict(point_id=m['point_id'], status=st, description=m['desc'], review_note=note))
    df = pd.DataFrame(rows); df.to_csv(OUT / 'control_point_review.csv', index=False)
    print(df.status.value_counts())


# ------------------------------------------------------------------------------------------------ fitting
def quality(f, n, nver, p95, scale, ext):
    cav = ['no independent check points: residuals are leave-one-out consistency of the fit, not accuracy']
    why = []
    if n < 8: why.append(f'only {n} usable points (small ortho or windows lost to no-data)')
    if p95 > 1.0: why.append(f'LOO p95 {p95:.2f} m > 1 m')
    if nver < 0.4 * n: why.append(f'only {nver} of {n} points verified visually (rest periodic lattice / straight road only)')
    if not 0.97 <= scale <= 1.035: why.append(f'scale {scale:.3f} unusually far from 1')
    if f.startswith('TT'): cav.append('TT base ortho shows ghosting/warping of table rows; a global similarity cannot absorb local base distortions (1-2 m possible)')
    if f.startswith('BIM2'): cav.append('BIM2 has four mapping orthos that are mutually offset by several metres; base chosen per flight by overlap and number of matched windows (base_ortho_selection.csv)')
    if f == 'TN19_20260411_87M_MAP2': cav.append('mapping ortho registered to MAP1 (not a thermal flight; provided for completeness)')
    if ext[0] < 100 or ext[1] < 60: cav.append(f'points cover only {ext[0]:.0f} x {ext[1]:.0f} m: rotation and scale poorly constrained')
    return ('weak: ' + '; '.join(why)) if why else 'good', '; '.join(cav)


def fit_tables(sel, rev):
    from orthoreg_register import ls_fit, apply_fit
    cp, tr = [], []
    for f, a in sel.items():
        pts = a['pts']; b = a['base']
        src = np.array([[m['E'], m['N']] for m in pts]); ref = src + np.array([[m['dE'], m['dN']] for m in pts])
        st = [rev.loc[m['point_id'], 'status'] for m in pts]
        use = np.array([s != 'rejected' for s in st]); res = {}
        for model in ('similarity', 'affine'):
            F = ls_fit(src[use], ref[use], model); rl = np.zeros(len(pts))
            for i in range(len(pts)):
                k = use.copy(); k[i] = False
                rl[i] = np.linalg.norm(apply_fit(ls_fit(src[k], ref[k], model), src[i])[0] - ref[i])
            res[model] = (F, rl)
        n = int(use.sum())
        ms, ma = np.median(res['similarity'][1][use]), np.median(res['affine'][1][use])
        ps, pa = np.percentile(res['similarity'][1][use], 95), np.percentile(res['affine'][1][use], 95)
        # affine only if it clearly helps: >= 30 % lower median LOO and >= 5 cm better
        prim = 'affine' if (n >= 10 and ma < 0.7 * ms and ms - ma >= 0.05 and pa < ps) else 'similarity'
        for i, m in enumerate(pts):
            note = f"zncc {m['zncc']:.2f}; 2nd-peak ratio {m['ratio']:.2f}"
            rn = rev.loc[m['point_id'], 'review_note']
            if rn: note += '; ' + str(rn)
            if 'interior' in m['desc']: note += '; periodic-lattice risk (no row end in window)'
            cp.append(dict(flight_id=f, base_ortho_flight_id=b, point_id=m['point_id'], E_src=round(m['E'], 3), N_src=round(m['N'], 3),
                           E_ref=round(ref[i, 0], 3), N_ref=round(ref[i, 1], 3), method='window_NCC_gradient',
                           status=st[i], description=rev.loc[m['point_id'], 'description'], residual_m=round(float(res[prim][1][i]), 3),
                           note=note, zncc=round(m['zncc'], 3), second_peak_ratio=round(m['ratio'], 3),
                           residual_similarity_m=round(float(res['similarity'][1][i]), 3), residual_affine_m=round(float(res['affine'][1][i]), 3),
                           used_in_fit=bool(use[i]), primary_model=prim))
        nver = sum(s == 'verified_visually' for s in st); ext = (np.ptp(src[use, 0]), np.ptp(src[use, 1]))
        for model in ('similarity', 'affine'):
            F, rl = res[model]; m_ = F['m']; r = rl[use]
            scale = float(np.sqrt(abs(np.linalg.det(m_)))); rot = float(np.degrees(np.arctan2(m_[1, 0] - m_[0, 1], m_[0, 0] + m_[1, 1])))
            q, cav = quality(f, n, nver, float(np.percentile(r, 95)), scale, ext)
            tr.append(dict(flight_id=f, base_ortho_flight_id=b, model=model, is_primary=int(model == prim), origin_E=round(F['c'][0], 3), origin_N=round(F['c'][1], 3),
                           m11=round(m_[0, 0], 7), m12=round(m_[0, 1], 7), m21=round(m_[1, 0], 7), m22=round(m_[1, 1], 7), tE=round(F['t'][0], 4), tN=round(F['t'][1], 4),
                           scale=round(scale, 5), rotation_deg=round(rot, 4), shift_E_at_origin_m=round(F['t'][0], 3), shift_N_at_origin_m=round(F['t'][1], 3),
                           shift_mag_m=round(float(np.hypot(*F['t'])), 3), n_points_used=n, n_points_total=len(pts), n_verified_visually=nver,
                           loo_median_m=round(float(np.median(r)), 3), loo_p95_m=round(float(np.percentile(r, 95)), 3), loo_max_m=round(float(r.max()), 3),
                           point_extent_E_m=round(ext[0]), point_extent_N_m=round(ext[1]), quality=q, caveats=cav))
    return pd.DataFrame(cp), pd.DataFrame(tr)


def cmd_build():
    sel = pickle.load(open(CACHE / 'selection.pkl', 'rb'))
    rev = pd.read_csv(OUT / 'control_point_review.csv', keep_default_na=False).set_index('point_id')
    cp, tr = fit_tables(sel, rev)
    bs = pd.read_csv(CACHE / 'base_selection.csv'); ch = bs[bs.chosen_as_base]
    for r in ch.itertuples():
        if r.flight_id not in sel:
            tr = pd.concat([tr, pd.DataFrame([dict(flight_id=r.flight_id, base_ortho_flight_id=r.base_ortho_flight_id, model='none', is_primary=1, n_points_used=0, n_points_total=0,
                            quality='failed: ' + FAIL_REASON.get(r.flight_id, 'no window matched'), caveats='')])], ignore_index=True)
    order = {f: i for i, f in enumerate(ch.flight_id)}
    cp = cp.sort_values(['flight_id', 'point_id'], key=lambda s: s.map(order) if s.name == 'flight_id' else s)
    tr = tr.sort_values(['flight_id', 'is_primary'], ascending=[True, False], key=lambda s: s.map(order) if s.name == 'flight_id' else s)
    cp.to_csv(OUT / 'ortho_control_points.csv', index=False); tr.to_csv(OUT / 'ortho_registration_transforms.csv', index=False)
    bs.to_csv(OUT / 'base_ortho_selection.csv', index=False)
    write_chain_csv(tr); crosscheck(tr)
    print(tr[tr.is_primary == 1][['flight_id', 'model', 'n_points_used', 'n_verified_visually', 'shift_mag_m', 'scale', 'rotation_deg', 'loo_median_m', 'loo_p95_m', 'quality']].to_string())


def write_chain_csv(tr):
    p = tr[(tr.is_primary == 1) & (tr.model != 'none') & (tr.flight_id != 'TN19_20260411_87M_MAP2')]
    good = p[p.quality == 'good']
    steps = [
        (1, 'thermal frame pixel (x_px, y_px; H20T 640x512)',
         'map position in the frame of the thermal flight (EPSG:32649; same frame as thermal_boxes.csv and the flight ortho)',
         'ray through the pixel (pinhole, 12 um pitch, f from XMP) rotated by gimbal yaw/pitch/roll, intersected with the horizontal plane RelativeAltitude below the GNSS/RTK camera position (flat terrain; optional LRF height); function frame_to_map() = projection_error.project()',
         'metadata/inventory/image_inventory.csv (GpsLatitude, GpsLongitude, RelativeAltitude, GimbalYawDegree, GimbalPitchDegree, GimbalRollDegree, focal_mm, LRFTargetDistance); model described in code/scripts/build_footprints.py and projection_error.py',
         '1.17 median / 2.60 p95 (nominal model V0, n=5204 detections of TN19, TNDL, TT); 0.97 / 2.28 for V4 but V4 includes a per-flight shift fitted on boxes; TT 39 m: 2.6 / 4.4',
         'nothing: the reference is the box centre on the flight ortho, so this is a consistency measurement between two products of one flight (no check points)',
         'metadata/projection_check/README.md: smooth systematic field (0.6 m near the area centre, 2.2 m beyond 90 m), not frame noise (0.43 m within a box); per-flight mean bias 0.1-1.6 m; flat-terrain assumption'),
        (2, 'map position in the frame of the thermal flight', 'map position in the frame of the base (mapping) ortho, EPSG:32649',
         'similarity (4 parameters), affine where is_primary says so: E_ref = origin_E + tE + m11 (E - origin_E) + m12 (N - origin_N); N_ref = origin_N + tN + m21 (E - origin_E) + m22 (N - origin_N); function ortho_to_base()',
         'ortho_registration_transforms.csv (row with is_primary = 1), fitted on ortho_control_points.csv (status != rejected)',
         'leave-one-out over control points: median %.2f-%.2f m and p95 %.2f-%.2f m over %d usable flights (%d flagged good); the shift itself is %.1f-%.1f m (scale %.3f-%.3f, rotation %.2f to %.2f deg)' % (
             p.loo_median_m.min(), p.loo_median_m.max(), p.loo_p95_m.min(), p.loo_p95_m.max(), len(p), len(good), p.shift_mag_m.min(), p.shift_mag_m.max(), p.scale.min(), p.scale.max(), p.rotation_deg.min(), p.rotation_deg.max()),
         'image content only (no boxes, labels or RTK values were used to find the points); see crosscheck_vs_association.csv for the comparison with the box-based association shifts (cross-check only)',
         'LOO measures consistency of the fit, not accuracy (no check points; both orthos have metre-level absolute errors); the periodic module lattice (1.92 m / 0.96 m) can bias windows without row ends; both orthos assume flat terrain'),
        (3, 'map position in the frame of the base ortho', 'reference cell of the grid (TN19 only: module_id; block, string_id, row_id in the grid)',
         'point-in-polygon of TN19_modules_master.gpkg layer modules (cells 1.904 x 0.960 m drawn on TN19_20260411_87M_MAP1); nearest cell within 1 m when the point is between cells; function base_to_cell()',
         'farm_01_thuan_nam_19/module_reference/TN19_modules_master.gpkg',
         '0.04 median / 0.12 p95 (grid_audit_notes.md section 4: cell edges vs module seams on MAP1; up to 0.16 m in 30 m windows, worse in blocks 3 and 5 at 0.09-0.17 m)',
         'heuristic audit by the data authors, not an independent survey; the grid omits strongly warped margin rows and is not a verified module inventory',
         'only TN19 and only for the base ortho MAP1 (the ortho the grid was drawn on); TNDL, TT and BIM2 have no grid, so the chain ends at step 2 there'),
        (4, 'whole chain: thermal frame pixel', 'reference cell (TN19)',
         'apply_chain(flight_id, thermal_file, x_px, y_px) in code/scripts/ortho_control_points.py; consistency test: python ortho_control_points.py test',
         'frame_chain_test.csv and frame_chain_test_summary.csv in this folder',
         'one frame has about one module of step-1 error, so only the within-one-cell fraction is a meaningful figure (see README.md)',
         'expected cell = association/thermal_box_to_cell.csv, whose registration was estimated with the boxes; the chain here uses an image-based registration, so the test is a cross-check, not a validation',
         'errors of the steps add (roughly in quadrature if independent, which is not shown)'),
    ]
    cols = ['step_no', 'from_frame', 'to_frame', 'operation', 'parameters_source', 'uncertainty_m', 'what_is_independent', 'notes']
    pd.DataFrame(steps, columns=cols).to_csv(OUT / 'projection_chain.csv', index=False)


def crosscheck(tr):
    """Our image-based ortho->base shift vs association/thermal_box_to_cell.csv (reg + drift) and registration_used.csv, at the box centres."""
    from orthoreg_register import apply_fit
    a = pd.read_csv(DS / 'metadata/association/thermal_box_to_cell.csv')
    bx = pd.concat([pd.read_csv(f) for f in sorted(DS.glob('farm_*/annotations/thermal_anomaly_obb/thermal_boxes.csv'))]).set_index('box_id')
    ru = pd.read_csv(DS / 'metadata/association/registration_used.csv').set_index('flight_id'); rows = []
    for f, g in a.groupby('flight_id'):
        r = tr[(tr.flight_id == f) & (tr.is_primary == 1)]
        if r.empty or r.iloc[0].model == 'none': continue
        r = r.iloc[0]; b = bx.loc[g.box_id]; pts = b[['centre_E', 'centre_N']].values
        F = dict(c=np.array([r.origin_E, r.origin_N]), m=np.array([[r.m11, r.m12], [r.m21, r.m22]]), t=np.array([r.tE, r.tN]))
        d = apply_fit(F, pts) - pts; box = g[['reg_dE', 'reg_dN']].values + g[['drift_dE', 'drift_dN']].values; diff = d - box
        rows.append(dict(flight_id=f, n_boxes=len(g), ours_mean_dE=round(d[:, 0].mean(), 3), ours_mean_dN=round(d[:, 1].mean(), 3),
                         assoc_mean_dE=round(box[:, 0].mean(), 3), assoc_mean_dN=round(box[:, 1].mean(), 3),
                         registration_used_prior_dE=round(ru.loc[f, 'prior_shift_E_mean'], 3) if f in ru.index else '',
                         registration_used_prior_dN=round(ru.loc[f, 'prior_shift_N_mean'], 3) if f in ru.index else '',
                         mean_diff_E=round(diff[:, 0].mean(), 3), mean_diff_N=round(diff[:, 1].mean(), 3),
                         per_box_diff_median_m=round(float(np.median(np.hypot(*diff.T))), 3), per_box_diff_p95_m=round(float(np.percentile(np.hypot(*diff.T), 95)), 3)))
    df = pd.DataFrame(rows); df.to_csv(OUT / 'crosscheck_vs_association.csv', index=False); print(df.to_string())


# ------------------------------------------------------------------------------------------------ chain
_CACHE = {}


def _tables(dataset=None):
    ds = Path(dataset) if dataset else DS; key = str(ds)
    if key not in _CACHE:
        tr = pd.read_csv(ds / 'metadata/ortho_registration/ortho_registration_transforms.csv')
        inv = pd.read_csv(ds / 'metadata/inventory/image_inventory.csv'); inv = inv[inv.sensor == 'T'].set_index(['flight_id', 'file'])
        _CACHE[key] = (tr[tr.is_primary == 1].set_index('flight_id'), inv, ds)
    return _CACHE[key]


def _cells(ds):
    key = ('cells', str(ds))
    if key not in _CACHE:
        from build_footprints import read_gpkg_layer
        from shapely import STRtree
        g = read_gpkg_layer(ds / 'farm_01_thuan_nam_19/module_reference/TN19_modules_master.gpkg', 'modules')
        geoms = list(g.geometry); _CACHE[key] = (g, geoms, STRtree(geoms))
    return _CACHE[key]


def frame_to_map(rows, x_px, y_px, height='relative_altitude'):
    """Step 1 (vectorised). rows: DataFrame with GpsLatitude, GpsLongitude, RelativeAltitude, GimbalYaw/Pitch/RollDegree, focal_mm, LRFTargetDistance."""
    from projection_error import project
    from pyproj import Transformer
    E0, N0 = Transformer.from_crs(4326, EPSG, always_xy=True).transform(rows.GpsLongitude.values, rows.GpsLatitude.values)
    alt = rows.RelativeAltitude.values.astype(float)
    if height == 'lrf':
        lrf = rows.LRFTargetDistance.values.astype(float); ok = np.isfinite(lrf) & (np.abs(lrf - alt) <= 6); alt = np.where(ok, lrf, alt)
    E, N, _, _ = project(np.asarray(x_px, float), np.asarray(y_px, float), np.asarray(E0), np.asarray(N0), alt,
                         rows.GimbalYawDegree.values.astype(float), rows.GimbalPitchDegree.values.astype(float),
                         rows.GimbalRollDegree.values.astype(float), rows.focal_mm.values.astype(float))
    return E, N


def ortho_to_base(flight_id, E, N, dataset=None):
    """Step 2: flight-ortho frame -> base-ortho frame with the primary transform of ortho_registration_transforms.csv."""
    tr, _, _ = _tables(dataset)
    if flight_id not in tr.index or tr.loc[flight_id, 'model'] == 'none':
        raise ValueError(f'no ortho->base transform for {flight_id}')
    r = tr.loc[flight_id]; E = np.asarray(E, float); N = np.asarray(N, float); dE, dN = E - r.origin_E, N - r.origin_N
    return r.origin_E + r.tE + r.m11 * dE + r.m12 * dN, r.origin_N + r.tN + r.m21 * dE + r.m22 * dN


def base_to_cell(E, N, dataset=None, max_dist=1.0):
    """Step 3 (TN19 / MAP1 only): module_id of the cell containing the point, else the nearest cell within max_dist. Returns (ids, distance to cell)."""
    from shapely.geometry import Point
    g, geoms, tree = _cells(Path(dataset) if dataset else DS); ids, dist = [], []
    for e, n in zip(np.atleast_1d(E), np.atleast_1d(N)):
        p = Point(e, n); i = int(tree.nearest(p)); d = geoms[i].distance(p)
        ids.append(g.module_id.iloc[i] if d <= max_dist else None); dist.append(d)
    return np.array(ids, dtype=object), np.array(dist)


def apply_chain(flight_id, thermal_file, x_px, y_px, dataset=None, height='relative_altitude'):
    """Frame pixel -> map (flight frame) -> base ortho -> TN19 reference cell, from released tables only.
    Returns a dict: E_map, N_map (frame of the thermal-flight ortho), E_base, N_base, base_ortho_flight_id,
    module_id and cell_distance_m (TN19 on MAP1 only, else None), ortho_base_quality, sigma_note."""
    tr, inv, ds = _tables(dataset)
    row = inv.loc[[(flight_id, thermal_file)]].reset_index()
    E, N = frame_to_map(row, [x_px], [y_px], height); Eb, Nb = ortho_to_base(flight_id, E, N, dataset)
    out = dict(flight_id=flight_id, E_map=float(E[0]), N_map=float(N[0]), E_base=float(Eb[0]), N_base=float(Nb[0]),
               base_ortho_flight_id=tr.loc[flight_id, 'base_ortho_flight_id'], module_id=None, cell_distance_m=None,
               ortho_base_quality=tr.loc[flight_id, 'quality'])
    if out['base_ortho_flight_id'] == 'TN19_20260411_87M_MAP1':
        ids, d = base_to_cell(Eb, Nb, dataset); out['module_id'] = ids[0]; out['cell_distance_m'] = float(d[0])
    out['sigma_note'] = 'frame->map ~1.0 m median / 2.3 m p95 (projection_check); ortho->base LOO %.2f m median; base->cell 0.04 m median' % tr.loc[flight_id, 'loo_median_m']
    return out


# ------------------------------------------------------------------------------------------------ test
def cmd_test():
    tr, inv, ds = _tables()
    a = pd.read_csv(ds / 'metadata/association/thermal_box_to_cell.csv'); a = a[a.status == 'unique']
    det = pd.concat([pd.read_csv(f) for f in sorted(ds.glob('farm_01*/annotations/thermal_anomaly_obb/thermal_box_frame_detections.csv'))])
    bx = pd.concat([pd.read_csv(f) for f in sorted(ds.glob('farm_01*/annotations/thermal_anomaly_obb/thermal_boxes.csv'))]).set_index('box_id')
    g, geoms, tree = _cells(ds); idx = {m: i for i, m in enumerate(g.module_id)}
    d = det.merge(a[['box_id', 'module_id']], on='box_id'); d = d[d.flight_id.isin(tr.index)]
    d = d[d.flight_id.map(lambda f: tr.loc[f, 'model'] != 'none')]
    adj = {}

    def neigh(m):
        if m not in adj:
            poly = geoms[idx[m]].buffer(0.05); adj[m] = {g.module_id.iloc[j] for j in tree.query(poly) if poly.intersects(geoms[j])}
        return adj[m]

    def score(got, exp):
        same = np.mean([x == m for x, m in zip(got, exp)]); near = np.mean([(x is not None) and (x == m or x in neigh(m)) for x, m in zip(got, exp)])
        return same, near
    cols = ['flight_id', 'file', 'GpsLatitude', 'GpsLongitude', 'RelativeAltitude', 'GimbalYawDegree', 'GimbalPitchDegree', 'GimbalRollDegree', 'focal_mm', 'LRFTargetDistance']
    rows = []
    for f, gg in d.groupby('flight_id'):
        gg = gg.merge(inv.reset_index()[cols], left_on=['flight_id', 'thermal_file'], right_on=['flight_id', 'file'], how='inner')
        for height in ('relative_altitude', 'lrf'):
            E, N = frame_to_map(gg, gg.x_px.values, gg.y_px.values, height); Eb, Nb = ortho_to_base(f, E, N)
            for variant, (e_, n_) in (('chain', (Eb, Nb)), ('no_ortho_to_base_step', (E, N))):
                ids, _ = base_to_cell(e_, n_); s, w = score(ids, gg.module_id.values)
                rows.append(dict(flight_id=f, height=height, variant=variant, n=len(gg), same_cell=s, within_one_cell=w))
        b = bx.loc[gg.box_id.unique()]; Eb, Nb = ortho_to_base(f, b.centre_E.values, b.centre_N.values); ids, _ = base_to_cell(Eb, Nb)
        s, w = score(ids, a.set_index('box_id').loc[b.index].module_id.values)
        rows.append(dict(flight_id=f, height='n/a', variant='box_centre_via_transform', n=len(b), same_cell=s, within_one_cell=w))
    r = pd.DataFrame(rows); r.to_csv(OUT / 'frame_chain_test.csv', index=False, float_format='%.4f')
    w = r.assign(w1=r.same_cell * r.n, w2=r.within_one_cell * r.n).groupby(['variant', 'height']).agg(n=('n', 'sum'), a=('w1', 'sum'), b=('w2', 'sum'))
    w['same_cell'] = w.a / w.n; w['within_one_cell'] = w.b / w.n; w = w.drop(columns=['a', 'b']).reset_index()
    w.to_csv(OUT / 'frame_chain_test_summary.csv', index=False, float_format='%.4f')
    print(r.to_string()); print(w.to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['match', 'select', 'review', 'build', 'test'])
    {'match': cmd_match, 'select': cmd_select, 'review': cmd_review, 'build': cmd_build, 'test': cmd_test}[ap.parse_args().cmd]()

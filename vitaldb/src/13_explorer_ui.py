#!/usr/bin/env python
"""
VitalDB Explorer
Stage 13 - Explorer UI
AI-WM Lab

Version: 2026-09-15-v2-native-download

Run:
    streamlit run src/13_explorer_ui.py

Purpose
-------
A researcher-facing Streamlit UI over the frozen VitalDB Explorer backend.

This UI provides:
1. Dataset overview.
2. Metadata-first cohort filtering.
3. Case browser and case metadata.
4. Canonical and exact signal inventory.
5. Laboratory inventory.
6. On-demand remote signal loading through frozen Stage 07.
7. Interactive waveform/numeric visualization.
8. Stage-10-inspired validity warnings for common clinical numerics.
9. Stage-12 download/export command builder (safe: UI does not silently launch
   bulk downloads).

Important
---------
- Source VitalDB data remain immutable.
- Merely browsing metadata does not download signal samples.
- Signal data are retrieved only when the user presses "Load selected signal".
- Stage 07 may retrieve a complete remote track before clipping a requested
  time window locally.
- Invalid values are not silently corrected.
- ECG-derived HR and CO2-derived RR are NOT presented as validated clinical
  measurements in this Stage-13 UI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

SCRIPT_VERSION = "2026-09-15-v2-native-download"
LAB_NAME = "AI-WM Lab"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

MASTER_PATH = METADATA_DIR / "master_case_index.csv"
TRACKS_PATH = METADATA_DIR / "tracks.csv"
DICTIONARY_PATH = METADATA_DIR / "canonical_signal_dictionary.csv"
LABS_PATH = METADATA_DIR / "labs.csv"
SIGNAL_AVAILABILITY_PATH = METADATA_DIR / "case_signal_availability.csv"
TRACK_AVAILABILITY_PATH = METADATA_DIR / "case_track_availability.csv"
LAB_AVAILABILITY_PATH = METADATA_DIR / "case_lab_availability.csv"
DEVICE_AVAILABILITY_PATH = METADATA_DIR / "case_device_availability.csv"
STAGE07_PATH = SRC_DIR / "07_build_remote_signal_retrieval_engine.py"
STAGE10_POLICY_PATH = RESULTS_DIR / "stage10" / "explorer_validity_policy.json"
STAGE12_PATH = SRC_DIR / "12_build_cohort_download_export_engine.py"

PREFERRED_TRACKS = {
    "ECG_II": "SNUADC/ECG_II",
    "PPG": "SNUADC/PLETH",
    "ART_WAVE": "SNUADC/ART",
    "CO2_WAVE": "Primus/CO2",
    "HR": "Solar8000/HR",
    "SPO2": "Solar8000/PLETH_SPO2",
    "ART_MAP": "Solar8000/ART_MBP",
    "ART_SBP": "Solar8000/ART_SBP",
    "ART_DBP": "Solar8000/ART_DBP",
    "ETCO2": "Solar8000/ETCO2",
    "BIS": "BIS/BIS",
}

COMMON_SIGNALS = [
    "ECG_II", "PPG", "ART_WAVE", "CO2_WAVE", "HR", "SPO2",
    "ART_MAP", "ART_SBP", "ART_DBP", "ETCO2", "BIS",
]


def require_files() -> List[Path]:
    return [
        MASTER_PATH, TRACKS_PATH, DICTIONARY_PATH, LABS_PATH,
        SIGNAL_AVAILABILITY_PATH, TRACK_AVAILABILITY_PATH,
        LAB_AVAILABILITY_PATH,
        STAGE07_PATH, STAGE10_POLICY_PATH, STAGE12_PATH,
    ]


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_stage07_engine():
    module_name = "vitaldb_stage07_for_stage13"
    spec = importlib.util.spec_from_file_location(module_name, STAGE07_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Stage 07: {STAGE07_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "RemoteSignalRetrievalEngine"):
        raise RuntimeError("Stage 07 does not expose RemoteSignalRetrievalEngine.")
    return module.RemoteSignalRetrievalEngine(
        tracks_path=TRACKS_PATH,
        dictionary_path=DICTIONARY_PATH,
        master_path=MASTER_PATH,
    )


def canonical_col(df: pd.DataFrame) -> str:
    for c in ("canonical_signal", "canonical_concept", "canonical_name"):
        if c in df.columns:
            return c
    raise KeyError("Canonical signal column not found in dictionary.")


def role_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("data_role", "role", "signal_role"):
        if c in df.columns:
            return c
    return None


def status_col(df: pd.DataFrame) -> Optional[str]:
    for c in ("mapping_status", "status", "semantic_status"):
        if c in df.columns:
            return c
    return None


def safe_display(v: Any) -> str:
    if pd.isna(v):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.3f}".rstrip("0").rstrip(".")
    return str(v)


def get_case_tracks(tracks: pd.DataFrame, dictionary: pd.DataFrame, caseid: int) -> pd.DataFrame:
    ccol = canonical_col(dictionary)
    cols = ["tname", ccol]
    rc = role_col(dictionary)
    sc = status_col(dictionary)
    if rc:
        cols.append(rc)
    if sc:
        cols.append(sc)
    meta = dictionary[cols].drop_duplicates("tname")
    x = tracks[tracks["caseid"] == caseid].copy()
    x = x.merge(meta, on="tname", how="left")
    return x


def numeric_window_validity(
    canonical: str,
    values: np.ndarray,
    policy: Dict[str, Any],
) -> Tuple[bool, str]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False, "No finite values in the selected interval."

    rules = policy.get("numeric_sanity_rules", {})
    rule = rules.get(canonical)
    if not rule:
        return True, "No Stage-10 hard numeric gate is defined for this signal."

    lo = float(rule["hard_min"])
    hi = float(rule["hard_max"])
    med = float(np.median(finite))
    if med < lo or med > hi:
        return False, (
            f"Selected-window median ({med:g}) violates the Stage-10 engineering "
            f"sanity range [{lo:g}, {hi:g}]."
        )
    return True, (
        f"Selected-window median ({med:g}) passes the Stage-10 engineering "
        f"sanity range [{lo:g}, {hi:g}]."
    )


def minmax_downsample(t: np.ndarray, y: np.ndarray, max_points: int = 12000) -> pd.DataFrame:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= max_points:
        return pd.DataFrame({"time_sec": t, "value": y})

    bins = max(1, max_points // 2)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    out_t, out_y = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        yy = y[a:b]
        tt = t[a:b]
        finite = np.isfinite(yy) & np.isfinite(tt)
        if not finite.any():
            continue
        idx_local = np.where(finite)[0]
        vals = yy[idx_local]
        imin = idx_local[int(np.argmin(vals))]
        imax = idx_local[int(np.argmax(vals))]
        for idx in sorted((imin, imax)):
            out_t.append(tt[idx])
            out_y.append(yy[idx])
    return pd.DataFrame({"time_sec": out_t, "value": out_y})


def safe_filename(text: str) -> str:
    text = str(text).replace("/", "__").replace("\\", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def parse_case_ids(text: str):
    tokens = [x for x in re.split(r"[\\s,;]+", text.strip()) if x]
    invalid = [x for x in tokens if not x.isdigit()]
    ids = list(dict.fromkeys(int(x) for x in tokens if x.isdigit()))
    return ids, invalid


def signal_csv_bytes(time_sec, values) -> bytes:
    return pd.DataFrame({
        "time_sec": np.asarray(time_sec, dtype=float),
        "value": np.asarray(values, dtype=float),
    }).to_csv(index=False).encode("utf-8")


def exact_track_for_canonical(dictionary: pd.DataFrame, canonical: str):
    ccol = canonical_col(dictionary)
    rows = dictionary[dictionary[ccol].astype(str) == str(canonical)].copy()
    if rows.empty:
        return None, "CANONICAL_SIGNAL_NOT_IN_DICTIONARY"
    preferred = PREFERRED_TRACKS.get(canonical)
    if preferred and (rows["tname"].astype(str) == preferred).any():
        return preferred, "PREFERRED_EXACT_TRACK"
    sc = status_col(dictionary)
    if sc:
        resolved = rows[rows[sc].astype(str).str.upper() == "RESOLVED"]
        names = resolved["tname"].astype(str).drop_duplicates().tolist()
        if len(names) == 1:
            return names[0], "UNIQUE_RESOLVED_TRACK"
    names = rows["tname"].astype(str).drop_duplicates().tolist()
    if len(names) == 1:
        return names[0], "UNIQUE_TRACK"
    return None, "AMBIGUOUS_CANONICAL_SOURCE_USE_EXACT_TRACK"


def build_native_export_plan(caseids, signals, tracks, dictionary, start_sec, end_sec):
    lookup = {(int(r.caseid), str(r.tname)): str(r.tid)
              for r in tracks[["caseid", "tname", "tid"]].itertuples(index=False)}
    rows = []
    for canonical in signals:
        tname, reason = exact_track_for_canonical(dictionary, canonical)
        for caseid in caseids:
            if tname is None:
                status, tid = "BLOCKED", ""
            else:
                tid = lookup.get((int(caseid), tname), "")
                status = "READY" if tid else "UNAVAILABLE"
                if not tid:
                    reason = "TRACK_NOT_AVAILABLE_FOR_CASE"
            rows.append({"caseid": int(caseid), "canonical_signal": canonical,
                         "tname": tname or "", "tid": tid,
                         "requested_start_sec": float(start_sec),
                         "requested_end_sec": float(end_sec),
                         "status": status, "reason": reason})
    return pd.DataFrame(rows)


def prepare_native_zip(plan_df: pd.DataFrame, policy: dict):
    engine = load_stage07_engine()
    results = []
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("download_plan.csv", plan_df.to_csv(index=False))
        for _, row in plan_df.iterrows():
            result = {"caseid": int(row.caseid), "canonical_signal": row.canonical_signal,
                      "tname": row.tname, "tid": row.tid, "status": "",
                      "sample_count": 0, "error": ""}
            if row.status != "READY":
                result["status"] = "SKIPPED_" + str(row.status)
                result["error"] = str(row.reason)
                results.append(result); continue
            try:
                resolved = engine.resolve_track(int(row.caseid), str(row.tname))
                if str(resolved["tid"]) != str(row.tid):
                    raise RuntimeError("TID mismatch between plan and Stage 07")
                sig = engine.fetch_track(int(row.caseid), str(row.tname),
                                         float(row.requested_start_sec), float(row.requested_end_sec))
                t = np.asarray(sig.time_sec, dtype=float).copy()
                y = np.asarray(sig.values, dtype=float).copy()
                if t.shape != y.shape:
                    raise RuntimeError("Timestamp/value shape mismatch")
                csvb = signal_csv_bytes(t, y)
                sha = hashlib.sha256(csvb).hexdigest()
                folder = f"case_{int(row.caseid):04d}"
                stem = safe_filename(row.tname)
                meta = {
                    "application": "VitalDB Explorer", "lab": LAB_NAME,
                    "stage13_version": SCRIPT_VERSION,
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "caseid": int(row.caseid), "canonical_signal": row.canonical_signal,
                    "tname": row.tname, "tid": str(row.tid),
                    "requested_start_sec": float(row.requested_start_sec),
                    "requested_end_sec": float(row.requested_end_sec),
                    "sample_count": int(len(y)), "finite_count": int(np.isfinite(y).sum()),
                    "missing_count": int(len(y)-np.isfinite(y).sum()),
                    "sampling_rate_hz": float(sig.sampling_rate_hz) if sig.sampling_rate_hz is not None else None,
                    "remote_url": str(sig.remote_url), "remote_format": str(sig.remote_format),
                    "response_bytes": int(sig.response_bytes), "response_sha256": str(sig.response_sha256),
                    "exported_csv_sha256": sha, "raw_values_modified": False,
                    "stage10_policy_version": policy.get("policy_version"),
                    "network_note": "Stage 07 may retrieve the complete remote track before local clipping."
                }
                zf.writestr(f"{folder}/{stem}.csv", csvb)
                zf.writestr(f"{folder}/{stem}.metadata.json", json.dumps(meta, indent=2))
                result.update(status="EXPORTED", sample_count=int(len(y)))
            except Exception as exc:
                result["status"] = "FAILED"
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
        rdf = pd.DataFrame(results)
        manifest = {
            "application": "VitalDB Explorer", "lab": LAB_NAME, "stage": 13,
            "script_version": SCRIPT_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "plan_rows": int(len(plan_df)),
            "exported_tracks": int((rdf.status == "EXPORTED").sum()),
            "failed_tracks": int((rdf.status == "FAILED").sum()),
            "skipped_tracks": int(rdf.status.astype(str).str.startswith("SKIPPED_").sum()),
            "raw_values_modified": False,
            "status": "PASS" if int((rdf.status == "FAILED").sum()) == 0 else "COMPLETED_WITH_FAILURES"
        }
        zf.writestr("download_execution_manifest.csv", rdf.to_csv(index=False))
        zf.writestr("export_manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("README.txt", "VitalDB Explorer Export\\nAI-WM Lab\\n\\nRaw VitalDB values; no silent calibration, normalization, interpolation, clipping, or correction.\\nUse frozen Stage 12 for large/resumable cohort downloads.\\n")
    return bio.getvalue(), rdf, manifest


def filter_master(
    master: pd.DataFrame,
    signal_avail: pd.DataFrame,
    required_signals: List[str],
    departments: List[str],
    sexes: List[str],
    age_range: Optional[Tuple[float, float]],
    emergency_choice: str,
    search_case: str,
) -> pd.DataFrame:
    df = master.copy()

    if required_signals:
        av = signal_avail[["caseid"]].copy()
        for sig in required_signals:
            col = f"has_{sig}"
            if col in signal_avail.columns:
                av[col] = signal_avail[col]
        needed_cols = [f"has_{s}" for s in required_signals if f"has_{s}" in av.columns]
        if needed_cols:
            mask = av[needed_cols].fillna(False).astype(bool).all(axis=1)
            allowed = set(pd.to_numeric(av.loc[mask, "caseid"], errors="coerce").dropna().astype(int))
            df = df[df["caseid"].isin(allowed)]

    if departments and "department" in df.columns:
        df = df[df["department"].astype(str).isin(departments)]

    if sexes and "sex" in df.columns:
        df = df[df["sex"].astype(str).isin(sexes)]

    if age_range is not None and "age" in df.columns:
        age = pd.to_numeric(df["age"], errors="coerce")
        df = df[(age >= age_range[0]) & (age <= age_range[1])]

    if emergency_choice != "All" and "emop" in df.columns:
        em = pd.to_numeric(df["emop"], errors="coerce").fillna(0).astype(int)
        df = df[em == (1 if emergency_choice == "Emergency" else 0)]

    if search_case.strip():
        q = search_case.strip()
        if q.isdigit():
            df = df[df["caseid"] == int(q)]

    return df


def render_header():
    st.title("VitalDB Explorer")
    st.caption(
        f"{LAB_NAME} · Biomedical signal and clinical metadata exploration · "
        f"Stage 13 UI · {SCRIPT_VERSION}"
    )
    st.info(
        "Metadata browsing is local. Signal samples are fetched from VitalDB only "
        "when you explicitly request a signal."
    )


def render_overview(master, tracks, dictionary, labs):
    st.subheader("Dataset Overview")
    a, b, c, d = st.columns(4)
    a.metric("Cases", f"{len(master):,}")
    b.metric("Subjects", f"{master['subjectid'].nunique():,}" if "subjectid" in master else "—")
    c.metric("Exact tracks", f"{tracks['tname'].nunique():,}")
    d.metric("Dictionary entries", f"{len(dictionary):,}")

    e, f, g, h = st.columns(4)
    e.metric("Track rows", f"{len(tracks):,}")
    f.metric("Lab measurements", f"{len(labs):,}")
    g.metric("Cases with labs", f"{labs['caseid'].nunique():,}")
    g2 = dictionary[canonical_col(dictionary)].nunique()
    h.metric("Canonical concepts", f"{g2:,}")

    st.markdown(
        """
        **Explorer architecture:** metadata-first cohort discovery → case detail →
        exact/canonical signal inventory → deliberate remote signal retrieval →
        validity-aware visualization → controlled Stage-12 export.
        """
    )


def render_cohort(master, signal_avail):
    st.subheader("Cohort Explorer")

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)

        available_common = [
            s for s in COMMON_SIGNALS if f"has_{s}" in signal_avail.columns
        ]
        required_signals = c1.multiselect(
            "Required canonical signals",
            options=available_common,
            default=[],
            help="Cases must contain every selected canonical signal.",
        )

        departments = []
        if "department" in master.columns:
            dep_options = sorted(
                x for x in master["department"].dropna().astype(str).unique()
                if x and x.lower() != "nan"
            )
            departments = c2.multiselect("Department", dep_options)

        sexes = []
        if "sex" in master.columns:
            sex_options = sorted(master["sex"].dropna().astype(str).unique())
            sexes = c3.multiselect("Sex", sex_options)

        c4, c5, c6 = st.columns(3)
        age_range = None
        if "age" in master.columns:
            ages = pd.to_numeric(master["age"], errors="coerce").dropna()
            if len(ages):
                amin, amax = int(math.floor(ages.min())), int(math.ceil(ages.max()))
                age_range = c4.slider("Age range", amin, amax, (amin, amax))

        emergency_choice = c5.selectbox(
            "Operation type",
            ["All", "Emergency", "Elective"],
        )
        search_case = c6.text_input("Exact case ID", placeholder="e.g. 1")

    filtered = filter_master(
        master, signal_avail, required_signals, departments, sexes,
        age_range, emergency_choice, search_case,
    )

    st.metric("Matching cases", f"{len(filtered):,}")

    preferred = [
        "caseid", "subjectid", "age", "sex", "height", "weight", "bmi",
        "asa", "emop", "department", "optype", "dx", "opname",
        "ane_type", "case_duration_min", "anesthesia_duration_min",
        "operation_duration_min", "exact_track_count", "canonical_signal_count",
        "waveform_track_count", "lab_measurement_count",
    ]
    cols = [c for c in preferred if c in filtered.columns]
    if not cols:
        cols = list(filtered.columns[:20])

    st.dataframe(
        filtered[cols].head(1000),
        use_container_width=True,
        hide_index=True,
    )
    if len(filtered) > 1000:
        st.caption("Showing the first 1,000 matching cases.")

    return filtered


def render_case_browser(master, tracks, dictionary, labs, signal_avail, filtered):
    st.subheader("Case Browser")
    candidate_ids = (
        pd.to_numeric(filtered["caseid"], errors="coerce").dropna().astype(int).tolist()
        if len(filtered)
        else []
    )
    if not candidate_ids:
        st.warning("No cases match the current cohort filters.")
        return None, None

    selected_case = st.selectbox(
        "Select case",
        options=candidate_ids,
        format_func=lambda x: f"Case {x}",
    )

    row = master[master["caseid"] == selected_case].iloc[0]

    a, b, c, d = st.columns(4)
    a.metric("Case ID", selected_case)
    b.metric("Age", safe_display(row.get("age", np.nan)))
    c.metric("Sex", safe_display(row.get("sex", np.nan)))
    d.metric("Department", safe_display(row.get("department", np.nan)))

    tabs = st.tabs(["Clinical metadata", "Signals", "Laboratory data"])

    with tabs[0]:
        important = [
            "subjectid", "age", "sex", "height", "weight", "bmi", "asa",
            "emop", "department", "optype", "dx", "opname", "approach",
            "position", "ane_type", "icu_days", "death_inhosp",
            "case_duration_min", "anesthesia_duration_min", "operation_duration_min",
            "exact_track_count", "canonical_signal_count", "device_count",
            "waveform_track_count", "numeric_track_count", "unique_lab_count",
        ]
        meta = [
            {"field": k, "value": safe_display(row.get(k))}
            for k in important if k in master.columns
        ]
        st.dataframe(pd.DataFrame(meta), use_container_width=True, hide_index=True)

    case_tracks = get_case_tracks(tracks, dictionary, selected_case)
    with tabs[1]:
        st.write(f"**{len(case_tracks):,} exact tracks available**")
        st.dataframe(case_tracks, use_container_width=True, hide_index=True)

    case_labs = labs[labs["caseid"] == selected_case].copy()
    with tabs[2]:
        if case_labs.empty:
            st.info("No laboratory measurements are available for this case.")
        else:
            st.write(
                f"**{len(case_labs):,} measurements · "
                f"{case_labs['name'].nunique():,} variables**"
            )
            st.caption(
                "Lab dt is preserved as source-relative timing. Units are not invented."
            )
            st.dataframe(case_labs, use_container_width=True, hide_index=True)

    return selected_case, case_tracks


def render_signal_viewer(selected_case, case_tracks, dictionary, policy):
    st.subheader("Remote Signal Viewer")
    if selected_case is None or case_tracks is None or case_tracks.empty:
        st.info("Select a case first.")
        return

    ccol = canonical_col(dictionary)
    rc = role_col(dictionary)

    display = case_tracks.copy()
    if ccol not in display.columns:
        display[ccol] = ""
    display["label"] = display.apply(
        lambda r: f"{r['tname']}  [{r.get(ccol, '')}]",
        axis=1,
    )

    selected_label = st.selectbox("Exact track", display["label"].tolist())
    selected = display[display["label"] == selected_label].iloc[0]
    tname = str(selected["tname"])
    canonical = str(selected.get(ccol, ""))
    role = str(selected.get(rc, "")) if rc else ""

    c1, c2, c3 = st.columns(3)
    start_sec = c1.number_input(
        "Start time (sec)", min_value=0.0, value=600.0, step=10.0
    )
    duration_sec = c2.number_input(
        "Duration (sec)", min_value=1.0, max_value=600.0, value=60.0, step=10.0
    )
    end_sec = float(start_sec + duration_sec)
    c3.metric("Requested end", f"{end_sec:g} sec")

    st.caption(
        "Important: Stage 07 may retrieve the complete remote track before clipping "
        "this requested interval locally."
    )

    key = f"loaded_signal_{selected_case}_{safe_key(tname)}_{start_sec}_{end_sec}"

    if st.button("Load selected signal", type="primary"):
        with st.spinner(f"Retrieving {tname} for Case {selected_case}…"):
            try:
                engine = load_stage07_engine()
                sig = engine.fetch_track(
                    caseid=int(selected_case),
                    tname=tname,
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                )
                st.session_state[key] = sig
            except Exception as exc:
                st.error(f"Signal retrieval failed: {type(exc).__name__}: {exc}")

    sig = st.session_state.get(key)
    if sig is None:
        return

    t = np.asarray(sig.time_sec, dtype=float)
    y = np.asarray(sig.values, dtype=float)
    finite = y[np.isfinite(y)]

    a, b, c, d = st.columns(4)
    a.metric("Samples", f"{len(y):,}")
    b.metric(
        "Sampling rate",
        f"{sig.sampling_rate_hz:g} Hz" if sig.sampling_rate_hz is not None else "Irregular / numeric",
    )
    c.metric("Finite", f"{len(finite):,}")
    d.metric("Missing", f"{len(y)-len(finite):,}")

    if finite.size:
        e, f, g, h = st.columns(4)
        e.metric("Minimum", f"{np.min(finite):.4g}")
        f.metric("Median", f"{np.median(finite):.4g}")
        g.metric("Maximum", f"{np.max(finite):.4g}")
        h.metric("Mean", f"{np.mean(finite):.4g}")

    if role.upper() == "NUMERIC" or canonical in {
        "HR", "SPO2", "ART_MAP", "ART_SBP", "ART_DBP", "ETCO2", "BIS"
    }:
        valid, reason = numeric_window_validity(canonical, y, policy)
        if valid:
            st.success("Display validity: PASS — " + reason)
        else:
            st.error(
                "Display validity: BLOCKED — " + reason +
                " The raw source values are retained for provenance but must not "
                "be presented as valid clinical measurements."
            )

    if canonical == "ART_WAVE" and finite.size and np.median(finite) <= 0:
        st.warning(
            "Arterial waveform has a non-positive median in this interval. "
            "It may be shown as a raw technical waveform, but calibrated pressure "
            "interpretation is blocked."
        )

    plot_df = minmax_downsample(t, y)
    st.line_chart(plot_df, x="time_sec", y="value", use_container_width=True)
    st.caption(
        "Visualization may be deterministically downsampled for display only. "
        "The retrieved source arrays are not modified."
    )

    st.markdown("#### Download current signal")
    csv_bytes = signal_csv_bytes(t, y)
    base = f"case_{int(selected_case):04d}_{safe_filename(tname)}_{float(start_sec):g}_{float(end_sec):g}s"
    dc1, dc2 = st.columns(2)
    current_meta = {
        "application": "VitalDB Explorer", "lab": LAB_NAME,
        "stage13_version": SCRIPT_VERSION,
        "caseid": int(sig.caseid), "canonical_signal": canonical,
        "tname": str(sig.tname), "tid": str(sig.tid),
        "requested_start_sec": sig.requested_start_sec,
        "requested_end_sec": sig.requested_end_sec,
        "sample_count": int(sig.sample_count), "finite_count": int(sig.finite_count),
        "missing_count": int(sig.missing_count),
        "sampling_rate_hz": float(sig.sampling_rate_hz) if sig.sampling_rate_hz is not None else None,
        "response_sha256": str(sig.response_sha256),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "raw_values_modified": False,
    }
    dc1.download_button("Download current signal CSV", csv_bytes, base + ".csv", "text/csv")
    dc2.download_button("Download signal metadata JSON", json.dumps(current_meta, indent=2).encode("utf-8"), base + ".metadata.json", "application/json")

    with st.expander("Retrieval provenance"):
        prov = {
            "caseid": sig.caseid,
            "tname": sig.tname,
            "tid": sig.tid,
            "canonical_signal": getattr(sig, "canonical_signal", canonical),
            "remote_url": sig.remote_url,
            "remote_format": sig.remote_format,
            "sample_count": sig.sample_count,
            "finite_count": sig.finite_count,
            "missing_count": sig.missing_count,
            "sampling_rate_hz": sig.sampling_rate_hz,
            "response_bytes": sig.response_bytes,
            "response_sha256": sig.response_sha256,
            "requested_start_sec": sig.requested_start_sec,
            "requested_end_sec": sig.requested_end_sec,
            "clipped": sig.clipped,
        }
        st.json(prov)


def safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)


def render_export_builder(selected_case: Optional[int], case_tracks: Optional[pd.DataFrame], dictionary, tracks, master, policy, filtered):
    st.subheader("Controlled Download / Export")
    st.write("Prepare a small multi-case/multi-signal ZIP directly inside the Explorer. Use frozen Stage 12 for large or resumable jobs.")

    source = st.radio("Case selection source", ["Enter case IDs", "Use current filtered cohort"], horizontal=True)
    if source == "Enter case IDs":
        text = st.text_input("Case IDs", value=str(selected_case) if selected_case is not None else "1",
                             help="Spaces, commas, and semicolons are accepted. Example: 1 2 3 or 1,2,3")
        caseids, invalid = parse_case_ids(text)
    else:
        caseids = pd.to_numeric(filtered["caseid"], errors="coerce").dropna().astype(int).tolist()
        invalid = []
        st.write(f"Current filtered cohort contains **{len(caseids):,} cases**.")

    if invalid:
        st.error("Invalid case ID token(s): " + ", ".join(invalid))
    valid_master = set(pd.to_numeric(master["caseid"], errors="raise").astype(int))
    unknown = [x for x in caseids if x not in valid_master]
    if unknown:
        st.error("Unknown VitalDB case ID(s): " + ", ".join(map(str, unknown[:20])))

    ccol = canonical_col(dictionary)
    concepts = sorted(dictionary[ccol].dropna().astype(str).unique())
    defaults = [x for x in ("ECG_II", "PPG", "HR") if x in concepts]
    signals = st.multiselect("Canonical signals", concepts, default=defaults)

    c1, c2, c3 = st.columns(3)
    start_sec = c1.number_input("Export start (sec)", min_value=0.0, value=600.0, step=10.0)
    end_sec = c2.number_input("Export end (sec)", min_value=0.1, value=660.0, step=10.0)
    max_cases = c3.number_input("Interactive export case limit", min_value=1, max_value=100, value=20, step=1)

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected cases", len(caseids)); m2.metric("Selected signals", len(signals)); m3.metric("Potential track requests", len(caseids)*len(signals))
    st.warning("Stage 07 may transfer the complete remote track before clipping the requested interval. Use Stage 12 for large/resumable acquisition.")

    blocked = bool(invalid or unknown or not caseids or not signals or end_sec <= start_sec or len(caseids) > int(max_cases))
    if end_sec <= start_sec:
        st.error("Export end must be greater than export start.")
    if len(caseids) > int(max_cases):
        st.error(f"Interactive export blocked: {len(caseids)} cases exceed the safety limit of {int(max_cases)}.")

    if not blocked:
        plan = build_native_export_plan(caseids, signals, tracks, dictionary, start_sec, end_sec)
        ready = int((plan.status == "READY").sum()); unavailable = int((plan.status == "UNAVAILABLE").sum()); amb = int((plan.status == "BLOCKED").sum())
        p1,p2,p3=st.columns(3); p1.metric("Ready",ready); p2.metric("Unavailable",unavailable); p3.metric("Blocked/ambiguous",amb)
        with st.expander("Review exact download plan"):
            st.dataframe(plan, use_container_width=True, hide_index=True)
        confirm = st.checkbox("I reviewed the case count, signals, interval, and full-track network-transfer warning.")
        if st.button("Prepare ZIP download", type="primary", disabled=(not confirm or ready == 0)):
            with st.spinner(f"Preparing {ready} available track request(s)…"):
                try:
                    zb, rdf, manifest = prepare_native_zip(plan, policy)
                    st.session_state["native_export_zip"] = zb
                    st.session_state["native_export_results"] = rdf
                    st.session_state["native_export_manifest"] = manifest
                except Exception as exc:
                    st.error(f"Export preparation failed: {type(exc).__name__}: {exc}")

    if "native_export_zip" in st.session_state:
        manifest = st.session_state["native_export_manifest"]
        rdf = st.session_state["native_export_results"]
        st.divider(); st.markdown("### Prepared export")
        a,b,c=st.columns(3); a.metric("Exported tracks",manifest["exported_tracks"]); b.metric("Failed tracks",manifest["failed_tracks"]); c.metric("Skipped tracks",manifest["skipped_tracks"])
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download prepared ZIP", st.session_state["native_export_zip"], f"VitalDB_Explorer_Export_{stamp}.zip", "application/zip", type="primary")
        st.caption("ZIP includes signal CSVs, per-track metadata JSON, download plan, execution manifest, export manifest, and README.")

    st.divider()
    with st.expander("Large/resumable export: Stage 12 command"):
        if caseids and signals and not invalid:
            ids=",".join(map(str,caseids)); sigs=",".join(signals)
            cmd=f"python src\\12_build_cohort_download_export_engine.py --case-ids {ids} --signals {sigs} --start-sec {float(start_sec):g} --end-sec {float(end_sec):g}"
            st.markdown("**Plan only:**"); st.code(cmd, language="bat")
            st.markdown("**Execute after reviewing the plan:**"); st.code(cmd+" --execute --confirm-download YES", language="bat")


def render_about():
    st.subheader("About")
    st.markdown(
        f"""
        ### VitalDB Explorer
        **Laboratory:** {LAB_NAME}

        This application was developed as a metadata-first research interface for
        exploring the VitalDB Open Dataset while preserving exact source provenance.

        **Current backend status:** Stages 00–12 are frozen after validation.
        Stage 13 is the researcher-facing UI validation stage.

        **Scientific safeguards**
        - Exact VitalDB `tname` and `tid` remain authoritative.
        - Source signal values are never silently corrected.
        - Invalid clinical numerics are blocked from valid-clinical display.
        - Negative/non-calibrated arterial waveform states are not interpreted as
          calibrated arterial pressure.
        - Raw PSD peaks are not labelled as physiological HR/RR.
        - The current ECG-derived rate algorithm remains experimental and is not
          exposed here as a validated clinical measurement.
        - CO2 waveform-derived respiratory rate remains an engineering estimate and
          is not exposed here as a validated clinical measurement.
        """
    )


def main():

    render_header()

    missing = [p for p in require_files() if not p.exists()]
    if missing:
        st.error("Stage 13 cannot start because required frozen inputs are missing.")
        st.code("\n".join(str(p) for p in missing))
        st.stop()

    master = load_csv(str(MASTER_PATH))
    tracks = load_csv(str(TRACKS_PATH))
    dictionary = load_csv(str(DICTIONARY_PATH))
    labs = load_csv(str(LABS_PATH))
    signal_avail = load_csv(str(SIGNAL_AVAILABILITY_PATH))
    policy = load_json(str(STAGE10_POLICY_PATH))

    with st.sidebar:
        st.markdown(f"## VitalDB Explorer")
        st.caption(LAB_NAME)
        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Cohort Explorer",
                "Case Browser",
                "Signal Viewer",
                "Download / Export",
                "About",
            ],
        )
        st.divider()
        st.caption(f"Stage 13 · {SCRIPT_VERSION}")
        st.caption("Frozen backend: Stages 00–12")

    # Cohort filters are persisted in session state by rendering them only on the
    # Cohort page. Other pages default to the entire master cohort unless a filtered
    # case list was saved.
    filtered = master
    if "filtered_caseids" in st.session_state:
        ids = set(st.session_state["filtered_caseids"])
        filtered = master[master["caseid"].isin(ids)]

    selected_case = st.session_state.get("selected_case")
    case_tracks = None

    if page == "Overview":
        render_overview(master, tracks, dictionary, labs)

    elif page == "Cohort Explorer":
        filtered = render_cohort(master, signal_avail)
        st.session_state["filtered_caseids"] = (
            pd.to_numeric(filtered["caseid"], errors="coerce")
            .dropna().astype(int).tolist()
        )
        st.caption(
            "The matching case IDs are retained while you move to Case Browser."
        )

    elif page == "Case Browser":
        selected_case, case_tracks = render_case_browser(
            master, tracks, dictionary, labs, signal_avail, filtered
        )
        if selected_case is not None:
            st.session_state["selected_case"] = int(selected_case)

    elif page == "Signal Viewer":
        if selected_case is None:
            ids = (
                pd.to_numeric(filtered["caseid"], errors="coerce")
                .dropna().astype(int).tolist()
            )
            if not ids:
                st.warning("No case is available. Adjust the cohort filters.")
                st.stop()
            selected_case = st.selectbox(
                "Select case",
                ids,
                format_func=lambda x: f"Case {x}",
            )
            st.session_state["selected_case"] = int(selected_case)
        else:
            st.caption(f"Current case: **{selected_case}**")
            if st.button("Choose another case"):
                st.session_state.pop("selected_case", None)
                st.rerun()

        case_tracks = get_case_tracks(tracks, dictionary, int(selected_case))
        render_signal_viewer(int(selected_case), case_tracks, dictionary, policy)

    elif page == "Download / Export":
        if selected_case is not None:
            case_tracks = get_case_tracks(tracks, dictionary, int(selected_case))
        render_export_builder(selected_case, case_tracks, dictionary, tracks, master, policy, filtered)

    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()

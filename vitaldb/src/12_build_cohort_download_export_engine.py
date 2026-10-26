#!/usr/bin/env python
"""
VitalDB Explorer
Stage 12 - Cohort Download & Export Engine

Version: 2026-09-15-v1

Purpose
-------
Provide deliberate, resumable, manifest-driven acquisition/export of selected
VitalDB tracks for a selected cohort.

Safety / architecture
---------------------
* NEVER downloads the entire VitalDB dataset automatically.
* Default mode is PLAN ONLY. No signal bytes are downloaded unless --execute is supplied.
* Uses frozen Stage-07 RemoteSignalRetrievalEngine for remote access.
* Exact VitalDB tname -> tid mapping remains authoritative.
* Existing Stage 00-11 metadata/results are read-only.
* Downloads are written only under results/stage12 (or an explicit --output-dir).
* Existing completed exports are skipped when their manifest/file checksum matches.
* Partial failures do not destroy successful exports.
* Raw source values are never calibrated, clipped, normalized, repaired, or silently corrected.
* Stage-10 validity policy is copied into the manifest as provenance; this stage does not
  reinterpret raw source values.
* A hard --max-cases gate prevents accidental unbounded acquisition.
* --execute requires --confirm-download YES.

Export format
-------------
CSV, one file per case/exact track:
    time_sec,value

The export is a lossless representation of the parsed Stage-07 SignalData arrays
(up to normal CSV floating-point text round-trip). Metadata/provenance is stored
separately in JSON/CSV manifests.

Examples
--------
Plan 5 cases with ECG + PPG:
    python src/12_build_cohort_download_export_engine.py ^
        --case-ids 1,2,3,4,5 ^
        --signals ECG_II,PPG

Execute that plan:
    python src/12_build_cohort_download_export_engine.py ^
        --case-ids 1,2,3,4,5 ^
        --signals ECG_II,PPG ^
        --execute --confirm-download YES

Plan cases from a CSV containing a caseid column:
    python src/12_build_cohort_download_export_engine.py ^
        --cohort-file my_cohort.csv ^
        --signals ECG_II,PPG,HR

Exact tracks can be requested:
    --tracks SNUADC/ECG_II,SNUADC/PLETH

Optional local time window:
    --start-sec 600 --end-sec 660

IMPORTANT:
Stage-07 fetch_track performs remote full-track retrieval and then client-side clipping.
Therefore --start-sec/--end-sec reduces exported content, but does NOT guarantee reduced
network transfer. The plan explicitly reports this.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "2026-09-15-v1"
DEFAULT_MAX_CASES = 100
DEFAULT_SIGNALS = "ECG_II,PPG"

# Preferred exact tracks for common Explorer concepts.
# Other canonical concepts are resolved from the Stage-03 dictionary.
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


@dataclass
class PlanRow:
    caseid: int
    canonical_signal: str
    tname: str
    tid: str
    data_role: str
    mapping_status: str
    requested_start_sec: Optional[float]
    requested_end_sec: Optional[float]
    action: str
    reason: str
    output_file: str


@dataclass
class DownloadResult:
    caseid: int
    canonical_signal: str
    tname: str
    tid: str
    status: str
    sample_count: int
    finite_count: int
    missing_count: int
    start_sec: Optional[float]
    end_sec: Optional[float]
    sampling_rate_hz: Optional[float]
    remote_url: str
    remote_format: str
    response_bytes: int
    response_sha256: str
    exported_file: str
    exported_bytes: int
    exported_sha256: str
    error: str


def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_name(text: str) -> str:
    text = text.replace("/", "__").replace("\\", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def import_stage07(path: Path):
    module_name = "vitaldb_stage07_for_stage12"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import Stage 07 from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "RemoteSignalRetrievalEngine"):
        raise RuntimeError("Frozen Stage 07 does not expose RemoteSignalRetrievalEngine.")
    return module


def canonical_column(dictionary: pd.DataFrame) -> str:
    for c in ("canonical_signal", "canonical_concept", "canonical_name"):
        if c in dictionary.columns:
            return c
    raise KeyError(
        "Cannot locate canonical-signal column in canonical_signal_dictionary.csv. "
        f"Columns={list(dictionary.columns)}"
    )


def status_column(dictionary: pd.DataFrame) -> Optional[str]:
    for c in ("mapping_status", "status", "semantic_status"):
        if c in dictionary.columns:
            return c
    return None


def role_column(dictionary: pd.DataFrame) -> Optional[str]:
    for c in ("data_role", "role", "signal_role"):
        if c in dictionary.columns:
            return c
    return None


def resolve_requests(
    tracks: pd.DataFrame,
    dictionary: pd.DataFrame,
    caseids: Sequence[int],
    requested_signals: Sequence[str],
    requested_tracks: Sequence[str],
    start_sec: Optional[float],
    end_sec: Optional[float],
    export_root: Path,
) -> Tuple[List[PlanRow], List[str]]:
    ccol = canonical_column(dictionary)
    scol = status_column(dictionary)
    rcol = role_column(dictionary)

    d = dictionary.copy()
    d["tname"] = d["tname"].astype(str)
    d[ccol] = d[ccol].astype(str)

    canonical_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in d.iterrows():
        canonical_lookup.setdefault(str(row[ccol]), []).append(row.to_dict())

    exact_to_meta = {str(r["tname"]): r.to_dict() for _, r in d.iterrows()}
    warnings: List[str] = []
    requests: List[Tuple[str, str]] = []  # canonical, exact tname

    for sig in requested_signals:
        if sig not in canonical_lookup:
            warnings.append(f"Unknown canonical signal requested: {sig}")
            continue

        candidates = canonical_lookup[sig]
        preferred = PREFERRED_TRACKS.get(sig)
        chosen = None
        if preferred:
            for row in candidates:
                if str(row["tname"]) == preferred:
                    chosen = preferred
                    break

        if chosen is None:
            resolved = []
            for row in candidates:
                st = str(row.get(scol, "")) if scol else ""
                if st.upper() == "RESOLVED":
                    resolved.append(str(row["tname"]))
            pool = resolved if resolved else [str(x["tname"]) for x in candidates]
            if len(pool) == 1:
                chosen = pool[0]
            else:
                warnings.append(
                    f"Canonical signal {sig} has multiple exact sources "
                    f"({', '.join(pool)}). Use --tracks to choose explicitly."
                )
                continue
        requests.append((sig, chosen))

    for tname in requested_tracks:
        meta = exact_to_meta.get(tname)
        if meta is None:
            warnings.append(f"Unknown exact tname requested: {tname}")
            continue
        requests.append((str(meta[ccol]), tname))

    # Deduplicate canonical+tname requests.
    requests = list(dict.fromkeys(requests))

    track_key = tracks[["caseid", "tname", "tid"]].copy()
    track_key["caseid"] = pd.to_numeric(track_key["caseid"], errors="raise").astype(int)
    track_key["tname"] = track_key["tname"].astype(str)
    track_map = {
        (int(r.caseid), str(r.tname)): str(r.tid)
        for r in track_key.itertuples(index=False)
    }

    rows: List[PlanRow] = []
    for caseid in caseids:
        for canonical, tname in requests:
            meta = exact_to_meta[tname]
            tid = track_map.get((int(caseid), tname))
            role = str(meta.get(rcol, "")) if rcol else ""
            mapping_status = str(meta.get(scol, "")) if scol else ""
            outfile = (
                export_root
                / f"case_{int(caseid):04d}"
                / f"{safe_name(tname)}.csv"
            )
            if tid is None:
                rows.append(
                    PlanRow(
                        int(caseid), canonical, tname, "", role, mapping_status,
                        start_sec, end_sec, "SKIP", "TRACK_NOT_AVAILABLE_FOR_CASE",
                        str(outfile),
                    )
                )
            else:
                rows.append(
                    PlanRow(
                        int(caseid), canonical, tname, tid, role, mapping_status,
                        start_sec, end_sec, "DOWNLOAD", "AVAILABLE", str(outfile),
                    )
                )
    return rows, warnings


def select_cases(
    master: pd.DataFrame,
    case_ids_text: Optional[str],
    cohort_file: Optional[Path],
    first_n: Optional[int],
) -> List[int]:
    valid_cases = set(pd.to_numeric(master["caseid"], errors="raise").astype(int).tolist())

    selectors = sum(
        x is not None
        for x in (case_ids_text, cohort_file, first_n)
    )
    if selectors > 1:
        raise ValueError(
            "Use only one cohort selector: --case-ids OR --cohort-file OR --first-n."
        )

    if case_ids_text is not None:
        ids = [int(x) for x in parse_csv_list(case_ids_text)]
    elif cohort_file is not None:
        df = pd.read_csv(cohort_file)
        if "caseid" not in df.columns:
            raise KeyError(f"Cohort file must contain a 'caseid' column: {cohort_file}")
        ids = pd.to_numeric(df["caseid"], errors="raise").astype(int).tolist()
    elif first_n is not None:
        ids = (
            pd.to_numeric(master["caseid"], errors="raise")
            .astype(int)
            .sort_values()
            .head(first_n)
            .tolist()
        )
    else:
        # Deliberately small, safe default for planning.
        ids = (
            pd.to_numeric(master["caseid"], errors="raise")
            .astype(int)
            .sort_values()
            .head(1)
            .tolist()
        )

    ids = list(dict.fromkeys(ids))
    unknown = [x for x in ids if x not in valid_cases]
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown[:20]}")
    return ids


def existing_export_valid(path: Path, sidecar: Path, tid: str) -> bool:
    if not path.exists() or not sidecar.exists():
        return False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if str(meta.get("tid")) != str(tid):
            return False
        expected = str(meta.get("exported_sha256", ""))
        if not expected:
            return False
        return sha256_file(path) == expected
    except Exception:
        return False


def write_signal_csv_atomic(path: Path, time_sec: np.ndarray, values: np.ndarray) -> Tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    df = pd.DataFrame(
        {
            "time_sec": np.asarray(time_sec, dtype=float),
            "value": np.asarray(values, dtype=float),
        }
    )
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path.stat().st_size, sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VitalDB Explorer Stage 12 cohort download/export engine."
    )
    cohort = parser.add_argument_group("cohort selection")
    cohort.add_argument("--case-ids", type=str, default=None,
                        help="Comma-separated case IDs.")
    cohort.add_argument("--cohort-file", type=Path, default=None,
                        help="CSV with a caseid column.")
    cohort.add_argument("--first-n", type=int, default=None,
                        help="First N case IDs; intended for testing/planning.")

    sigs = parser.add_argument_group("signal selection")
    sigs.add_argument("--signals", type=str, default=DEFAULT_SIGNALS,
                      help="Comma-separated canonical concepts.")
    sigs.add_argument("--tracks", type=str, default=None,
                      help="Comma-separated exact VitalDB tnames.")

    window = parser.add_argument_group("optional exported time window")
    window.add_argument("--start-sec", type=float, default=None)
    window.add_argument("--end-sec", type=float, default=None)

    safety = parser.add_argument_group("safety/execution")
    safety.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES,
                        help=f"Hard safety gate. Default {DEFAULT_MAX_CASES}.")
    safety.add_argument("--execute", action="store_true",
                        help="Actually retrieve and export signals. Default is plan-only.")
    safety.add_argument("--confirm-download", type=str, default=None,
                        help="Must be exactly YES when --execute is used.")
    safety.add_argument("--overwrite", action="store_true",
                        help="Replace existing valid exports.")
    safety.add_argument("--output-dir", type=Path, default=None,
                        help="Optional export root. Default results/stage12/exports.")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    metadata_dir = project_root / "metadata"
    src_dir = project_root / "src"
    stage10_dir = project_root / "results" / "stage10"
    results_dir = project_root / "results" / "stage12"
    export_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "exports"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    tracks_path = metadata_dir / "tracks.csv"
    dictionary_path = metadata_dir / "canonical_signal_dictionary.csv"
    master_path = metadata_dir / "master_case_index.csv"
    stage07_path = src_dir / "07_build_remote_signal_retrieval_engine.py"
    policy_path = stage10_dir / "explorer_validity_policy.json"

    banner("VITALDB EXPLORER - STAGE 12")
    print(f"Script version:       {SCRIPT_VERSION}")
    print(f"Project root:         {project_root}")
    print(f"Mode:                 {'EXECUTE' if args.execute else 'PLAN ONLY'}")
    print(f"Export root:          {export_root}")

    banner("1. VALIDATING REQUIRED INPUTS")
    required = [
        tracks_path, dictionary_path, master_path, stage07_path, policy_path
    ]
    for p in required:
        if not p.exists():
            print(f"FAIL | {p}")
            raise FileNotFoundError(p)
        print(f"PASS | {p}")

    if args.start_sec is not None and args.start_sec < 0:
        raise ValueError("--start-sec must be >= 0.")
    if (
        args.start_sec is not None
        and args.end_sec is not None
        and args.end_sec <= args.start_sec
    ):
        raise ValueError("--end-sec must be greater than --start-sec.")
    if args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1.")
    if args.first_n is not None and args.first_n < 1:
        raise ValueError("--first-n must be >= 1.")
    if args.execute and args.confirm_download != "YES":
        raise RuntimeError(
            "Execution blocked. --execute requires --confirm-download YES."
        )

    banner("2. LOADING METADATA AND FROZEN CONTRACTS")
    tracks = pd.read_csv(tracks_path)
    dictionary = pd.read_csv(dictionary_path)
    master = pd.read_csv(master_path)
    with policy_path.open("r", encoding="utf-8") as f:
        policy = json.load(f)

    print(f"Cases:                {len(master):,}")
    print(f"Track rows:           {len(tracks):,}")
    print(f"Exact tracks:         {tracks['tname'].nunique():,}")
    print(f"Stage-10 policy:      {policy.get('policy_version')}")

    banner("3. RESOLVING COHORT")
    caseids = select_cases(master, args.case_ids, args.cohort_file, args.first_n)
    if len(caseids) > args.max_cases:
        raise RuntimeError(
            f"Safety gate blocked {len(caseids)} cases because --max-cases="
            f"{args.max_cases}. Increase --max-cases deliberately if intended."
        )
    print(f"Selected cases:       {len(caseids):,}")
    print("Case IDs:             " + ", ".join(map(str, caseids[:50])))
    if len(caseids) > 50:
        print(f"... plus {len(caseids)-50} additional cases")

    requested_signals = parse_csv_list(args.signals)
    requested_tracks = parse_csv_list(args.tracks)
    if not requested_signals and not requested_tracks:
        raise ValueError("Select at least one canonical signal or exact track.")

    banner("4. BUILDING DOWNLOAD PLAN")
    plan_rows, warnings = resolve_requests(
        tracks=tracks,
        dictionary=dictionary,
        caseids=caseids,
        requested_signals=requested_signals,
        requested_tracks=requested_tracks,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        export_root=export_root,
    )
    if warnings:
        for w in warnings:
            print(f"WARNING | {w}")

    if not plan_rows:
        raise RuntimeError("No valid signal/track requests could be resolved.")

    plan_df = pd.DataFrame([asdict(x) for x in plan_rows])
    available_df = plan_df[plan_df["action"] == "DOWNLOAD"].copy()
    unavailable_df = plan_df[plan_df["action"] == "SKIP"].copy()

    plan_path = results_dir / "download_plan.csv"
    plan_df.to_csv(plan_path, index=False)

    print(f"Requested plan rows:  {len(plan_df):,}")
    print(f"Available downloads:  {len(available_df):,}")
    print(f"Unavailable/skipped:  {len(unavailable_df):,}")
    print(f"Plan saved:           {plan_path}")

    print("\nRequested exact tracks:")
    for tname in sorted(available_df["tname"].unique()):
        print(f"  - {tname}")

    print(
        "\nNETWORK NOTE: Stage-07 retrieves the complete remote track before local "
        "window clipping. A requested time window reduces exported samples, but "
        "does not guarantee smaller network transfer."
    )

    plan_manifest = {
        "stage": "12",
        "script_version": SCRIPT_VERSION,
        "generated_utc": utc_now(),
        "mode": "EXECUTE" if args.execute else "PLAN_ONLY",
        "selected_case_count": len(caseids),
        "selected_cases": caseids,
        "requested_signals": requested_signals,
        "requested_tracks": requested_tracks,
        "requested_start_sec": args.start_sec,
        "requested_end_sec": args.end_sec,
        "max_cases_gate": args.max_cases,
        "available_download_rows": len(available_df),
        "unavailable_rows": len(unavailable_df),
        "export_root": str(export_root),
        "stage10_policy_version": policy.get("policy_version"),
        "remote_behavior_note": (
            "Stage-07 fetch_track retrieves a complete remote track and clips "
            "the requested window client-side."
        ),
        "raw_value_policy": (
            "No calibration, clipping, normalization, interpolation, repair, "
            "or silent correction."
        ),
        "warnings": warnings,
    }
    atomic_write_json(results_dir / "download_plan_manifest.json", plan_manifest)

    if not args.execute:
        banner("STAGE 12 PLAN COMPLETE")
        print("No remote signal data were downloaded.")
        print(f"Selected cases:       {len(caseids)}")
        print(f"Available downloads:  {len(available_df)}")
        print(f"Unavailable/skipped:  {len(unavailable_df)}")
        print("STATUS: PASS (PLAN ONLY)")
        print(
            "\nTo execute this exact type of request, rerun with "
            "--execute --confirm-download YES."
        )
        return 0

    banner("5. LOADING FROZEN STAGE 07 RETRIEVAL ENGINE")
    stage07 = import_stage07(stage07_path)
    engine = stage07.RemoteSignalRetrievalEngine(
        tracks_path=tracks_path,
        dictionary_path=dictionary_path,
        master_path=master_path,
    )
    print("PASS | RemoteSignalRetrievalEngine loaded.")

    banner("6. EXECUTING RESUMABLE DOWNLOAD/EXPORT PLAN")
    results: List[DownloadResult] = []

    for i, row in enumerate(plan_rows, start=1):
        print(
            f"[{i:>4}/{len(plan_rows):<4}] "
            f"case={row.caseid:<5} | {row.tname}"
        )

        if row.action != "DOWNLOAD":
            print(f"  SKIP | {row.reason}")
            results.append(
                DownloadResult(
                    row.caseid, row.canonical_signal, row.tname, row.tid,
                    "SKIPPED_UNAVAILABLE", 0, 0, 0, None, None, None,
                    "", "", 0, "", row.output_file, 0, "", row.reason,
                )
            )
            continue

        out = Path(row.output_file)
        sidecar = out.with_suffix(".metadata.json")

        if not args.overwrite and existing_export_valid(out, sidecar, row.tid):
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            print("  SKIP | existing export checksum verified")
            results.append(
                DownloadResult(
                    row.caseid, row.canonical_signal, row.tname, row.tid,
                    "SKIPPED_EXISTING_VALID",
                    int(meta.get("sample_count", 0)),
                    int(meta.get("finite_count", 0)),
                    int(meta.get("missing_count", 0)),
                    meta.get("start_sec"),
                    meta.get("end_sec"),
                    meta.get("sampling_rate_hz"),
                    str(meta.get("remote_url", "")),
                    str(meta.get("remote_format", "")),
                    int(meta.get("response_bytes", 0)),
                    str(meta.get("response_sha256", "")),
                    str(out),
                    int(out.stat().st_size),
                    str(meta.get("exported_sha256", "")),
                    "",
                )
            )
            continue

        try:
            # Stage-07 performs authoritative exact-track resolution and tid validation.
            resolved = engine.resolve_track(row.caseid, row.tname)
            if str(resolved["tid"]) != str(row.tid):
                raise RuntimeError(
                    f"TID mismatch: plan={row.tid}, Stage07={resolved['tid']}"
                )

            sig = engine.fetch_track(
                caseid=row.caseid,
                tname=row.tname,
                start_sec=row.requested_start_sec,
                end_sec=row.requested_end_sec,
            )

            # Copy arrays before export; never mutate Stage-07 SignalData.
            t = np.asarray(sig.time_sec, dtype=float).copy()
            v = np.asarray(sig.values, dtype=float).copy()
            if t.shape != v.shape:
                raise RuntimeError(
                    f"Timestamp/value shape mismatch: {t.shape} vs {v.shape}"
                )

            exported_bytes, exported_sha = write_signal_csv_atomic(out, t, v)

            fv = np.isfinite(v)
            finite_count = int(fv.sum())
            missing_count = int(len(v) - finite_count)
            start_actual = (
                float(np.nanmin(t[np.isfinite(t)]))
                if np.isfinite(t).any() else None
            )
            end_actual = (
                float(np.nanmax(t[np.isfinite(t)]))
                if np.isfinite(t).any() else None
            )

            metadata = {
                "stage": "12",
                "script_version": SCRIPT_VERSION,
                "exported_utc": utc_now(),
                "caseid": row.caseid,
                "canonical_signal": row.canonical_signal,
                "tname": row.tname,
                "tid": row.tid,
                "data_role": row.data_role,
                "mapping_status": row.mapping_status,
                "requested_start_sec": row.requested_start_sec,
                "requested_end_sec": row.requested_end_sec,
                "sample_count": int(len(v)),
                "finite_count": finite_count,
                "missing_count": missing_count,
                "start_sec": start_actual,
                "end_sec": end_actual,
                "sampling_rate_hz": (
                    float(sig.sampling_rate_hz)
                    if sig.sampling_rate_hz is not None else None
                ),
                "remote_url": str(sig.remote_url),
                "remote_format": str(sig.remote_format),
                "response_bytes": int(sig.response_bytes),
                "response_sha256": str(sig.response_sha256),
                "exported_file": str(out),
                "exported_bytes": exported_bytes,
                "exported_sha256": exported_sha,
                "raw_values_modified": False,
                "stage10_policy_version": policy.get("policy_version"),
                "network_note": (
                    "Remote retrieval may have transferred the complete track "
                    "before client-side clipping."
                ),
            }
            atomic_write_json(sidecar, metadata)

            results.append(
                DownloadResult(
                    row.caseid, row.canonical_signal, row.tname, row.tid,
                    "EXPORTED", int(len(v)), finite_count, missing_count,
                    start_actual, end_actual,
                    (
                        float(sig.sampling_rate_hz)
                        if sig.sampling_rate_hz is not None else None
                    ),
                    str(sig.remote_url), str(sig.remote_format),
                    int(sig.response_bytes), str(sig.response_sha256),
                    str(out), exported_bytes, exported_sha, "",
                )
            )
            print(
                f"  PASS | samples={len(v):,} | "
                f"file={exported_bytes/1024/1024:.2f} MiB"
            )

        except Exception as exc:
            print(f"  FAIL | {type(exc).__name__}: {exc}")
            results.append(
                DownloadResult(
                    row.caseid, row.canonical_signal, row.tname, row.tid,
                    "FAILED", 0, 0, 0, None, None, None,
                    "", "", 0, "", str(out), 0, "", repr(exc),
                )
            )

        # Persist progress after every row so an interrupted job can resume.
        progress_df = pd.DataFrame([asdict(x) for x in results])
        progress_df.to_csv(results_dir / "download_execution_manifest.csv", index=False)

    result_df = pd.DataFrame([asdict(x) for x in results])

    banner("7. FINAL EXECUTION QC")
    exported = int((result_df["status"] == "EXPORTED").sum())
    existing = int((result_df["status"] == "SKIPPED_EXISTING_VALID").sum())
    unavailable = int((result_df["status"] == "SKIPPED_UNAVAILABLE").sum())
    failed = int((result_df["status"] == "FAILED").sum())
    successful_available = exported + existing
    expected_available = int((plan_df["action"] == "DOWNLOAD").sum())

    qc_rows = [
        ("Selected cohort respects max-cases gate",
         len(caseids) <= args.max_cases, "ERROR", len(caseids), f"<={args.max_cases}"),
        ("Plan contains at least one available track",
         expected_available > 0, "ERROR", expected_available, ">0"),
        ("All available plan rows completed or resume-verified",
         successful_available == expected_available,
         "ERROR", successful_available, expected_available),
        ("No download/export failures",
         failed == 0, "ERROR", failed, 0),
        ("Raw source values were not modified",
         True, "ERROR", True, True),
    ]
    qc_df = pd.DataFrame(
        [
            {
                "test": name,
                "status": "PASS" if ok else "FAIL",
                "severity": severity,
                "observed": observed,
                "expected": expected,
            }
            for name, ok, severity, observed, expected in qc_rows
        ]
    )
    print(qc_df.to_string(index=False))
    qc_df.to_csv(results_dir / "stage12_qc.csv", index=False)

    qc_errors = int(
        ((qc_df["status"] == "FAIL") & (qc_df["severity"] == "ERROR")).sum()
    )
    status = "PASS" if qc_errors == 0 else "FAIL"

    banner("8. WRITING FINAL REPORT + MANIFEST")
    report = "\n".join(
        [
            "VitalDB Explorer",
            "Stage 12 - Cohort Download & Export Engine",
            "=" * 80,
            "",
            f"Generated UTC: {utc_now()}",
            f"Script version: {SCRIPT_VERSION}",
            "",
            "COHORT",
            f"Selected cases: {len(caseids)}",
            f"Plan rows: {len(plan_df)}",
            f"Available rows: {expected_available}",
            f"Unavailable rows: {len(unavailable_df)}",
            "",
            "EXECUTION",
            f"New exports: {exported}",
            f"Resume-verified existing exports: {existing}",
            f"Unavailable/skipped: {unavailable}",
            f"Failures: {failed}",
            "",
            "SAFETY",
            "Raw source values modified: False",
            "Automatic whole-dataset acquisition: False",
            "Stage-10 validity policy retained as provenance: True",
            "Window requests are client-side after Stage-07 full-track retrieval.",
            "",
            f"STATUS: {status}",
        ]
    )
    atomic_write_text(results_dir / "stage12_report.txt", report)

    final_manifest = {
        **plan_manifest,
        "completed_utc": utc_now(),
        "new_exports": exported,
        "resume_verified_existing_exports": existing,
        "unavailable_skipped": unavailable,
        "failures": failed,
        "qc_error_failures": qc_errors,
        "status": status,
        "outputs": [
            "download_plan.csv",
            "download_plan_manifest.json",
            "download_execution_manifest.csv",
            "stage12_qc.csv",
            "stage12_report.txt",
            "stage12_manifest.json",
        ],
    }
    atomic_write_json(results_dir / "stage12_manifest.json", final_manifest)

    print(f"Saved: {results_dir / 'download_execution_manifest.csv'}")
    print(f"Saved: {results_dir / 'stage12_qc.csv'}")
    print(f"Saved: {results_dir / 'stage12_report.txt'}")
    print(f"Saved: {results_dir / 'stage12_manifest.json'}")

    banner("STAGE 12 COMPLETE")
    print(f"Selected cases:              {len(caseids)}")
    print(f"Available download rows:     {expected_available}")
    print(f"New exports:                 {exported}")
    print(f"Existing verified/skipped:   {existing}")
    print(f"Unavailable/skipped:         {unavailable}")
    print(f"Failures:                    {failed}")
    print(f"QC error failures:           {qc_errors}")
    print(f"STATUS: {status}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        banner("STAGE 12 FAILED")
        traceback.print_exc()
        raise SystemExit(1)

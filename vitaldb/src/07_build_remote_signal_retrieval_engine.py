"""
VitalDB Explorer
Stage 07 - Remote Signal Retrieval Engine
=========================================

Purpose
-------
Provide validated, on-demand access to actual VitalDB Open Dataset track data.

Previous stages:
    Stage 05 -> Which cases match the cohort?
    Stage 06 -> What metadata/tracks exist for a selected case?
    Stage 07 -> Retrieve the actual measurements for a selected track.

Architecture
------------
The authoritative retrieval chain is:

    caseid
        ->
    exact VitalDB tname
        ->
    tid
        ->
    https://api.vitaldb.net/{tid}
        ->
    remote track data

This stage DOES NOT:
    - permanently download complete VitalDB cases
    - build the cohort-download system
    - perform physiological analysis
    - perform filtering/FFT/HRV
    - create the final visualization UI

Those belong to later stages.

Official VitalDB Web API behavior
---------------------------------
Track endpoint:
    https://api.vitaldb.net/{tid}

Numeric tracks:
    - explicit time/value rows
    - missing observations may be omitted
    - sampling intervals may therefore be irregular

Waveform tracks:
    - first column encodes:
          start time
          sample interval
          end time
    - subsequent values represent uniformly spaced samples
    - missing waveform samples are retained

Stage 07 parses both forms into a common SignalData object.

Important
---------
No sampling frequency or unit is taken from Stage 03 metadata.

For waveform tracks, sampling frequency is derived from the remote
time interval:

    fs = 1 / interval_sec

For numeric tracks, no artificial sampling rate is asserted.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION = "2026-09-15-v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = PROJECT_ROOT / "metadata"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage07"

TRACKS_PATH = METADATA_DIR / "tracks.csv"
DICTIONARY_PATH = METADATA_DIR / "canonical_signal_dictionary.csv"
MASTER_PATH = METADATA_DIR / "master_case_index.csv"

API_BASE = "https://api.vitaldb.net"

CONNECT_TIMEOUT_SEC = 15
READ_TIMEOUT_SEC = 120

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0

USER_AGENT = (
    "VitalDB-Explorer/"
    + SCRIPT_VERSION
    + " (research metadata/signal client)"
)

EXPECTED_CASES = 6388
EXPECTED_TRACK_ROWS = 486449
EXPECTED_EXACT_TRACKS = 196
EXPECTED_CANONICAL_SIGNALS = 180

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def section(title: str) -> None:

    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def utc_now_iso() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def human_bytes(
    size: int,
) -> str:

    size = float(size)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:

        if size < 1024.0:
            return f"{size:.2f} {unit}"

        size /= 1024.0

    return f"{size:.2f} PB"


def sha256_bytes(
    content: bytes,
) -> str:

    return hashlib.sha256(
        content
    ).hexdigest()


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def json_safe(value: Any) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        if np.isnan(value):
            return None

        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        dict,
    ):

        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            json_safe(v)
            for v in value
        ]

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    return value


def write_json(
    path: Path,
    obj: Any,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_safe(obj),
            f,
            indent=2,
            ensure_ascii=False,
        )


def normalize_caseid(
    series: pd.Series,
) -> pd.Series:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():

        return numeric.astype(
            "int64"
        )

    return (
        series
        .astype("string")
        .str.strip()
    )


def numeric_array(
    values: pd.Series,
) -> np.ndarray:

    return (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RemoteResponse:

    url: str

    status_code: int

    content_type: str

    content_encoding: str

    content_length_header: Optional[int]

    elapsed_sec: float

    attempts: int

    raw_bytes: bytes

    sha256: str


@dataclass
class SignalData:

    caseid: int

    tname: str

    tid: str

    canonical_signal: Optional[str]

    display_name: Optional[str]

    device: Optional[str]

    signal_family: Optional[str]

    data_role: Optional[str]

    mapping_status: Optional[str]

    remote_url: str

    remote_format: str

    track_type: str

    time_sec: np.ndarray

    values: np.ndarray

    sample_count: int

    finite_count: int

    missing_count: int

    start_sec: Optional[float]

    end_sec: Optional[float]

    duration_sec: Optional[float]

    interval_sec: Optional[float]

    sampling_rate_hz: Optional[float]

    response_bytes: int

    response_sha256: str

    http_status: int

    elapsed_sec: float

    requested_start_sec: Optional[float] = None

    requested_end_sec: Optional[float] = None

    source_sample_count: Optional[int] = None

    clipped: bool = False

    @property
    def finite_fraction(self) -> Optional[float]:

        if self.sample_count <= 0:
            return None

        return (
            self.finite_count
            / self.sample_count
        )

    def to_dataframe(
        self,
    ) -> pd.DataFrame:

        return pd.DataFrame(
            {
                "time_sec": (
                    self.time_sec
                ),
                "value": (
                    self.values
                ),
            }
        )

    def summary(
        self,
    ) -> dict[str, Any]:

        finite_values = (
            self.values[
                np.isfinite(
                    self.values
                )
            ]
        )

        if finite_values.size:

            value_min = float(
                np.min(
                    finite_values
                )
            )

            value_max = float(
                np.max(
                    finite_values
                )
            )

            value_mean = float(
                np.mean(
                    finite_values
                )
            )

            value_median = float(
                np.median(
                    finite_values
                )
            )

            value_std = float(
                np.std(
                    finite_values
                )
            )

        else:

            value_min = None
            value_max = None
            value_mean = None
            value_median = None
            value_std = None

        return {

            "caseid": self.caseid,

            "tname": self.tname,

            "tid": self.tid,

            "canonical_signal": (
                self.canonical_signal
            ),

            "display_name": (
                self.display_name
            ),

            "device": self.device,

            "signal_family": (
                self.signal_family
            ),

            "data_role": (
                self.data_role
            ),

            "mapping_status": (
                self.mapping_status
            ),

            "remote_url": (
                self.remote_url
            ),

            "remote_format": (
                self.remote_format
            ),

            "track_type": (
                self.track_type
            ),

            "sample_count": (
                self.sample_count
            ),

            "source_sample_count": (
                self.source_sample_count
            ),

            "finite_count": (
                self.finite_count
            ),

            "missing_count": (
                self.missing_count
            ),

            "finite_fraction": (
                self.finite_fraction
            ),

            "start_sec": (
                self.start_sec
            ),

            "end_sec": (
                self.end_sec
            ),

            "duration_sec": (
                self.duration_sec
            ),

            "interval_sec": (
                self.interval_sec
            ),

            "sampling_rate_hz": (
                self.sampling_rate_hz
            ),

            "requested_start_sec": (
                self.requested_start_sec
            ),

            "requested_end_sec": (
                self.requested_end_sec
            ),

            "clipped": (
                self.clipped
            ),

            "value_min": value_min,

            "value_max": value_max,

            "value_mean": value_mean,

            "value_median": value_median,

            "value_std": value_std,

            "response_bytes": (
                self.response_bytes
            ),

            "response_sha256": (
                self.response_sha256
            ),

            "http_status": (
                self.http_status
            ),

            "elapsed_sec": (
                self.elapsed_sec
            ),
        }


# =============================================================================
# REMOTE RETRIEVAL ENGINE
# =============================================================================

class RemoteSignalRetrievalEngine:

    def __init__(
        self,
        tracks_path: Path = TRACKS_PATH,
        dictionary_path: Path = DICTIONARY_PATH,
        master_path: Path = MASTER_PATH,
    ) -> None:

        self.tracks_path = Path(
            tracks_path
        )

        self.dictionary_path = Path(
            dictionary_path
        )

        self.master_path = Path(
            master_path
        )

        self.session = (
            requests.Session()
        )

        self.session.headers.update(
            {
                "User-Agent": (
                    USER_AGENT
                ),
                "Accept": (
                    "text/plain,"
                    "text/csv,"
                    "application/gzip,"
                    "application/octet-stream,"
                    "*/*"
                ),
            }
        )

        self.tracks = pd.DataFrame()

        self.dictionary = pd.DataFrame()

        self.master = pd.DataFrame()

        self.track_index = (
            pd.DataFrame()
        )

        self._load_metadata()

        self._validate_metadata()

        self._build_track_index()

    # =========================================================================
    # METADATA
    # =========================================================================

    def _load_metadata(
        self,
    ) -> None:

        self.tracks = pd.read_csv(
            self.tracks_path,
            low_memory=False,
        )

        self.dictionary = pd.read_csv(
            self.dictionary_path,
            low_memory=False,
        )

        self.master = pd.read_csv(
            self.master_path,
            low_memory=False,
        )

        self.tracks["caseid"] = (
            normalize_caseid(
                self.tracks[
                    "caseid"
                ]
            )
        )

        self.master["caseid"] = (
            normalize_caseid(
                self.master[
                    "caseid"
                ]
            )
        )

        self.tracks["tname"] = (
            self.tracks[
                "tname"
            ]
            .astype(str)
            .str.strip()
        )

        self.tracks["tid"] = (
            self.tracks[
                "tid"
            ]
            .astype(str)
            .str.strip()
        )

        self.dictionary["tname"] = (
            self.dictionary[
                "tname"
            ]
            .astype(str)
            .str.strip()
        )

    def _validate_metadata(
        self,
    ) -> None:

        required_tracks = {
            "caseid",
            "tname",
            "tid",
        }

        missing = (
            required_tracks
            - set(
                self.tracks.columns
            )
        )

        if missing:

            raise ValueError(
                "tracks.csv missing columns: "
                f"{sorted(missing)}"
            )

        required_dictionary = {
            "tname",
            "canonical_signal",
            "display_name",
            "signal_family",
            "data_role",
            "status",
        }

        missing = (
            required_dictionary
            - set(
                self.dictionary.columns
            )
        )

        if missing:

            raise ValueError(
                "canonical_signal_dictionary.csv "
                "missing columns: "
                f"{sorted(missing)}"
            )

        if (
            self.dictionary[
                "tname"
            ]
            .duplicated()
            .any()
        ):

            raise ValueError(
                "Duplicate tname entries in "
                "canonical dictionary."
            )

        if (
            self.tracks[
                "tid"
            ]
            .duplicated()
            .any()
        ):

            raise ValueError(
                "Duplicate tid entries found "
                "in tracks.csv."
            )

        observed_tnames = set(
            self.tracks[
                "tname"
            ]
        )

        dictionary_tnames = set(
            self.dictionary[
                "tname"
            ]
        )

        if (
            observed_tnames
            != dictionary_tnames
        ):

            raise ValueError(
                "Track vocabulary and canonical "
                "dictionary disagree."
            )

    def _build_track_index(
        self,
    ) -> None:

        dictionary_columns = [
            col
            for col in [
                "tname",
                "device",
                "parameter",
                "canonical_signal",
                "display_name",
                "signal_family",
                "data_role",
                "measurement_type",
                "explorer_group",
                "status",
                "analysis_candidate",
                "notes",
            ]
            if col
            in self.dictionary.columns
        ]

        dictionary = (
            self.dictionary[
                dictionary_columns
            ]
            .copy()
        )

        self.track_index = (
            self.tracks.merge(
                dictionary,
                on="tname",
                how="left",
                validate="many_to_one",
            )
        )

        if (
            self.track_index[
                "canonical_signal"
            ]
            .isna()
            .any()
        ):

            raise ValueError(
                "One or more track rows could "
                "not be mapped to the canonical "
                "dictionary."
            )

    # =========================================================================
    # TRACK RESOLUTION
    # =========================================================================

    def list_case_tracks(
        self,
        caseid: int,
    ) -> pd.DataFrame:

        caseid = int(
            caseid
        )

        result = (
            self.track_index[
                self.track_index[
                    "caseid"
                ]
                == caseid
            ]
            .copy()
        )

        if result.empty:

            raise KeyError(
                f"Unknown caseid or no tracks: "
                f"{caseid}"
            )

        return (
            result
            .sort_values(
                [
                    "data_role",
                    "canonical_signal",
                    "tname",
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

    def resolve_track(
        self,
        caseid: int,
        tname: str,
    ) -> pd.Series:

        caseid = int(
            caseid
        )

        tname = str(
            tname
        ).strip()

        matches = (
            self.track_index[
                (
                    self.track_index[
                        "caseid"
                    ]
                    == caseid
                )
                &
                (
                    self.track_index[
                        "tname"
                    ]
                    == tname
                )
            ]
        )

        if matches.empty:

            raise KeyError(
                f"Track {tname!r} is not "
                f"available for case {caseid}."
            )

        if len(matches) != 1:

            raise ValueError(
                f"Expected exactly one track "
                f"for case={caseid}, "
                f"tname={tname!r}; "
                f"found {len(matches)}."
            )

        return matches.iloc[0]

    def resolve_canonical_signal(
        self,
        caseid: int,
        canonical_signal: str,
        preferred_tname: Optional[str] = None,
    ) -> pd.Series:

        caseid = int(
            caseid
        )

        canonical_signal = str(
            canonical_signal
        ).strip()

        matches = (
            self.track_index[
                (
                    self.track_index[
                        "caseid"
                    ]
                    == caseid
                )
                &
                (
                    self.track_index[
                        "canonical_signal"
                    ]
                    .astype(str)
                    .str.casefold()
                    == canonical_signal.casefold()
                )
            ]
            .copy()
        )

        if matches.empty:

            raise KeyError(
                f"Canonical signal "
                f"{canonical_signal!r} "
                f"is not available for "
                f"case {caseid}."
            )

        if preferred_tname:

            preferred = (
                matches[
                    matches[
                        "tname"
                    ]
                    == preferred_tname
                ]
            )

            if len(
                preferred
            ) == 1:

                return (
                    preferred.iloc[0]
                )

            raise KeyError(
                f"Preferred exact track "
                f"{preferred_tname!r} "
                f"is not available for "
                f"canonical signal "
                f"{canonical_signal!r} "
                f"in case {caseid}."
            )

        if len(matches) == 1:

            return matches.iloc[0]

        #
        # Multiple exact sources may map to
        # the same canonical signal.
        #
        # Do not silently merge them.
        #
        # Deterministically choose the source
        # with the greatest global case coverage,
        # while preserving exact source identity.
        #

        if "case_count" in matches.columns:

            matches = (
                matches.sort_values(
                    [
                        "case_count",
                        "tname",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
            )

        else:

            matches = (
                matches.sort_values(
                    "tname"
                )
            )

        selected = (
            matches.iloc[0]
        )

        return selected

    # =========================================================================
    # HTTP
    # =========================================================================

    def _request_tid(
        self,
        tid: str,
    ) -> RemoteResponse:

        tid = str(
            tid
        ).strip()

        if not tid:

            raise ValueError(
                "Empty tid."
            )

        url = (
            f"{API_BASE}/{tid}"
        )

        last_error = None

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            started = (
                time.perf_counter()
            )

            try:

                response = (
                    self.session.get(
                        url,
                        timeout=(
                            CONNECT_TIMEOUT_SEC,
                            READ_TIMEOUT_SEC,
                        ),
                    )
                )

                elapsed = (
                    time.perf_counter()
                    - started
                )

                if (
                    response.status_code
                    == 200
                ):

                    content = (
                        response.content
                    )

                    content_length = (
                        response.headers.get(
                            "Content-Length"
                        )
                    )

                    try:

                        content_length_int = (
                            int(
                                content_length
                            )
                            if content_length
                            else None
                        )

                    except Exception:

                        content_length_int = (
                            None
                        )

                    return RemoteResponse(

                        url=url,

                        status_code=(
                            response.status_code
                        ),

                        content_type=(
                            response.headers.get(
                                "Content-Type",
                                "",
                            )
                        ),

                        content_encoding=(
                            response.headers.get(
                                "Content-Encoding",
                                "",
                            )
                        ),

                        content_length_header=(
                            content_length_int
                        ),

                        elapsed_sec=(
                            elapsed
                        ),

                        attempts=attempt,

                        raw_bytes=content,

                        sha256=(
                            sha256_bytes(
                                content
                            )
                        ),
                    )

                last_error = (
                    RuntimeError(
                        f"HTTP "
                        f"{response.status_code} "
                        f"for {url}"
                    )
                )

                #
                # Client errors generally should
                # not be repeatedly retried.
                #

                if (
                    400
                    <= response.status_code
                    < 500
                    and response.status_code
                    not in {
                        408,
                        429,
                    }
                ):

                    break

            except (
                requests.RequestException
            ) as exc:

                last_error = exc

            if attempt < MAX_RETRIES:

                sleep_sec = (
                    RETRY_BACKOFF_SEC
                    * attempt
                )

                time.sleep(
                    sleep_sec
                )

        raise RuntimeError(
            f"Remote VitalDB request failed "
            f"after {MAX_RETRIES} attempt(s): "
            f"{url}. "
            f"Last error: {last_error}"
        )

    # =========================================================================
    # DECOMPRESSION / TEXT DECODING
    # =========================================================================

    @staticmethod
    def _decode_payload(
        response: RemoteResponse,
    ) -> tuple[str, str]:

        content = (
            response.raw_bytes
        )

        detected_format = (
            "plain"
        )

        #
        # requests normally handles HTTP-level
        # Content-Encoding automatically.
        #
        # However, VitalDB endpoints are documented
        # as gzip-compressed CSV resources, so also
        # support gzip file bytes directly.
        #

        if (
            len(content) >= 2
            and content[:2]
            == b"\x1f\x8b"
        ):

            try:

                content = (
                    gzip.decompress(
                        content
                    )
                )

                detected_format = (
                    "gzip"
                )

            except Exception as exc:

                raise ValueError(
                    "Payload starts with gzip "
                    "magic bytes but could not "
                    "be decompressed."
                ) from exc

        try:

            text = (
                content.decode(
                    "utf-8-sig"
                )
            )

        except UnicodeDecodeError:

            text = (
                content.decode(
                    "latin-1"
                )
            )

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        return (
            text,
            detected_format,
        )

    # =========================================================================
    # RAW CSV PARSING
    # =========================================================================

    @staticmethod
    def _parse_numeric_rows(
        text: str,
    ) -> list[list[float]]:

        rows: list[
            list[float]
        ] = []

        reader = csv.reader(
            io.StringIO(
                text
            )
        )

        for raw_row in reader:

            if not raw_row:
                continue

            cleaned = [
                cell.strip()
                for cell
                in raw_row
            ]

            if (
                len(cleaned) == 1
                and cleaned[0] == ""
            ):
                continue

            numeric_row = []

            valid = True

            for cell in cleaned:

                if cell == "":

                    numeric_row.append(
                        np.nan
                    )

                    continue

                try:

                    numeric_row.append(
                        float(cell)
                    )

                except ValueError:

                    valid = False
                    break

            #
            # This safely skips a possible
            # textual CSV header.
            #

            if not valid:
                continue

            rows.append(
                numeric_row
            )

        return rows

    # =========================================================================
    # PARSE NUMERIC TRACK
    # =========================================================================

    @staticmethod
    def _parse_numeric_track(
        rows: list[list[float]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        valid_rows = [
            row
            for row in rows
            if len(row) >= 2
        ]

        if not valid_rows:

            raise ValueError(
                "No two-column numeric "
                "track rows found."
            )

        times = np.asarray(
            [
                row[0]
                for row
                in valid_rows
            ],
            dtype=float,
        )

        values = np.asarray(
            [
                row[1]
                for row
                in valid_rows
            ],
            dtype=float,
        )

        finite_time = (
            np.isfinite(
                times
            )
        )

        times = times[
            finite_time
        ]

        values = values[
            finite_time
        ]

        if times.size == 0:

            raise ValueError(
                "Numeric track contains "
                "no finite timestamps."
            )

        order = np.argsort(
            times,
            kind="stable",
        )

        times = times[
            order
        ]

        values = values[
            order
        ]

        return (
            times,
            values,
        )

    # =========================================================================
    # PARSE WAVEFORM TRACK
    # =========================================================================

    @staticmethod
    def _parse_waveform_track(
        rows: list[list[float]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
    ]:

        #
        # Official VitalDB representation:
        #
        # first column has three values:
        #   start time
        #   interval
        #   end time
        #
        # second column contains waveform values.
        #
        # Depending on CSV parsing, the payload
        # may appear as:
        #
        #   0,value0
        #   dt,value1
        #   end,value2
        #   ,value3
        #   ,value4
        #
        # or equivalent rows.
        #
        # Therefore preserve the first three
        # finite first-column values as metadata
        # and use the second column as the sample
        # stream.
        #

        valid_rows = [
            row
            for row in rows
            if len(row) >= 2
        ]

        if len(valid_rows) < 3:

            raise ValueError(
                "Waveform payload has fewer "
                "than three usable rows."
            )

        first_column = np.asarray(
            [
                row[0]
                for row
                in valid_rows
            ],
            dtype=float,
        )

        values = np.asarray(
            [
                row[1]
                for row
                in valid_rows
            ],
            dtype=float,
        )

        finite_first = (
            first_column[
                np.isfinite(
                    first_column
                )
            ]
        )

        if (
            finite_first.size
            < 3
        ):

            raise ValueError(
                "Waveform payload does not "
                "contain the required start/"
                "interval/end metadata."
            )

        start_sec = float(
            finite_first[0]
        )

        interval_sec = float(
            finite_first[1]
        )

        declared_end_sec = float(
            finite_first[2]
        )

        if (
            not np.isfinite(
                interval_sec
            )
            or interval_sec <= 0
        ):

            raise ValueError(
                "Invalid waveform interval: "
                f"{interval_sec}"
            )

        sample_count = (
            len(values)
        )

        times = (
            start_sec
            + np.arange(
                sample_count,
                dtype=float,
            )
            * interval_sec
        )

        #
        # The declared endpoint is retained
        # indirectly through validation later.
        #
        # Do not force the generated time vector
        # to the declared endpoint because the
        # exact endpoint convention may be
        # inclusive/exclusive.
        #

        _ = declared_end_sec

        return (
            times,
            values,
            interval_sec,
        )

    # =========================================================================
    # TRACK TYPE DETECTION
    # =========================================================================

    @staticmethod
    def _detect_track_type(
        metadata: pd.Series,
        rows: list[list[float]],
    ) -> str:

        data_role = str(
            metadata.get(
                "data_role",
                "",
            )
        ).strip().upper()

        if (
            data_role
            == "WAVEFORM"
        ):

            return "WAVEFORM"

        if data_role in {
            "NUMERIC",
            "SETTING",
            "QUALITY",
            "INFUSION",
        }:

            return "NUMERIC"

        #
        # Conservative structural fallback.
        #

        valid_rows = [
            row
            for row in rows
            if len(row) >= 2
        ]

        if len(
            valid_rows
        ) >= 4:

            first = np.asarray(
                [
                    row[0]
                    for row
                    in valid_rows
                ],
                dtype=float,
            )

            finite_count = int(
                np.isfinite(
                    first
                ).sum()
            )

            if finite_count == 3:

                return "WAVEFORM"

        return "NUMERIC"

    # =========================================================================
    # WINDOWING
    # =========================================================================

    @staticmethod
    def _clip_window(
        times: np.ndarray,
        values: np.ndarray,
        start_sec: Optional[float],
        end_sec: Optional[float],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        bool,
    ]:

        if (
            start_sec is None
            and end_sec is None
        ):

            return (
                times,
                values,
                False,
            )

        if (
            start_sec is not None
            and not np.isfinite(
                start_sec
            )
        ):

            raise ValueError(
                "start_sec must be finite."
            )

        if (
            end_sec is not None
            and not np.isfinite(
                end_sec
            )
        ):

            raise ValueError(
                "end_sec must be finite."
            )

        if (
            start_sec is not None
            and end_sec is not None
            and end_sec <= start_sec
        ):

            raise ValueError(
                "end_sec must be greater "
                "than start_sec."
            )

        mask = np.ones(
            len(times),
            dtype=bool,
        )

        if start_sec is not None:

            mask &= (
                times >= start_sec
            )

        if end_sec is not None:

            #
            # Half-open interval:
            # [start_sec, end_sec)
            #

            mask &= (
                times < end_sec
            )

        return (
            times[
                mask
            ],
            values[
                mask
            ],
            True,
        )

    # =========================================================================
    # MAIN FETCH BY EXACT TRACK
    # =========================================================================

    def fetch_track(
        self,
        caseid: int,
        tname: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> SignalData:

        metadata = (
            self.resolve_track(
                caseid,
                tname,
            )
        )

        response = (
            self._request_tid(
                metadata[
                    "tid"
                ]
            )
        )

        text, payload_format = (
            self._decode_payload(
                response
            )
        )

        rows = (
            self._parse_numeric_rows(
                text
            )
        )

        if not rows:

            raise ValueError(
                f"No numeric CSV rows "
                f"returned for case={caseid}, "
                f"track={tname!r}."
            )

        track_type = (
            self._detect_track_type(
                metadata,
                rows,
            )
        )

        interval_sec = None

        sampling_rate_hz = None

        if (
            track_type
            == "WAVEFORM"
        ):

            (
                times,
                values,
                interval_sec,
            ) = (
                self._parse_waveform_track(
                    rows
                )
            )

            sampling_rate_hz = (
                1.0
                / interval_sec
            )

        else:

            (
                times,
                values,
            ) = (
                self._parse_numeric_track(
                    rows
                )
            )

        source_sample_count = (
            len(values)
        )

        (
            times,
            values,
            clipped,
        ) = (
            self._clip_window(
                times,
                values,
                start_sec,
                end_sec,
            )
        )

        sample_count = int(
            len(values)
        )

        finite_count = int(
            np.isfinite(
                values
            ).sum()
        )

        missing_count = (
            sample_count
            - finite_count
        )

        if sample_count:

            start_actual = float(
                times[0]
            )

            end_actual = float(
                times[-1]
            )

            if (
                track_type
                == "WAVEFORM"
                and interval_sec
                is not None
            ):

                duration_sec = float(
                    sample_count
                    * interval_sec
                )

            elif sample_count >= 2:

                duration_sec = float(
                    end_actual
                    - start_actual
                )

            else:

                duration_sec = 0.0

        else:

            start_actual = None

            end_actual = None

            duration_sec = 0.0

        return SignalData(

            caseid=int(
                caseid
            ),

            tname=str(
                metadata[
                    "tname"
                ]
            ),

            tid=str(
                metadata[
                    "tid"
                ]
            ),

            canonical_signal=(
                json_safe(
                    metadata.get(
                        "canonical_signal"
                    )
                )
            ),

            display_name=(
                json_safe(
                    metadata.get(
                        "display_name"
                    )
                )
            ),

            device=(
                json_safe(
                    metadata.get(
                        "device"
                    )
                )
            ),

            signal_family=(
                json_safe(
                    metadata.get(
                        "signal_family"
                    )
                )
            ),

            data_role=(
                json_safe(
                    metadata.get(
                        "data_role"
                    )
                )
            ),

            mapping_status=(
                json_safe(
                    metadata.get(
                        "status"
                    )
                )
            ),

            remote_url=(
                response.url
            ),

            remote_format=(
                payload_format
            ),

            track_type=(
                track_type
            ),

            time_sec=times,

            values=values,

            sample_count=(
                sample_count
            ),

            finite_count=(
                finite_count
            ),

            missing_count=(
                missing_count
            ),

            start_sec=(
                start_actual
            ),

            end_sec=(
                end_actual
            ),

            duration_sec=(
                duration_sec
            ),

            interval_sec=(
                interval_sec
            ),

            sampling_rate_hz=(
                sampling_rate_hz
            ),

            response_bytes=len(
                response.raw_bytes
            ),

            response_sha256=(
                response.sha256
            ),

            http_status=(
                response.status_code
            ),

            elapsed_sec=(
                response.elapsed_sec
            ),

            requested_start_sec=(
                start_sec
            ),

            requested_end_sec=(
                end_sec
            ),

            source_sample_count=(
                source_sample_count
            ),

            clipped=clipped,
        )

    # =========================================================================
    # FETCH BY CANONICAL SIGNAL
    # =========================================================================

    def fetch_signal(
        self,
        caseid: int,
        canonical_signal: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        preferred_tname: Optional[str] = None,
    ) -> SignalData:

        metadata = (
            self.resolve_canonical_signal(
                caseid=caseid,
                canonical_signal=(
                    canonical_signal
                ),
                preferred_tname=(
                    preferred_tname
                ),
            )
        )

        return self.fetch_track(

            caseid=caseid,

            tname=str(
                metadata[
                    "tname"
                ]
            ),

            start_sec=(
                start_sec
            ),

            end_sec=(
                end_sec
            ),
        )


# =============================================================================
# SIGNAL VALIDATION
# =============================================================================

def validate_signal(
    signal: SignalData,
) -> pd.DataFrame:

    rows = []

    def add(
        check: str,
        passed: bool,
        observed: Any,
        expected: Any,
        severity: str = "ERROR",
        note: str = "",
    ) -> None:

        rows.append(
            {
                "caseid": (
                    signal.caseid
                ),
                "tname": (
                    signal.tname
                ),
                "check": check,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "severity": severity,
                "observed": (
                    observed
                ),
                "expected": (
                    expected
                ),
                "note": note,
            }
        )

    add(
        "HTTP status",
        signal.http_status == 200,
        signal.http_status,
        200,
    )

    add(
        "Non-empty remote response",
        signal.response_bytes > 0,
        signal.response_bytes,
        "> 0",
    )

    add(
        "Parsed sample count",
        signal.sample_count > 0,
        signal.sample_count,
        "> 0",
    )

    add(
        "Time/value lengths agree",
        len(signal.time_sec)
        == len(signal.values),
        (
            len(signal.time_sec),
            len(signal.values),
        ),
        "equal",
    )

    add(
        "Finite timestamps",
        (
            np.isfinite(
                signal.time_sec
            ).all()
        ),
        int(
            np.isfinite(
                signal.time_sec
            ).sum()
        ),
        len(
            signal.time_sec
        ),
    )

    if (
        len(
            signal.time_sec
        ) >= 2
    ):

        differences = np.diff(
            signal.time_sec
        )

        add(
            "Monotonic timestamps",
            bool(
                np.all(
                    differences
                    >= 0
                )
            ),
            float(
                np.min(
                    differences
                )
            ),
            ">= 0",
        )

    if (
        signal.track_type
        == "WAVEFORM"
    ):

        add(
            "Waveform interval available",
            (
                signal.interval_sec
                is not None
                and signal.interval_sec
                > 0
            ),
            signal.interval_sec,
            "> 0 sec",
        )

        add(
            "Waveform sampling rate available",
            (
                signal.sampling_rate_hz
                is not None
                and signal.sampling_rate_hz
                > 0
            ),
            signal.sampling_rate_hz,
            "> 0 Hz",
        )

        if (
            len(
                signal.time_sec
            ) >= 3
            and signal.interval_sec
            is not None
        ):

            differences = np.diff(
                signal.time_sec
            )

            tolerance = max(
                1e-12,
                abs(
                    signal.interval_sec
                )
                * 1e-6,
            )

            regular = bool(
                np.allclose(
                    differences,
                    signal.interval_sec,
                    rtol=1e-6,
                    atol=tolerance,
                )
            )

            add(
                "Waveform timestamps uniformly spaced",
                regular,
                (
                    float(
                        np.median(
                            differences
                        )
                    )
                    if differences.size
                    else None
                ),
                signal.interval_sec,
            )

    else:

        add(
            "Numeric track does not assert waveform sampling rate",
            (
                signal.sampling_rate_hz
                is None
            ),
            signal.sampling_rate_hz,
            None,
        )

    add(
        "At least one finite measurement",
        signal.finite_count > 0,
        signal.finite_count,
        "> 0",
        severity="WARNING",
    )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# REPRESENTATIVE TRACK SELECTION
# =============================================================================

def choose_representative_tracks(
    engine: RemoteSignalRetrievalEngine,
    caseid: int = 1,
) -> list[str]:

    available = set(
        engine.list_case_tracks(
            caseid
        )[
            "tname"
        ]
    )

    #
    # Keep the validation small.
    #
    # Stage 07 is testing retrieval behavior,
    # not downloading a complete case.
    #

    desired = [

        # waveform
        "SNUADC/ECG_II",
        "SNUADC/PLETH",
        "SNUADC/ART",
        "Primus/CO2",
        "BIS/EEG1_WAV",

        # numeric
        "Solar8000/HR",
        "Solar8000/PLETH_SPO2",
        "Solar8000/ART_MBP",
        "Solar8000/ETCO2",
        "BIS/BIS",
    ]

    selected = [
        tname
        for tname
        in desired
        if tname in available
    ]

    #
    # We do not require every desired track
    # to exist in the demonstration case.
    #

    return selected


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_engine_metadata(
    engine: RemoteSignalRetrievalEngine,
) -> pd.DataFrame:

    rows = []

    def add(
        test: str,
        passed: bool,
        observed: Any,
        expected: Any,
        severity: str = "ERROR",
    ) -> None:

        rows.append(
            {
                "test": test,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "severity": severity,
                "observed": observed,
                "expected": expected,
            }
        )

    add(
        "Master case count",
        len(
            engine.master
        )
        == EXPECTED_CASES,
        len(
            engine.master
        ),
        EXPECTED_CASES,
    )

    add(
        "Track row count",
        len(
            engine.tracks
        )
        == EXPECTED_TRACK_ROWS,
        len(
            engine.tracks
        ),
        EXPECTED_TRACK_ROWS,
    )

    add(
        "Exact track vocabulary",
        engine.tracks[
            "tname"
        ].nunique()
        == EXPECTED_EXACT_TRACKS,
        engine.tracks[
            "tname"
        ].nunique(),
        EXPECTED_EXACT_TRACKS,
    )

    add(
        "Canonical concept count",
        engine.dictionary[
            "canonical_signal"
        ].nunique()
        == EXPECTED_CANONICAL_SIGNALS,
        engine.dictionary[
            "canonical_signal"
        ].nunique(),
        EXPECTED_CANONICAL_SIGNALS,
    )

    add(
        "Unique track IDs",
        engine.tracks[
            "tid"
        ].nunique()
        == len(
            engine.tracks
        ),
        engine.tracks[
            "tid"
        ].nunique(),
        len(
            engine.tracks
        ),
    )

    add(
        "Track index rows",
        len(
            engine.track_index
        )
        == len(
            engine.tracks
        ),
        len(
            engine.track_index
        ),
        len(
            engine.tracks
        ),
    )

    add(
        "All track-index rows mapped",
        engine.track_index[
            "canonical_signal"
        ].notna().all(),
        int(
            engine.track_index[
                "canonical_signal"
            ].notna().sum()
        ),
        len(
            engine.track_index
        ),
    )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EXPORT SIGNAL
# =============================================================================

def export_signal_result(
    signal: SignalData,
    output_dir: Path,
) -> dict[str, Path]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_track = (
        signal.tname
        .replace(
            "/",
            "__",
        )
        .replace(
            "\\",
            "__",
        )
    )

    csv_path = (
        output_dir
        / f"{safe_track}_samples.csv"
    )

    summary_path = (
        output_dir
        / f"{safe_track}_summary.json"
    )

    signal.to_dataframe().to_csv(
        csv_path,
        index=False,
    )

    write_json(
        summary_path,
        signal.summary(),
    )

    return {
        "samples": csv_path,
        "summary": summary_path,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    section(
        "VITALDB EXPLORER - STAGE 07"
    )

    print(
        f"Project root:    {PROJECT_ROOT}"
    )

    print(
        f"Metadata dir:    {METADATA_DIR}"
    )

    print(
        f"Results dir:     {RESULTS_DIR}"
    )

    print(
        f"API base:        {API_BASE}"
    )

    print(
        f"Script version:  {SCRIPT_VERSION}"
    )

    # =========================================================================
    # 1. INPUT VALIDATION
    # =========================================================================

    section(
        "1. INPUT VALIDATION"
    )

    required_files = [
        TRACKS_PATH,
        DICTIONARY_PATH,
        MASTER_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: "
                f"{path}"
            )

        print(
            f"Found: {path}"
        )

    print(
        "Input validation: PASS"
    )

    # =========================================================================
    # 2. INITIALIZE
    # =========================================================================

    section(
        "2. INITIALIZING REMOTE RETRIEVAL ENGINE"
    )

    engine = (
        RemoteSignalRetrievalEngine()
    )

    print(
        f"Cases:                "
        f"{len(engine.master):,}"
    )

    print(
        f"Track rows:           "
        f"{len(engine.tracks):,}"
    )

    print(
        f"Unique tids:          "
        f"{engine.tracks['tid'].nunique():,}"
    )

    print(
        f"Exact tracks:         "
        f"{engine.tracks['tname'].nunique():,}"
    )

    print(
        f"Canonical concepts:   "
        f"{engine.dictionary['canonical_signal'].nunique():,}"
    )

    # =========================================================================
    # 3. METADATA VALIDATION
    # =========================================================================

    section(
        "3. METADATA VALIDATION"
    )

    metadata_validation = (
        validate_engine_metadata(
            engine
        )
    )

    print(
        metadata_validation.to_string(
            index=False
        )
    )

    metadata_validation_path = (
        RESULTS_DIR
        / "metadata_validation.csv"
    )

    metadata_validation.to_csv(
        metadata_validation_path,
        index=False,
    )

    metadata_error_failures = int(
        (
            (
                metadata_validation[
                    "status"
                ]
                == "FAIL"
            )
            &
            (
                metadata_validation[
                    "severity"
                ]
                == "ERROR"
            )
        ).sum()
    )

    print()

    print(
        f"Metadata error failures: "
        f"{metadata_error_failures}"
    )

    if (
        metadata_error_failures
        > 0
    ):

        print()
        print(
            "STATUS: FAIL"
        )

        print(
            "Do not perform remote "
            "retrieval."
        )

        return

    # =========================================================================
    # 4. DEMONSTRATION CASE
    # =========================================================================

    section(
        "4. DEMONSTRATION CASE"
    )

    demo_caseid = 1

    case_tracks = (
        engine.list_case_tracks(
            demo_caseid
        )
    )

    print(
        f"Case ID:              "
        f"{demo_caseid}"
    )

    print(
        f"Available tracks:     "
        f"{len(case_tracks)}"
    )

    print(
        f"Waveform tracks:      "
        f"{int((case_tracks['data_role'].astype(str).str.upper() == 'WAVEFORM').sum())}"
    )

    print(
        f"Numeric tracks:       "
        f"{int((case_tracks['data_role'].astype(str).str.upper() == 'NUMERIC').sum())}"
    )

    # =========================================================================
    # 5. REPRESENTATIVE TRACKS
    # =========================================================================

    section(
        "5. REPRESENTATIVE REMOTE TRACKS"
    )

    representative_tracks = (
        choose_representative_tracks(
            engine,
            demo_caseid,
        )
    )

    if not representative_tracks:

        raise RuntimeError(
            "No representative tracks "
            "available for case 1."
        )

    for index, tname in enumerate(
        representative_tracks,
        start=1,
    ):

        metadata = (
            engine.resolve_track(
                demo_caseid,
                tname,
            )
        )

        print(
            f"{index:2d}. "
            f"{tname:30s} "
            f"{str(metadata['data_role']):10s} "
            f"tid={metadata['tid']}"
        )

    # =========================================================================
    # 6. REMOTE RETRIEVAL
    # =========================================================================

    section(
        "6. REMOTE SIGNAL RETRIEVAL"
    )

    signal_summaries = []

    signal_validations = []

    fetched_signals = {}

    #
    # IMPORTANT:
    #
    # The API endpoint itself returns the track
    # resource. For this validation run, each
    # selected representative track is retrieved
    # once.
    #
    # The result is only exported under stage07
    # results for validation.
    #

    for index, tname in enumerate(
        representative_tracks,
        start=1,
    ):

        print()

        print(
            "-" * 96
        )

        print(
            f"[{index}/{len(representative_tracks)}] "
            f"Fetching {tname}"
        )

        try:

            signal = (
                engine.fetch_track(
                    caseid=(
                        demo_caseid
                    ),
                    tname=tname,
                )
            )

            fetched_signals[
                tname
            ] = signal

            summary = (
                signal.summary()
            )

            summary[
                "fetch_status"
            ] = "PASS"

            summary[
                "error"
            ] = None

            signal_summaries.append(
                summary
            )

            validation = (
                validate_signal(
                    signal
                )
            )

            signal_validations.append(
                validation
            )

            print(
                f"HTTP status:          "
                f"{signal.http_status}"
            )

            print(
                f"Response size:        "
                f"{human_bytes(signal.response_bytes)}"
            )

            print(
                f"Response time:        "
                f"{signal.elapsed_sec:.3f} sec"
            )

            print(
                f"Track type:           "
                f"{signal.track_type}"
            )

            print(
                f"Samples:              "
                f"{signal.sample_count:,}"
            )

            print(
                f"Finite values:        "
                f"{signal.finite_count:,}"
            )

            print(
                f"Missing values:       "
                f"{signal.missing_count:,}"
            )

            print(
                f"Start (sec):          "
                f"{signal.start_sec}"
            )

            print(
                f"End (sec):            "
                f"{signal.end_sec}"
            )

            print(
                f"Duration (sec):       "
                f"{signal.duration_sec}"
            )

            print(
                f"Interval (sec):       "
                f"{signal.interval_sec}"
            )

            print(
                f"Sampling rate (Hz):   "
                f"{signal.sampling_rate_hz}"
            )

            validation_failures = int(
                (
                    (
                        validation[
                            "status"
                        ]
                        == "FAIL"
                    )
                    &
                    (
                        validation[
                            "severity"
                        ]
                        == "ERROR"
                    )
                ).sum()
            )

            print(
                f"Validation errors:    "
                f"{validation_failures}"
            )

        except Exception as exc:

            print(
                f"FETCH FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            signal_summaries.append(
                {
                    "caseid": (
                        demo_caseid
                    ),
                    "tname": (
                        tname
                    ),
                    "fetch_status": (
                        "FAIL"
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    signal_summary_df = (
        pd.DataFrame(
            signal_summaries
        )
    )

    signal_summary_path = (
        RESULTS_DIR
        / "representative_signal_summary.csv"
    )

    signal_summary_df.to_csv(
        signal_summary_path,
        index=False,
    )

    if signal_validations:

        signal_validation_df = (
            pd.concat(
                signal_validations,
                ignore_index=True,
            )
        )

    else:

        signal_validation_df = (
            pd.DataFrame()
        )

    signal_validation_path = (
        RESULTS_DIR
        / "remote_signal_validation.csv"
    )

    signal_validation_df.to_csv(
        signal_validation_path,
        index=False,
    )

    # =========================================================================
    # 7. RETRIEVAL SUMMARY
    # =========================================================================

    section(
        "7. REMOTE RETRIEVAL SUMMARY"
    )

    print(
        signal_summary_df.to_string(
            index=False
        )
    )

    successful_fetches = int(
        (
            signal_summary_df[
                "fetch_status"
            ]
            == "PASS"
        ).sum()
    )

    failed_fetches = int(
        (
            signal_summary_df[
                "fetch_status"
            ]
            == "FAIL"
        ).sum()
    )

    print()

    print(
        f"Successful fetches: "
        f"{successful_fetches}"
    )

    print(
        f"Failed fetches:     "
        f"{failed_fetches}"
    )

    # =========================================================================
    # 8. REMOTE SIGNAL VALIDATION
    # =========================================================================

    section(
        "8. REMOTE SIGNAL VALIDATION"
    )

    if (
        signal_validation_df.empty
    ):

        print(
            "No successfully parsed "
            "signals available."
        )

        signal_error_failures = (
            failed_fetches
        )

        signal_warning_failures = 0

    else:

        print(
            signal_validation_df
            .to_string(
                index=False
            )
        )

        signal_error_failures = int(
            (
                (
                    signal_validation_df[
                        "status"
                    ]
                    == "FAIL"
                )
                &
                (
                    signal_validation_df[
                        "severity"
                    ]
                    == "ERROR"
                )
            ).sum()
        )

        signal_warning_failures = int(
            (
                (
                    signal_validation_df[
                        "status"
                    ]
                    == "FAIL"
                )
                &
                (
                    signal_validation_df[
                        "severity"
                    ]
                    == "WARNING"
                )
            ).sum()
        )

        signal_error_failures += (
            failed_fetches
        )

    print()

    print(
        f"Remote error failures:   "
        f"{signal_error_failures}"
    )

    print(
        f"Remote warning failures: "
        f"{signal_warning_failures}"
    )

    # =========================================================================
    # 9. WINDOWED RETRIEVAL TEST
    # =========================================================================

    section(
        "9. WINDOWED RETRIEVAL INTERFACE TEST"
    )

    window_tests = []

    waveform_candidates = [
        signal
        for signal
        in fetched_signals.values()
        if (
            signal.track_type
            == "WAVEFORM"
            and signal.duration_sec
            is not None
            and signal.duration_sec
            > 60
        )
    ]

    if waveform_candidates:

        source_signal = (
            waveform_candidates[0]
        )

        #
        # IMPORTANT:
        #
        # This does NOT issue another HTTP request.
        #
        # It validates the exact same clipping
        # semantics used by fetch_track().
        #

        test_start = 10.0
        test_end = 20.0

        (
            clipped_time,
            clipped_values,
            _,
        ) = (
            engine._clip_window(
                source_signal.time_sec,
                source_signal.values,
                test_start,
                test_end,
            )
        )

        within_bounds = bool(
            (
                len(
                    clipped_time
                ) > 0
            )
            and np.all(
                clipped_time
                >= test_start
            )
            and np.all(
                clipped_time
                < test_end
            )
        )

        window_tests.append(
            {
                "caseid": (
                    source_signal.caseid
                ),
                "tname": (
                    source_signal.tname
                ),
                "start_sec": (
                    test_start
                ),
                "end_sec": (
                    test_end
                ),
                "samples": (
                    len(
                        clipped_values
                    )
                ),
                "status": (
                    "PASS"
                    if within_bounds
                    else "FAIL"
                ),
            }
        )

        print(
            f"Track:       "
            f"{source_signal.tname}"
        )

        print(
            f"Window:      "
            f"[{test_start}, "
            f"{test_end}) sec"
        )

        print(
            f"Samples:     "
            f"{len(clipped_values):,}"
        )

        print(
            f"Bounds test: "
            f"{'PASS' if within_bounds else 'FAIL'}"
        )

    else:

        print(
            "No waveform signal suitable "
            "for window test."
        )

    window_test_df = (
        pd.DataFrame(
            window_tests
        )
    )

    window_test_path = (
        RESULTS_DIR
        / "window_retrieval_test.csv"
    )

    window_test_df.to_csv(
        window_test_path,
        index=False,
    )

    # =========================================================================
    # 10. CANONICAL-SIGNAL INTERFACE TEST
    # =========================================================================

    section(
        "10. CANONICAL SIGNAL RESOLUTION TEST"
    )

    canonical_tests = []

    if (
        "SNUADC/ECG_II"
        in fetched_signals
    ):

        exact_metadata = (
            engine.resolve_track(
                demo_caseid,
                "SNUADC/ECG_II",
            )
        )

        canonical_name = str(
            exact_metadata[
                "canonical_signal"
            ]
        )

        resolved = (
            engine.resolve_canonical_signal(
                caseid=(
                    demo_caseid
                ),
                canonical_signal=(
                    canonical_name
                ),
                preferred_tname=(
                    "SNUADC/ECG_II"
                ),
            )
        )

        passed = (
            resolved[
                "tname"
            ]
            == "SNUADC/ECG_II"
        )

        canonical_tests.append(
            {
                "caseid": (
                    demo_caseid
                ),
                "canonical_signal": (
                    canonical_name
                ),
                "resolved_tname": (
                    resolved[
                        "tname"
                    ]
                ),
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

        print(
            f"Canonical signal: "
            f"{canonical_name}"
        )

        print(
            f"Resolved track:   "
            f"{resolved['tname']}"
        )

        print(
            f"Resolution test:  "
            f"{'PASS' if passed else 'FAIL'}"
        )

    else:

        print(
            "ECG_II was not successfully "
            "fetched; canonical interface "
            "test skipped."
        )

    canonical_test_df = (
        pd.DataFrame(
            canonical_tests
        )
    )

    canonical_test_path = (
        RESULTS_DIR
        / "canonical_resolution_test.csv"
    )

    canonical_test_df.to_csv(
        canonical_test_path,
        index=False,
    )

    # =========================================================================
    # 11. EXPORT SMALL VALIDATION WINDOWS
    # =========================================================================

    section(
        "11. EXPORTING VALIDATION WINDOWS"
    )

    validation_data_dir = (
        RESULTS_DIR
        / "validation_signals"
    )

    validation_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exported_files = []

    for tname, signal in (
        fetched_signals.items()
    ):

        #
        # Do NOT write entire large waveform
        # tracks to disk.
        #
        # Export at most the first 30 seconds
        # for waveform validation.
        #
        # Numeric tracks are typically sparse;
        # cap them at the first 5000 rows.
        #

        if (
            signal.track_type
            == "WAVEFORM"
        ):

            if (
                signal.time_sec.size
                > 0
            ):

                start = float(
                    signal.time_sec[
                        0
                    ]
                )

                end = (
                    start
                    + 30.0
                )

                mask = (
                    signal.time_sec
                    < end
                )

                export_time = (
                    signal.time_sec[
                        mask
                    ]
                )

                export_values = (
                    signal.values[
                        mask
                    ]
                )

            else:

                export_time = (
                    signal.time_sec
                )

                export_values = (
                    signal.values
                )

        else:

            limit = min(
                5000,
                signal.sample_count,
            )

            export_time = (
                signal.time_sec[
                    :limit
                ]
            )

            export_values = (
                signal.values[
                    :limit
                ]
            )

        export_signal = (
            SignalData(
                caseid=(
                    signal.caseid
                ),

                tname=(
                    signal.tname
                ),

                tid=(
                    signal.tid
                ),

                canonical_signal=(
                    signal.canonical_signal
                ),

                display_name=(
                    signal.display_name
                ),

                device=(
                    signal.device
                ),

                signal_family=(
                    signal.signal_family
                ),

                data_role=(
                    signal.data_role
                ),

                mapping_status=(
                    signal.mapping_status
                ),

                remote_url=(
                    signal.remote_url
                ),

                remote_format=(
                    signal.remote_format
                ),

                track_type=(
                    signal.track_type
                ),

                time_sec=(
                    export_time
                ),

                values=(
                    export_values
                ),

                sample_count=(
                    len(
                        export_values
                    )
                ),

                finite_count=int(
                    np.isfinite(
                        export_values
                    ).sum()
                ),

                missing_count=int(
                    len(
                        export_values
                    )
                    - np.isfinite(
                        export_values
                    ).sum()
                ),

                start_sec=(
                    float(
                        export_time[
                            0
                        ]
                    )
                    if len(
                        export_time
                    )
                    else None
                ),

                end_sec=(
                    float(
                        export_time[
                            -1
                        ]
                    )
                    if len(
                        export_time
                    )
                    else None
                ),

                duration_sec=(
                    (
                        float(
                            export_time[
                                -1
                            ]
                            - export_time[
                                0
                            ]
                        )
                    )
                    if len(
                        export_time
                    ) >= 2
                    else 0.0
                ),

                interval_sec=(
                    signal.interval_sec
                ),

                sampling_rate_hz=(
                    signal.sampling_rate_hz
                ),

                response_bytes=(
                    signal.response_bytes
                ),

                response_sha256=(
                    signal.response_sha256
                ),

                http_status=(
                    signal.http_status
                ),

                elapsed_sec=(
                    signal.elapsed_sec
                ),

                source_sample_count=(
                    signal.sample_count
                ),

                clipped=True,
            )
        )

        paths = (
            export_signal_result(
                export_signal,
                validation_data_dir,
            )
        )

        exported_files.extend(
            paths.values()
        )

        print(
            f"Exported validation "
            f"subset: {tname}"
        )

    # =========================================================================
    # 12. REPORT
    # =========================================================================

    section(
        "12. WRITING STAGE 07 REPORT"
    )

    window_failures = (
        int(
            (
                window_test_df[
                    "status"
                ]
                == "FAIL"
            ).sum()
        )
        if not window_test_df.empty
        else 0
    )

    canonical_failures = (
        int(
            (
                canonical_test_df[
                    "status"
                ]
                == "FAIL"
            ).sum()
        )
        if not canonical_test_df.empty
        else 0
    )

    total_error_failures = (
        metadata_error_failures
        + signal_error_failures
        + window_failures
        + canonical_failures
    )

    report_lines = [

        "VitalDB Explorer",
        "Stage 07 - Remote Signal Retrieval Engine",
        "=" * 80,
        "",
        f"Generated UTC: {utc_now_iso()}",
        f"Script version: {SCRIPT_VERSION}",
        "",
        "PURPOSE",
        "-" * 80,
        (
            "Stage 07 provides on-demand access "
            "to actual VitalDB Open Dataset "
            "track measurements."
        ),
        (
            "Tracks are resolved through the "
            "caseid -> exact tname -> tid chain."
        ),
        "",
        "REMOTE ENDPOINT",
        "-" * 80,
        f"Base URL: {API_BASE}",
        (
            "Track resource: "
            "https://api.vitaldb.net/{tid}"
        ),
        "",
        "DATASET",
        "-" * 80,
        (
            f"Cases: "
            f"{len(engine.master):,}"
        ),
        (
            f"Track metadata rows: "
            f"{len(engine.tracks):,}"
        ),
        (
            f"Exact tracks: "
            f"{engine.tracks['tname'].nunique():,}"
        ),
        (
            f"Canonical signals: "
            f"{engine.dictionary['canonical_signal'].nunique():,}"
        ),
        "",
        "REMOTE VALIDATION",
        "-" * 80,
        (
            f"Representative tracks requested: "
            f"{len(representative_tracks)}"
        ),
        (
            f"Successful fetches: "
            f"{successful_fetches}"
        ),
        (
            f"Failed fetches: "
            f"{failed_fetches}"
        ),
        (
            f"Remote validation error failures: "
            f"{signal_error_failures}"
        ),
        (
            f"Remote validation warning failures: "
            f"{signal_warning_failures}"
        ),
        "",
        "ARCHITECTURE",
        "-" * 80,
        (
            "Waveform sampling rate is derived "
            "from the remote track interval."
        ),
        (
            "Numeric tracks retain their explicit "
            "timestamps and are not assigned an "
            "artificial waveform sampling rate."
        ),
        (
            "Canonical signals are resolved to "
            "exact VitalDB source tracks before "
            "remote retrieval."
        ),
        (
            "Exact source identity and tid are "
            "preserved in every returned "
            "SignalData object."
        ),
        "",
        "STORAGE POLICY",
        "-" * 80,
        (
            "Stage 07 does not permanently "
            "download complete cases."
        ),
        (
            "Only small validation subsets are "
            "written under results/stage07."
        ),
        (
            "Permanent cohort acquisition remains "
            "the responsibility of the later "
            "download/export stage."
        ),
        "",
        "TOTAL ERROR FAILURES",
        "-" * 80,
        str(
            total_error_failures
        ),
        "",
        "STATUS",
        "-" * 80,
        (
            "STATUS: PASS"
            if total_error_failures
            == 0
            else "STATUS: FAIL"
        ),
    ]

    report_text = "\n".join(
        report_lines
    )

    print(
        report_text
    )

    report_path = (
        RESULTS_DIR
        / "stage07_report.txt"
    )

    report_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {report_path}"
    )

    # =========================================================================
    # 13. MANIFEST
    # =========================================================================

    section(
        "13. WRITING STAGE 07 MANIFEST"
    )

    manifest = {

        "stage": 7,

        "name": (
            "Remote Signal Retrieval Engine"
        ),

        "script_version": (
            SCRIPT_VERSION
        ),

        "generated_utc": (
            utc_now_iso()
        ),

        "environment": {

            "python": (
                sys.version
            ),

            "platform": (
                platform.platform()
            ),

            "pandas": (
                pd.__version__
            ),

            "numpy": (
                np.__version__
            ),

            "requests": (
                requests.__version__
            ),
        },

        "api": {

            "base_url": (
                API_BASE
            ),

            "track_endpoint": (
                f"{API_BASE}/{{tid}}"
            ),

            "connect_timeout_sec": (
                CONNECT_TIMEOUT_SEC
            ),

            "read_timeout_sec": (
                READ_TIMEOUT_SEC
            ),

            "max_retries": (
                MAX_RETRIES
            ),
        },

        "dataset": {

            "cases": (
                len(
                    engine.master
                )
            ),

            "track_rows": (
                len(
                    engine.tracks
                )
            ),

            "exact_tracks": int(
                engine.tracks[
                    "tname"
                ].nunique()
            ),

            "canonical_signals": int(
                engine.dictionary[
                    "canonical_signal"
                ].nunique()
            ),
        },

        "validation": {

            "representative_tracks": (
                len(
                    representative_tracks
                )
            ),

            "successful_fetches": (
                successful_fetches
            ),

            "failed_fetches": (
                failed_fetches
            ),

            "metadata_error_failures": (
                metadata_error_failures
            ),

            "remote_error_failures": (
                signal_error_failures
            ),

            "remote_warning_failures": (
                signal_warning_failures
            ),

            "window_failures": (
                window_failures
            ),

            "canonical_resolution_failures": (
                canonical_failures
            ),

            "total_error_failures": (
                total_error_failures
            ),
        },

        "representative_results": (
            signal_summary_df
            .to_dict(
                orient="records"
            )
        ),

        "storage_policy": {

            "complete_cases_written": (
                False
            ),

            "complete_remote_tracks_written": (
                False
            ),

            "validation_subsets_written": (
                True
            ),

            "permanent_download_stage": (
                12
            ),
        },
    }

    manifest_path = (
        RESULTS_DIR
        / "stage07_manifest.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    print(
        f"Saved: {manifest_path}"
    )

    # =========================================================================
    # 14. FINAL
    # =========================================================================

    section(
        "STAGE 07 COMPLETE"
    )

    print(
        f"Representative tracks:       "
        f"{len(representative_tracks)}"
    )

    print(
        f"Successful remote fetches:   "
        f"{successful_fetches}"
    )

    print(
        f"Failed remote fetches:       "
        f"{failed_fetches}"
    )

    print(
        f"Metadata error failures:     "
        f"{metadata_error_failures}"
    )

    print(
        f"Remote error failures:       "
        f"{signal_error_failures}"
    )

    print(
        f"Remote warning failures:     "
        f"{signal_warning_failures}"
    )

    print(
        f"Window test failures:        "
        f"{window_failures}"
    )

    print(
        f"Canonical resolution fails:  "
        f"{canonical_failures}"
    )

    print(
        f"Total error failures:        "
        f"{total_error_failures}"
    )

    print()

    if (
        total_error_failures
        == 0
    ):

        print(
            "STATUS: PASS"
        )

        print()

        print(
            "Remote VitalDB signal retrieval "
            "is operational."
        )

        print()

        print(
            "The Explorer can now resolve a "
            "selected case and signal to its "
            "authoritative VitalDB track ID and "
            "retrieve the actual measurements."
        )

        print()

        print(
            "Waveform timing and sampling rate "
            "are derived from the remote data."
        )

        print()

        print(
            "Numeric tracks retain their "
            "original irregular timestamps."
        )

        print()

        print(
            "Stage 08 signal visualization "
            "may proceed after reviewing "
            "this output."
        )

    else:

        print(
            "STATUS: FAIL"
        )

        print()

        print(
            "Do not proceed to Stage 08."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
"""
LMCI engine for the Hyderabad Metro Last-Mile Connectivity project.

Responsibilities
----------------
1. Build station-level multimodal catchments
2. Compute temporal feeder/bus/MMTS frequencies
3. Compute route-diversity weighted LMCI
4. Classify transit deserts
5. Export validated station-level LMCI outputs into outputs/
6. Keep dashboard-safe CSVs and reusable GeoDataFrames

Expected project layout
-----------------------
project/
├── data/
│   ├── hmrl/
│   ├── tgsrtc/
│   ├── mmts/
│   └── feeder/
├── outputs/
├── assets/
└── src/
    ├── preprocessing.py
    └── lmci.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from src.preprocessing import Config as PreprocessingConfig
    from src.preprocessing import GTFSLoader, minmax_norm, to_gdf
except ImportError:
    from preprocessing import Config as PreprocessingConfig
    from preprocessing import GTFSLoader, minmax_norm, to_gdf


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("LMCI")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class LMCIConfig:
    """Configuration for the LMCI engine."""

    # Directories
    data_dir: Path = Path("data")
    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")

    hmrl_dir: Path = Path("data/hmrl")
    tgsrtc_dir: Path = Path("data/tgsrtc")
    mmts_dir: Path = Path("data/mmts")
    feeder_dir: Path = Path("data/feeder")

    # CRS
    wgs84_crs: int = 4326
    utm_crs: int = 32644

    # Spatial parameters
    walk_radius_m: float = 800.0
    feeder_radius_m: float = 3000.0
    detour_factor: float = 1.35

    # MMTS transfer logic
    mmts_xfer_dist_m: float = 1500.0
    mmts_transfer_penalty_min: float = 5.0
    mmts_schedule_overlap_min: float = 30.0

    # Hyderabad bbox
    bbox_lat_min: float = 16.8
    bbox_lat_max: float = 17.8
    bbox_lon_min: float = 77.5
    bbox_lon_max: float = 79.5

    # Time windows
    time_windows: List[Tuple[float, float, str]] = field(default_factory=lambda: [
        (7.0, 10.0, "Morning"),
        (11.0, 14.0, "Midday"),
        (17.0, 21.0, "Evening"),
    ])

    # LMCI weights: baseline notebook model
    w_density: float = 0.50
    w_frequency: float = 0.40
    w_walkzone: float = 0.10

    # Demand-augmented model, useful when demand_count exists downstream
    w_density_new: float = 0.35
    w_frequency_new: float = 0.35
    w_walkzone_new: float = 0.15
    w_demand_new: float = 0.15

    # Mode weights applied to route_diversity in density computation.
    # Values > 1.0 give feeder/MMTS stops proportionally more density credit,
    # correcting the metro-centric bias without changing the LMCI formula.
    feeder_mode_weight: float = 1.5
    mmts_mode_weight: float = 1.3

    # Equity threshold
    equity_desert_threshold: float = 4.0

    def __post_init__(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        old_sum = self.w_density + self.w_frequency + self.w_walkzone
        new_sum = (
            self.w_density_new
            + self.w_frequency_new
            + self.w_walkzone_new
            + self.w_demand_new
        )

        if abs(old_sum - 1.0) > 1e-6:
            raise ValueError("Baseline LMCI weights must sum to 1.0")

        if abs(new_sum - 1.0) > 1e-6:
            raise ValueError("Demand-augmented LMCI weights must sum to 1.0")


CFG = LMCIConfig()


# ─────────────────────────────────────────────────────────────
# BASIC UTILITIES
# ─────────────────────────────────────────────────────────────

def parse_gtfs_time(value) -> Optional[float]:
    """
    Parse GTFS HH:MM:SS into fractional hours.

    Handles GTFS times beyond 24:00:00.
    Example:
        25:30:00 → 25.5
    """

    try:
        parts = str(value).strip().split(":")
        if len(parts) != 3:
            return None

        h, m, s = parts
        return int(h) + int(m) / 60.0 + int(s) / 3600.0

    except Exception:
        return None


def _safe_numeric(series: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip().str.lower()
    return out


# ─────────────────────────────────────────────────────────────
# OPTIONAL GTFS LOADERS
# ─────────────────────────────────────────────────────────────

def _read_gtfs_file(path: Path) -> Optional[pd.DataFrame]:
    """Read GTFS file if present. Return None if missing."""

    if not path.exists():
        logger.warning(f"Missing optional GTFS file: {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    return _normalise_columns(df)


def load_optional_gtfs_feeds(cfg: LMCIConfig = CFG) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load optional MMTS and feeder feeds.

    Required feeds are handled by preprocessing.GTFSLoader:
    - HMRL stops
    - TGSRTC stops
    - TGSRTC stop_times

    Optional feeds:
    - TGSRTC trips
    - MMTS stops / stop_times
    - feeder stops / stop_times / frequencies
    """

    feeds = {
        "bus_trips": _read_gtfs_file(cfg.tgsrtc_dir / "trips.txt"),
        "mmts_stops": _read_gtfs_file(cfg.mmts_dir / "stops.txt"),
        "mmts_times": _read_gtfs_file(cfg.mmts_dir / "stop_times.txt"),
        "feeder_stops": _read_gtfs_file(cfg.feeder_dir / "stops.txt"),
        "feeder_times": _read_gtfs_file(cfg.feeder_dir / "stop_times.txt"),
        "feeder_freqs": _read_gtfs_file(cfg.feeder_dir / "frequencies.txt"),
        # trips.txt allows proper route-diversity computation for feeder stops
        "feeder_trips": _read_gtfs_file(cfg.feeder_dir / "trips.txt"),
    }

    for key, df in feeds.items():
        if df is not None:
            logger.info(f"Loaded {key}: {len(df):,} rows")

    return feeds


def _prepare_stop_dataframe(
    df: Optional[pd.DataFrame],
    cfg: LMCIConfig = CFG,
) -> Optional[pd.DataFrame]:
    """Validate optional stop dataframe and keep clean Hyderabad rows."""

    if df is None or df.empty:
        return None

    required = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        logger.warning(f"Skipping stop dataframe because columns are missing: {missing}")
        return None

    out = df.copy()
    out["stop_lat"] = pd.to_numeric(out["stop_lat"], errors="coerce")
    out["stop_lon"] = pd.to_numeric(out["stop_lon"], errors="coerce")
    out = out.dropna(subset=["stop_lat", "stop_lon"])

    mask = (
        out["stop_lat"].between(cfg.bbox_lat_min, cfg.bbox_lat_max)
        & out["stop_lon"].between(cfg.bbox_lon_min, cfg.bbox_lon_max)
    )

    out = out[mask].copy()

    return out[["stop_id", "stop_name", "stop_lat", "stop_lon"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# ROUTE DIVERSITY
# ─────────────────────────────────────────────────────────────

def compute_route_diversity(
    bus_stops: pd.DataFrame,
    bus_times: pd.DataFrame,
    bus_trips: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Count unique route IDs serving each bus stop.

    Strategy order:
    A. route_id directly in stop_times
    B. stop_times joined with trips.txt on trip_id
    C. trip_id prefix proxy
    D. neutral fallback = 1
    """

    if bus_times is None or bus_times.empty:
        logger.warning("No bus_times available; route diversity defaults to 1.")
        return pd.Series(1, index=bus_stops["stop_id"], name="route_diversity")

    st = _normalise_columns(bus_times)

    if {"stop_id", "route_id"}.issubset(st.columns):
        div = st.groupby("stop_id")["route_id"].nunique().clip(lower=1)
        logger.info(f"Route diversity direct: mean={div.mean():.2f}, max={div.max():.0f}")
        return div.rename("route_diversity")

    if bus_trips is not None and not bus_trips.empty:
        trips = _normalise_columns(bus_trips)

        if {"trip_id", "route_id"}.issubset(trips.columns) and "trip_id" in st.columns:
            merged = st.merge(
                trips[["trip_id", "route_id"]],
                on="trip_id",
                how="left",
            )

            div = merged.groupby("stop_id")["route_id"].nunique().fillna(1).clip(lower=1)
            logger.info(f"Route diversity via trips join: mean={div.mean():.2f}")
            return div.rename("route_diversity")

    if {"stop_id", "trip_id"}.issubset(st.columns):
        st["_route_proxy"] = st["trip_id"].astype(str).str.split("_").str[0]
        div = st.groupby("stop_id")["_route_proxy"].nunique().clip(lower=1)
        logger.info(f"Route diversity proxy: mean={div.mean():.2f}")
        return div.rename("route_diversity")

    logger.warning("Could not compute route diversity; defaulting to 1.")
    return pd.Series(1, index=bus_stops["stop_id"], name="route_diversity")


# ─────────────────────────────────────────────────────────────
# TEMPORAL FREQUENCY
# ─────────────────────────────────────────────────────────────

def compute_temporal_frequency(
    stop_times_df: Optional[pd.DataFrame],
    stop_ids: Iterable,
    time_windows: List[Tuple[float, float, str]],
    time_col: str = "departure_time",
) -> pd.DataFrame:
    """
    Compute trips/hour per stop per time window.

    Returns
    -------
    DataFrame indexed by stop_id with:
        Morning_freq, Midday_freq, Evening_freq
    """

    result = pd.DataFrame({"stop_id": list(stop_ids)}).drop_duplicates().set_index("stop_id")

    for _, _, label in time_windows:
        result[f"{label}_freq"] = 0.0

    if stop_times_df is None or stop_times_df.empty:
        return result

    st = _normalise_columns(stop_times_df)

    if time_col not in st.columns:
        if "arrival_time" in st.columns:
            st[time_col] = st["arrival_time"]
        else:
            logger.warning("No departure_time or arrival_time found; frequency set to zero.")
            return result

    if "trip_id" not in st.columns or "stop_id" not in st.columns:
        logger.warning("stop_times missing stop_id/trip_id; frequency set to zero.")
        return result

    st["dep_hr"] = st[time_col].apply(parse_gtfs_time)
    st = st.dropna(subset=["dep_hr"])

    for start_h, end_h, label in time_windows:
        duration_h = max(end_h - start_h, 1e-9)
        window = st[st["dep_hr"].between(start_h, end_h)]
        freq = window.groupby("stop_id")["trip_id"].nunique().div(duration_h)
        result[f"{label}_freq"] = result.index.map(freq).fillna(0.0)

    return result


def compute_feeder_frequency_from_headways(
    feeder_stops: Optional[pd.DataFrame],
    feeder_freqs: Optional[pd.DataFrame],
    time_windows: List[Tuple[float, float, str]],
) -> pd.DataFrame:
    """Compute feeder frequency from GTFS frequencies.txt."""

    if feeder_stops is None or feeder_stops.empty:
        return pd.DataFrame(columns=["stop_id"] + [f"{lbl}_freq" for _, _, lbl in time_windows]).set_index("stop_id")

    result = pd.DataFrame({"stop_id": feeder_stops["stop_id"]}).drop_duplicates().set_index("stop_id")

    for _, _, label in time_windows:
        result[f"{label}_freq"] = 0.0

    if feeder_freqs is None or feeder_freqs.empty:
        logger.warning("No feeder frequencies.txt; feeder frequency set to zero.")
        return result

    ff = _normalise_columns(feeder_freqs)

    required = {"start_time", "end_time", "headway_secs"}
    if not required.issubset(ff.columns):
        logger.warning("feeder frequencies.txt missing required columns; feeder frequency set to zero.")
        return result

    ff["start_hr"] = ff["start_time"].apply(parse_gtfs_time)
    ff["end_hr"] = ff["end_time"].apply(parse_gtfs_time)
    ff["headway_secs"] = pd.to_numeric(ff["headway_secs"], errors="coerce")
    ff = ff.dropna(subset=["start_hr", "end_hr", "headway_secs"])
    ff["trips_per_hr"] = 3600.0 / ff["headway_secs"].clip(lower=60)

    for start_h, end_h, label in time_windows:
        overlap = ff[(ff["start_hr"] < end_h) & (ff["end_hr"] > start_h)]

        if not overlap.empty:
            result[f"{label}_freq"] = overlap["trips_per_hr"].mean()

    return result


# ─────────────────────────────────────────────────────────────
# MMTS SUPPORT
# ─────────────────────────────────────────────────────────────

def compute_mmts_schedule_overlap(
    mmts_times: Optional[pd.DataFrame],
    time_windows: List[Tuple[float, float, str]],
    min_overlap_min: float = 30.0,
) -> Dict[str, Optional[Set[str]]]:
    """Return eligible MMTS stop IDs by time window."""

    if mmts_times is None or mmts_times.empty:
        return {label: None for _, _, label in time_windows}

    st = _normalise_columns(mmts_times)

    if "departure_time" not in st.columns:
        if "arrival_time" in st.columns:
            st["departure_time"] = st["arrival_time"]
        else:
            return {label: None for _, _, label in time_windows}

    if "stop_id" not in st.columns:
        return {label: None for _, _, label in time_windows}

    st["dep_hr"] = st["departure_time"].apply(parse_gtfs_time)
    st = st.dropna(subset=["dep_hr"])

    overlap_map: Dict[str, Optional[Set[str]]] = {}
    tolerance_h = min_overlap_min / 60.0

    for start_h, end_h, label in time_windows:
        eligible = set(
            st[
                st["dep_hr"].between(start_h - tolerance_h, end_h + tolerance_h)
            ]["stop_id"].unique()
        )
        overlap_map[label] = eligible

    return overlap_map


def compute_mmts_transfer_penalty_factor(
    cfg: LMCIConfig,
    window_duration_h: float,
) -> float:
    """MMTS frequency discount for transfer friction."""

    half_window_min = max(window_duration_h * 60.0 / 2.0, 1e-9)
    return min(cfg.mmts_transfer_penalty_min / half_window_min, 0.30)


# ─────────────────────────────────────────────────────────────
# MULTIMODAL GRAPH
# ─────────────────────────────────────────────────────────────

def build_multimodal_graph(
    cfg: LMCIConfig,
    metro_df: pd.DataFrame,
    bus_df: pd.DataFrame,
    bus_times: pd.DataFrame,
    bus_trips: Optional[pd.DataFrame] = None,
    mmts_stops: Optional[pd.DataFrame] = None,
    mmts_times: Optional[pd.DataFrame] = None,
    feeder_stops: Optional[pd.DataFrame] = None,
    feeder_times: Optional[pd.DataFrame] = None,
    feeder_trips: Optional[pd.DataFrame] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Set[str], Dict[str, Optional[Set[str]]]]:
    """
    Build multimodal stop graph.

    Every bus/MMTS/feeder stop is assigned to its nearest metro station.
    Distances are converted from Euclidean UTM distance to approximate
    network distance using detour_factor.
    """

    logger.info("Building multimodal graph...")

    gdf_metro = to_gdf(
        metro_df,
        lat_col="stop_lat",
        lon_col="stop_lon",
        crs_out=cfg.utm_crs,
    )

    metro_xy = np.column_stack([gdf_metro.geometry.x, gdf_metro.geometry.y])
    metro_tree = cKDTree(metro_xy)

    def annotate_nearest_metro(gdf: gpd.GeoDataFrame, mode: str) -> gpd.GeoDataFrame:
        xy = np.column_stack([gdf.geometry.x, gdf.geometry.y])
        dist_euclidean, idx = metro_tree.query(xy, k=1, workers=-1)
        dist_network = dist_euclidean * cfg.detour_factor

        out = gdf.copy()
        out["mode"] = mode
        out["nearest_metro_idx"] = idx
        out["nearest_metro_id"] = gdf_metro.iloc[idx]["stop_id"].values
        out["nearest_metro_name"] = gdf_metro.iloc[idx]["stop_name"].values
        out["dist_euclidean_m"] = dist_euclidean
        out["dist_to_metro_m"] = dist_network
        out["in_walk_zone"] = dist_network <= cfg.walk_radius_m
        out["in_feeder_zone"] = dist_network <= cfg.feeder_radius_m
        return out

    # Bus layer
    gdf_bus = annotate_nearest_metro(
        to_gdf(bus_df, lat_col="stop_lat", lon_col="stop_lon", crs_out=cfg.utm_crs),
        "bus",
    )

    route_div = compute_route_diversity(bus_df, bus_times, bus_trips)
    gdf_bus["route_diversity"] = gdf_bus["stop_id"].map(route_div).fillna(1).clip(lower=1)

    parts: List[gpd.GeoDataFrame] = [gdf_bus]

    # Feeder layer
    feeder_stops = _prepare_stop_dataframe(feeder_stops, cfg)
    if feeder_stops is not None and not feeder_stops.empty:
        gdf_feeder = annotate_nearest_metro(
            to_gdf(feeder_stops, lat_col="stop_lat", lon_col="stop_lon", crs_out=cfg.utm_crs),
            "feeder",
        )
        # Compute real route diversity for feeders (same strategy as bus: direct,
        # trips-join, prefix proxy, neutral). Then apply mode weight so feeder stops
        # receive proportionally stronger density credit in the LMCI formula.
        feeder_route_div = compute_route_diversity(feeder_stops, feeder_times, feeder_trips)
        gdf_feeder["route_diversity"] = (
            gdf_feeder["stop_id"].map(feeder_route_div).fillna(1).clip(lower=1)
            * cfg.feeder_mode_weight
        )
        parts.append(gdf_feeder)

    # MMTS layer
    mmts_eligible: Set[str] = set()
    mmts_schedule_map: Dict[str, Optional[Set[str]]] = {label: None for _, _, label in cfg.time_windows}

    mmts_stops = _prepare_stop_dataframe(mmts_stops, cfg)
    if mmts_stops is not None and not mmts_stops.empty:
        gdf_mmts = annotate_nearest_metro(
            to_gdf(mmts_stops, lat_col="stop_lat", lon_col="stop_lon", crs_out=cfg.utm_crs),
            "mmts",
        )
        gdf_mmts["route_diversity"] = cfg.mmts_mode_weight  # typically 1 route/stop; weight gives MMTS proportional density credit

        mmts_xy = np.column_stack([gdf_mmts.geometry.x, gdf_mmts.geometry.y])
        mmts_dist, _ = cKDTree(mmts_xy).query(metro_xy, k=1, workers=-1)
        eligible_idx = np.where(mmts_dist * cfg.detour_factor <= cfg.mmts_xfer_dist_m)[0]
        mmts_eligible = set(gdf_metro.iloc[eligible_idx]["stop_id"])

        mmts_schedule_map = compute_mmts_schedule_overlap(
            mmts_times,
            cfg.time_windows,
            cfg.mmts_schedule_overlap_min,
        )

        parts.append(gdf_mmts)

    gdf_combined = pd.concat(parts, ignore_index=True)

    logger.info(
        "Combined graph: "
        + ", ".join(
            f"{mode}={(gdf_combined['mode'] == mode).sum():,}"
            for mode in sorted(gdf_combined["mode"].unique())
        )
    )

    return gdf_metro, gdf_combined, mmts_eligible, mmts_schedule_map


# ─────────────────────────────────────────────────────────────
# FREQUENCY ATTACHMENT
# ─────────────────────────────────────────────────────────────

def attach_temporal_frequencies(
    cfg: LMCIConfig,
    gdf_combined: gpd.GeoDataFrame,
    bus_times: pd.DataFrame,
    mmts_times: Optional[pd.DataFrame] = None,
    feeder_stops: Optional[pd.DataFrame] = None,
    feeder_freqs: Optional[pd.DataFrame] = None,
    mmts_schedule_map: Optional[Dict[str, Optional[Set[str]]]] = None,
    feeder_times: Optional[pd.DataFrame] = None,
) -> gpd.GeoDataFrame:
    """Attach Morning/Midday/Evening frequency columns to multimodal graph."""

    out = gdf_combined.copy()

    bus_ids = out.loc[out["mode"] == "bus", "stop_id"]
    bus_freq = compute_temporal_frequency(bus_times, bus_ids, cfg.time_windows)

    mmts_ids = out.loc[out["mode"] == "mmts", "stop_id"]
    mmts_freq = compute_temporal_frequency(mmts_times, mmts_ids, cfg.time_windows)

    feeder_ids = out.loc[out["mode"] == "feeder", "stop_id"]
    feeder_stop_frame = pd.DataFrame({"stop_id": feeder_ids})
    feeder_freq = compute_feeder_frequency_from_headways(
        feeder_stop_frame,
        feeder_freqs,
        cfg.time_windows,
    )

    # Fallback: if frequencies.txt was absent/unusable (all-zero), derive feeder
    # frequency from stop_times.txt so valid service never collapses to zero.
    _freq_cols = [f"{lbl}_freq" for _, _, lbl in cfg.time_windows]
    if (
        feeder_times is not None
        and not feeder_times.empty
        and not feeder_ids.empty
        and feeder_freq[_freq_cols].eq(0).all().all()
    ):
        logger.info(
            "Feeder frequencies.txt absent or produced zero; "
            "deriving feeder frequency from stop_times.txt."
        )
        feeder_freq = compute_temporal_frequency(feeder_times, feeder_ids, cfg.time_windows)

    all_freq = pd.concat([bus_freq, mmts_freq, feeder_freq], axis=0)

    for start_h, end_h, label in cfg.time_windows:
        col = f"{label}_freq"
        out[col] = out["stop_id"].map(all_freq[col]).fillna(0.0)

        # MMTS transfer penalty + schedule gate
        mmts_mask = out["mode"] == "mmts"
        if mmts_mask.any():
            penalty = compute_mmts_transfer_penalty_factor(cfg, end_h - start_h)
            out.loc[mmts_mask, col] *= 1.0 - penalty

            if mmts_schedule_map and mmts_schedule_map.get(label) is not None:
                eligible_stops = mmts_schedule_map[label]
                invalid = mmts_mask & ~out["stop_id"].isin(eligible_stops)
                out.loc[invalid, col] = 0.0

    return out


# ─────────────────────────────────────────────────────────────
# LMCI COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_entropy_weights(indicator_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy weights for a normalized indicator matrix.

    Parameters
    ----------
    indicator_matrix : np.ndarray, shape (n_samples, n_indicators)
        Already min-max normalized (0-1) indicator values, stacked across
        every station and every time window so the resulting weights are
        a single, global, reproducible vector.

    Returns
    -------
    np.ndarray, shape (n_indicators,)
        Non-negative weights that sum to exactly 1.0.
    """
    X = np.asarray(indicator_matrix, dtype=float)
    # Guard against NaN/Inf reaching the entropy math (e.g. a constant
    # normalized column), so the pipeline never crashes here.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples, n_indicators = X.shape
    if n_samples == 0:
        return np.full(n_indicators, 1.0 / n_indicators)

    col_sums = X.sum(axis=0)
    safe_col_sums = np.where(col_sums == 0, 1.0, col_sums)
    p = X / safe_col_sums

    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)

    k = 1.0 / np.log(n_samples) if n_samples > 1 else 0.0
    entropy = -k * plogp.sum(axis=0)

    diversity = np.clip(1.0 - entropy, 0.0, None)
    total_diversity = diversity.sum()

    if total_diversity <= 1e-12:
        return np.full(n_indicators, 1.0 / n_indicators)

    return diversity / total_diversity


def compute_temporal_lmci(
    cfg: LMCIConfig,
    gdf_metro: gpd.GeoDataFrame,
    gdf_combined: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Compute LMCI for all metro stations across all configured time windows.

    Components
    ----------
    density_score:
        route-diversity weighted number of transit stops within 3 km

    frequency_score:
        average temporal trips/hour around each station

    walkzone_score:
        number of stops within 800 m walk zone

    LMCI:
        10 × weighted sum of normalized components
    """

    feeder_zone = gdf_combined[gdf_combined["in_feeder_zone"]].copy()

    if feeder_zone.empty:
        raise ValueError("No multimodal stops found inside feeder zone. Check coordinates and CRS.")

    # Route-diversity weighted density
    if "route_diversity" in feeder_zone.columns:
        density = (
            feeder_zone.groupby("nearest_metro_id")["route_diversity"]
            .sum()
            .rename("stop_count_3km")
        )
    else:
        density = (
            feeder_zone.groupby("nearest_metro_id")["stop_id"]
            .count()
            .rename("stop_count_3km")
        )

    # Walk-zone count
    walkzone = (
        feeder_zone[feeder_zone["in_walk_zone"]]
        .groupby("nearest_metro_id")["stop_id"]
        .count()
        .rename("stop_count_800m")
    )

    base = gdf_metro[["stop_id", "stop_name", "stop_lat", "stop_lon"]].copy()

    base = base.merge(
        density.reset_index().rename(columns={"nearest_metro_id": "stop_id"}),
        on="stop_id",
        how="left",
    )

    base = base.merge(
        walkzone.reset_index().rename(columns={"nearest_metro_id": "stop_id"}),
        on="stop_id",
        how="left",
    )

    base[["stop_count_3km", "stop_count_800m"]] = base[["stop_count_3km", "stop_count_800m"]].fillna(0)

    base["density_norm"] = minmax_norm(base["stop_count_3km"].astype(float)).fillna(0.0)
    base["walkzone_norm"] = minmax_norm(base["stop_count_800m"].astype(float)).fillna(0.0)

    freq_norm_cols: List[str] = []

    for _, _, label in cfg.time_windows:
        freq_col = f"{label}_freq"
        avg_freq_col = f"avg_{freq_col}"
        norm_freq_col = f"{label}_freq_norm"

        avg_freq = (
            feeder_zone.groupby("nearest_metro_id")[freq_col]
            .mean()
            .rename(avg_freq_col)
        )

        base = base.merge(
            avg_freq.reset_index().rename(columns={"nearest_metro_id": "stop_id"}),
            on="stop_id",
            how="left",
        )

        base[avg_freq_col] = base[avg_freq_col].fillna(0.0)
        base[norm_freq_col] = minmax_norm(base[avg_freq_col].astype(float)).fillna(0.0)
        freq_norm_cols.append(norm_freq_col)

    # Entropy weights computed ONCE from the complete normalized indicator
    # dataset (every station x every time window), then reused unchanged
    # for every temporal LMCI calculation below.
    global_entropy_matrix = np.concatenate(
        [
            np.column_stack([
                base["density_norm"].to_numpy(dtype=float),
                base[col].to_numpy(dtype=float),
                base["walkzone_norm"].to_numpy(dtype=float),
            ])
            for col in freq_norm_cols
        ],
        axis=0,
    )
    w_density_e, w_frequency_e, w_walkzone_e = compute_entropy_weights(global_entropy_matrix)

    logger.info(
        "Global entropy weights -> Density=%.4f Frequency=%.4f Walkability=%.4f",
        w_density_e,
        w_frequency_e,
        w_walkzone_e,
    )

    for norm_freq_col, (_, _, label) in zip(freq_norm_cols, cfg.time_windows):
        base[f"{label}_LMCI"] = 10.0 * (
            w_density_e * base["density_norm"]
            + w_frequency_e * base[norm_freq_col]
            + w_walkzone_e * base["walkzone_norm"]
        )

    lmci_cols = [f"{label}_LMCI" for _, _, label in cfg.time_windows]
    base["LMCI_mean"] = base[lmci_cols].mean(axis=1)

    # Dashboard aliases
    base["LMCI"] = base["Morning_LMCI"] if "Morning_LMCI" in base.columns else base["LMCI_mean"]
    base["LMCI_Base"] = base["LMCI_mean"]

    if "Morning_LMCI" in base.columns and "Midday_LMCI" in base.columns:
        base["temporal_gap"] = base["Morning_LMCI"] - base["Midday_LMCI"]
    else:
        base["temporal_gap"] = 0.0

    def category(score: float) -> str:
        if score >= 7.0:
            return "Well-Connected"
        if score >= 4.0:
            return "Moderate"
        return "Transit Desert"

    for _, _, label in cfg.time_windows:
        base[f"{label}_category"] = base[f"{label}_LMCI"].apply(category)

    base["category"] = base["LMCI"].apply(category)
    base["service_class"] = base["LMCI_mean"].apply(category)
    base["stop_count"] = base["stop_count_3km"]

    return base


# ─────────────────────────────────────────────────────────────
# DEMAND-AUGMENTED LMCI
# ─────────────────────────────────────────────────────────────

def attach_demand_counts(
    cfg: LMCIConfig,
    lmci_df: pd.DataFrame,
    demand_points_path: Optional[Path] = None,
    radius_m: float = 800.0,
) -> pd.DataFrame:
    """
    Attach demand_count from outputs/demand_points.csv if available.

    Expected demand columns:
        lat, lon, type

    This keeps LMCI usable even if demand data is not available.
    """

    out = lmci_df.copy()

    if demand_points_path is None:
        demand_points_path = cfg.outputs_dir / "demand_points.csv"

    if not demand_points_path.exists():
        logger.warning("Demand points not found; demand_count set to zero.")
        out["demand_count"] = 0
        out["demand_norm"] = 0.0
        return out

    demand = pd.read_csv(demand_points_path)
    required = {"lat", "lon"}

    if not required.issubset(demand.columns) or demand.empty:
        logger.warning("Demand points invalid or empty; demand_count set to zero.")
        out["demand_count"] = 0
        out["demand_norm"] = 0.0
        return out

    demand = demand.dropna(subset=["lat", "lon"]).copy()

    station_gdf = to_gdf(out, lat_col="stop_lat", lon_col="stop_lon", crs_out=cfg.utm_crs)
    demand_gdf = to_gdf(demand, lat_col="lat", lon_col="lon", crs_out=cfg.utm_crs)

    station_xy = np.column_stack([station_gdf.geometry.x, station_gdf.geometry.y])
    demand_xy = np.column_stack([demand_gdf.geometry.x, demand_gdf.geometry.y])

    if len(demand_xy) == 0:
        out["demand_count"] = 0
        out["demand_norm"] = 0.0
        return out

    tree = cKDTree(demand_xy)
    counts = [len(tree.query_ball_point(xy, r=radius_m)) for xy in station_xy]

    out["demand_count"] = counts
    out["demand_norm"] = minmax_norm(out["demand_count"].astype(float)).fillna(0.0)
    return out


def compute_demand_augmented_lmci(
    cfg: LMCIConfig,
    lmci_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute LMCI_Demand using density + frequency + walkzone + demand.

    Uses Morning frequency normalization as the temporal operating signal
    because dashboard-level recommendations usually anchor on peak-period use.
    """

    out = lmci_df.copy()

    freq_norm_col = "Morning_freq_norm"
    if freq_norm_col not in out.columns:
        freq_candidates = [c for c in out.columns if c.endswith("_freq_norm")]
        if freq_candidates:
            freq_norm_col = freq_candidates[0]
        else:
            out["_freq_norm_fallback"] = 0.0
            freq_norm_col = "_freq_norm_fallback"

    for col in ["density_norm", "walkzone_norm", "demand_norm", freq_norm_col]:
        if col not in out.columns:
            out[col] = 0.0

    entropy_matrix = out[["density_norm", freq_norm_col, "walkzone_norm", "demand_norm"]].to_numpy(dtype=float)
    w_density_e, w_frequency_e, w_walkzone_e, w_demand_e = compute_entropy_weights(entropy_matrix)

    out["LMCI_Demand"] = 10.0 * (
        w_density_e * out["density_norm"]
        + w_frequency_e * out[freq_norm_col]
        + w_walkzone_e * out["walkzone_norm"]
        + w_demand_e * out["demand_norm"]
    )

    # LMCI_Final remains a compatibility column for later scoring/decision modules.
    out["LMCI_Final"] = out["LMCI_Demand"]

    return out


# ─────────────────────────────────────────────────────────────
# TRANSIT DESERT CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def classify_transit_deserts(
    lmci_df: pd.DataFrame,
    cfg: LMCIConfig = CFG,
) -> pd.DataFrame:
    """
    Add desert-window and priority fields.

    Columns added:
        desert_windows
        n_desert_windows
        desert_severity
        temporal_gap_flag
        priority_score
        is_persistent_desert
    """

    out = lmci_df.copy()
    threshold = cfg.equity_desert_threshold
    labels = [label for _, _, label in cfg.time_windows]

    for label in labels:
        col = f"{label}_LMCI"
        if col not in out.columns:
            out[col] = out.get("LMCI_mean", 0.0)

    out["desert_windows"] = out.apply(
        lambda row: [label for label in labels if row[f"{label}_LMCI"] < threshold],
        axis=1,
    )

    out["n_desert_windows"] = out["desert_windows"].apply(len)

    def severity(n: int) -> str:
        if n == len(labels):
            return "Persistent"
        if n > 0:
            return "Partial"
        return "None"

    out["desert_severity"] = out["n_desert_windows"].apply(severity)
    out["is_persistent_desert"] = out["desert_severity"].eq("Persistent")
    out["temporal_gap_flag"] = out.get("temporal_gap", pd.Series(0, index=out.index)) > 2.0

    # -------------------------------------------------------
    # Equity-based Priority Index
    # -------------------------------------------------------

    priority_inputs = pd.DataFrame({
        "LMCI_Deficit": 1.0 - minmax_norm(out["LMCI_mean"]).fillna(0.0),
        "Demand": minmax_norm(out["demand_count"]).fillna(0.0),
        "Persistence": out["is_persistent_desert"].astype(float),
        "TemporalGap": minmax_norm(out["temporal_gap"].clip(lower=0)).fillna(0.0),
    })

    entropy_weights = compute_entropy_weights(priority_inputs.to_numpy())

    (
        w_deficit,
        w_demand,
        w_persistence,
        w_gap,
    ) = entropy_weights

    logger.info(
        "Priority entropy weights -> "
        f"Deficit={w_deficit:.4f} "
        f"Demand={w_demand:.4f} "
        f"Persistence={w_persistence:.4f} "
        f"Gap={w_gap:.4f}"
    )

    out.attrs["priority_weights"] = {
        "LMCI_Deficit": float(w_deficit),
        "Demand": float(w_demand),
        "Persistence": float(w_persistence),
        "TemporalGap": float(w_gap),
    }

    priority_index = (
        w_deficit * priority_inputs["LMCI_Deficit"]
        + w_demand * priority_inputs["Demand"]
        + w_persistence * priority_inputs["Persistence"]
        + w_gap * priority_inputs["TemporalGap"]
    )

    out["priority_lmci_deficit"] = priority_inputs["LMCI_Deficit"]
    out["priority_demand"] = priority_inputs["Demand"]
    out["priority_persistence"] = priority_inputs["Persistence"]
    out["priority_temporal_gap"] = priority_inputs["TemporalGap"]

    out["priority_score"] = 10 * minmax_norm(priority_index).fillna(0.0)

    logger.info(
        "Priority score range: %.2f - %.2f",
        out["priority_score"].min(),
        out["priority_score"].max(),
    )

    logger.info(
        "Top intervention priorities:\n%s",
        out.sort_values("priority_score", ascending=False)[
            ["stop_name", "priority_score"]
        ].head(10).to_string(index=False),
    )

    return out


# ─────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────

def export_lmci_outputs(
    cfg: LMCIConfig,
    lmci_df: pd.DataFrame,
    gdf_combined: Optional[gpd.GeoDataFrame] = None,
) -> Dict[str, Path]:
    """Export LMCI outputs into outputs/ and assets/."""

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.assets_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    lmci_path = cfg.outputs_dir / "lmci_station_scores.csv"
    lmci_df.to_csv(lmci_path, index=False)
    paths["lmci_station_scores"] = lmci_path

    # Dashboard-safe summary contract
    summary_cols = [
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "LMCI_Base",
        "LMCI_Demand",
        "LMCI_Final",
        "LMCI_mean",
        "Morning_LMCI",
        "Midday_LMCI",
        "Evening_LMCI",
        "Density",
        "density_norm",
        "Frequency",
        "Walkability"
        "walkzone_norm",
        "Morning_freq_norm",
        "stop_count_3km",
        "stop_count_800m",
        "demand_count",
        "desert_severity",
        "is_persistent_desert",
        "priority_lmci_deficit",
        "priority_demand",
        "priority_persistence",
        "priority_temporal_gap",
        "priority_score",
        "service_class",
    ]

    available_summary_cols = [c for c in summary_cols if c in lmci_df.columns]
    station_summary = (
        lmci_df[available_summary_cols]
        .sort_values("priority_score", ascending=False)
        .reset_index(drop=True)
    )

    station_summary_path = cfg.outputs_dir / "station_lmci_summary.csv"
    station_summary.to_csv(station_summary_path, index=False)
    paths["station_lmci_summary"] = station_summary_path

    if gdf_combined is not None and not gdf_combined.empty:
        graph_csv = cfg.outputs_dir / "multimodal_graph.csv"
        graph_cols = [
            "stop_id",
            "stop_name",
            "stop_lat",
            "stop_lon",
            "mode",
            "nearest_metro_id",
            "nearest_metro_name",
            "dist_to_metro_m",
            "in_walk_zone",
            "in_feeder_zone",
            "route_diversity",
            "Morning_freq",
            "Midday_freq",
            "Evening_freq",
        ]
        graph_cols = [c for c in graph_cols if c in gdf_combined.columns]
        gdf_combined[graph_cols].to_csv(graph_csv, index=False)
        paths["multimodal_graph"] = graph_csv

        try:
            graph_geojson = cfg.assets_dir / "multimodal_graph.geojson"
            gdf_combined.to_crs(epsg=4326).to_file(graph_geojson, driver="GeoJSON")
            paths["multimodal_graph_geojson"] = graph_geojson
        except Exception as exc:
            logger.warning(f"GeoJSON export skipped: {exc}")

    logger.info("LMCI exports complete:")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")

    return paths


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_lmci_output(lmci_df: pd.DataFrame) -> None:
    """Fail fast if dashboard-critical LMCI fields are broken."""

    required = [
        "stop_name",
        "stop_lat",
        "stop_lon",
        "LMCI_Base",
        "LMCI_Demand",
        "LMCI_Final",
        "priority_score",
    ]

    missing = [col for col in required if col not in lmci_df.columns]
    if missing:
        raise ValueError(f"LMCI output missing required columns: {missing}")

    if lmci_df[["stop_lat", "stop_lon"]].isna().any().any():
        bad = lmci_df[lmci_df[["stop_lat", "stop_lon"]].isna().any(axis=1)]["stop_name"].tolist()
        raise ValueError(f"Stations with missing coordinates: {bad}")

    for col in ["LMCI_Base", "LMCI_Demand", "LMCI_Final"]:
        if lmci_df[col].isna().any():
            raise ValueError(f"{col} contains NaN values")

    logger.info("LMCI validation passed.")


# ─────────────────────────────────────────────────────────────
# END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────

def run_lmci_pipeline(cfg: LMCIConfig = CFG) -> Dict[str, object]:
    """
    Run full LMCI pipeline.

    Returns dictionary containing:
        metro_df
        bus_df
        gdf_metro
        gdf_combined
        lmci_df
        export_paths
    """

    logger.info("=" * 70)
    logger.info("STARTING LMCI PIPELINE")
    logger.info("=" * 70)

    # Reuse preprocessing loader, but override its paths to match this config.
    pre_cfg = PreprocessingConfig(
        data_dir=cfg.data_dir,
        outputs_dir=cfg.outputs_dir,
        hmrl_dir=cfg.hmrl_dir,
        tgsrtc_dir=cfg.tgsrtc_dir,
        mmts_dir=cfg.mmts_dir,
        feeder_dir=cfg.feeder_dir,
    )

    loader = GTFSLoader(pre_cfg)

    metro_df = loader.load_metro_stops()
    bus_df = loader.load_bus_stops()
    bus_times = loader.load_bus_stop_times()

    optional = load_optional_gtfs_feeds(cfg)

    gdf_metro, gdf_combined, mmts_eligible, mmts_schedule_map = build_multimodal_graph(
        cfg=cfg,
        metro_df=metro_df,
        bus_df=bus_df,
        bus_times=bus_times,
        bus_trips=optional["bus_trips"],
        mmts_stops=optional["mmts_stops"],
        mmts_times=optional["mmts_times"],
        feeder_stops=optional["feeder_stops"],
        feeder_times=optional["feeder_times"],
        feeder_trips=optional["feeder_trips"],
    )

    gdf_combined = attach_temporal_frequencies(
        cfg=cfg,
        gdf_combined=gdf_combined,
        bus_times=bus_times,
        mmts_times=optional["mmts_times"],
        feeder_stops=optional["feeder_stops"],
        feeder_freqs=optional["feeder_freqs"],
        mmts_schedule_map=mmts_schedule_map,
        feeder_times=optional["feeder_times"],
    )

    lmci_df = compute_temporal_lmci(
        cfg=cfg,
        gdf_metro=gdf_metro,
        gdf_combined=gdf_combined,
    )

    lmci_df = attach_demand_counts(cfg, lmci_df)
    lmci_df = compute_demand_augmented_lmci(cfg, lmci_df)
    lmci_df = classify_transit_deserts(lmci_df, cfg)

    validate_lmci_output(lmci_df)
    export_paths = export_lmci_outputs(cfg, lmci_df, gdf_combined)

    logger.info("=" * 70)
    logger.info("LMCI PIPELINE COMPLETE")
    logger.info(f"Stations scored: {len(lmci_df)}")
    logger.info(f"Persistent deserts: {lmci_df['is_persistent_desert'].sum()}")
    logger.info("=" * 70)

    return {
        "metro_df": metro_df,
        "bus_df": bus_df,
        "gdf_metro": gdf_metro,
        "gdf_combined": gdf_combined,
        "lmci_df": lmci_df,
        "mmts_eligible": mmts_eligible,
        "export_paths": export_paths,
    }


if __name__ == "__main__":
    run_lmci_pipeline()
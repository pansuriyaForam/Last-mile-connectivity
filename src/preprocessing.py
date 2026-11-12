"""
Core preprocessing pipeline for the Hyderabad Metro Last-Mile Connectivity project.

Responsibilities
----------------
1. Load GTFS feeds
2. Validate coordinates
3. Build GeoDataFrames
4. Load POI + school demand signals
5. Extract clean demand points
6. Save all generated artifacts into outputs/

Author: Pansuriya Foram Rasikbhai
"""

from __future__ import annotations
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger("preprocessing")


# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────

@dataclass
class Config:
    """Global configuration for preprocessing pipeline."""

    # Base folders
    data_dir: Path = Path("data")
    outputs_dir: Path = Path("outputs")

    # GTFS folders
    hmrl_dir: Path = Path("data/hmrl")
    tgsrtc_dir: Path = Path("data/tgsrtc")
    mmts_dir: Path = Path("data/mmts")
    feeder_dir: Path = Path("data/feeder")

    # Demand signal files
    poi_path: Path = Path("data/external/poi_business.geojson")
    school_path: Path = Path("data/external/Affordable_Schools.geojson")

    # Coordinate reference systems
    wgs84_crs: int = 4326
    metric_crs: int = 32644

    # Metro influence corridor
    metro_influence_corridor_m: float = 3000.0

    # Hyderabad bbox
    bbox_lat_min: float = 16.8
    bbox_lat_max: float = 17.8
    bbox_lon_min: float = 77.5
    bbox_lon_max: float = 79.5

    def __post_init__(self):
        if self.metro_influence_corridor_m <= 0:
            raise ValueError(
                "metro_influence_corridor_m must be positive."
            )

CFG = Config()


# ──────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────

EARTH_RADIUS_M = 6_371_000.0


def minmax_norm(series: pd.Series) -> pd.Series:
    """Min-max normalization."""

    s = series.astype(float)

    if s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - s.min()) / (s.max() - s.min())

def to_gdf(
        df: pd.DataFrame,
        lat_col: str = "lat",
        lon_col: str = "lon",
        crs_in: int = CFG.wgs84_crs,
        crs_out: int = CFG.wgs84_crs,
) -> gpd.GeoDataFrame:
    """Convert dataframe to GeoDataFrame."""

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=f"EPSG:{crs_in}"
    )

    if crs_out != crs_in:
        gdf = gdf.to_crs(epsg=crs_out)

    return gdf

"""
    Research Note:
    The demand layer is intentionally restricted to destinations within the
    Metro Influence Corridor because locations outside this corridor cannot
    reasonably benefit from metro-based last-mile interventions. This keeps
    the optimization problem focused on realistic accessibility improvements.
"""

def filter_demand_within_metro_corridor(
    demand_df: pd.DataFrame,
    metro_df: pd.DataFrame,
    cfg: Config = CFG,
) -> pd.DataFrame:
    """
    Restrict demand points to the Metro Influence Corridor.

    The corridor is defined as the union of 3 km buffers around all
    Hyderabad Metro stations.

    This ensures the optimization problem focuses only on destinations
    that can realistically benefit from metro-based last-mile
    improvements.
    """

    if demand_df.empty:
        return demand_df

    if metro_df.empty:
        logger.warning(
            "Metro station dataset is empty. Corridor filtering skipped."
        )
        return demand_df

    metro = to_gdf(
        metro_df,
        lat_col="stop_lat",
        lon_col="stop_lon",
        crs_out=cfg.metric_crs,
    )

    demand = to_gdf(
        demand_df,
        lat_col="lat",
        lon_col="lon",
        crs_out=cfg.metric_crs,
    )

    buffered = metro.buffer(cfg.metro_influence_corridor_m)

    try:
        corridor = buffered.union_all()
    except AttributeError:
        corridor = buffered.unary_union

    mask = demand.geometry.within(corridor)

    filtered = demand.loc[mask].copy()

    filtered = filtered.to_crs(epsg=cfg.wgs84_crs)

    filtered["lat"] = filtered.geometry.y
    filtered["lon"] = filtered.geometry.x

    logger.info(
        "Demand composition after corridor filtering:\n"
        f"  Business: {(filtered['type'] == 'business').sum()}\n"
        f"  School: {(filtered['type'] == 'school').sum()}"
    )

    return pd.DataFrame(filtered.drop(columns="geometry"))


def haversine_matrix(
    lat_a: Sequence[float],
    lon_a: Sequence[float],
    lat_b: Sequence[float],
    lon_b: Sequence[float],
) -> np.ndarray:
    """Vectorised haversine distance matrix in metres."""

    la = np.deg2rad(np.asarray(lat_a, dtype=float)[:, np.newaxis])
    lo = np.deg2rad(np.asarray(lon_a, dtype=float)[:, np.newaxis])

    lb = np.deg2rad(np.asarray(lat_b, dtype=float)[np.newaxis, :])
    mb = np.deg2rad(np.asarray(lon_b, dtype=float)[np.newaxis, :])

    dlat = lb - la
    dlon = mb - lo

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(la) * np.cos(lb) * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


# ──────────────────────────────────────────────────────────
# GTFS LOADER
# ──────────────────────────────────────────────────────────

class GTFSLoader:
    """Load and validate GTFS feeds."""

    HMRL_FILES = ["stops.txt"]
    TGSRTC_FILES = [
        "stops.txt",
        "stop_times.txt",
        "trips.txt",
        "routes.txt"
    ]

    MMTS_FILES = [
        "stops.txt",
        "stop_times.txt"
    ]

    FEEDER_FILES = [
        "stops.txt",
        "stop_times.txt",
        "frequencies.txt"
    ]

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg

    def _require_files(self, directory: Path, files: List[str]):
        missing = [f for f in files if not (directory / f).exists()]

        if missing:
            raise FileNotFoundError(
                f"Missing GTFS files in {directory}: {missing}"
            )

    def _load_csv(self, path: Path, **kwargs: Any) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False, **kwargs)
        df.columns = df.columns.str.strip().str.lower()

        logger.info(f"Loaded {path.name}: {len(df):,} rows")

        return df

    def _coerce_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"stop_lat", "stop_lon"}

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing coordinate columns: {sorted(missing)}"
            )

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        before = len(df)

        df = df.dropna(subset=["stop_lat", "stop_lon"])

        dropped = before - len(df)

        if dropped:
            logger.warning(
                "Dropped %d rows with invalid coordinates",
                dropped,
            )

        return df

    def _bbox_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (
            df["stop_lat"].between(
                self.cfg.bbox_lat_min,
                self.cfg.bbox_lat_max
            )
            &
            df["stop_lon"].between(
                self.cfg.bbox_lon_min,
                self.cfg.bbox_lon_max
            )
        )

        return df[mask].reset_index(drop=True)

    def load_metro_stops(self) -> pd.DataFrame:
        """
        Load ALL Hyderabad Metro stations.

        Important:
        ------------
        GTFS contains:

        location_type = 1  → actual station
        location_type = 0  → platforms

        We ONLY keep parent stations to avoid:
        - duplicated plotting
        - duplicated demand mapping
        - noisy graphs
        - incorrect accessibility calculations
        """

        self._require_files(
            self.cfg.hmrl_dir,
            self.HMRL_FILES
        )

        df = self._load_csv(
            self.cfg.hmrl_dir / "stops.txt"
        )

        df = self._coerce_coords(df)

        # Ensure location_type exists
        if "location_type" not in df.columns:
            raise ValueError(
                "location_type column missing in metro GTFS"
            )

        df["location_type"] = pd.to_numeric(
            df["location_type"],
            errors="coerce"
        ).fillna(0)

        # KEEP ONLY PARENT STATIONS
        # location_type = 1 → actual stations
        metro = df[df["location_type"] == 1].copy()

        # Remove helper infrastructure names
        metro = metro[
            ~metro["stop_name"].str.contains(
                r"Arm|Lift|Escalator|Staircase",
                case=False,
                na=False
            )
        ]

        metro = self._bbox_filter(metro)

        # Remove duplicate station names if any
        metro = metro.drop_duplicates(
            subset=["stop_id"]
        ).reset_index(drop=True)

        logger.info(
            f"Total Hyderabad Metro stations loaded: {len(metro)}"
        )

        return metro[
            [
                "stop_id",
                "stop_name",
                "stop_lat",
                "stop_lon",
            ]
        ].reset_index(drop=True)

    def load_bus_stops(self) -> pd.DataFrame:
        """Load TGSRTC stops."""

        self._require_files(
            self.cfg.tgsrtc_dir,
            ["stops.txt"]
        )

        df = self._load_csv(
            self.cfg.tgsrtc_dir / "stops.txt"
        )

        df = self._coerce_coords(df)
        df = self._bbox_filter(df)

        return df[
            ["stop_id", "stop_name", "stop_lat", "stop_lon"]
        ].reset_index(drop=True)

    def load_bus_stop_times(self) -> pd.DataFrame:
        self._require_files(
            self.cfg.tgsrtc_dir,
            ["stop_times.txt"]
        )

        return self._load_csv(
            self.cfg.tgsrtc_dir / "stop_times.txt"
        )


# ──────────────────────────────────────────────────────────
# DEMAND SIGNAL EXTRACTION
# ──────────────────────────────────────────────────────────


def extract_latlon(
    gdf: gpd.GeoDataFrame,
    label: str,
    cfg: Config = CFG,
) -> pd.DataFrame:
    """
    Extract lat/lon from GeoJSON.

    GeoJSON:
        geometry.x = lon
        geometry.y = lat
    """

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=cfg.wgs84_crs)

    elif gdf.crs.to_epsg() != cfg.wgs84_crs:
        gdf = gdf.to_crs(epsg=cfg.wgs84_crs)

    df = pd.DataFrame({
        "lon": gdf.geometry.x,
        "lat": gdf.geometry.y,
        "type": label,
    })

    before_null = len(df)

    df = df.dropna(subset=["lat", "lon"])

    dropped_null = before_null - len(df)

    before_bbox = len(df)

    mask = (
            df["lat"].between(cfg.bbox_lat_min, cfg.bbox_lat_max)
            &
            df["lon"].between(cfg.bbox_lon_min, cfg.bbox_lon_max)
    )

    df = df[mask].reset_index(drop=True)

    dropped_bbox = before_bbox - len(df)

    logger.info(
        f"[{label}] "
        f"null_dropped={dropped_null} "
        f"bbox_dropped={dropped_bbox} "
        f"kept={len(df)}"
    )

    return df


# ──────────────────────────────────────────────────────────
# LOAD DEMAND POINTS
# ──────────────────────────────────────────────────────────


def load_demand_points(
    cfg: Config = CFG,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Load POI + school demand signals.

    Returns
    -------
    DataFrame:
        lat | lon | type
    """

    logger.info("Loading demand signal layers...")

    gdf_business = gpd.read_file(cfg.poi_path)
    gdf_schools = gpd.read_file(cfg.school_path)

    logger.info(
        f"Business POIs: {len(gdf_business):,}"
    )

    logger.info(
        f"Affordable schools: {len(gdf_schools):,}"
    )

    business_df = extract_latlon(
        gdf_business,
        "business"
    )

    school_df = extract_latlon(
        gdf_schools,
        "school"
    )

    demand_points_df = pd.concat(
        [business_df, school_df],
        ignore_index=True,
    )

    before = len(demand_points_df)

    demand_points_df = (
        demand_points_df
        .drop_duplicates(subset=["lat", "lon", "type"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Removed {before - len(demand_points_df)} duplicate demand points."
    )

    loader = GTFSLoader(cfg)
    metro_df = loader.load_metro_stops()

    demand_points_df = filter_demand_within_metro_corridor(
        demand_points_df,
        metro_df,
        cfg,
    )

    demand_points_df = demand_points_df[
        demand_points_df["lat"].between(-90, 90)
        &
        demand_points_df["lon"].between(-180, 180)
        ].reset_index(drop=True)

    logger.info(
        f"Final demand points: {len(demand_points_df):,}"
    )

    if save_output:
        cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

        demand_points_df["source"] = demand_points_df["type"]

        demand_points_df["inside_metro_corridor"] = True

        out_csv = cfg.outputs_dir / "demand_points.csv"

        demand_points_df.to_csv(out_csv, index=False)

        logger.info(f"Saved demand points → {out_csv}")

    return demand_points_df


# ──────────────────────────────────────────────────────────
# STATION EXPORT
# ──────────────────────────────────────────────────────────


def export_station_coordinates(
    station_df: pd.DataFrame,
    output_name: str = "station_coordinates.csv",
    cfg: Config = CFG,
):
    """Save station coordinate reference."""

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    out_path = cfg.outputs_dir / output_name

    station_df.to_csv(out_path, index=False)

    logger.info(f"Saved station coordinates → {out_path}")


# ──────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────


def run_preprocessing_pipeline(cfg: Config = CFG):
    """Run complete preprocessing pipeline."""

    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    loader = GTFSLoader(cfg)

    # Metro stations
    metro_stops = loader.load_metro_stops()

    # Bus stops
    bus_stops = loader.load_bus_stops()

    # Demand points
    demand_points = load_demand_points(cfg)

    # Export station coordinates
    export_station_coordinates(metro_stops)

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)

    return {
        "metro_stops": metro_stops,
        "bus_stops": bus_stops,
        "demand_points": demand_points,
    }


# ──────────────────────────────────────────────────────────
# SCRIPT ENTRY
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing_pipeline()
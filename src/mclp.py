"""
src/mclp.py
────────────────────────────────────────────────────────────
MCLP engine for the Hyderabad Metro Last-Mile Connectivity project.

MCLP = Maximal Covering Location Problem

Purpose
-------
Select the best metro stations / intervention locations that maximize
weighted demand coverage within a service radius.

This module is intentionally deterministic and explainable:
- no solver dependency
- greedy approximation
- dashboard-safe CSV exports
- reproducible ranked recommendations

Inputs expected from previous modules
-------------------------------------
outputs/station_lmci_summary.csv
outputs/lmci_station_scores.csv
outputs/demand_points.csv
outputs/lmci_entropy_weights.csv (optional, for weighted opportunity scoring)

Exports
-------
outputs/mclp_selected_stations.csv
outputs/mclp_coverage_by_k.csv
outputs/mclp_demand_assignment.csv
outputs/mclp_candidate_scores.csv
outputs/ablation_results.csv
assets/mclp_selected_stations.geojson
assets/mclp_demand_assignment.geojson
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from src.preprocessing import minmax_norm, to_gdf
except ImportError:
    from preprocessing import minmax_norm, to_gdf


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("MCLP")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class MCLPConfig:
    """Configuration for MCLP optimization."""

    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")

    station_scores_path: Path = Path("outputs/station_lmci_summary.csv")
    fallback_station_scores_path: Path = Path("outputs/lmci_station_scores.csv")
    demand_points_path: Path = Path("outputs/demand_points.csv")

    # CRS
    wgs84_crs: int = 4326
    metric_crs: int = 32644

    # Coverage assumptions
    coverage_radius_m: float = 800.0
    detour_factor: float = 1.35

    # Optimization defaults
    max_k: int = 10
    default_k: int = 10

    # Demand weighting
    business_weight: float = 1.00
    school_weight: float = 1.25
    default_demand_weight: float = 1.00

    # Candidate prioritization
    underserved_bonus: float = 1.20

    # If true, candidate score considers both uncovered demand and LMCI deficit.
    use_equity_weighting: bool = True

    def __post_init__(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


CFG = MCLPConfig()


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def load_station_scores(cfg: MCLPConfig = CFG) -> pd.DataFrame:
    """
    Load station-level LMCI outputs.

    Tries primary path first, then fallback path.
    Ensures required columns exist and coordinates are valid.
    """
    if cfg.station_scores_path.exists():
        path = cfg.station_scores_path
    elif cfg.fallback_station_scores_path.exists():
        path = cfg.fallback_station_scores_path
    else:
        raise FileNotFoundError(
            "No station LMCI file found. Run src/lmci.py first. Expected one of: "
            f"{cfg.station_scores_path}, {cfg.fallback_station_scores_path}"
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Station score file missing columns: {missing}")

    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
    df = df.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)

    logger.info(f"Loaded station scores: {len(df):,} stations from {path}")
    return df


def load_demand_points(cfg: MCLPConfig = CFG) -> pd.DataFrame:
    """
    Load demand points created by preprocessing.py.

    Validates required columns, cleans coordinates, and adds a unique demand_id.
    """
    if not cfg.demand_points_path.exists():
        raise FileNotFoundError(
            f"Demand points not found: {cfg.demand_points_path}. "
            "Run src/preprocessing.py first."
        )

    df = pd.read_csv(cfg.demand_points_path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"lat", "lon"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Demand file missing required columns: {missing}")

    if "type" not in df.columns:
        df["type"] = "unknown"

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Demand file is empty after coordinate cleaning.")

    df["demand_id"] = np.arange(len(df), dtype=int)

    logger.info(f"Loaded demand points: {len(df):,}")
    return df


# ─────────────────────────────────────────────────────────────
# DEMAND WEIGHTING
# ─────────────────────────────────────────────────────────────

def assign_demand_weights(
    demand_df: pd.DataFrame,
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """
    Assign weights to demand points by type.

    Schools get higher weight (1.25) to prioritize educational access.
    Business/POI get base weight (1.00).
    Unknown types get default weight (1.00).
    """
    out = demand_df.copy()

    def weight_for_type(value: str) -> float:
        text = str(value).lower().strip()

        if "school" in text:
            return cfg.school_weight
        if "business" in text or "poi" in text:
            return cfg.business_weight
        return cfg.default_demand_weight

    out["base_weight"] = out["type"].apply(weight_for_type)
    out["demand_weight"] = out["base_weight"].astype(float)

    return out


def load_lmci_weights(cfg: MCLPConfig = CFG) -> Dict[str, float]:
    """
    Load entropy weights from LMCI analysis if available.

    These weights represent the relative importance of different
    accessibility components (density, frequency, walkability)
    as determined by the entropy weighting in lmci.py.

    Falls back to default weights if lmci_entropy_weights.csv not found.
    """
    weights_path = cfg.outputs_dir / "lmci_entropy_weights.csv"

    if weights_path.exists():
        try:
            weights_df = pd.read_csv(weights_path)
            # Assuming columns: variable, weight
            weights = dict(zip(weights_df["variable"], weights_df["weight"]))
            logger.info(f"Loaded LMCI weights from {weights_path}")
            return weights
        except Exception as e:
            logger.warning(f"Could not load LMCI weights: {e}. Using defaults.")

    # Fallback weights based on typical LMCI analysis
    return {
        "density_norm": 0.3524,
        "morning_freq_norm": 0.3265,
        "walkzone_norm": 0.3211
    }


def infer_opportunity_score(stations_df: pd.DataFrame, cfg: MCLPConfig = CFG) -> pd.Series:
    """
    Estimate Accessibility Opportunity using LMCI entropy weights.

    Opportunity measures how much existing surrounding infrastructure
    can amplify a future intervention. Stations with high existing
    infrastructure have higher opportunity scores.

    Uses entropy-weighted combination of:
        density_norm - surrounding population/workplace density
        walkzone_norm - walkable area coverage
        morning_freq_norm - morning peak frequency

    The weights are loaded from lmci_entropy_weights.csv if available,
    otherwise using default weights that sum to approximately 1.0.

    Returns
    -------
    pd.Series scaled between 0.5 and 1.5
    """
    cols = [
        "density_norm",
        "walkzone_norm",
        "morning_freq_norm",
    ]

    available = [c for c in cols if c in stations_df.columns]

    if not available:
        logger.warning(
            "No opportunity columns found. Using neutral opportunity score."
        )
        return pd.Series(1.0, index=stations_df.index)

    # Load LMCI weights or use defaults
    weights = load_lmci_weights(cfg)

    # Calculate weighted opportunity score
    opportunity = pd.Series(0.0, index=stations_df.index)
    valid_weights = 0

    for col in available:
        if col in weights:
            weight = weights[col]
            opportunity += stations_df[col].fillna(0) * weight
            valid_weights += weight
        else:
            # Use equal weight if variable not in weights dict
            logger.debug(f"No weight found for {col}, using equal weight")
            opportunity += stations_df[col].fillna(0) * (1.0 / len(available))
            valid_weights += (1.0 / len(available))

    # Normalize by sum of weights used
    if valid_weights > 0:
        opportunity = opportunity / valid_weights

    # Scale to 0.5-1.5 range (slight amplification)
    return 0.5 + opportunity


def enrich_station_candidates(
    stations_df: pd.DataFrame,
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """
    Add candidate-level equity weights for intervention prioritization.

    This function computes:
    1. need_score - how much a station needs intervention (based on LMCI deficit)
    2. opportunity_score - how much existing infrastructure amplifies intervention
    3. candidate_equity_weight - combined need × opportunity

    Priority score is computed in `lmci.py` and represents intervention need derived from LMCI deficit and accessibility indicators.

    Lower LMCI means higher intervention need. Persistent deserts get
    an additional policy-priority multiplier via the underserved_bonus.
    """
    out = stations_df.copy()

    # ------------------------
    # Priority Score
    # ------------------------

    if "priority_score" not in out.columns:
        logger.warning("priority_score not found. Using neutral priority.")
        out["priority_score"] = 5.0

    out["priority_score"] = pd.to_numeric(
        out["priority_score"],
        errors="coerce"
    ).fillna(5.0)

    # Normalize 0–1 for consistent scaling
    out["priority_norm"] = minmax_norm(out["priority_score"])

    # Handle desert flags if present
    if "is_persistent_desert" in out.columns:
        out["is_persistent_desert"] = out["is_persistent_desert"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        out["is_persistent_desert"] = False

    if "desert_severity" not in out.columns:
        out["desert_severity"] = np.where(out["is_persistent_desert"], "Persistent", "None")

    if cfg.use_equity_weighting:
        # need_score combines LMCI deficit, demand, persistence, and connectivity gap
        # Already encoded in priority_score from lmci.py
        out["need_score"] = (
                1.0 +
                out["priority_norm"] * (cfg.underserved_bonus - 1.0)
        )

        # opportunity_score uses entropy-weighted infrastructure metrics
        out["opportunity_score"] = infer_opportunity_score(out, cfg)

        # candidate_equity_weight combines need and opportunity for balanced selection
        out["candidate_equity_weight"] = (
                out["need_score"]
                * out["opportunity_score"]
        )

    else:
        # Neutral weights when equity weighting is disabled
        out["need_score"] = 1.0
        out["opportunity_score"] = 1.0
        out["candidate_equity_weight"] = 1.0

    return out


# ─────────────────────────────────────────────────────────────
# COVERAGE MATRIX
# ─────────────────────────────────────────────────────────────

def build_coverage_matrix(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    cfg: MCLPConfig = CFG,
) -> Tuple[List[Set[int]], np.ndarray, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build station-to-demand coverage sets using spatial indexing.

    Uses cKDTree for efficient spatial querying. Network distances
    are approximated by applying detour_factor to Euclidean distances.

    Returns
    -------
    coverage_sets:
        list where coverage_sets[i] contains demand_id values covered by station i

    distance_matrix_m:
        approximate network distances from station i to demand j

    station_gdf:
        projected station GeoDataFrame

    demand_gdf:
        projected demand GeoDataFrame
    """
    station_gdf = to_gdf(
        stations_df,
        lat_col="stop_lat",
        lon_col="stop_lon",
        crs_out=cfg.metric_crs,
    )

    demand_gdf = to_gdf(
        demand_df,
        lat_col="lat",
        lon_col="lon",
        crs_out=cfg.metric_crs,
    )

    station_xy = np.column_stack([station_gdf.geometry.x, station_gdf.geometry.y])
    demand_xy = np.column_stack([demand_gdf.geometry.x, demand_gdf.geometry.y])

    demand_tree = cKDTree(demand_xy)

    # Adjust radius for detour factor (network distance = Euclidean × detour_factor)
    euclidean_radius = cfg.coverage_radius_m / max(cfg.detour_factor, 1e-9)

    coverage_sets: List[Set[int]] = []

    for xy in station_xy:
        covered_idx = demand_tree.query_ball_point(xy, r=euclidean_radius)
        covered_ids = set(demand_df.iloc[covered_idx]["demand_id"].astype(int).tolist())
        coverage_sets.append(covered_ids)

    # Full matrix for assignment export
    diff = station_xy[:, None, :] - demand_xy[None, :, :]
    distance_matrix_m = np.sqrt((diff ** 2).sum(axis=2)) * cfg.detour_factor

    logger.info(
        f"Coverage matrix built: stations={len(stations_df)}, demand={len(demand_df)}, "
        f"radius={cfg.coverage_radius_m:.0f}m network-adjusted"
    )

    return coverage_sets, distance_matrix_m, station_gdf, demand_gdf


# ─────────────────────────────────────────────────────────────
# GREEDY MCLP
# ─────────────────────────────────────────────────────────────

def _weighted_coverage_value(
    demand_ids: Set[int],
    demand_weight_map: Dict[int, float],
) -> float:
    """Calculate total weighted demand for a set of demand IDs."""
    return float(sum(demand_weight_map.get(i, 0.0) for i in demand_ids))


def greedy_mclp(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    coverage_sets: Sequence[Set[int]],
    k: int,
    cfg: MCLPConfig = CFG,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Greedy MCLP approximation for selecting optimal intervention locations.

    At every step, select the station with the largest marginal gain in
    uncovered weighted demand.

    Tie-breaking criteria (in priority order):
    1. higher equity-weighted marginal gain (need × opportunity)
    2. higher raw marginal gain (unweighted demand coverage)
    3. higher priority_score (LMCI deficit, from lmci.py)
    4. station name alphabetical (deterministic)

    The verbose parameter controls logging output. Set to False for
    batch processing to avoid cluttering the console.

    Returns
    -------
    selected_df:
        DataFrame with selected stations and their coverage metrics

    summary:
        Dictionary with overall coverage statistics
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if len(stations_df) != len(coverage_sets):
        raise ValueError("stations_df and coverage_sets length mismatch.")

    k = min(k, len(stations_df))

    demand_weight_map = dict(
        zip(demand_df["demand_id"].astype(int), demand_df["demand_weight"].astype(float))
    )

    total_weighted_demand = max(float(demand_df["demand_weight"].sum()), 1e-9)

    selected_indices: List[int] = []
    covered: Set[int] = set()
    rows: List[Dict[str, object]] = []

    station_work = stations_df.reset_index(drop=True).copy()

    for rank in range(1, k + 1):
        best_idx: Optional[int] = None
        best_tuple: Optional[Tuple[float, float, float, str]] = None
        best_new_covered: Set[int] = set()

        # Evaluate all unselected stations
        for idx, candidate_set in enumerate(coverage_sets):
            if idx in selected_indices:
                continue

            newly_covered = candidate_set - covered
            raw_gain = _weighted_coverage_value(newly_covered, demand_weight_map)

            equity_weight = float(station_work.loc[idx, "candidate_equity_weight"])
            equity_gain = raw_gain * equity_weight

            priority = float(station_work.loc[idx, "priority_score"])
            station_name = str(station_work.loc[idx, "stop_name"])

            # Python tuple comparison is ascending, so use values where
            # larger is better (equity_gain, raw_gain, priority)
            score_tuple = (
                equity_gain,
                raw_gain,
                priority,
                station_name,
            )

            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best_idx = idx
                best_new_covered = newly_covered

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        covered |= best_new_covered

        cumulative_weighted = _weighted_coverage_value(covered, demand_weight_map)
        marginal_weighted = _weighted_coverage_value(best_new_covered, demand_weight_map)

        row = station_work.loc[best_idx].to_dict()
        row.update({
            "selection_rank": rank,
            "k": k,
            "marginal_demand_points": len(best_new_covered),
            "marginal_weighted_demand": marginal_weighted,
            "cumulative_demand_points": len(covered),
            "cumulative_weighted_demand": cumulative_weighted,
            "coverage_pct": 100.0 * cumulative_weighted / total_weighted_demand,
        })

        rows.append(row)

        # Only log if verbose is True (suppresses logging in coverage_by_k computation)
        if verbose:
            logger.info(
                f"Rank {rank}: {row['stop_name']} | "
                f"marginal={marginal_weighted:.2f} | "
                f"coverage={row['coverage_pct']:.2f}%"
            )

    selected_df = pd.DataFrame(rows)

    summary = {
        "k": k,
        "selected_count": len(selected_df),
        "covered_demand_points": len(covered),
        "total_demand_points": len(demand_df),
        "covered_weighted_demand": _weighted_coverage_value(covered, demand_weight_map),
        "total_weighted_demand": total_weighted_demand,
        "coverage_pct": 100.0 * _weighted_coverage_value(covered, demand_weight_map) / total_weighted_demand,
        "covered_ids": covered,
        "selected_indices": selected_indices,
    }

    return selected_df, summary


# ─────────────────────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────────────────────

def run_mclp_ablation_study(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    coverage_sets: List[Set[int]],
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """
    Evaluate the contribution of equity weighting in MCLP.

    Ablation variants
    -----------------
    1. Full Model:
       Equity weighting enabled (need_score × opportunity_score).

    2. Demand-Only Baseline:
       Equity weighting disabled (candidate_equity_weight = 1.0).

    3. Need-Only Variant:
       Only need_score used (opportunity_score = 1.0).
       Tests whether infrastructure amplification matters.

    4. Opportunity-Only Variant:
       Only opportunity_score used (need_score = 1.0).
       Tests whether equity need matters beyond infrastructure.

    5. No Desert Bonus:
       Equity weighting enabled but underserved_bonus = 1.0.
       Tests the marginal impact of the persistent-desert policy preference.

    The coverage geometry, demand points, demand weights, service
    radius, and candidate locations remain unchanged across all
    variants. Only the candidate scoring mechanism is modified.

    Returns
    -------
    pd.DataFrame
        One row per ablation variant containing:
        - variant name
        - whether equity weighting was used
        - description of the variant
        - number of stations selected (k)
        - selected station names
        - covered demand points (unweighted count)
        - weighted demand covered
        - coverage percentage
        - mean priority score of selected stations
        - persistent deserts selected
        - overlap with Full Model selections (Jaccard similarity)
        - rank correlation with Full Model candidate ranking
    """

    logger.info("=" * 70)
    logger.info("STARTING MCLP ABLATION STUDY")
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────
    # First, run the Full Model to establish the reference
    # ─────────────────────────────────────────────────────────

    full_cfg = replace(cfg, use_equity_weighting=True)
    full_stations = enrich_station_candidates(stations_df.copy(), full_cfg)
    full_selected_df, full_summary = greedy_mclp(
        stations_df=full_stations,
        demand_df=demand_df,
        coverage_sets=coverage_sets,
        k=cfg.default_k,
        cfg=full_cfg,
    )
    full_selected_names = set(full_selected_df["stop_name"].astype(str).tolist())

    # Build full candidate ranking for correlation analysis
    full_candidates = build_candidate_scores(
        stations_df=full_stations,
        demand_df=demand_df,
        coverage_sets=coverage_sets,
        cfg=full_cfg,
    )
    full_ranking = full_candidates.set_index("stop_id")["candidate_rank"].to_dict()

    # ─────────────────────────────────────────────────────────
    # Define all ablation variants
    # ─────────────────────────────────────────────────────────

    variants = [
        {
            "variant_name": "Full Model",
            "use_equity": True,
            "underserved_bonus": cfg.underserved_bonus,
            "description": "Equity weighting enabled (need × opportunity) with desert bonus",
        },
        {
            "variant_name": "Demand-Only Baseline",
            "use_equity": False,
            "underserved_bonus": cfg.underserved_bonus,
            "description": "Equity weighting disabled — pure coverage maximization",
        },
        {
            "variant_name": "Need-Only (No Opportunity)",
            "use_equity": True,
            "underserved_bonus": cfg.underserved_bonus,
            "description": "Only need_score used; opportunity_score = 1.0",
            "override_opportunity": True,
        },
        {
            "variant_name": "Opportunity-Only (No Need)",
            "use_equity": True,
            "underserved_bonus": cfg.underserved_bonus,
            "description": "Only opportunity_score used; need_score = 1.0",
            "override_need": True,
        },
        {
            "variant_name": "No Desert Bonus",
            "use_equity": True,
            "underserved_bonus": 1.0,
            "description": "Equity weighting enabled but persistent-desert bonus removed",
        },
    ]

    results = []

    for variant in variants:

        variant_name = variant["variant_name"]
        use_equity = variant["use_equity"]
        bonus = variant["underserved_bonus"]
        description = variant["description"]
        override_opportunity = variant.get("override_opportunity", False)
        override_need = variant.get("override_need", False)

        logger.info(f"Running ablation variant: {variant_name}")

        # Create isolated configuration
        variant_cfg = replace(
            cfg,
            use_equity_weighting=use_equity,
            underserved_bonus=bonus,
        )

        # Enrich candidates with variant configuration
        variant_stations = enrich_station_candidates(
            stations_df.copy(),
            variant_cfg,
        )

        # Apply overrides if specified
        if override_opportunity:
            variant_stations["opportunity_score"] = 1.0
            variant_stations["candidate_equity_weight"] = (
                variant_stations["need_score"] * variant_stations["opportunity_score"]
            )
            logger.info("  → Overriding opportunity_score to 1.0")

        if override_need:
            variant_stations["need_score"] = 1.0
            variant_stations["candidate_equity_weight"] = (
                variant_stations["need_score"] * variant_stations["opportunity_score"]
            )
            logger.info("  → Overriding need_score to 1.0")

        # Run greedy MCLP
        selected_df, summary = greedy_mclp(
            stations_df=variant_stations,
            demand_df=demand_df,
            coverage_sets=coverage_sets,
            k=variant_cfg.default_k,
            cfg=variant_cfg,
        )

        # ─────────────────────────────────────────────────────
        # Extract metrics
        # ─────────────────────────────────────────────────────

        selected_names = selected_df["stop_name"].astype(str).tolist()
        selected_names_set = set(selected_names)

        # Jaccard similarity with Full Model
        if len(full_selected_names | selected_names_set) > 0:
            jaccard_similarity = len(full_selected_names & selected_names_set) / len(full_selected_names | selected_names_set)
        else:
            jaccard_similarity = 0.0

        # Count how many Full Model stations are also selected
        overlap_count = len(full_selected_names & selected_names_set)

        # Persistent deserts selected
        persistent_selected = 0
        if "is_persistent_desert" in selected_df.columns:
            persistent_selected = int(
                selected_df["is_persistent_desert"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .sum()
            )

        # Mean priority of selected stations
        mean_priority = np.nan
        if "priority_score" in selected_df.columns:
            mean_priority = pd.to_numeric(
                selected_df["priority_score"],
                errors="coerce",
            ).mean()

        # Build variant candidate ranking for correlation
        variant_candidates = build_candidate_scores(
            stations_df=variant_stations,
            demand_df=demand_df,
            coverage_sets=coverage_sets,
            cfg=variant_cfg,
        )
        variant_ranking = variant_candidates.set_index("stop_id")["candidate_rank"].to_dict()

        # Spearman rank correlation with Full Model
        common_ids = set(full_ranking.keys()) & set(variant_ranking.keys())
        if len(common_ids) > 2:
            full_ranks = [full_ranking[sid] for sid in common_ids]
            variant_ranks = [variant_ranking[sid] for sid in common_ids]
            from scipy.stats import spearmanr
            rank_corr, _ = spearmanr(full_ranks, variant_ranks)
        else:
            rank_corr = np.nan

        # Weighted demand covered
        weighted_coverage = float(
            summary.get("covered_weighted_demand", 0.0)
        )

        # Coverage percentage
        coverage_pct = float(summary.get("coverage_pct", 0.0))

        # Number of demand points covered
        covered_points = int(summary.get("covered_demand_points", 0))

        # Mean candidate_equity_weight of selected stations
        mean_equity_weight = np.nan
        if "candidate_equity_weight" in selected_df.columns:
            mean_equity_weight = pd.to_numeric(
                selected_df["candidate_equity_weight"],
                errors="coerce",
            ).mean()

        # Mean need_score of selected stations
        mean_need = np.nan
        if "need_score" in selected_df.columns:
            mean_need = pd.to_numeric(
                selected_df["need_score"],
                errors="coerce",
            ).mean()

        # Mean opportunity_score of selected stations
        mean_opportunity = np.nan
        if "opportunity_score" in selected_df.columns:
            mean_opportunity = pd.to_numeric(
                selected_df["opportunity_score"],
                errors="coerce",
            ).mean()

        results.append({
            "variant": variant_name,
            "equity_weighting": use_equity,
            "underserved_bonus": bonus,
            "description": description,
            "k": variant_cfg.default_k,
            "selected_stations": " | ".join(selected_names),
            "n_selected": len(selected_names),
            "covered_demand_points": covered_points,
            "weighted_demand_covered": weighted_coverage,
            "coverage_pct": coverage_pct,
            "mean_selected_priority": mean_priority,
            "mean_equity_weight": mean_equity_weight,
            "mean_need_score": mean_need,
            "mean_opportunity_score": mean_opportunity,
            "persistent_deserts_selected": persistent_selected,
            "overlap_with_full_model": overlap_count,
            "jaccard_similarity": jaccard_similarity,
            "rank_correlation_with_full": rank_corr,
        })

        logger.info(
            f"  {variant_name}: "
            f"coverage={coverage_pct:.2f}%, "
            f"deserts={persistent_selected}, "
            f"overlap={overlap_count}/{len(full_selected_names)}, "
            f"Jaccard={jaccard_similarity:.3f}, "
            f"rank_corr={rank_corr:.3f}"
        )

    ablation_df = pd.DataFrame(results)

    output_path = cfg.outputs_dir / "ablation_results.csv"
    ablation_df.to_csv(output_path, index=False)

    logger.info(f"Ablation results saved: {output_path}")
    logger.info("=" * 70)
    logger.info("MCLP ABLATION STUDY COMPLETE")
    logger.info("=" * 70)

    return ablation_df


# ─────────────────────────────────────────────────────────────
# COVERAGE CURVE
# ─────────────────────────────────────────────────────────────

def compute_coverage_by_k(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    coverage_sets: Sequence[Set[int]],
    max_k: int,
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """
    Run greedy MCLP for k = 1...max_k and return coverage curve.

    This helps decision-makers understand the trade-off between
    number of interventions and coverage achieved.

    The logging is suppressed (verbose=False) to avoid console clutter
    since this runs k times.
    """
    rows: List[Dict[str, object]] = []
    max_k = min(max_k, len(stations_df))

    for k in range(1, max_k + 1):
        selected_df, summary = greedy_mclp(
            stations_df=stations_df,
            demand_df=demand_df,
            coverage_sets=coverage_sets,
            k=k,
            cfg=cfg,
            verbose=False,  # Suppress logging for coverage curve
        )

        selected_names = [] if selected_df.empty else selected_df["stop_name"].tolist()

        rows.append({
            "k": k,
            "selected_stations": " | ".join(selected_names),
            "covered_demand_points": summary["covered_demand_points"],
            "total_demand_points": summary["total_demand_points"],
            "covered_weighted_demand": summary["covered_weighted_demand"],
            "total_weighted_demand": summary["total_weighted_demand"],
            "coverage_pct": summary["coverage_pct"],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# ASSIGNMENT EXPORT
# ─────────────────────────────────────────────────────────────

def assign_demand_to_selected_stations(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    distance_matrix_m: np.ndarray,
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """
    Assign each covered demand point to its nearest selected station.

    This creates a mapping for dashboard storytelling:
    - which demand point is served by which selected station
    - what distance is assumed (network-adjusted)
    - what type of demand it represents (school, business, etc.)

    Only demand points within the coverage radius are assigned.
    """
    if selected_df.empty:
        return pd.DataFrame()

    station_lookup = stations_df.reset_index(drop=True).copy()
    selected_stop_ids = selected_df["stop_id"].tolist()
    selected_indices = station_lookup.index[station_lookup["stop_id"].isin(selected_stop_ids)].tolist()

    rows: List[Dict[str, object]] = []

    for demand_pos, demand_row in demand_df.reset_index(drop=True).iterrows():
        selected_distances = distance_matrix_m[selected_indices, demand_pos]
        nearest_pos = int(np.argmin(selected_distances))
        nearest_station_idx = selected_indices[nearest_pos]
        nearest_dist = float(selected_distances[nearest_pos])

        if nearest_dist <= cfg.coverage_radius_m:
            station_row = station_lookup.loc[nearest_station_idx]

            rows.append({
                "demand_id": int(demand_row["demand_id"]),
                "lat": float(demand_row["lat"]),
                "lon": float(demand_row["lon"]),
                "type": demand_row.get("type", "unknown"),
                "demand_weight": float(demand_row["demand_weight"]),
                "assigned_station_id": station_row["stop_id"],
                "assigned_station_name": station_row["stop_name"],
                "distance_to_station_m": nearest_dist,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# CANDIDATE SCORE EXPORT
# ─────────────────────────────────────────────────────────────

def build_candidate_scores(
    stations_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    coverage_sets: Sequence[Set[int]],
    cfg: MCLPConfig = CFG,
) -> pd.DataFrame:
    """
    Create ranked candidate table before greedy selection.

    This shows all stations with their individual scores, enabling
    transparency in the selection process.

    Key scores exported:
    - weighted_demand_if_selected: demand that would be covered
    - priority_score: LMCI deficit / need from lmci.py
    - opportunity_score: infrastructure amplification potential
    - candidate_equity_weight: need × opportunity
    - AIP_score: demand × equity_weight
    - intervention_score: demand × priority (raw intervention potential)
    - marginal_priority_gain: same as intervention_score (for analysis)

    Stations are ranked by AIP_score (descending)
    to show which would be selected first.
    """
    demand_weight_map = dict(
        zip(demand_df["demand_id"].astype(int), demand_df["demand_weight"].astype(float))
    )

    rows = []

    for idx, station in stations_df.reset_index(drop=True).iterrows():
        covered_ids = coverage_sets[idx]
        raw_weight = _weighted_coverage_value(covered_ids, demand_weight_map)
        equity_weight = float(station.get("candidate_equity_weight", 1.0))
        priority_score = float(station.get("priority_score", 5.0))

        # Calculate various scores for analysis
        aip_score = raw_weight * equity_weight  # Accessibility Improvement Potential
        intervention_score = raw_weight * priority_score  # Raw intervention value

        row = station.to_dict()

        row.update({
            "covered_demand_points_if_selected": len(covered_ids),
            "weighted_demand_if_selected": raw_weight,
            "priority_score": priority_score,
            "opportunity_score": station.get("opportunity_score", 1.0),
            "candidate_equity_weight": equity_weight,
            "AIP_score": aip_score,
            "intervention_score": intervention_score,
            "marginal_priority_gain": raw_weight * priority_score,  # For later analysis
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["AIP_score", "weighted_demand_if_selected"],
        ascending=False,
    ).reset_index(drop=True)

    out["candidate_rank"] = np.arange(1, len(out) + 1)

    return out


# ─────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────

def export_mclp_outputs(
    cfg: MCLPConfig,
    selected_df: pd.DataFrame,
    coverage_by_k_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    candidate_scores_df: pd.DataFrame,
) -> Dict[str, Path]:
    """
    Save all MCLP outputs to CSV and GeoJSON formats.

    CSV outputs are for analysis and dashboard consumption.
    GeoJSON outputs are for mapping and spatial visualization.
    """
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.assets_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    selected_path = cfg.outputs_dir / "mclp_selected_stations.csv"
    selected_df.to_csv(selected_path, index=False)
    paths["selected_stations"] = selected_path

    coverage_path = cfg.outputs_dir / "mclp_coverage_by_k.csv"
    coverage_by_k_df.to_csv(coverage_path, index=False)
    paths["coverage_by_k"] = coverage_path

    assignment_path = cfg.outputs_dir / "mclp_demand_assignment.csv"
    assignment_df.to_csv(assignment_path, index=False)
    paths["demand_assignment"] = assignment_path

    candidate_path = cfg.outputs_dir / "mclp_candidate_scores.csv"
    candidate_scores_df.to_csv(candidate_path, index=False)
    paths["candidate_scores"] = candidate_path

    # GeoJSON exports for dashboard/map usage
    try:
        if not selected_df.empty:
            selected_gdf = to_gdf(
                selected_df,
                lat_col="stop_lat",
                lon_col="stop_lon",
                crs_out=cfg.wgs84_crs,
            )
            selected_geojson = cfg.assets_dir / "mclp_selected_stations.geojson"
            selected_gdf.to_file(selected_geojson, driver="GeoJSON")
            paths["selected_stations_geojson"] = selected_geojson
    except Exception as exc:
        logger.warning(f"Selected stations GeoJSON export skipped: {exc}")

    try:
        if not assignment_df.empty:
            assignment_gdf = to_gdf(
                assignment_df,
                lat_col="lat",
                lon_col="lon",
                crs_out=cfg.wgs84_crs,
            )
            assignment_geojson = cfg.assets_dir / "mclp_demand_assignment.geojson"
            assignment_gdf.to_file(assignment_geojson, driver="GeoJSON")
            paths["demand_assignment_geojson"] = assignment_geojson
    except Exception as exc:
        logger.warning(f"Demand assignment GeoJSON export skipped: {exc}")

    logger.info("MCLP exports complete:")
    for key, path in paths.items():
        logger.info(f"  {key}: {path}")

    return paths


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_mclp_outputs(
    selected_df: pd.DataFrame,
    coverage_by_k_df: pd.DataFrame,
    assignment_df: pd.DataFrame,
    candidate_scores_df: pd.DataFrame,
) -> None:
    """
    Basic fail-fast validation for dashboard reliability.

    Ensures all expected outputs are present and have required columns.
    """
    if selected_df.empty:
        raise ValueError("MCLP selected_df is empty. No stations selected.")

    required_selected = {
        "stop_id",
        "stop_name",
        "selection_rank",
        "coverage_pct",
        "marginal_weighted_demand",
    }
    missing_selected = required_selected - set(selected_df.columns)
    if missing_selected:
        raise ValueError(f"Selected station output missing columns: {missing_selected}")

    if coverage_by_k_df.empty:
        raise ValueError("coverage_by_k_df is empty.")

    if "coverage_pct" not in coverage_by_k_df.columns:
        raise ValueError("coverage_by_k_df missing coverage_pct.")

    if candidate_scores_df.empty:
        raise ValueError("candidate_scores_df is empty.")

    # Assignment can be empty only if selected stations cover zero demand.
    if assignment_df.empty:
        logger.warning("Demand assignment is empty. Check coverage radius and demand coordinates.")

    logger.info("MCLP validation passed.")


# ─────────────────────────────────────────────────────────────
# END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────

def run_mclp_pipeline(
    cfg: MCLPConfig = CFG,
    k: Optional[int] = None,
) -> Dict[str, object]:
    """
    Run complete MCLP pipeline end-to-end.

    Parameters
    ----------
    cfg:
        MCLPConfig with all configuration parameters

    k:
        Number of intervention stations to select.
        Defaults to cfg.default_k.

    Returns
    -------
    dict containing:
        - stations: loaded station data with candidate enrichment
        - demand: loaded demand points with weights
        - coverage_sets: station-to-demand coverage sets
        - distance_matrix_m: station-demand distance matrix
        - station_gdf: station GeoDataFrame
        - demand_gdf: demand GeoDataFrame
        - selected_df: selected stations with metrics
        - coverage_by_k_df: coverage curve for k=1..max_k
        - assignment_df: demand point assignment to selected stations
        - candidate_scores_df: all stations with candidate scores
        - ablation_df: ablation study results
        - summary: overall coverage summary
        - export_paths: paths to all exported files
    """
    logger.info("=" * 70)
    logger.info("STARTING MCLP PIPELINE")
    logger.info("=" * 70)

    if k is None:
        k = cfg.default_k

    # 1. Load and prepare data
    stations = load_station_scores(cfg)
    stations = enrich_station_candidates(stations, cfg)

    demand = load_demand_points(cfg)
    demand = assign_demand_weights(demand, cfg)

    # 2. Build coverage matrix
    coverage_sets, distance_matrix_m, station_gdf, demand_gdf = build_coverage_matrix(
        stations_df=stations,
        demand_df=demand,
        cfg=cfg,
    )

    # 3. Run greedy MCLP for target k
    selected_df, summary = greedy_mclp(
        stations_df=stations,
        demand_df=demand,
        coverage_sets=coverage_sets,
        k=k,
        cfg=cfg,
    )

    # 4. Compute coverage curve (k=1..max_k)
    coverage_by_k_df = compute_coverage_by_k(
        stations_df=stations,
        demand_df=demand,
        coverage_sets=coverage_sets,
        max_k=cfg.max_k,
        cfg=cfg,
    )

    # 5. Assign demand points to selected stations
    assignment_df = assign_demand_to_selected_stations(
        stations_df=stations,
        demand_df=demand,
        selected_df=selected_df,
        distance_matrix_m=distance_matrix_m,
        cfg=cfg,
    )

    # 6. Build candidate scores for all stations
    candidate_scores_df = build_candidate_scores(
        stations_df=stations,
        demand_df=demand,
        coverage_sets=coverage_sets,
        cfg=cfg,
    )

    # 7. Ablation Study
    ablation_df = run_mclp_ablation_study(
        stations_df=stations,
        demand_df=demand,
        coverage_sets=coverage_sets,
        cfg=cfg,
    )

    # 8. Validate outputs
    validate_mclp_outputs(
        selected_df=selected_df,
        coverage_by_k_df=coverage_by_k_df,
        assignment_df=assignment_df,
        candidate_scores_df=candidate_scores_df,
    )

    # 9. Export all outputs
    export_paths = export_mclp_outputs(
        cfg=cfg,
        selected_df=selected_df,
        coverage_by_k_df=coverage_by_k_df,
        assignment_df=assignment_df,
        candidate_scores_df=candidate_scores_df,
    )

    logger.info("=" * 70)
    logger.info("MCLP PIPELINE COMPLETE")
    logger.info(f"Selected stations: {len(selected_df)}")
    logger.info(f"Weighted coverage: {summary['coverage_pct']:.2f}%")
    logger.info("=" * 70)

    return {
        "stations": stations,
        "demand": demand,
        "coverage_sets": coverage_sets,
        "distance_matrix_m": distance_matrix_m,
        "station_gdf": station_gdf,
        "demand_gdf": demand_gdf,
        "selected_df": selected_df,
        "coverage_by_k_df": coverage_by_k_df,
        "assignment_df": assignment_df,
        "candidate_scores_df": candidate_scores_df,
        "ablation_df": ablation_df,
        "summary": summary,
        "export_paths": export_paths,
    }


if __name__ == "__main__":
    run_mclp_pipeline()
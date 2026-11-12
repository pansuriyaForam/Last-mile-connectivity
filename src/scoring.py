"""
Scoring and insight layer for the Hyderabad Metro Last-Mile
Connectivity project.

Purpose
-------
Convert technical LMCI + MCLP outputs into demo-safe, pitch-ready
station priorities and explainable recommendation tables.

This module sits AFTER:
1. src/preprocessing.py
2. src/lmci.py
3. src/mclp.py

Inputs
------
outputs/station_lmci_summary.csv
outputs/lmci_station_scores.csv
outputs/mclp_selected_stations.csv
outputs/mclp_candidate_scores.csv
outputs/mclp_coverage_by_k.csv
outputs/mclp_demand_assignment.csv

Exports
-------
outputs/station_priority_scores.csv
outputs/demand_service_mismatch.csv
outputs/conversion_insights_top5.csv
outputs/executive_summary_metrics.csv
outputs/scoring_audit_report.csv
assets/station_priority_scores.geojson
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    from src.preprocessing import to_gdf
except ImportError:
    from preprocessing import to_gdf


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("SCORING")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class ScoringConfig:
    """Configuration for scoring layer."""

    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")

    station_summary_path: Path = Path("outputs/station_lmci_summary.csv")
    station_scores_fallback_path: Path = Path("outputs/lmci_station_scores.csv")
    mclp_selected_path: Path = Path("outputs/mclp_selected_stations.csv")
    mclp_candidate_scores_path: Path = Path("outputs/mclp_candidate_scores.csv")
    mclp_coverage_by_k_path: Path = Path("outputs/mclp_coverage_by_k.csv")
    mclp_assignment_path: Path = Path("outputs/mclp_demand_assignment.csv")

    metric_crs: int = 32644
    wgs84_crs: int = 4326

    # Classification thresholds (percentile-based for mismatch)
    high_quantile: float = 0.67
    low_quantile: float = 0.33

    # Absolute priority thresholds for band classification
    # priority_index is on a 0-10 scale from lmci.py
    critical_threshold: float = 8.0
    high_threshold: float = 6.0
    medium_threshold: float = 4.0

    # Output size
    top_n_insights: int = 5

    def __post_init__(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


CFG = ScoringConfig()


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def _load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning(f"Missing optional scoring input: {path}")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    logger.info(f"Loaded {path.name}: {len(df):,} rows")
    return df



def load_station_summary(cfg: ScoringConfig = CFG) -> pd.DataFrame:
    """Load station LMCI summary."""

    if cfg.station_summary_path.exists():
        path = cfg.station_summary_path
    elif cfg.station_scores_fallback_path.exists():
        path = cfg.station_scores_fallback_path
    else:
        raise FileNotFoundError(
            "No LMCI station output found. Run src/lmci.py first."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"stop_id", "stop_name", "stop_lat", "stop_lon"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Station summary missing required columns: {missing}")

    df["stop_lat"] = pd.to_numeric(df["stop_lat"], errors="coerce")
    df["stop_lon"] = pd.to_numeric(df["stop_lon"], errors="coerce")
    df = df.dropna(subset=["stop_lat", "stop_lon"]).reset_index(drop=True)

    logger.info(f"Loaded station summary: {len(df):,} stations")
    return df



def load_scoring_inputs(cfg: ScoringConfig = CFG) -> Dict[str, Optional[pd.DataFrame]]:
    """Load all available scoring inputs."""

    station_df = load_station_summary(cfg)

    return {
        "stations": station_df,
        "mclp_selected": _load_csv_if_exists(cfg.mclp_selected_path),
        "mclp_candidate_scores": _load_csv_if_exists(cfg.mclp_candidate_scores_path),
        "mclp_coverage_by_k": _load_csv_if_exists(cfg.mclp_coverage_by_k_path),
        "mclp_assignment": _load_csv_if_exists(cfg.mclp_assignment_path),
    }


# ─────────────────────────────────────────────────────────────
# COLUMN HELPERS
# ─────────────────────────────────────────────────────────────

def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None



def _ensure_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


# ─────────────────────────────────────────────────────────────
# MERGE MCLP SIGNALS
# ─────────────────────────────────────────────────────────────

def attach_mclp_signals(
    station_df: pd.DataFrame,
    mclp_selected: Optional[pd.DataFrame],
    mclp_candidate_scores: Optional[pd.DataFrame],
    mclp_assignment: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Attach MCLP-derived signals to station summary."""

    out = station_df.copy()

    # Candidate score signal
    if mclp_candidate_scores is not None and not mclp_candidate_scores.empty:
        candidate_cols = [
            "stop_id",
            "candidate_rank",
            "covered_demand_points_if_selected",
            "weighted_demand_if_selected",
            "equity_weighted_candidate_score",
        ]
        candidate_cols = [c for c in candidate_cols if c in mclp_candidate_scores.columns]

        out = out.merge(
            mclp_candidate_scores[candidate_cols].drop_duplicates("stop_id"),
            on="stop_id",
            how="left",
        )
    else:
        out["candidate_rank"] = np.nan
        out["covered_demand_points_if_selected"] = 0
        out["weighted_demand_if_selected"] = 0.0
        out["equity_weighted_candidate_score"] = 0.0

    # Selected station signal
    out["mclp_selected"] = False
    out["mclp_selection_rank"] = np.nan
    out["mclp_marginal_weighted_demand"] = 0.0
    out["mclp_cumulative_coverage_pct"] = 0.0

    if mclp_selected is not None and not mclp_selected.empty:
        selected_cols = [
            "stop_id",
            "selection_rank",
            "marginal_weighted_demand",
            "coverage_pct",
        ]
        selected_cols = [c for c in selected_cols if c in mclp_selected.columns]

        selected_small = mclp_selected[selected_cols].drop_duplicates("stop_id")
        out = out.merge(
            selected_small,
            on="stop_id",
            how="left",
        )

        out["mclp_selected"] = out["selection_rank"].notna()
        out["mclp_selection_rank"] = out["selection_rank"]
        out["mclp_marginal_weighted_demand"] = _ensure_numeric(out, "marginal_weighted_demand")
        out["mclp_cumulative_coverage_pct"] = _ensure_numeric(out, "coverage_pct")

        out = out.drop(
            columns=[c for c in ["selection_rank", "marginal_weighted_demand", "coverage_pct"] if c in out.columns],
            errors="ignore",
        )

    # Assignment count signal
    if mclp_assignment is not None and not mclp_assignment.empty:
        if "assigned_station_id" in mclp_assignment.columns:
            assignment_count = (
                mclp_assignment.groupby("assigned_station_id")
                .size()
                .rename("assigned_demand_count")
                .reset_index()
                .rename(columns={"assigned_station_id": "stop_id"})
            )

            out = out.merge(assignment_count, on="stop_id", how="left")
        else:
            out["assigned_demand_count"] = 0
    else:
        out["assigned_demand_count"] = 0

    fill_cols = [
        "covered_demand_points_if_selected",
        "weighted_demand_if_selected",
        "equity_weighted_candidate_score",
        "assigned_demand_count",
    ]

    for col in fill_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out


# ─────────────────────────────────────────────────────────────
# DEMAND-SERVICE MISMATCH
# ─────────────────────────────────────────────────────────────

def classify_demand_service_mismatch(
    df: pd.DataFrame,
    cfg: ScoringConfig = CFG,
) -> pd.DataFrame:
    """
    Classify each station into explainable demand-service buckets.

    Classes
    -------
    High Demand - Low Service:
        strongest intervention opportunity

    High Demand - High Service:
        already important, protect capacity

    Low Demand - High Service:
        possible oversupply / lower immediate priority

    Low Demand - Low Service:
        monitor or bundle with network expansion

    Balanced:
        middle-zone station
    """

    out = df.copy()

    demand_col = _first_existing_column(
        out,
        [
            "demand_count",
            "weighted_demand_if_selected",
            "covered_demand_points_if_selected",
            "assigned_demand_count",
        ],
    )

    lmci_col = _first_existing_column(
        out,
        ["LMCI_new2", "LMCI_new", "LMCI_old", "LMCI_mean", "LMCI"],
    )

    if demand_col is None:
        out["demand_signal"] = 0.0
    else:
        out["demand_signal"] = _ensure_numeric(out, demand_col)

    if lmci_col is None:
        out["service_signal"] = 0.0
    else:
        out["service_signal"] = _ensure_numeric(out, lmci_col)

    demand_high = out["demand_signal"].quantile(cfg.high_quantile)
    demand_low = out["demand_signal"].quantile(cfg.low_quantile)
    service_high = out["service_signal"].quantile(cfg.high_quantile)
    service_low = out["service_signal"].quantile(cfg.low_quantile)

    def bucket(row: pd.Series) -> str:
        demand = row["demand_signal"]
        service = row["service_signal"]

        if demand >= demand_high and service <= service_low:
            return "High Demand - Low Service"
        if demand >= demand_high and service >= service_high:
            return "High Demand - High Service"
        if demand <= demand_low and service >= service_high:
            return "Low Demand - High Service"
        if demand <= demand_low and service <= service_low:
            return "Low Demand - Low Service"
        return "Balanced"

    out["mismatch_class"] = out.apply(bucket, axis=1)

    out["is_high_demand_low_service"] = out["mismatch_class"].eq("High Demand - Low Service")
    out["is_possible_overserved"] = out["mismatch_class"].eq("Low Demand - High Service")

    out["demand_percentile"] = out["demand_signal"].rank(pct=True) * 100.0
    out["service_percentile"] = out["service_signal"].rank(pct=True) * 100.0

    return out


# ─────────────────────────────────────────────────────────────
# PRIORITY SCORING
# ─────────────────────────────────────────────────────────────

def compute_station_priority_scores(df: pd.DataFrame, cfg: ScoringConfig = CFG) -> pd.DataFrame:
    """
    Compute final investment/intervention priority index.

    This function simply reports what LMCI already computed.
    No re-optimization or weighting occurs here.
    """
    out = df.copy()

    # priority_score MUST come from LMCI - no fallback allowed
    # In lmci.py, this is computed in classify_transit_deserts()
    if "priority_score" not in out.columns:
        raise ValueError(
            "priority_score missing. Run lmci.py before scoring.py."
        )

    # Rename to priority_index for consistency with academic terminology
    out["priority_index"] = _ensure_numeric(out, "priority_score")

    # Validate that priority_rank already exists from LMCI
    if "priority_rank" not in out.columns:
        # Compute rank if LMCI didn't export it (fallback for older versions)
        out = out.sort_values(
            ["priority_index", "demand_signal"],
            ascending=[False, False],
        ).reset_index(drop=True)
        out["priority_rank"] = np.arange(1, len(out) + 1)

    # Priority bands using absolute thresholds from config
    # This answers "which stations are objectively bad?" rather than
    # "which stations are worst relative to the rest?"
    def priority_band(score: float) -> str:
        if score >= cfg.critical_threshold:
            return "Critical"
        if score >= cfg.high_threshold:
            return "High"
        if score >= cfg.medium_threshold:
            return "Medium"
        return "Low"

    out["priority_band"] = out["priority_index"].apply(priority_band)

    # Add priority reason for dashboard interpretability
    def priority_reason(row: pd.Series) -> str:
        reasons = []

        # LMCI deficit (from lmci.py)
        if "priority_lmci_deficit" in row.index:
            if row.get("priority_lmci_deficit", 0) > 0.7:
                reasons.append("Very Poor Accessibility")
            elif row.get("priority_lmci_deficit", 0) > 0.4:
                reasons.append("Poor Accessibility")

        # Demand signal
        if "demand_signal" in row.index and row.get("demand_signal", 0) > 0:
            if row.get("demand_signal", 0) > out["demand_signal"].quantile(0.75):
                reasons.append("High Demand")

        # Persistent desert (from lmci.py)
        if row.get("is_persistent_desert", False):
            reasons.append("Persistent Desert")

        # Temporal gap (from lmci.py)
        if row.get("temporal_gap", 0) > 2.0:
            reasons.append("Temporal Variability")

        # Mismatch class (if available)
        mismatch = row.get("mismatch_class", "")
        if mismatch == "High Demand - Low Service":
            reasons.append("Demand-Service Mismatch")

        if not reasons:
            return "Monitor"
        return " | ".join(reasons[:3])  # Limit to 3 reasons for readability

    out["priority_reason"] = out.apply(priority_reason, axis=1)

    # Export priority_percentile alongside priority_index for dashboard flexibility
    out["priority_percentile"] = out["priority_index"].rank(pct=True) * 100.0

    # Sort by priority_index (highest first)
    out = out.sort_values(
        ["priority_index", "demand_signal"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return out


# ─────────────────────────────────────────────────────────────
# CONVERSION INSIGHTS
# ─────────────────────────────────────────────────────────────

def generate_recommendation(row: pd.Series) -> str:
    """Generate deterministic station-level recommendation."""

    mismatch = row.get("mismatch_class", "Balanced")
    selected = bool(row.get("mclp_selected", False))
    desert = str(row.get("desert_severity", "None"))
    station = row.get("stop_name", "station")

    # High Demand - Low Service is the strongest signal
    if mismatch == "High Demand - Low Service" and selected:
        return f"Prioritize {station} for immediate last-mile intervention; it combines high latent demand, weak service, and strong MCLP coverage gain."

    if mismatch == "High Demand - Low Service":
        return f"Investigate {station} as a near-term feeder/micromobility candidate due to demand-service mismatch."

    # MCLP selected stations are coverage-maximizing
    if selected:
        return f"Use {station} as a coverage-maximizing intervention node based on MCLP selection."

    # Persistent deserts need long-term monitoring
    if desert == "Persistent":
        return f"Monitor {station} as a persistent transit desert; bundle it with nearby station-level improvements if direct intervention is costly."

    # High Service but Low Demand suggests oversupply
    if mismatch == "Low Demand - High Service":
        return f"Deprioritize new investment at {station}; current service appears stronger than nearby demand signal."

    # High Demand with High Service means protect existing quality
    if mismatch == "High Demand - High Service":
        return f"Protect service quality at {station}; it is demand-heavy but already relatively well served."

    # Default monitoring case
    return f"Keep {station} in monitoring pipeline; no urgent intervention signal dominates."



def build_conversion_insights(
    scored_df: pd.DataFrame,
    cfg: ScoringConfig = CFG,
) -> pd.DataFrame:
    """Build top-N pitch-ready station insights."""

    out = scored_df.copy()
    out["recommendation"] = out.apply(generate_recommendation, axis=1)

    cols = [
        "priority_rank",
        "stop_id",
        "stop_name",
        "priority_index",
        "priority_band",
        "priority_reason",
        "priority_percentile",
        "mismatch_class",
        "demand_signal",
        "service_signal",
        "LMCI_old",
        "LMCI_new",
        "LMCI_new2",
        "Morning_LMCI",
        "Midday_LMCI",
        "Evening_LMCI",
        "temporal_gap",
        "mclp_selected",
        "mclp_selection_rank",
        "mclp_marginal_weighted_demand",
        "covered_demand_points_if_selected",
        "weighted_demand_if_selected",
        "desert_severity",
        "recommendation",
    ]

    cols = [c for c in cols if c in out.columns]

    return out[cols].head(cfg.top_n_insights).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────

def build_executive_summary(
    scored_df: pd.DataFrame,
    coverage_by_k: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create one-row summary metrics for dashboard cards."""

    df = scored_df.copy()

    total_stations = len(df)
    critical_count = int((df["priority_band"] == "Critical").sum()) if "priority_band" in df.columns else 0
    high_count = int((df["priority_band"] == "High").sum()) if "priority_band" in df.columns else 0
    persistent_deserts = int(df.get("is_persistent_desert", pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"]).sum())
    high_demand_low_service = int(df.get("is_high_demand_low_service", pd.Series(False, index=df.index)).sum())

    top_station = df.iloc[0]["stop_name"] if not df.empty else None
    top_index = float(df.iloc[0]["priority_index"]) if not df.empty else 0.0
    top_percentile = float(df.iloc[0]["priority_percentile"]) if not df.empty else 0.0

    best_coverage_k5 = np.nan
    if coverage_by_k is not None and not coverage_by_k.empty and "coverage_pct" in coverage_by_k.columns:
        row = coverage_by_k[coverage_by_k["k"] == 5]
        if not row.empty:
            best_coverage_k5 = float(row.iloc[0]["coverage_pct"])
        else:
            best_coverage_k5 = float(coverage_by_k["coverage_pct"].max())

    summary = pd.DataFrame([
        {
            "total_stations_scored": total_stations,
            "critical_priority_stations": critical_count,
            "high_priority_stations": high_count,
            "persistent_transit_deserts": persistent_deserts,
            "high_demand_low_service_stations": high_demand_low_service,
            "top_priority_station": top_station,
            "top_priority_index": top_index,
            "top_priority_percentile": top_percentile,
            "mclp_coverage_pct_at_k5": best_coverage_k5,
        }
    ])

    return summary


# ─────────────────────────────────────────────────────────────
# AUDIT REPORT
# ─────────────────────────────────────────────────────────────

def build_scoring_audit_report(scored_df: pd.DataFrame, cfg: ScoringConfig = CFG) -> pd.DataFrame:
    """Create scoring audit table for explainability."""

    checks = []

    def add_check(name: str, passed: bool, detail: str):
        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        })

    add_check(
        "station_rows_available",
        len(scored_df) > 0,
        f"stations={len(scored_df)}",
    )

    add_check(
        "priority_index_not_null",
        "priority_index" in scored_df.columns and scored_df["priority_index"].notna().all(),
        "priority_index computed for all stations",
    )

    add_check(
        "priority_index_available",
        "priority_index" in scored_df.columns,
        "priority_index exists from LMCI",
    )

    add_check(
        "priority_rank_available",
        "priority_rank" in scored_df.columns,
        "priority_rank exists from LMCI",
    )

    add_check(
        "priority_range_valid",
        "priority_index" in scored_df.columns and scored_df["priority_index"].between(0, 10).all(),
        f"priority_index range: {scored_df['priority_index'].min():.2f} - {scored_df['priority_index'].max():.2f} (0-10 scale)",
    )

    add_check(
        "priority_reason_available",
        "priority_reason" in scored_df.columns,
        "priority_reason generated for all stations",
    )

    add_check(
        "coordinates_valid",
        scored_df[["stop_lat", "stop_lon"]].notna().all().all() if {"stop_lat", "stop_lon"}.issubset(scored_df.columns) else False,
        "station coordinates available",
    )

    add_check(
        "mismatch_class_available",
        "mismatch_class" in scored_df.columns,
        "demand-service mismatch classes generated",
    )

    add_check(
        "mclp_signal_available",
        "equity_weighted_candidate_score" in scored_df.columns,
        "MCLP candidate score attached",
    )

    add_check(
        "priority_band_available",
        "priority_band" in scored_df.columns,
        "priority bands generated",
    )

    return pd.DataFrame(checks)


# ─────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────

def export_scoring_outputs(
    cfg: ScoringConfig,
    scored_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    insights_df: pd.DataFrame,
    executive_summary_df: pd.DataFrame,
    audit_df: pd.DataFrame,
) -> Dict[str, Path]:
    """Export scoring outputs to outputs/ and assets/."""

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.assets_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    priority_path = cfg.outputs_dir / "station_priority_scores.csv"
    scored_df.to_csv(priority_path, index=False)
    paths["station_priority_scores"] = priority_path

    mismatch_path = cfg.outputs_dir / "demand_service_mismatch.csv"
    mismatch_df.to_csv(mismatch_path, index=False)
    paths["demand_service_mismatch"] = mismatch_path

    insights_path = cfg.outputs_dir / "conversion_insights_top5.csv"
    insights_df.to_csv(insights_path, index=False)
    paths["conversion_insights_top5"] = insights_path

    summary_path = cfg.outputs_dir / "executive_summary_metrics.csv"
    executive_summary_df.to_csv(summary_path, index=False)
    paths["executive_summary_metrics"] = summary_path

    audit_path = cfg.outputs_dir / "scoring_audit_report.csv"
    audit_df.to_csv(audit_path, index=False)
    paths["scoring_audit_report"] = audit_path

    try:
        priority_gdf = to_gdf(
            scored_df,
            lat_col="stop_lat",
            lon_col="stop_lon",
            crs_out=cfg.wgs84_crs,
        )
        geojson_path = cfg.assets_dir / "station_priority_scores.geojson"
        priority_gdf.to_file(geojson_path, driver="GeoJSON")
        paths["station_priority_scores_geojson"] = geojson_path
    except Exception as exc:
        logger.warning(f"Priority GeoJSON export skipped: {exc}")

    logger.info("Scoring exports complete:")
    for key, path in paths.items():
        logger.info(f"  {key}: {path}")

    return paths


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_scoring_output(scored_df: pd.DataFrame) -> None:
    """Fail fast on dashboard-critical scoring issues."""

    required = [
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "priority_index",
        "priority_rank",
        "priority_band",
        "priority_reason",
        "priority_percentile",
        "mismatch_class",
    ]

    missing = [col for col in required if col not in scored_df.columns]
    if missing:
        raise ValueError(f"Scoring output missing required columns: {missing}")

    if scored_df["priority_index"].isna().any():
        raise ValueError("priority_index contains NaN values")

    # priority_index should be on a 0-10 scale (from lmci.py)
    if not scored_df["priority_index"].between(0, 10).all():
        raise ValueError(
            f"priority_index must be between 0 and 10. "
            f"Range: {scored_df['priority_index'].min():.2f} - {scored_df['priority_index'].max():.2f}"
        )

    if scored_df[["stop_lat", "stop_lon"]].isna().any().any():
        raise ValueError("Station coordinates contain NaN values")

    logger.info("Scoring validation passed.")


# ─────────────────────────────────────────────────────────────
# END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────

def run_scoring_pipeline(cfg: ScoringConfig = CFG) -> Dict[str, object]:
    """Run complete scoring pipeline."""

    logger.info("=" * 70)
    logger.info("STARTING SCORING PIPELINE")
    logger.info("=" * 70)

    inputs = load_scoring_inputs(cfg)

    station_df = inputs["stations"]

    merged = attach_mclp_signals(
        station_df=station_df,
        mclp_selected=inputs["mclp_selected"],
        mclp_candidate_scores=inputs["mclp_candidate_scores"],
        mclp_assignment=inputs["mclp_assignment"],
    )

    mismatch_df = classify_demand_service_mismatch(merged, cfg)
    scored_df = compute_station_priority_scores(mismatch_df, cfg)
    insights_df = build_conversion_insights(scored_df, cfg)
    executive_summary_df = build_executive_summary(scored_df, inputs["mclp_coverage_by_k"])
    audit_df = build_scoring_audit_report(scored_df, cfg)

    validate_scoring_output(scored_df)

    export_paths = export_scoring_outputs(
        cfg=cfg,
        scored_df=scored_df,
        mismatch_df=mismatch_df,
        insights_df=insights_df,
        executive_summary_df=executive_summary_df,
        audit_df=audit_df,
    )

    logger.info("=" * 70)
    logger.info("SCORING PIPELINE COMPLETE")
    logger.info(f"Stations scored: {len(scored_df)}")
    logger.info(f"Top station: {scored_df.iloc[0]['stop_name']} ({scored_df.iloc[0]['priority_index']:.2f})")
    logger.info("=" * 70)

    return {
        "scored_df": scored_df,
        "mismatch_df": mismatch_df,
        "insights_df": insights_df,
        "executive_summary_df": executive_summary_df,
        "audit_df": audit_df,
        "export_paths": export_paths,
    }


if __name__ == "__main__":
    run_scoring_pipeline()
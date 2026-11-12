"""
src/simulation.py
────────────────────────────────────────────────────────────
Scenario modelling engine for the Hyderabad Metro
Urban Mobility Accessibility & Decision Support Platform.

Purpose
-------
Model the relative accessibility impact of hypothetical interventions:
- feeder routes
- e-rickshaw loops
- bus frequency improvements
- micromobility access
- walkability upgrades
- multimodal integration upgrades

This module DOES NOT modify GTFS and does NOT produce ridership forecasts.
It performs explainable analytical simulations on top of:
- LMCI (Last-Mile Connectivity Index)
- MCLP (Maximum Coverage Location Problem)
- scoring outputs

All outputs are indicative accessibility estimates based on multimodal
connectivity assumptions — not AFC/smartcard or passenger count data.

Core philosophy:
- deterministic
- explainable
- investor/demo-safe
- policy-friendly

Inputs
------
outputs/station_priority_scores.csv
outputs/mclp_selected_stations.csv
outputs/station_lmci_summary.csv
outputs/lmci_station_scores.csv

Exports
-------
outputs/simulation_station_impacts.csv
outputs/simulation_network_summary.csv
outputs/simulation_intervention_ranking.csv
outputs/simulation_scenarios.csv
outputs/simulation_audit_report.csv
assets/simulation_station_impacts.geojson
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
    from src.preprocessing import minmax_norm, to_gdf
    from src.scoring import PriorityConfig  # reuse thresholds
except ImportError:
    from preprocessing import minmax_norm, to_gdf
    # fallback if scoring not available
    class PriorityConfig:
        high_threshold = 8.0
        medium_threshold = 6.0
        low_threshold = 4.0


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("SIMULATION")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """
    Configuration for simulation engine.

    The scenario effectiveness coefficients (feeder_effectiveness, etc.) are
    expert‑assigned exploratory parameters intended for policy experimentation.
    They are **not** empirically derived and should be treated as such.
    The results reported in this study use the default values defined below.
    Sensitivity analysis is presented separately.
    """

    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")

    station_priority_path: Path = Path("outputs/station_priority_scores.csv")
    mclp_selected_path: Path = Path("outputs/mclp_selected_stations.csv")
    station_summary_path: Path = Path("outputs/station_lmci_summary.csv")
    lmci_station_scores_path: Path = Path("outputs/lmci_station_scores.csv")

    metric_crs: int = 32644
    wgs84_crs: int = 4326

    # Scenario effectiveness (0-1) – subjective, expert‑assigned.
    # These values are not calibrated against observed data; they are
    # intended to illustrate the relative potential of different
    # intervention types. Sensitivity analysis is provided separately.
    feeder_effectiveness: float = 0.90
    micromobility_effectiveness: float = 0.55
    frequency_effectiveness: float = 0.70
    multimodal_effectiveness: float = 1.00
    walkability_effectiveness: float = 0.45

    # LMCI gain parameters
    lmci_improvement_cap: float = 3.0
    # Calibration parameter α (alpha). It scales the deficit‑based gain
    # so that a station with LMCI=0 and effectiveness=1 would gain
    # exactly the cap (3.0) – i.e. 30% of the full deficit (10).
    # This value is chosen to avoid unrealistic single‑intervention
    # improvements while preserving numerical spread across stations.
    # It is a calibration parameter, not a statistically estimated one.
    lmci_gain_alpha: float = 0.3

    # Policy multipliers (restored from original)
    desert_recovery_factor: float = 0.15
    feeder_bonus_factor: float = 0.12
    temporal_stability_factor: float = 0.06
    multimodal_bonus_factor: float = 0.10
    temporal_gap_threshold: float = 1.5   # hours

    # Priority band thresholds (reuse from scoring)
    priority_high: float = PriorityConfig.high_threshold
    priority_medium: float = PriorityConfig.medium_threshold
    priority_low: float = PriorityConfig.low_threshold

    # Scenario sizes
    top_interventions_count: int = 10

    def __post_init__(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)


CFG = SimulationConfig()


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def _load_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logger.warning(f"Missing optional simulation input: {path}")
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    logger.info(f"Loaded {path.name}: {len(df):,} rows")
    return df


def load_simulation_inputs(cfg: SimulationConfig = CFG) -> Dict[str, Optional[pd.DataFrame]]:
    """Load all required simulation inputs."""

    station_priority = _load_csv_if_exists(cfg.station_priority_path)

    if station_priority is None:
        raise FileNotFoundError(
            "station_priority_scores.csv not found. Run src/scoring.py first."
        )

    return {
        "station_priority": station_priority,
        "mclp_selected": _load_csv_if_exists(cfg.mclp_selected_path),
        "station_summary": _load_csv_if_exists(cfg.station_summary_path),
        "lmci_station_scores": _load_csv_if_exists(cfg.lmci_station_scores_path),
    }


# ─────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────

def define_intervention_scenarios() -> pd.DataFrame:
    """
    Define intervention catalog with effectiveness scores.
    Cost bands are descriptive; no numeric division.
    """

    scenarios = [
        {
            "scenario_id": "S1",
            "scenario_name": "Feeder Shuttle Integration",
            "intervention_type": "feeder",
            "effectiveness": CFG.feeder_effectiveness,
            "coverage_gain": 1.25,
            "walkability_gain": 1.0,
            "frequency_gain": 1.0,
            "cost_band": "Medium",
        },
        {
            "scenario_id": "S2",
            "scenario_name": "E-Rickshaw Last-Mile Loop",
            "intervention_type": "micromobility",
            "effectiveness": CFG.micromobility_effectiveness,
            "coverage_gain": 1.15,
            "walkability_gain": 1.0,
            "frequency_gain": 1.0,
            "cost_band": "Low",
        },
        {
            "scenario_id": "S3",
            "scenario_name": "Bus Frequency Enhancement",
            "intervention_type": "frequency",
            "effectiveness": CFG.frequency_effectiveness,
            "coverage_gain": 1.0,
            "walkability_gain": 1.0,
            "frequency_gain": 1.25,
            "cost_band": "Medium",
        },
        {
            "scenario_id": "S4",
            "scenario_name": "Integrated Multimodal Hub",
            "intervention_type": "multimodal",
            "effectiveness": CFG.multimodal_effectiveness,
            "coverage_gain": 1.3,
            "walkability_gain": 1.2,
            "frequency_gain": 1.1,
            "cost_band": "High",
        },
        {
            "scenario_id": "S5",
            "scenario_name": "Pedestrian Access Upgrade",
            "intervention_type": "walkability",
            "effectiveness": CFG.walkability_effectiveness,
            "coverage_gain": 1.0,
            "walkability_gain": 1.2,
            "frequency_gain": 1.0,
            "cost_band": "Low",
        },
    ]

    return pd.DataFrame(scenarios)


# ─────────────────────────────────────────────────────────────
# PREPARE INPUTS
# ─────────────────────────────────────────────────────────────

def prepare_simulation_inputs(
    station_df: pd.DataFrame,
    mclp_selected: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Prepare station-level simulation features with proper normalization."""

    out = station_df.copy()

    # Ensure priority_index exists (rename from final_priority_score if needed)
    if "priority_index" not in out.columns:
        if "final_priority_score" in out.columns:
            out["priority_index"] = out["final_priority_score"]
            logger.info("Renamed 'final_priority_score' to 'priority_index'")
        else:
            out["priority_index"] = 0.0
            logger.warning("No priority_index found, defaulting to 0")

    numeric_candidates = [
        "priority_index",
        "LMCI_old",
        "LMCI_new",
        "LMCI_new2",
        "LMCI_mean",
        "Morning_LMCI",
        "Midday_LMCI",
        "Evening_LMCI",
        "temporal_gap",
        "demand_signal",
        "weighted_demand_if_selected",
        "covered_demand_points_if_selected",
    ]

    for col in numeric_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    # Normalize demand once across all stations (fixes the [0,x] bug)
    if "demand_signal" in out.columns:
        out["demand_norm"] = minmax_norm(out["demand_signal"])
        logger.info(f"Demand normalized across {len(out)} stations")
    else:
        out["demand_norm"] = 0.0
        logger.warning("No demand_signal found, demand_norm defaulted to 0")

    if "mclp_selected" not in out.columns:
        out["mclp_selected"] = False

    if mclp_selected is not None and not mclp_selected.empty:
        selected_ids = set(mclp_selected["stop_id"].astype(str))
        out["mclp_selected"] = out["stop_id"].astype(str).isin(selected_ids)

    if "desert_severity" not in out.columns:
        out["desert_severity"] = "None"

    # Ensure lat/lon are present for GeoJSON
    for col in ["stop_lat", "stop_lon"]:
        if col not in out.columns:
            out[col] = np.nan

    return out


# ─────────────────────────────────────────────────────────────
# COMPUTE LMCI GAIN (deficit‑based)
# ─────────────────────────────────────────────────────────────

def compute_lmci_gain(
    station_row: pd.Series,
    scenario_row: pd.Series,
    cfg: SimulationConfig = CFG,
) -> float:
    """
    Calculate LMCI gain using deficit and effectiveness.

    gain = min(effectiveness * deficit * alpha, cap, deficit)

    The alpha parameter scales the improvement; with alpha=0.3 and
    cap=3.0, a station at LMCI=0 with effectiveness=1 and full
    policy multipliers would reach the cap (3.0).
    This is a calibration parameter, not a statistically estimated one.
    """
    current_lmci = float(
        station_row.get(
            "LMCI_new2",
            station_row.get("LMCI_new", station_row.get("LMCI_old", 0.0)),
        )
    )
    deficit = 10.0 - current_lmci
    effectiveness = float(scenario_row.get("effectiveness", 0.5))
    raw_gain = effectiveness * deficit * cfg.lmci_gain_alpha
    # Cap by both the improvement cap and the remaining deficit
    return min(raw_gain, cfg.lmci_improvement_cap, deficit)


# ─────────────────────────────────────────────────────────────
# COMPUTE IMPACT MULTIPLIER (policy‑oriented)
# ─────────────────────────────────────────────────────────────

def compute_impact_multiplier(
    station_row: pd.Series,
    scenario_row: pd.Series,
    cfg: SimulationConfig = CFG,
) -> float:
    """
    Restore the original policy multipliers (additive).

    - persistent desert  → +desert_recovery_factor
    - MCLP selected      → +feeder_bonus_factor
    - temporal gap > threshold → +temporal_stability_factor
    - multimodal scenario→ +multimodal_bonus_factor

    (An alternative would be multiplicative: 1.15 * 1.12 * ... but
     we keep additive for simplicity and consistency with the original.)
    """
    multiplier = 1.0

    is_desert = str(station_row.get("desert_severity", "None")).lower() == "persistent"
    is_mclp = bool(station_row.get("mclp_selected", False))
    temporal_gap = float(station_row.get("temporal_gap", 0.0))
    is_multimodal = scenario_row.get("intervention_type") == "multimodal"

    if is_desert:
        multiplier += cfg.desert_recovery_factor
    if is_mclp:
        multiplier += cfg.feeder_bonus_factor
    if temporal_gap > cfg.temporal_gap_threshold:
        multiplier += cfg.temporal_stability_factor
    if is_multimodal:
        multiplier += cfg.multimodal_bonus_factor

    return multiplier


# ─────────────────────────────────────────────────────────────
# STATION IMPACT MODEL (multiplicative, no magic constants)
# ─────────────────────────────────────────────────────────────

def estimate_station_impact(
    station_row: pd.Series,
    scenario_row: pd.Series,
    cfg: SimulationConfig = CFG,
) -> Dict[str, object]:
    """
    Model station-level accessibility impact.

    impact = lmci_gain × (priority_index/10) × (1+demand_norm) × impact_multiplier

    The score is a dimensionless relative ranking metric, not a bounded index.
    Higher values indicate stronger expected accessibility improvement.
    """
    lmci_gain = compute_lmci_gain(station_row, scenario_row, cfg)
    current_lmci = float(
        station_row.get(
            "LMCI_new2",
            station_row.get("LMCI_new", station_row.get("LMCI_old", 0.0)),
        )
    )
    simulated_lmci = current_lmci + lmci_gain

    priority_index = float(station_row.get("priority_index", 0.0))
    demand_norm = float(station_row.get("demand_norm", 0.0))
    multiplier = compute_impact_multiplier(station_row, scenario_row, cfg)

    # Multiplicative impact – no arbitrary weights
    impact_score = (
        lmci_gain
        * (priority_index / 10.0)
        * (1.0 + demand_norm)
        * multiplier
    )
    impact_score = round(impact_score, 3)

    return {
        "stop_id": station_row["stop_id"],
        "stop_name": station_row["stop_name"],
        "scenario_id": scenario_row["scenario_id"],
        "scenario_name": scenario_row["scenario_name"],
        "intervention_type": scenario_row["intervention_type"],
        "cost_band": scenario_row["cost_band"],
        "priority_index": priority_index,
        "current_lmci": current_lmci,
        "simulated_lmci": simulated_lmci,
        "lmci_gain": lmci_gain,
        "lmci_deficit": 10.0 - current_lmci,
        "effectiveness": float(scenario_row["effectiveness"]),
        "impact_multiplier": multiplier,
        "simulation_impact_score": impact_score,
        "demand_norm": demand_norm,
        "mclp_selected": bool(station_row.get("mclp_selected", False)),
        "desert_severity": station_row.get("desert_severity", "None"),
        "stop_lat": station_row.get("stop_lat", np.nan),
        "stop_lon": station_row.get("stop_lon", np.nan),
    }


# ─────────────────────────────────────────────────────────────
# RUN ALL SCENARIOS
# ─────────────────────────────────────────────────────────────

def run_station_scenarios(
    station_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> pd.DataFrame:
    """Run all intervention scenarios across all stations."""

    rows: List[Dict[str, object]] = []

    for _, station in station_df.iterrows():
        for _, scenario in scenarios_df.iterrows():
            rows.append(
                estimate_station_impact(
                    station_row=station,
                    scenario_row=scenario,
                    cfg=cfg,
                )
            )

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("Simulation produced zero rows.")

    return out


# ─────────────────────────────────────────────────────────────
# RANK BEST INTERVENTIONS
# ─────────────────────────────────────────────────────────────

def rank_best_interventions(
    simulation_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> pd.DataFrame:
    """Pick best intervention scenario per station based on impact score."""

    # Sort and select best per station
    out = (
        simulation_df
        .sort_values(
            ["stop_name", "simulation_impact_score"],
            ascending=[True, False],
        )
        .groupby("stop_name", as_index=False)
        .first()
    )

    out = out.sort_values(
        "simulation_impact_score",
        ascending=False,
    ).reset_index(drop=True)

    out["intervention_rank"] = np.arange(1, len(out) + 1)

    # Add priority band using thresholds from scoring
    def get_priority_band(priority):
        if priority >= cfg.priority_high:
            return "High"
        elif priority >= cfg.priority_medium:
            return "Medium-High"
        elif priority >= cfg.priority_low:
            return "Medium-Low"
        else:
            return "Low"

    out["priority_band"] = out["priority_index"].apply(get_priority_band)

    return out


# ─────────────────────────────────────────────────────────────
# NETWORK SUMMARY
# ─────────────────────────────────────────────────────────────

def build_network_summary(
    best_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> pd.DataFrame:
    """Create executive simulation metrics."""

    top_interventions = best_df.head(cfg.top_interventions_count)

    # Fix: ensure correct parentheses for boolean condition
    critical_stations_improved = int(
        (
            (best_df["priority_index"] >= cfg.priority_medium)
            &
            (best_df["lmci_gain"] > 0.05)
        ).sum()
    )

    metrics = {
        "stations_simulated": len(best_df),
        "scenarios_tested": simulation_df["scenario_id"].nunique(),
        "top_interventions_evaluated": len(top_interventions),
        "mean_impact_score": float(best_df["simulation_impact_score"].mean()),
        "mean_lmci_gain": float(best_df["lmci_gain"].mean()),
        "stations_improved": int((best_df["lmci_gain"] > 0.05).sum()),
        "critical_stations_improved": critical_stations_improved,
        "persistent_deserts_targeted": int(
            top_interventions["desert_severity"]
            .astype(str)
            .str.lower()
            .eq("persistent")
            .sum()
        ),
        "multimodal_projects_recommended": int(
            top_interventions["intervention_type"]
            .astype(str)
            .str.lower()
            .eq("multimodal")
            .sum()
        ),
        "feeder_projects_recommended": int(
            top_interventions["intervention_type"]
            .astype(str)
            .str.lower()
            .eq("feeder")
            .sum()
        ),
    }

    return pd.DataFrame([metrics])


# ─────────────────────────────────────────────────────────────
# SCENARIO RANKING
# ─────────────────────────────────────────────────────────────

def rank_intervention_types(
    simulation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate performance by intervention type."""

    grouped = (
        simulation_df
        .groupby(
            [
                "intervention_type",
                "scenario_name",
                "cost_band",
            ],
            as_index=False,
        )
        .agg({
            "lmci_gain": "mean",
            "simulation_impact_score": "mean",
            "stop_id": "count",
        })
        .rename(columns={"stop_id": "stations_affected"})
    )

    grouped = grouped.sort_values(
        "simulation_impact_score",
        ascending=False,
    ).reset_index(drop=True)

    grouped["intervention_type_rank"] = np.arange(1, len(grouped) + 1)

    return grouped


# ─────────────────────────────────────────────────────────────
# AUDIT REPORT
# ─────────────────────────────────────────────────────────────

def build_simulation_audit_report(
    simulation_df: pd.DataFrame,
    best_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> pd.DataFrame:
    """Create comprehensive explainability audit report."""

    checks = []

    def add(name: str, passed: bool, detail: str):
        checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": detail,
        })

    add(
        "simulation_rows_created",
        len(simulation_df) > 0,
        f"rows={len(simulation_df)}",
    )

    add(
        "best_interventions_available",
        len(best_df) > 0,
        f"rows={len(best_df)}",
    )

    add(
        "simulation_impact_score_nonnegative",
        (simulation_df["simulation_impact_score"] >= 0).all(),
        "all impact scores are non-negative",
    )

    add(
        "lmci_gain_nonnegative",
        (simulation_df["lmci_gain"] >= 0).all(),
        "all lmci gains are non-negative",
    )

    add(
        "lmci_not_exceed_cap",
        (simulation_df["lmci_gain"] <= cfg.lmci_improvement_cap).all(),
        f"no lmci gain exceeds cap of {cfg.lmci_improvement_cap}",
    )

    add(
        "simulated_lmci_within_bounds",
        (simulation_df["simulated_lmci"] <= 10.0).all(),
        "all simulated LMCI values ≤ 10",
    )

    add(
        "priority_between_0_and_10",
        simulation_df["priority_index"].between(0, 10).all(),
        "all priority_index values are in [0,10]",
    )

    add(
        "scenario_id_unique",
        simulation_df["scenario_id"].nunique() >= 3,
        f"unique scenarios={simulation_df['scenario_id'].nunique()}",
    )

    expected_rows = simulation_df["stop_id"].nunique() * simulation_df["scenario_id"].nunique()
    add(
        "scenario_count_correct",
        len(simulation_df) == expected_rows,
        f"expected {expected_rows}, got {len(simulation_df)}",
    )

    add(
        "priority_index_present",
        "priority_index" in best_df.columns,
        "priority_index column exists",
    )

    add(
        "demand_norm_valid",
        best_df["demand_norm"].between(0, 1).all(),
        "all demand_norm values between 0 and 1",
    )

    add(
        "best_scenario_per_station",
        len(best_df) == simulation_df["stop_id"].nunique(),
        f"one best scenario per {simulation_df['stop_id'].nunique()} station",
    )

    add(
        "impact_score_sorted",
        best_df["simulation_impact_score"].is_monotonic_decreasing,
        "impact scores sorted descending",
    )

    return pd.DataFrame(checks)


# ─────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────

def export_simulation_outputs(
    simulation_df: pd.DataFrame,
    best_df: pd.DataFrame,
    network_summary_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    cfg: SimulationConfig = CFG,
) -> Dict[str, Path]:
    """Export all simulation outputs with enriched GeoJSON."""

    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.assets_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    impacts_path = cfg.outputs_dir / "simulation_station_impacts.csv"
    best_df.to_csv(impacts_path, index=False)
    paths["simulation_station_impacts"] = impacts_path

    summary_path = cfg.outputs_dir / "simulation_network_summary.csv"
    network_summary_df.to_csv(summary_path, index=False)
    paths["simulation_network_summary"] = summary_path

    ranking_path = cfg.outputs_dir / "simulation_intervention_ranking.csv"
    ranking_df.to_csv(ranking_path, index=False)
    paths["simulation_intervention_ranking"] = ranking_path

    scenarios_path = cfg.outputs_dir / "simulation_scenarios.csv"
    simulation_df.to_csv(scenarios_path, index=False)
    paths["simulation_scenarios"] = scenarios_path

    audit_path = cfg.outputs_dir / "simulation_audit_report.csv"
    audit_df.to_csv(audit_path, index=False)
    paths["simulation_audit_report"] = audit_path

    # Enhanced GeoJSON with all key metrics
    try:
        geojson_df = best_df.copy()

        # Keep lat/lon columns – do NOT filter them out
        geojson_cols = [
            "stop_id", "stop_name", "priority_index", "priority_band",
            "simulation_impact_score", "scenario_name", "intervention_type",
            "lmci_gain", "simulated_lmci", "current_lmci", "demand_norm",
            "stop_lat", "stop_lon"   # essential for GeoJSON
        ]
        available_cols = [c for c in geojson_cols if c in geojson_df.columns]
        geojson_df = geojson_df[available_cols]

        # Ensure we have lat/lon
        if "stop_lat" in geojson_df.columns and "stop_lon" in geojson_df.columns:
            gdf = to_gdf(
                geojson_df,
                lat_col="stop_lat",
                lon_col="stop_lon",
                crs_out=cfg.wgs84_crs,
            )

            geojson_path = cfg.assets_dir / "simulation_station_impacts.geojson"
            gdf.to_file(geojson_path, driver="GeoJSON")
            paths["simulation_station_impacts_geojson"] = geojson_path
        else:
            logger.warning("stop_lat / stop_lon missing – GeoJSON not created")

    except Exception as exc:
        logger.warning(f"Simulation GeoJSON export skipped: {exc}")

    logger.info("Simulation exports complete:")
    for key, path in paths.items():
        logger.info(f"  {key}: {path}")

    return paths


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_simulation_outputs(best_df: pd.DataFrame) -> None:
    """Fail-fast validation for dashboard safety."""

    required = [
        "stop_id",
        "stop_name",
        "scenario_name",
        "simulated_lmci",
        "lmci_gain",
        "simulation_impact_score",
        "priority_index",
        "demand_norm",
    ]

    missing = [c for c in required if c not in best_df.columns]

    if missing:
        raise ValueError(f"Simulation output missing columns: {missing}")

    if best_df["simulation_impact_score"].isna().any():
        raise ValueError("simulation_impact_score contains NaN values")

    if (best_df["lmci_gain"] < 0).any():
        raise ValueError("Negative LMCI gains detected")

    if (best_df["lmci_gain"] > 3.0).any():
        raise ValueError("LMCI gains exceed cap of 3.0")

    if (best_df["simulated_lmci"] > 10.0).any():
        raise ValueError("Simulated LMCI exceeds 10.0")

    if not best_df["demand_norm"].between(0, 1).all():
        raise ValueError("demand_norm values outside [0,1] range")

    if not best_df["priority_index"].between(0, 10).all():
        raise ValueError("priority_index values outside [0,10] range")

    if best_df["scenario_id"].nunique() < 5:
        logger.warning(f"Only {best_df['scenario_id'].nunique()} scenarios found, expected 5")

    logger.info("Simulation validation passed.")


# ─────────────────────────────────────────────────────────────
# END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────

def run_simulation_pipeline(
    cfg: SimulationConfig = CFG,
) -> Dict[str, object]:
    """Run full simulation pipeline."""

    logger.info("=" * 70)
    logger.info("STARTING SIMULATION PIPELINE")
    logger.info("=" * 70)

    # Load inputs
    inputs = load_simulation_inputs(cfg)

    # Prepare simulation inputs
    logger.info("Preparing simulation inputs")
    station_df = prepare_simulation_inputs(
        station_df=inputs["station_priority"],
        mclp_selected=inputs["mclp_selected"],
    )

    # Define scenarios
    logger.info("Defining intervention scenarios")
    scenarios_df = define_intervention_scenarios()

    # Compute station impacts
    logger.info("Computing station impacts")
    simulation_df = run_station_scenarios(
        station_df=station_df,
        scenarios_df=scenarios_df,
        cfg=cfg,
    )

    # Rank best interventions
    logger.info("Ranking best interventions")
    best_df = rank_best_interventions(
        simulation_df=simulation_df,
        cfg=cfg,
    )

    # Build summary
    logger.info("Building summary")
    network_summary_df = build_network_summary(
        best_df=best_df,
        simulation_df=simulation_df,
        cfg=cfg,
    )

    # Rank intervention types
    logger.info("Ranking intervention types")
    ranking_df = rank_intervention_types(simulation_df)

    # Audit
    logger.info("Building audit report")
    audit_df = build_simulation_audit_report(
        simulation_df=simulation_df,
        best_df=best_df,
        cfg=cfg,
    )

    # Validate
    logger.info("Validating outputs")
    validate_simulation_outputs(best_df)

    # Export
    logger.info("Exporting outputs")
    export_paths = export_simulation_outputs(
        simulation_df=simulation_df,
        best_df=best_df,
        network_summary_df=network_summary_df,
        ranking_df=ranking_df,
        audit_df=audit_df,
        cfg=cfg,
    )

    logger.info("=" * 70)
    logger.info("SIMULATION PIPELINE COMPLETE")
    logger.info(f"Stations simulated: {len(best_df)}")
    if len(best_df) > 0:
        logger.info(
            f"Top intervention: {best_df.iloc[0]['stop_name']} → "
            f"{best_df.iloc[0]['scenario_name']}"
        )
        logger.info(f"Top impact score: {best_df.iloc[0]['simulation_impact_score']:.3f}")
    logger.info("=" * 70)

    return {
        "simulation_df": simulation_df,
        "best_df": best_df,
        "network_summary_df": network_summary_df,
        "ranking_df": ranking_df,
        "audit_df": audit_df,
        "export_paths": export_paths,
    }


if __name__ == "__main__":
    run_simulation_pipeline()
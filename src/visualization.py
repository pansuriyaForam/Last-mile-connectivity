"""
src/visualization.py
────────────────────────────────────────────────────────────
Publication-Grade Visualization Engine
for the Hyderabad Metro Last-Mile Connectivity Project.

Purpose
-------
Generate deterministic, publication-quality figures from the
analytical outputs produced by:

- preprocessing.py
- scoring.py
- mclp.py
- simulation.py

The visualization layer is strictly downstream of the analytical
pipeline. It does not modify source data, introduce new analytical
weights, estimate ridership, or alter model outputs.

Primary objectives
------------------
1. Communicate LMCI spatial and distributional patterns.
2. Visualize station-level accessibility priorities.
3. Illustrate demand-service mismatch.
4. Show temporal accessibility instability.
5. Visualize MCLP coverage and marginal coverage gains.
6. Compare simulated intervention strategies.
7. Visualize station-level intervention impacts.
8. Provide an ablation/sensitivity figure for methodological analysis.
9. Produce publication-ready geospatial figures.
10. Produce concise dashboard-compatible visuals.

Research Visualization Principles
----------------------------------
- deterministic
- reproducible
- publication-quality
- high resolution
- accessible typography
- explicit units
- scientifically interpretable axes
- no unnecessary decorative elements
- no misleading 3D effects
- no truncated axes where they distort interpretation
- consistent terminology with the analytical pipeline
- figures suitable for academic papers and presentations

Analytical Integrity
--------------------
All plotted quantities must originate from project outputs.

The visualization module must NOT:
- invent new metrics,
- introduce undocumented weights,
- infer passenger counts,
- convert accessibility scores into ridership forecasts,
- modify LMCI/MCLP/simulation results,
- silently replace missing variables with unrelated metrics.

Simulation outputs represent indicative accessibility-impact scenarios,
not observed ridership forecasts.

Ablation / Sensitivity Visualization
-------------------------------------
At least one figure will explicitly examine the contribution or
sensitivity of major methodological components.

Depending on the available output data, this may compare:
- full LMCI vs component-removal variants,
- entropy-weighted vs alternative weighting,
- temporal components enabled vs disabled,
- MCLP-related prioritization enabled vs disabled,
- or another experimentally supported model variant.

An ablation figure will only be generated when the underlying
CSV contains sufficient evidence for the comparison. No synthetic
ablation values will be created inside this module.

Expected Outputs
----------------
assets/plots/
├── lmci_distribution.png
├── top_priority_stations.png
├── mismatch_distribution.png
├── temporal_gap_analysis.png
├── mclp_coverage_curve.png
├── intervention_ranking.png
├── simulation_impact.png
├── network_summary_dashboard.png
├── top5_conversion_insights.png
├── priority_map.png
└── ablation_study.png

Output Quality
--------------
Default export resolution: 300 DPI.

Figures should be suitable for:
- research paper inclusion,
- thesis/dissertation figures,
- presentations,
- technical reports,
- Streamlit dashboards.

The same analytical figure should not require modification of the
underlying CSV outputs to be rendered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s — %(message)s",
)

logger = logging.getLogger("VISUALIZATION")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

@dataclass
class VisualizationConfig:
    """
    Central configuration for the visualization engine.

    All analytical values are read from existing project outputs.
    This module does not generate new analytical scores.
    """

    # ─────────────────────────────────────────────────────────
    # DIRECTORIES
    # ─────────────────────────────────────────────────────────

    outputs_dir: Path = Path("outputs")
    assets_dir: Path = Path("assets")
    plots_dir: Path = Path("assets/plots")

    # ─────────────────────────────────────────────────────────
    # CORE INPUT FILES
    # ─────────────────────────────────────────────────────────

    station_priority_path: Path = (
        Path("outputs/station_priority_scores.csv")
    )

    mismatch_path: Path = (
        Path("outputs/demand_service_mismatch.csv")
    )

    lmci_scores_path: Path = (
        Path("outputs/lmci_station_scores.csv")
    )

    station_summary_path: Path = (
        Path("outputs/station_lmci_summary.csv")
    )

    # ─────────────────────────────────────────────────────────
    # MCLP INPUTS
    # ─────────────────────────────────────────────────────────

    mclp_candidate_path: Path = (
        Path("outputs/mclp_candidate_scores.csv")
    )

    mclp_selected_path: Path = (
        Path("outputs/mclp_selected_stations.csv")
    )

    mclp_coverage_path: Path = (
        Path("outputs/mclp_coverage_by_k.csv")
    )

    mclp_assignment_path: Path = (
        Path("outputs/mclp_demand_assignement.csv")
    )

    # ─────────────────────────────────────────────────────────
    # SIMULATION INPUTS
    # ─────────────────────────────────────────────────────────

    simulation_scenarios_path: Path = (
        Path("outputs/simulation_scenarios.csv")
    )

    simulation_impacts_path: Path = (
        Path("outputs/simulation_station_impacts.csv")
    )

    simulation_ranking_path: Path = (
        Path("outputs/simulation_intervention_ranking.csv")
    )

    simulation_summary_path: Path = (
        Path("outputs/simulation_network_summary.csv")
    )

    # ─────────────────────────────────────────────────────────
    # SUPPORTING OUTPUTS
    # ─────────────────────────────────────────────────────────

    executive_summary_path: Path = (
        Path("outputs/executive_summary_metrics.csv")
    )

    conversion_insights_path: Path = (
        Path("outputs/conversion_insights_top5.csv")
    )

    scoring_audit_path: Path = (
        Path("outputs/scoring_audit_report.csv")
    )

    simulation_audit_path: Path = (
        Path("outputs/simulation_audit_report.csv")
    )

    # ─────────────────────────────────────────────────────────
    # ABLATION INPUT
    # ─────────────────────────────────────────────────────────

    ablation_path: Path = (
        Path("outputs/ablation_results.csv")
    )

    # ─────────────────────────────────────────────────────────
    # GEOGRAPHIC DATA
    # ─────────────────────────────────────────────────────────

    station_coordinates_path: Path = (
        Path("outputs/station_coordinates.csv")
    )

    demand_points_path: Path = (
        Path("outputs/demand_points.csv")
    )

    multimodel_graph_path: Path = (
        Path("outputs/multimodel_graph.csv")
    )

    conversion_insights_path: Path = (
        Path("outputs/conversion_insights_top5.csv")
    )

    # ─────────────────────────────────────────────────────────
    # FIGURE SIZES
    # ─────────────────────────────────────────────────────────

    figsize_small: Tuple[int, int] = (8, 5)
    figsize_medium: Tuple[int, int] = (10, 6)
    figsize_large: Tuple[int, int] = (12, 7)
    figsize_wide: Tuple[int, int] = (14, 7)
    figsize_square: Tuple[int, int] = (9, 9)

    # ─────────────────────────────────────────────────────────
    # OUTPUT QUALITY
    # ─────────────────────────────────────────────────────────

    dpi: int = 300

    # Export both raster and vector formats for paper use.
    save_png: bool = True
    save_pdf: bool = True

    # ─────────────────────────────────────────────────────────
    # TYPOGRAPHY
    # ─────────────────────────────────────────────────────────

    title_size: int = 16
    axis_label_size: int = 12
    tick_size: int = 10
    annotation_size: int = 9
    legend_size: int = 10

    # ─────────────────────────────────────────────────────────
    # FIGURE PARAMETERS
    # ─────────────────────────────────────────────────────────

    lmci_bins: int = 12
    top_station_count: int = 10
    top_simulation_count: int = 10
    top_conversion_count: int = 5

    # ─────────────────────────────────────────────────────────
    # MAP PARAMETERS
    # ─────────────────────────────────────────────────────────

    map_point_size: float = 55
    map_alpha: float = 0.85

    def __post_init__(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


CFG = VisualizationConfig()


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────

def _load_csv(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and normalize column names.

    Parameters
    ----------
    path:
        CSV path.

    required:
        If True, missing files raise an exception.
        If False, missing files return None.
    """

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Required visualization input missing: {path}"
            )

        logger.warning(
            f"Optional visualization input missing: {path}"
        )
        return None

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    logger.info(
        f"Loaded {path.name}: {len(df):,} rows"
    )

    return df


def load_visualization_inputs(
    cfg: VisualizationConfig = CFG,
) -> Dict[str, pd.DataFrame]:
    """
    Load the core datasets required by the visualization pipeline.

    Additional datasets can be loaded by individual figure functions
    when needed.
    """

    return {
        "station_priority": _load_csv(cfg.station_priority_path),
        "mismatch": _load_csv(cfg.mismatch_path),
        "lmci_station": _load_csv(cfg.lmci_scores_path),
        "mclp_coverage": _load_csv(cfg.mclp_coverage_path),
        "simulation_station_impacts": _load_csv(
            cfg.simulation_impacts_path
        ),
        "simulation_ranking": _load_csv(
            cfg.simulation_ranking_path
        ),
    }


# ─────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────

def save_plot(
    fig,
    path: Path | str,
    cfg: VisualizationConfig = CFG,
):
    """
    Save a matplotlib figure safely.

    Accepts either pathlib.Path or string paths and guarantees
    PNG output with the configured publication DPI.
    """

    path = Path(path)

    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guarantee PNG extension
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=cfg.dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    logger.info(f"Saved plot: {path}")

    return path


# ─────────────────────────────────────────────────────────────
# PLOT STYLE HELPERS
# ─────────────────────────────────────────────────────────────

def _format_axes(
    ax: plt.Axes,
    cfg: VisualizationConfig = CFG,
) -> None:
    """Apply consistent publication-oriented axis formatting."""

    ax.title.set_fontsize(cfg.title_size)

    ax.xaxis.label.set_fontsize(cfg.axis_label_size)
    ax.yaxis.label.set_fontsize(cfg.axis_label_size)

    ax.tick_params(
        axis="both",
        labelsize=cfg.tick_size,
    )

    # Remove unnecessary top/right borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars(
    ax: plt.Axes,
    horizontal: bool = False,
    fontsize: int = 9,
) -> None:
    """Add numeric labels to bars."""

    for patch in ax.patches:

        if horizontal:
            width = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2

            ax.text(
                width,
                y,
                f" {width:.2f}",
                va="center",
                fontsize=fontsize,
            )

        else:
            height = patch.get_height()
            x = patch.get_x() + patch.get_width() / 2

            ax.text(
                x,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — LMCI DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_lmci_distribution(
    station_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Plot the distribution of final LMCI scores across stations.
    Uses LMCI_Final as the canonical final accessibility metric.
    """

    logger.info("Generating LMCI distribution")

    if "LMCI_Final" not in station_df.columns:
        raise ValueError(
            "LMCI_Final column missing from station priority data"
        )

    values = pd.to_numeric(
        station_df["LMCI_Final"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise ValueError(
            "No valid LMCI_Final values available for plotting"
        )

    fig, ax = plt.subplots(
        figsize=cfg.figsize_square
    )

    x_positions = np.zeros(len(values))
    jitter = np.random.default_rng(42).normal(
        loc=0,
        scale=0.07,
        size=len(values),
    )

    ax.scatter(
        x_positions + jitter,
        values.to_numpy(),
        s=28,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.35,
        alpha=0.8,
        zorder=3,
    )

    ax.boxplot(
        values.to_numpy(),
        vert=True,
        widths=0.25,
        patch_artist=True,
        boxprops={"facecolor": "lightblue", "alpha": 0.7},
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"color": "#2f4f4f"},
        capprops={"color": "#2f4f4f"},
        flierprops={"marker": "o", "markersize": 4},
    )

    mean_lmci = values.mean()

    ax.axhline(
        mean_lmci,
        linestyle="--",
        linewidth=1.5,
        color="green",
        label=f"Mean = {mean_lmci:.2f}",
    )

    ax.set_title(
        "Distribution of Final LMCI Across Hyderabad Metro Stations"
    )

    ax.set_xlabel(
        "Metro Stations"
    )

    ax.set_ylabel(
        "Final Last-Mile Connectivity Index (LMCI)"
    )

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 6.5)

    ax.set_xticks([])

    ax.legend(
        fontsize=cfg.legend_size,
        frameon=False,
    )

    _format_axes(ax, cfg)

    path = save_plot(
        fig,
        cfg.plots_dir / "01_lmci_distribution",
        cfg,
    )

    logger.info(
        f"LMCI range: {values.min():.2f}–{values.max():.2f}; "
        f"mean={mean_lmci:.2f}; n={len(values)}"
    )

    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — TOP PRIORITY STATIONS
# ─────────────────────────────────────────────────────────────

def plot_top_priority_stations(
    station_df: pd.DataFrame,
    top_n: int = 10,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Plot the highest-priority metro stations.

    Uses priority_index as the canonical intervention-priority metric.
    """

    logger.info(
        f"Generating top {top_n} priority station figure"
    )

    required = {
        "stop_name",
        "priority_index",
    }

    missing = required - set(station_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns for top-priority figure: {sorted(missing)}"
        )

    df = station_df[
        ["stop_name", "priority_index"]
    ].copy()

    df["priority_index"] = pd.to_numeric(
        df["priority_index"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["priority_index"])
        .sort_values(
            "priority_index",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "priority_index",
            ascending=True,
        )
    )

    if df.empty:
        raise ValueError(
            "No valid priority_index values available"
        )

    fig, ax = plt.subplots(
        figsize=cfg.figsize_large
    )

    ax.barh(
        df["stop_name"],
        df["priority_index"],
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title(
        f"Top {len(df)} Metro Stations by Intervention Priority"
    )

    ax.set_xlabel(
        "Priority Index"
    )

    ax.set_ylabel(
        "Metro Station"
    )

    ax.set_xlim(0, 10)

    _annotate_bars(
        ax,
        horizontal=True,
        fontsize=cfg.annotation_size,
    )

    _format_axes(ax, cfg)

    path = save_plot(
        fig,
        cfg.plots_dir / "02_top_priority_stations",
        cfg,
    )

    logger.info(
        "Top priority station: "
        f"{df.iloc[-1]['stop_name']} "
        f"(priority_index={df.iloc[-1]['priority_index']:.2f})"
    )

    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 3 — DEMAND-SERVICE MISMATCH DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_mismatch_distribution(
    mismatch_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Plot the distribution of demand-service mismatch classes.

    The mismatch classification is taken directly from the
    `mismatch_class` field produced by the analytical pipeline.

    Purpose
    -------
    Visualize how Metro stations are distributed across different
    demand-service accessibility conditions.

    Input
    -----
    demand_service_mismatch.csv

    Output
    ------
    assets/plots/03_demand_service_mismatch.png
    """

    required = {"stop_id", "stop_name", "mismatch_class"}

    missing = required - set(mismatch_df.columns)
    if missing:
        raise ValueError(
            f"Missing columns for mismatch distribution: {sorted(missing)}"
        )

    # Remove missing classifications rather than silently treating
    # them as a meaningful category.
    df = mismatch_df.dropna(subset=["mismatch_class"]).copy()

    if df.empty:
        raise ValueError("No valid mismatch classifications available.")

    # Count stations in each mismatch class.
    counts = (
        df["mismatch_class"]
        .astype(str)
        .str.strip()
        .value_counts()
    )

    total = counts.sum()

    # Keep the observed classes deterministic.
    # This avoids hard-coding categories that may change in future runs.
    counts = counts.sort_values(ascending=True)

    percentages = (counts / total) * 100

    fig, ax = plt.subplots(figsize=cfg.figsize_large)

    bars = ax.barh(
        counts.index,
        counts.values,
        edgecolor="black",
        linewidth=0.8,
    )

    # Add station count and percentage to each bar.
    for bar, count, pct in zip(
        bars,
        counts.values,
        percentages.values,
    ):
        ax.text(
            bar.get_width() + max(counts.values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({pct:.1f}%)",
            va="center",
            fontsize=11,
        )

    ax.set_title(
        "Distribution of Demand-Service Mismatch Classes",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Number of Metro Stations",
        fontsize=12,
    )

    ax.set_ylabel(
        "Mismatch Class",
        fontsize=12,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    # Give annotation labels some breathing room.
    ax.set_xlim(
        0,
        max(counts.values) * 1.18,
    )

    # Keep the figure visually clean for paper inclusion.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = save_plot(
        fig,
        cfg.plots_dir / "03_demand_service_mismatch.png",
        cfg,
    )

    logger.info(
        "Figure 3 generated: %d stations across %d mismatch classes",
        total,
        len(counts),
    )

    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 4 — TEMPORAL LMCI PROFILE
# ─────────────────────────────────────────────────────────────

def plot_temporal_lmci_profile(
    station_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Visualize the temporal distribution of station-level LMCI.

    Compares Morning, Midday, and Evening LMCI distributions across
    all Metro stations.

    Purpose
    -------
    Demonstrate temporal variation in last-mile accessibility.

    Input
    -----
    station_priority_scores.csv

    Output
    ------
    assets/plots/04_temporal_lmci_profile.png
    """

    required = {
        "Morning_LMCI",
        "Midday_LMCI",
        "Evening_LMCI",
    }

    missing = required - set(station_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns for temporal LMCI figure: {sorted(missing)}"
        )

    temporal_df = station_df[
        ["Morning_LMCI", "Midday_LMCI", "Evening_LMCI"]
    ].apply(pd.to_numeric, errors="coerce")

    temporal_df = temporal_df.dropna(how="all")

    if temporal_df.empty:
        raise ValueError(
            "No valid temporal LMCI observations available."
        )

    data = [
        temporal_df["Morning_LMCI"].dropna(),
        temporal_df["Midday_LMCI"].dropna(),
        temporal_df["Evening_LMCI"].dropna(),
    ]

    labels = [
        "Morning",
        "Midday",
        "Evening",
    ]

    fig, ax = plt.subplots(figsize=cfg.figsize_medium)

    # Distribution summary.
    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=False,
        widths=0.55,
        showmeans=True,
        meanline=True,
        medianprops={"linewidth": 2},
        meanprops={
            "linestyle": "--",
            "linewidth": 1.5,
        },
    )

    # Add lightly jittered observations so the figure retains
    # station-level information without becoming a line chart.
    rng = np.random.default_rng(42)

    for index, values in enumerate(data, start=1):

        jitter = rng.normal(
            index,
            0.045,
            size=len(values),
        )

        ax.scatter(
            jitter,
            values,
            s=18,
            alpha=0.45,
            edgecolors="none",
        )

    # Calculate temporal means for annotation.
    means = [
        float(values.mean())
        for values in data
    ]

    for index, mean_value in enumerate(means, start=1):

        ax.annotate(
            f"Mean = {mean_value:.2f}",
            xy=(index, mean_value),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_title(
        "Temporal Variation in Last-Mile Connectivity",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Time Period",
        fontsize=12,
    )

    ax.set_ylabel(
        "LMCI Score",
        fontsize=12,
    )

    ax.set_ylim(
        0,
        10.5,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = save_plot(
        fig,
        cfg.plots_dir / "04_temporal_lmci_profile.png",
        cfg,
    )

    logger.info(
        "Figure 4 generated: temporal LMCI profile "
        "(Morning=%.2f, Midday=%.2f, Evening=%.2f)",
        means[0],
        means[1],
        means[2],
    )

    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 5 — MCLP COVERAGE CURVE
# ─────────────────────────────────────────────────────────────

def plot_mclp_coverage_curve(
    coverage_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Visualize MCLP coverage efficiency as intervention stations
    are progressively selected.

    Purpose
    -------
    Demonstrate the relationship between the number of selected
    intervention stations and weighted demand coverage.

    The figure also reports marginal coverage gain between
    consecutive values of k.

    Input
    -----
    mclp_coverage_by_k.csv

    Output
    ------
    assets/plots/05_mclp_coverage_curve.png
    """

    required = {
        "k",
        "coverage_pct",
        "covered_weighted_demand",
        "total_weighted_demand",
    }

    missing = required - set(coverage_df.columns)

    if missing:
        raise ValueError(
            f"Missing columns for MCLP coverage figure: {sorted(missing)}"
        )

    df = coverage_df.copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df
        .dropna(subset=["k", "coverage_pct"])
        .sort_values("k")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No valid MCLP coverage observations available."
        )

    # Calculate marginal coverage gain.
    df["marginal_coverage_gain"] = (
        df["coverage_pct"].diff()
    )

    # First point has no previous intervention count.
    df.loc[
        df.index[0],
        "marginal_coverage_gain"
    ] = np.nan

    fig, ax = plt.subplots(
        figsize=cfg.figsize_large
    )

    # Main coverage curve.
    ax.plot(
        df["k"],
        df["coverage_pct"],
        marker="o",
        linewidth=2.5,
        markersize=7,
    )

    # Highlight k=5 — project's selected configuration
    selected_k = 5
    selected = df[df["k"] == selected_k]

    if not selected.empty:
        selected_row = selected.iloc[0]
        k5_coverage = float(selected_row["coverage_pct"])

        ax.scatter(
            selected_row["k"],
            k5_coverage,
            s=120,
            zorder=5,
        )

        ax.annotate(
            f"Selected solution\n"
            f"k = {selected_k}\n"
            f"Coverage = {k5_coverage:.1f}%",
            xy=(
                selected_row["k"],
                k5_coverage,
            ),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                alpha=0.9,
            ),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.2,
            ),
        )

        logger.info(
            "MCLP coverage at k=5: %.2f%%",
            k5_coverage,
        )

    # Annotate coverage values.
    for _, row in df.iterrows():

        ax.annotate(
            f"{row['coverage_pct']:.1f}%",
            (
                row["k"],
                row["coverage_pct"],
            ),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_title(
        "MCLP Weighted Demand Coverage vs. Selected Stations",
        fontsize=16,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Number of Selected Intervention Stations (k)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Weighted Demand Coverage (%)",
        fontsize=12,
    )

    ax.set_ylim(
        0,
        min(105, max(df["coverage_pct"]) * 1.15),
    )

    ax.set_xticks(
        df["k"].astype(int)
    )

    ax.grid(
        axis="both",
        linestyle="--",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Research-friendly explanatory note
    ax.text(
        0.01,
        -0.16,
        "Higher coverage indicates greater weighted demand served by "
        "the selected intervention locations.",
        transform=ax.transAxes,
        fontsize=9.5,
        alpha=0.75,
    )

    path = save_plot(
        fig,
        cfg.plots_dir / "05_mclp_coverage_curve.png",
        cfg,
    )

    logger.info(
        "Figure 5 generated: MCLP coverage from %.1f%% to %.1f%% "
        "across k=%d to k=%d",
        df["coverage_pct"].iloc[0],
        df["coverage_pct"].iloc[-1],
        int(df["k"].iloc[0]),
        int(df["k"].iloc[-1]),
    )

    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 6 — INTERVENTION STRATEGY RANKING
# ─────────────────────────────────────────────────────────────

def plot_intervention_ranking(
    ranking_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Figure 6: Rank intervention strategies by mean simulation impact.

    Source
    ------
    simulation_intervention_ranking.csv

    Interpretation
    --------------
    Each intervention type is ranked according to its mean
    station-level simulation impact score.

    The figure also reports mean LMCI gain and the number of
    stations affected, providing context for the ranking.
    """

    required = {
        "scenario_name",
        "intervention_type",
        "cost_band",
        "lmci_gain",
        "simulation_impact_score",
        "stations_affected",
        "intervention_type_rank",
    }

    missing = required - set(ranking_df.columns)

    if missing:
        raise ValueError(
            f"Figure 6 missing columns: {sorted(missing)}"
        )

    df = ranking_df.copy()

    # Numeric conversion
    for col in [
        "lmci_gain",
        "simulation_impact_score",
        "stations_affected",
        "intervention_type_rank",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(
            subset=[
                "scenario_name",
                "simulation_impact_score",
            ]
        )
        .sort_values(
            "simulation_impact_score",
            ascending=True,
        )
    )

    if df.empty:
        raise ValueError(
            "No valid intervention ranking data available."
        )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=cfg.figsize_large
    )

    bars = ax.barh(
        df["scenario_name"],
        df["simulation_impact_score"],
        height=0.62,
    )

    # ---------------------------------------------------------
    # Value labels
    # ---------------------------------------------------------

    max_score = df["simulation_impact_score"].max()

    for bar, (_, row) in zip(bars, df.iterrows()):

        score = row["simulation_impact_score"]

        ax.text(
            bar.get_width() + max_score * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    ax.set_title(
        "Intervention Strategy Ranking",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Mean Simulation Impact Score",
        fontsize=12,
    )

    ax.set_ylabel(
        "Intervention Strategy",
        fontsize=12,
    )

    ax.set_xlim(
        0,
        max_score * 1.18 if max_score > 0 else 1,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        linestyle="--",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------------------------------
    # Add rank / LMCI gain / cost information
    # ---------------------------------------------------------

    for i, (_, row) in enumerate(df.iterrows()):

        ax.text(
            0.01,
            i,
            (
                f"LMCI +{row['lmci_gain']:.2f}  |  "
                f"{int(row['stations_affected'])} stations  |  "
                f"Cost: {row['cost_band']}"
            ),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            alpha=0.75,
        )

    # ---------------------------------------------------------
    # Research note
    # ---------------------------------------------------------

    ax.text(
        0.01,
        -0.14,
        "Higher scores indicate greater simulated accessibility "
        "impact under the defined intervention assumptions.",
        transform=ax.transAxes,
        fontsize=9.5,
        alpha=0.75,
    )

    path = cfg.plots_dir / "06_intervention_ranking.png"

    save_plot(
        fig,
        path,
        cfg,
    )

    logger.info(
        "Figure 6 generated: %d intervention strategies ranked",
        len(df),
    )

    logger.info(
        "Top intervention strategy: %s",
        df.iloc[-1]["scenario_name"],
    )

    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 7 — TOP STATION-LEVEL SIMULATED IMPACTS
# ─────────────────────────────────────────────────────────────

def plot_simulation_impact(
    simulation_df: pd.DataFrame,
    top_n: int = 10,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Figure 7: Top station-level simulated intervention impacts.

    Source
    ------
    simulation_station_impacts.csv

    The input contains the winning intervention scenario for each
    station, as produced by rank_best_interventions().

    Therefore, each station appears once and represents its
    highest-scoring intervention scenario.
    """

    required = {
        "stop_id",
        "stop_name",
        "scenario_name",
        "intervention_type",
        "simulation_impact_score",
        "lmci_gain",
        "current_lmci",
        "simulated_lmci",
        "priority_index",
    }

    missing = required - set(simulation_df.columns)

    if missing:
        raise ValueError(
            f"Figure 7 missing columns: {sorted(missing)}"
        )

    df = simulation_df.copy()

    # ---------------------------------------------------------
    # Numeric conversion
    # ---------------------------------------------------------

    numeric_cols = [
        "simulation_impact_score",
        "lmci_gain",
        "current_lmci",
        "simulated_lmci",
        "priority_index",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "stop_name",
            "simulation_impact_score",
        ]
    )

    # ---------------------------------------------------------
    # Keep highest-impact stations
    # ---------------------------------------------------------

    df = (
        df.sort_values(
            "simulation_impact_score",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    if df.empty:
        raise ValueError(
            "No valid station-level simulation results available."
        )

    # Reverse for horizontal bar chart
    df = df.sort_values(
        "simulation_impact_score",
        ascending=True,
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=cfg.figsize_large
    )

    bars = ax.barh(
        df["stop_name"],
        df["simulation_impact_score"],
        height=0.62,
    )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    max_score = df["simulation_impact_score"].max()

    for bar, (_, row) in zip(bars, df.iterrows()):

        score = row["simulation_impact_score"]

        ax.text(
            bar.get_width() + max_score * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    ax.set_title(
        f"Top {len(df)} Station-Level Intervention Impacts",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Simulation Impact Score",
        fontsize=12,
    )

    ax.set_ylabel(
        "Metro Station",
        fontsize=12,
    )

    ax.set_xlim(
        0,
        max_score * 1.20 if max_score > 0 else 1,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        linestyle="--",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------------------------------
    # Add intervention labels
    # ---------------------------------------------------------

    for i, (_, row) in enumerate(df.iterrows()):

        ax.text(
            0.01,
            i,
            (
                f"{row['scenario_name']}  |  "
                f"LMCI +{row['lmci_gain']:.2f}"
            ),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            alpha=0.75,
        )

    # ---------------------------------------------------------
    # Research interpretation note
    # ---------------------------------------------------------

    ax.text(
        0.01,
        -0.14,
        "Each station is represented by its highest-impact simulated "
        "intervention scenario.",
        transform=ax.transAxes,
        fontsize=9.5,
        alpha=0.75,
    )

    path = cfg.plots_dir / "07_simulation_impact.png"

    save_plot(
        fig,
        path,
        cfg,
    )

    logger.info(
        "Figure 7 generated: top %d station-level interventions",
        len(df),
    )

    logger.info(
        "Highest simulated impact: %s → %s (%.3f)",
        df.iloc[-1]["stop_name"],
        df.iloc[-1]["scenario_name"],
        df.iloc[-1]["simulation_impact_score"],
    )

    return path


# ─────────────────────────────────────────────────────────────
# FIGURE 8 — ABLATION STUDY
# ─────────────────────────────────────────────────────────────

def plot_ablation_study(
    cfg: VisualizationConfig = CFG,
) -> Optional[Path]:
    """
    Figure 8: Ablation study of equity weighting in MCLP.

    Visualizes the results of the MCLP ablation experiment comparing
    five model variants:

    1. Full Model — equity weighting enabled (need × opportunity) with desert bonus
    2. Demand-Only Baseline — pure coverage maximization
    3. Need-Only — opportunity_score set to 1.0
    4. Opportunity-Only — need_score set to 1.0
    5. No Desert Bonus — equity weighting without persistent-desert bonus

    The figure is a multi-panel visualization showing:
    - Panel A: Coverage percentage by variant
    - Panel B: Jaccard similarity to Full Model
    - Panel C: Persistent deserts selected by variant
    - Panel D: Rank correlation with Full Model

    Only generated when ablation_results.csv exists and contains
    sufficient variants for meaningful comparison.

    Input
    -----
    outputs/ablation_results.csv

    Output
    ------
    assets/plots/08_ablation_study.png
    """

    logger.info("Generating ablation study figure")

    ablation_df = _load_csv(cfg.ablation_path, required=False)

    if ablation_df is None:
        logger.warning(
            "Ablation results not found. Skipping Figure 8. "
            "Run mclp.py to generate ablation_results.csv."
        )
        return None

    if len(ablation_df) < 2:
        logger.warning(
            "Ablation results contain fewer than 2 variants. "
            "Skipping Figure 8 — insufficient data for comparison."
        )
        return None

    # ─────────────────────────────────────────────────────────
    # Data preparation
    # ─────────────────────────────────────────────────────────

    required_cols = {
        "variant",
        "coverage_pct",
        "jaccard_similarity",
        "persistent_deserts_selected",
        "rank_correlation_with_full",
        "overlap_with_full_model",
        "n_selected",
    }

    missing = required_cols - set(ablation_df.columns)

    if missing:
        logger.warning(
            f"Ablation results missing columns: {sorted(missing)}. "
            "Skipping Figure 8."
        )
        return None

    # Convert numeric columns
    numeric_cols = [
        "coverage_pct",
        "jaccard_similarity",
        "persistent_deserts_selected",
        "rank_correlation_with_full",
        "overlap_with_full_model",
        "n_selected",
    ]

    for col in numeric_cols:
        ablation_df[col] = pd.to_numeric(
            ablation_df[col],
            errors="coerce",
        )

    df = ablation_df.dropna(
        subset=["variant", "coverage_pct"]
    )

    if df.empty:
        logger.warning(
            "No valid ablation data after cleaning. Skipping Figure 8."
        )
        return None

    # ─────────────────────────────────────────────────────────
    # Color scheme
    # ─────────────────────────────────────────────────────────

    # Full Model gets a distinct color; other variants are distinguishable
    variant_colors = {
        "Full Model": "#1f77b4",              # blue
        "Demand-Only Baseline": "#d62728",     # red
        "Need-Only (No Opportunity)": "#ff7f0e",  # orange
        "Opportunity-Only (No Need)": "#2ca02c",  # green
        "No Desert Bonus": "#9467bd",          # purple
    }

    def get_color(variant_name: str) -> str:
        return variant_colors.get(variant_name, "#7f7f7f")

    # ─────────────────────────────────────────────────────────
    # Create 2×2 panel figure
    # ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 12),
    )

    # Flatten for easy indexing
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    # Sort: Full Model first, then alphabetically
    variant_order = ["Full Model"] + sorted(
        [v for v in df["variant"].unique() if v != "Full Model"]
    )
    # Keep only variants that exist in the data
    variant_order = [v for v in variant_order if v in df["variant"].values]

    df_plot = df.set_index("variant").loc[variant_order].reset_index()

    bar_width = 0.55
    x_positions = np.arange(len(df_plot))

    # ─────────────────────────────────────────────────────────
    # Panel A: Coverage Percentage
    # ─────────────────────────────────────────────────────────

    colors_a = [get_color(v) for v in df_plot["variant"]]

    bars_a = ax_a.bar(
        x_positions,
        df_plot["coverage_pct"],
        bar_width,
        color=colors_a,
        edgecolor="black",
        linewidth=0.8,
    )

    # Add value labels
    for bar, pct in zip(bars_a, df_plot["coverage_pct"]):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(df_plot["coverage_pct"]) * 0.015,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax_a.set_title(
        "A. Weighted Demand Coverage",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    ax_a.set_ylabel(
        "Coverage (%)",
        fontsize=12,
    )

    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(
        df_plot["variant"],
        rotation=25,
        ha="right",
        fontsize=9,
    )

    ax_a.set_ylim(
        0,
        max(df_plot["coverage_pct"]) * 1.15 if max(df_plot["coverage_pct"]) > 0 else 100,
    )

    ax_a.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ─────────────────────────────────────────────────────────
    # Panel B: Jaccard Similarity to Full Model
    # ─────────────────────────────────────────────────────────

    # Jaccard is 1.0 for Full Model (self-comparison), skip it for other variants
    df_non_full = df_plot[df_plot["variant"] != "Full Model"].copy()

    if not df_non_full.empty:
        x_pos_b = np.arange(len(df_non_full))
        colors_b = [get_color(v) for v in df_non_full["variant"]]

        bars_b = ax_b.bar(
            x_pos_b,
            df_non_full["jaccard_similarity"],
            bar_width,
            color=colors_b,
            edgecolor="black",
            linewidth=0.8,
        )

        # Add value labels
        for bar, jaccard, overlap, n_sel in zip(
            bars_b,
            df_non_full["jaccard_similarity"],
            df_non_full["overlap_with_full_model"],
            df_non_full["n_selected"],
        ):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{jaccard:.3f}\n({int(overlap)}/{int(n_sel)})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax_b.set_title(
            "B. Jaccard Similarity with Full Model",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )

        ax_b.set_ylabel(
            "Jaccard Similarity",
            fontsize=12,
        )

        ax_b.set_xticks(x_pos_b)
        ax_b.set_xticklabels(
            df_non_full["variant"],
            rotation=25,
            ha="right",
            fontsize=9,
        )

        ax_b.set_ylim(0, 1.15)

        ax_b.grid(
            axis="y",
            linestyle="--",
            alpha=0.25,
        )

        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)

        # Add reference line at Jaccard = 0.5
        ax_b.axhline(
            y=0.5,
            linestyle=":",
            color="gray",
            alpha=0.5,
            linewidth=1,
        )
    else:
        ax_b.text(
            0.5, 0.5,
            "No non-Full variants to compare",
            ha="center",
            va="center",
            transform=ax_b.transAxes,
            fontsize=12,
        )
        ax_b.set_title(
            "B. Jaccard Similarity with Full Model",
            fontsize=14,
            fontweight="bold",
        )

    # ─────────────────────────────────────────────────────────
    # Panel C: Persistent Deserts Selected
    # ─────────────────────────────────────────────────────────

    colors_c = [get_color(v) for v in df_plot["variant"]]

    bars_c = ax_c.bar(
        x_positions,
        df_plot["persistent_deserts_selected"],
        bar_width,
        color=colors_c,
        edgecolor="black",
        linewidth=0.8,
    )

    # Add value labels
    max_deserts = max(df_plot["persistent_deserts_selected"].max(), 1)

    for bar, deserts in zip(bars_c, df_plot["persistent_deserts_selected"]):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_deserts * 0.05,
            f"{int(deserts)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax_c.set_title(
        "C. Persistent Transit Deserts Selected",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    ax_c.set_ylabel(
        "Number of Deserts Selected",
        fontsize=12,
    )

    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels(
        df_plot["variant"],
        rotation=25,
        ha="right",
        fontsize=9,
    )

    ax_c.set_ylim(
        0,
        max_deserts * 1.25 if max_deserts > 0 else 5,
    )

    ax_c.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ─────────────────────────────────────────────────────────
    # Panel D: Rank Correlation with Full Model
    # ─────────────────────────────────────────────────────────

    if not df_non_full.empty:
        x_pos_d = np.arange(len(df_non_full))
        colors_d = [get_color(v) for v in df_non_full["variant"]]

        bars_d = ax_d.bar(
            x_pos_d,
            df_non_full["rank_correlation_with_full"],
            bar_width,
            color=colors_d,
            edgecolor="black",
            linewidth=0.8,
        )

        # Add value labels
        for bar, corr in zip(bars_d, df_non_full["rank_correlation_with_full"]):
            label = f"{corr:.3f}" if not np.isnan(corr) else "N/A"

            ax_d.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03 if not np.isnan(corr) else bar.get_height() / 2,
                label,
                ha="center",
                va="bottom" if not np.isnan(corr) else "center",
                fontsize=10,
                fontweight="bold",
            )

        ax_d.set_title(
            "D. Spearman Rank Correlation with Full Model",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )

        ax_d.set_ylabel(
            "Spearman ρ",
            fontsize=12,
        )

        ax_d.set_xticks(x_pos_d)
        ax_d.set_xticklabels(
            df_non_full["variant"],
            rotation=25,
            ha="right",
            fontsize=9,
        )

        ax_d.set_ylim(-0.1, 1.1)

        ax_d.grid(
            axis="y",
            linestyle="--",
            alpha=0.25,
        )

        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)

        # Add reference lines
        ax_d.axhline(
            y=0.9,
            linestyle=":",
            color="gray",
            alpha=0.5,
            linewidth=1,
        )
        ax_d.axhline(
            y=0.7,
            linestyle=":",
            color="gray",
            alpha=0.5,
            linewidth=1,
        )
    else:
        ax_d.text(
            0.5, 0.5,
            "No non-Full variants to compare",
            ha="center",
            va="center",
            transform=ax_d.transAxes,
            fontsize=12,
        )
        ax_d.set_title(
            "D. Spearman Rank Correlation with Full Model",
            fontsize=14,
            fontweight="bold",
        )

    # ─────────────────────────────────────────────────────────
    # Overall figure title
    # ─────────────────────────────────────────────────────────

    fig.suptitle(
        "Ablation Study: Contribution of Equity Weighting in MCLP",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    # Research note at bottom
    fig.text(
        0.5,
        -0.02,
        "Full Model = equity weighting (need × opportunity) with desert bonus. "
        "All variants use identical coverage geometry, demand points, and k={}.".format(
            int(df_plot["n_selected"].iloc[0]) if "n_selected" in df_plot.columns else "5"
        ),
        ha="center",
        fontsize=10,
        fontstyle="italic",
        alpha=0.7,
    )

    fig.tight_layout()

    path = save_plot(
        fig,
        cfg.plots_dir / "08_ablation_study.png",
        cfg,
    )

    logger.info(
        "Figure 8 generated: ablation study with %d variants",
        len(df_plot),
    )

    # Log key findings for paper writing
    full_row = df_plot[df_plot["variant"] == "Full Model"]

    if not full_row.empty:
        full_cov = full_row["coverage_pct"].iloc[0]
        full_deserts = full_row["persistent_deserts_selected"].iloc[0]

        logger.info(
            "Full Model: coverage=%.2f%%, deserts=%d",
            full_cov,
            int(full_deserts),
        )

    for _, row in df_plot[df_plot["variant"] != "Full Model"].iterrows():
        logger.info(
            "%s: coverage=%.2f%%, Jaccard=%.3f, deserts=%d, rank_corr=%.3f",
            row["variant"],
            row["coverage_pct"],
            row["jaccard_similarity"],
            int(row["persistent_deserts_selected"]),
            row["rank_correlation_with_full"],
        )

    return path

# ─────────────────────────────────────────────────────────────
# FIGURE 9 — STATION PRIORITY MAP
# ─────────────────────────────────────────────────────────────

def plot_priority_map(
    station_df: pd.DataFrame,
    coordinates_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Figure 9: Spatial distribution of station intervention priority.

    Stations are positioned using their geographic coordinates and
    represented by their existing priority_index.

    This is a descriptive spatial visualization only. No new
    analytical metric is calculated.

    Inputs
    ------
    station_priority_scores.csv
    station_coordinates.csv

    Output
    ------
    assets/plots/09_priority_map.png
    """

    logger.info("Generating Figure 9 — station priority map")

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    station_required = {
        "stop_id",
        "stop_name",
        "priority_index",
    }

    coord_required = {
        "stop_id",
    }

    missing_station = station_required - set(station_df.columns)
    missing_coord = coord_required - set(coordinates_df.columns)

    if missing_station:
        raise ValueError(
            f"Figure 9 missing station columns: "
            f"{sorted(missing_station)}"
        )

    if missing_coord:
        raise ValueError(
            f"Figure 9 missing coordinate columns: "
            f"{sorted(missing_coord)}"
        )

    # ---------------------------------------------------------
    # Identify coordinate columns
    # ---------------------------------------------------------

    latitude_candidates = [
        "latitude",
        "lat",
        "Latitude",
        "stop_lat",
    ]

    longitude_candidates = [
        "longitude",
        "lon",
        "lng",
        "Longitude",
        "stop_lon",
    ]

    lat_col = next(
        (c for c in latitude_candidates if c in coordinates_df.columns),
        None,
    )

    lon_col = next(
        (c for c in longitude_candidates if c in coordinates_df.columns),
        None,
    )

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Figure 9 requires latitude and longitude columns "
            "in station_coordinates.csv."
        )

    # ---------------------------------------------------------
    # Prepare data
    # ---------------------------------------------------------

    priority = station_df[
        [
            "stop_id",
            "stop_name",
            "priority_index",
        ]
    ].copy()

    coordinates = coordinates_df[
        [
            "stop_id",
            lat_col,
            lon_col,
        ]
    ].copy()

    priority["priority_index"] = pd.to_numeric(
        priority["priority_index"],
        errors="coerce",
    )

    coordinates[lat_col] = pd.to_numeric(
        coordinates[lat_col],
        errors="coerce",
    )

    coordinates[lon_col] = pd.to_numeric(
        coordinates[lon_col],
        errors="coerce",
    )

    df = priority.merge(
        coordinates,
        on="stop_id",
        how="inner",
    )

    df = df.dropna(
        subset=[
            "priority_index",
            lat_col,
            lon_col,
        ]
    )

    if df.empty:
        raise ValueError(
            "No valid station coordinates could be matched "
            "with station priority scores."
        )

    # ---------------------------------------------------------
    # Create figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )

    scatter = ax.scatter(
        df[lon_col],
        df[lat_col],
        c=df["priority_index"],
        s=75,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
    )

    # ---------------------------------------------------------
    # Colorbar
    # ---------------------------------------------------------

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Intervention Priority Index",
        fontsize=12,
    )

    cbar.ax.tick_params(
        labelsize=10,
    )

    # ---------------------------------------------------------
    # Label the five highest-priority stations
    # ---------------------------------------------------------

    top_stations = (
        df.sort_values(
            "priority_index",
            ascending=False,
        )
        .head(5)
    )

    for _, row in top_stations.iterrows():

        ax.annotate(
            row["stop_name"],
            xy=(
                row[lon_col],
                row[lat_col],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
        )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    ax.set_title(
        "Spatial Distribution of Station Intervention Priority",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel(
        "Longitude",
        fontsize=12,
    )

    ax.set_ylabel(
        "Latitude",
        fontsize=12,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.grid(
        linestyle="--",
        alpha=0.25,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Equal geographic aspect ratio.
    ax.set_aspect("equal", adjustable="datalim")

    # ---------------------------------------------------------
    # Research note
    # ---------------------------------------------------------

    fig.text(
        0.5,
        0.015,
        "Higher priority values indicate greater intervention priority "
        "under the project's analytical framework.",
        ha="center",
        fontsize=9.5,
        alpha=0.7,
    )

    path = save_plot(
        fig,
        cfg.plots_dir / "09_priority_map.png",
        cfg,
    )

    logger.info(
        "Figure 9 generated: spatial priority distribution "
        "for %d stations",
        len(df),
    )

    logger.info(
        "Highest-priority mapped station: %s (priority_index=%.2f)",
        top_stations.iloc[0]["stop_name"],
        top_stations.iloc[0]["priority_index"],
    )

    return path

# ─────────────────────────────────────────────────────────────
# FIGURE 10 — TOP-5 INTERVENTION INSIGHTS
# ─────────────────────────────────────────────────────────────

def plot_top5_conversion_insights(
    insights_df: pd.DataFrame,
    cfg: VisualizationConfig = CFG,
) -> Path:
    """
    Figure 10: Top-5 intervention opportunities.

    Uses the existing conversion_insights_top5.csv output.
    No new analytical metric is calculated.

    The figure presents the five highest-priority intervention
    opportunities in a compact, publication-oriented format.

    Input
    -----
    outputs/conversion_insights_top5.csv

    Output
    ------
    assets/plots/10_top5_intervention_insights.png
    """

    logger.info(
        "Generating Figure 10 — top-5 intervention insights"
    )

    # ---------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------

    required = {
        "stop_name",
    }

    missing = required - set(insights_df.columns)

    if missing:
        raise ValueError(
            f"Figure 10 missing columns: {sorted(missing)}"
        )

    df = insights_df.copy()

    # ---------------------------------------------------------
    # Identify the primary ranking metric
    #
    # We deliberately use an existing metric rather than
    # constructing a new score inside visualization.py.
    # ---------------------------------------------------------

    ranking_candidates = [
        "priority_index",
        "simulation_impact_score",
        "lmci_gain",
        "LMCI_Final",
    ]

    ranking_col = next(
        (
            col
            for col in ranking_candidates
            if col in df.columns
        ),
        None,
    )

    if ranking_col is None:
        raise ValueError(
            "Figure 10 requires an existing ranking metric. "
            f"Expected one of: {ranking_candidates}"
        )

    df[ranking_col] = pd.to_numeric(
        df[ranking_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "stop_name",
            ranking_col,
        ]
    )

    if df.empty:
        raise ValueError(
            "No valid intervention insight records available."
        )

    # Keep the top five based on the existing analytical metric.
    df = (
        df.sort_values(
            ranking_col,
            ascending=False,
        )
        .head(5)
        .copy()
    )

    # Reverse for horizontal ranking.
    df = df.sort_values(
        ranking_col,
        ascending=True,
    )

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.barh(
        df["stop_name"],
        df[ranking_col],
        height=0.58,
        edgecolor="black",
        linewidth=0.7,
    )

    # ---------------------------------------------------------
    # Numeric labels
    # ---------------------------------------------------------

    max_value = df[ranking_col].max()

    for bar, value in zip(
        bars,
        df[ranking_col],
    ):
        ax.text(
            bar.get_width() + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # ---------------------------------------------------------
    # Add intervention information when available
    # ---------------------------------------------------------

    intervention_col = next(
        (
            col
            for col in [
                "intervention_type",
                "scenario_name",
                "recommended_intervention",
                "intervention",
            ]
            if col in df.columns
        ),
        None,
    )

    if intervention_col is not None:

        for i, (_, row) in enumerate(df.iterrows()):

            intervention = str(
                row[intervention_col]
            )

            ax.text(
                0.01,
                i,
                intervention,
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8.5,
                alpha=0.75,
            )

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    metric_labels = {
        "priority_index": "Intervention Priority Index",
        "simulation_impact_score": "Simulation Impact Score",
        "lmci_gain": "Simulated LMCI Gain",
        "LMCI_Final": "Final LMCI",
    }

    metric_label = metric_labels.get(
        ranking_col,
        ranking_col,
    )

    ax.set_title(
        "Top Five Intervention Opportunities",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        metric_label,
        fontsize=12,
    )

    ax.set_ylabel(
        "Metro Station",
        fontsize=12,
    )

    ax.set_xlim(
        0,
        max_value * 1.18 if max_value > 0 else 1,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25,
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---------------------------------------------------------
    # Research interpretation note
    # ---------------------------------------------------------

    fig.text(
        0.5,
        0.015,
        "The ranking reflects the analytical outputs of the "
        "last-mile intervention framework; it does not represent "
        "observed ridership or a ridership forecast.",
        ha="center",
        fontsize=9.5,
        alpha=0.7,
    )

    path = save_plot(
        fig,
        cfg.plots_dir / "10_top5_intervention_insights.png",
        cfg,
    )

    logger.info(
        "Figure 10 generated: %d top intervention opportunities",
        len(df),
    )

    logger.info(
        "Highest-ranked station: %s (%s=%.2f)",
        df.iloc[-1]["stop_name"],
        ranking_col,
        df.iloc[-1][ranking_col],
    )

    return path

# ─────────────────────────────────────────────────────────────
# VISUALIZATION PIPELINE
# ─────────────────────────────────────────────────────────────

def generate_all_visualizations(
    cfg: VisualizationConfig = CFG,
) -> Dict[str, Path]:
    """Generate all project visualizations."""

    logger.info("=" * 70)
    logger.info("STARTING VISUALIZATION PIPELINE")
    logger.info("=" * 70)

    data = load_visualization_inputs(cfg)

    paths: Dict[str, Path] = {}

    # Figure 1
    paths["lmci_distribution"] = plot_lmci_distribution(
        data["station_priority"],
        cfg,
    )

    # Figure 2
    paths["top_priority_stations"] = plot_top_priority_stations(
        data["station_priority"],
        top_n=cfg.top_station_count,
        cfg=cfg,
    )

    # Figure 3
    paths["demand_service_mismatch"] = plot_mismatch_distribution(
        data["mismatch"],
        cfg,
    )

    # Figure 4
    paths["temporal_lmci_profile"] = plot_temporal_lmci_profile(
        data["station_priority"],
        cfg,
    )

    # Figure 5
    paths["mclp_coverage_curve"] = plot_mclp_coverage_curve(
        data["mclp_coverage"],
        cfg,
    )

    # Figure 6
    paths["intervention_ranking"] = plot_intervention_ranking(
        data["simulation_ranking"],
        cfg,
    )

    # Figure 7
    paths["simulation_impact"] = plot_simulation_impact(
        data["simulation_station_impacts"],
        cfg=cfg,
    )

    # Figure 8 — Ablation Study (optional — generated only if data exists)
    ablation_path = plot_ablation_study(cfg)
    if ablation_path is not None:
        paths["ablation_study"] = ablation_path

    # Figure 9 — Station Priority Map
    paths["priority_map"] = plot_priority_map(
        data["station_priority"],
        _load_csv(cfg.station_coordinates_path),
        cfg,
    )

    # Figure 10 — Top-5 Intervention Insights
    paths["top5_intervention_insights"] = plot_top5_conversion_insights(
        _load_csv(cfg.conversion_insights_path),
        cfg,
    )

    logger.info("=" * 70)
    logger.info("VISUALIZATION PIPELINE COMPLETE")
    logger.info(f"Generated plots: {len(paths)}")
    logger.info("=" * 70)

    return paths


if __name__ == "__main__":
    generate_all_visualizations()
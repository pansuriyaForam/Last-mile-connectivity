# Hyderabad Metro Last-Mile Intelligence Platform

A geospatial analytics and optimization platform for evaluating multimodal transit accessibility across the Hyderabad Metro network.

The platform integrates GTFS transit feeds, OpenStreetMap data, accessibility modeling, equity analysis, and optimization techniques to identify underserved areas and recommend high-impact interventions for improving last-mile connectivity.

---

## Project Overview

Urban transit systems often struggle with last-mile accessibility, where commuters face challenges reaching transit stations despite significant infrastructure investments.

This project develops a data-driven framework to:

* Measure station-level accessibility
* Detect transit deserts
* Analyze temporal variations in service quality
* Evaluate equity across different regions
* Simulate intervention strategies
* Prioritize high-impact improvements under resource constraints

The Hyderabad Metro network is used as a real-world case study.

---

## Key Highlights

* Analysis of 57 Hyderabad Metro stations
* Integration of Metro, Bus, and MMTS transit networks
* 14,000+ geospatial demand points
* Temporal accessibility assessment
* Equity-aware optimization framework
* Interactive Streamlit dashboard
* Research paper and reproducible methodology

---

## Methodology

### 1. Data Integration

The platform combines:

* Hyderabad Metro Rail GTFS feeds
* TGSRTC bus GTFS feeds
* MMTS suburban rail GTFS feeds
* OpenStreetMap business points of interest
* Educational institution datasets
* Feeder service information

---

### 2. Last-Mile Connectivity Index (LMCI)

LMCI is a composite accessibility metric that incorporates:

* Service frequency
* Stop density
* Walkability
* Multimodal accessibility

The index enables station-level comparison and identification of accessibility gaps across the network.

---

### 3. Equity Assessment

Accessibility is evaluated not only by demand coverage but also by service equity.

The framework identifies underserved regions and highlights stations experiencing persistent accessibility disadvantages across different time periods.

---

### 4. Optimization Framework

An Equity-Weighted Maximum Coverage Location Problem (MCLP) model is used to prioritize interventions.

The optimization framework:

* Maximizes demand coverage
* Prioritizes low-accessibility areas
* Supports constrained resource allocation
* Generates intervention recommendations

---

### 5. Scenario Simulation

The platform can simulate:

* Feeder route additions
* Multimodal integration strategies
* Accessibility improvements
* Coverage expansion scenarios

---

## Dashboard Features

The Streamlit application provides:

* Network overview
* LMCI analysis
* Transit desert identification
* Equity assessment
* Optimization results
* Scenario simulation
* Interactive visualizations

---

## Repository Structure

```text
.
├── app.py
├── config.yaml
├── requirements.txt
│
├── Data/
│   ├── hmrl/
│   ├── tgsrtc/
│   ├── mmts/
│   ├── feeder/
│   └── external/
│
├── src/
│   ├── preprocessing.py
│   ├── lmci.py
│   ├── mclp.py
│   ├── scoring.py
│   ├── simulation.py
│   └── visualization.py
│
├── assets/
│   ├── plots
│
├── outputs/
└── docs/
```

---

## Installation

```bash
git clone https://github.com/pansuriyaForam/hyderabad-metro-last-mile-intelligence.git

cd hyderabad-metro-last-mile-intelligence

pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

---

## Research Contributions

This work introduces:

* A multimodal Last-Mile Connectivity Index (LMCI)
* Temporal accessibility assessment methodology
* Equity-aware transit evaluation framework
* Optimization-based intervention planning approach
* Interactive decision-support platform for transit analysis

---

## Limitations

* No real-time traffic integration
* Demand approximations based on available datasets
* Limited field validation
* Static transit schedules

---

## Future Work

* Real-time GTFS integration
* Demand forecasting
* Dynamic routing optimization
* Multi-city benchmarking
* Deployment as a planning intelligence platform

---

## Research Paper

**Title:** Temporal Equity Assessment and Optimization of Last-Mile Connectivity in Hyderabad Metro

**Status:** Manuscript in Preparation

The repository will include the full paper upon completion.

---

## Technologies Used

* Python
* Pandas
* GeoPandas
* NumPy
* Scikit-learn
* SciPy
* Folium
* Matplotlib
* Plotly
* Streamlit
* OpenStreetMap
* GTFS

---

## License

This project is intended for research, educational, and urban mobility planning purposes.

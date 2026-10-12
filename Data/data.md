# Data Sources

This directory contains all datasets used in the *Hyderabad Metro Last-Mile Intelligence Platform* project. The data has been collected from official transit agencies and publicly available geospatial repositories to support multimodal transportation analysis, accessibility assessment, and Last-Mile Connectivity Index (LMCI) calculations.

> **Accessed:** July 2026

---

## Data Directory Structure

```text
Data/
├── hmrl/       # Hyderabad Metro Rail GTFS data
├── tgsrtc/     # Telangana State RTC GTFS data
├── mmts/       # MMTS Hyderabad GTFS data
├── feeder/     # Hyderabad Metro feeder service information
└── external/   # Supplementary geospatial datasets
```

---

## Hyderabad Metro Rail (HMRL)

- **Source**: Hyderabad Metro Rail (HMRL) GTFS Dataset
- **URL**: https://aikosh.indiaai.gov.in/home/datasets/details/hyderabad_metro_rail_hmrl_general_transit_feed_specification_gtfs_data.html
- **Organization**: Hyderabad Metro Rail Limited (HMRL) / Open Data Telangana
- **License**: Open Government License (India)

### Files Used
- `agency.txt`
- `calendar.txt`
- `fare_attributes.txt`
- `fare_rules.txt`
- `routes.txt`
- `stop_times.txt`
- `stops.txt`
- `trips.txt`

### Usage
Used for metro station locations, routes, schedules, and network analysis.

---

## Telangana State Road Transport Corporation (TGSRTC)
- **Source**: TGSRTC Open Data Initiative
- **URL**: https://www.tgsrtc.telangana.gov.in/opendata.php
- **GTFS Request Form**: https://docs.google.com/forms/d/e/1FAIpQLScwhvSJvhDiFUQfe0gngnxhcabpE95n01ANDa6SM3jE65R6ow/viewform
- **Organization**: Telangana State Road Transport Corporation (TGSRTC)

### Files Used
- `agency.txt`
- `calendar.txt`
- `fare_attributes.txt`
- `fare_rules.txt`
- `routes.txt`
- `stop_times.txt`
- `stops.txt`
- `trips.txt`

### Usage
Used for bus stop accessibility analysis and multimodal integration.

> **Attribution**: "Contains data provided by TGSRTC."

---

## Multi-Modal Transport System (MMTS)
- **Source**: GTFS for MMTS Hyderabad
- **URL**: https://data.telangana.gov.in/dataset/gtfs-mmts-hyderabad
- **Organization**: South Central Railway
- **License**: Open Government License (India)


### Files Used
- `agency.txt`
- `calendar.txt`
- `routes.txt`
- `stop_times.txt`
- `stops.txt`
- `trips.txt`

### Usage
Used for suburban rail connectivity analysis and intermodal transportation assessment.

--- 

## Feeder Services
- **Source**: Hyderabad Metro Last-Mile Connectivity Portal
- **URL**: https://ltmetro.com/last-mile-connectivity/
- **Organization**: L&T Metro Rail (Hyderabad) Limited

### Data Used
- Feeder routes
- Service operators
- Vehicle types
- Service timings
- Fare information
- Station pickup points

### Usage
Used to evaluate station-level last-mile connectivity and feeder service availability.

---

## External Geospatial Data

### Commercial & Industrial Buildings
- **Source**: Hyderabad Open GIS Data (Lakeer)
- **URL**: https://github.com/Lakeer-org/hyderabad-open-gis-data/blob/master/5.%20Economy/5.1%20Business%20Density/Commercial%20_%20Industrial%20Buildings%20and%20Zones%20(points).geojson

### Affordable Schools
- **Source**: Hyderabad Open GIS Data (Lakeer)
- **URL**: https://github.com/Lakeer-org/hyderabad-open-gis-data/blob/master/2.%20Urban%20Poverty/2.1%20Education/Affordable%20Schools%20(Govt%20_%20Pvt%20Aided).geojson

### Usage
These GeoJSON datasets were used to provide contextual indicators of commercial activity and educational accessibility around transit stations.

---

## Preprocessing
The following preprocessing steps were performed where applicable:
- Extracted GTFS archives.
- Standardized station and stop names.
- Validated geographic coordinates.
- Converted datasets into analysis-ready formats.
- Integrated multiple transit modes into a unified pipeline.
- Performed spatial joins for proximity and density calculations.

---

### Data Availability
All datasets are publicly available from their respective providers and were accessed for academic and research purposes. Users should refer to the original sources for the latest versions, licensing terms, and usage conditions.

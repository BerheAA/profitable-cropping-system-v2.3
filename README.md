Kansas Cropping Systems Explorer – Irrigation v2

This is an irrigation-aware, Kansas-focused version of the Cropping Systems Explorer.

Main features:
- Kansas county choropleth map (dominant recommended system by county, irrigation-aware).
- Multi-objective suitability scoring: profit, water use, yield stability, GHG, cover crops, irrigation performance.
- Explicit irrigation fields: capacity, seasonal irrigation, triggers, efficiency, shortfall events.
- Climate tab using ERA5-Land via Open-Meteo.
- Soil & SSURGO tab using USDA Soil Data Access (point query).
- Profit & risk visualization including irrigation costs and profit per mm.
- ML & trial data tab with an irrigation-enabled trial schema and simple RandomForest training (if scikit-learn is installed).

How to run (local):
-------------------
1. Create and activate an environment with at least:

   pip install streamlit pandas numpy plotly requests scikit-learn

2. From this folder, run:

   streamlit run app.py

Data:
-----
- data/ks_counties_fips_centroids.csv
- data/ks_cropping_systems_kansas.csv

Replace or expand these CSVs with your real Kansas county centroids and cropping systems
derived from APSIM/DSSAT, NASS, and SSURGO (including irrigation scenarios and capacities).

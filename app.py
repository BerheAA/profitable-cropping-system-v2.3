
import os
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# Optional ML
try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None

# -------------------------------------------------------------------
# Basic config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Kansas Cropping Systems Explorer – Irrigation v2",
    layout="wide",
)

if "__file__" in globals():
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
else:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_kansas_data():
    """Load Kansas counties (all FIPS starting with '20') and cropping systems.

    Counties are derived from the Plotly US counties GeoJSON so all 105 Kansas
    counties are available for:
      - maps
      - Climate & Water county dropdown
      - Soils & SSURGO county dropdown

    If a local CSV with extra attributes exists (ks_counties_fips_centroids.csv),
    it is used to enrich the GeoJSON-derived table (regions, climate summaries, etc.).
    """
    counties_path = DATA_DIR / "ks_counties_fips_centroids.csv"
    systems_path = DATA_DIR / "ks_cropping_systems_kansas.csv"

    # -------------------------
    # Load all Kansas counties from GeoJSON
    # -------------------------
    try:
        geo = load_us_counties_geojson()
    except Exception:
        geo = None

    counties_geo_rows = []
    if geo is not None:
        features = geo.get("features", [])
        for feat in features:
            fips = str(feat.get("id", "")).zfill(5)
            # Kansas FIPS codes all start with "20"
            if not fips.startswith("20"):
                continue
            props = feat.get("properties", {}) or {}
            name = props.get("NAME") or props.get("name") or "Unknown"
            geom = feat.get("geometry", {}) or {}
            coords = geom.get("coordinates", [])

            # Approximate centroid as mean of all vertices (no shapely dependency)
            xs = []
            ys = []

            def collect_points(c):
                if not c:
                    return
                if isinstance(c[0], (float, int)):
                    # single [lon, lat]
                    if len(c) >= 2:
                        xs.append(c[0])
                        ys.append(c[1])
                else:
                    for sub in c:
                        collect_points(sub)

            collect_points(coords)
            if xs and ys:
                lon = float(sum(xs) / len(xs))
                lat = float(sum(ys) / len(ys))
            else:
                # Fallback: rough Kansas center
                lat, lon = 39.0, -98.0

            counties_geo_rows.append(
                {
                    "fips": fips,
                    "county": name,
                    "state": "Kansas",
                    "state_abbr": "KS",
                    "lat": lat,
                    "lon": lon,
                }
            )

    counties_geo = pd.DataFrame(counties_geo_rows)

    # -------------------------
    # Enrich with local CSV if available (region, climate summaries, etc.)
    # -------------------------
    if counties_path.exists():
        counties_attr = pd.read_csv(counties_path, dtype={"fips": str})
        counties_attr["fips"] = counties_attr["fips"].astype(str).str.zfill(5)
        # Drop duplicate columns before merge
        for col in ["county", "state", "state_abbr", "lat", "lon"]:
            if col in counties_attr.columns:
                counties_attr = counties_attr.drop(columns=[col])
        counties = counties_geo.merge(counties_attr, on="fips", how="left")
    else:
        counties = counties_geo.copy()
        # minimal placeholders if region/precip columns are referenced elsewhere
        if "region" not in counties.columns:
            counties["region"] = "Unknown"

    # -------------------------
    # Load cropping systems
    # -------------------------
    if systems_path.exists():
        systems = pd.read_csv(systems_path)
    else:
        systems = pd.DataFrame(
            {
                "system_id": [1, 2, 3, 4],
                "county": ["Riley", "Riley", "Dickinson", "Butler"],
                "county_fips": ["20161", "20161", "20041", "20015"],
                "region": [
                    "Northeast / Flint Hills",
                    "Northeast / Flint Hills",
                    "Central",
                    "South-Central",
                ],
                "crop": ["Corn", "Corn", "Wheat", "Sorghum"],
                "system_name": [
                    "Corn–Soybean (no-till, rainfed)",
                    "Corn–Soybean–Cover (strip-till)",
                    "Wheat–Soybean (reduced tillage)",
                    "Sorghum–Wheat–Cover (limited irrigation)",
                ],
                "rotation": [
                    "Corn → Soybean",
                    "Corn → Soybean + winter cover",
                    "Wheat → Soybean",
                    "Sorghum → Wheat + cover",
                ],
                "irrigation_system": [
                    "Rainfed",
                    "Rainfed",
                    "Rainfed",
                    "Center pivot (limited)",
                ],
                "soil_texture": ["Silt loam", "Silt loam", "Loam", "Sandy loam"],
                "awc_mm": [180, 185, 160, 140],
                "baseline_yield_t_ha": [11.0, 10.5, 4.2, 6.5],
                "yield_cv": [0.18, 0.17, 0.22, 0.25],
                "yield_penalty_from_water_deficit_t_ha": [0.1, 0.05, 0.0, 0.3],
                "net_return_usd_ha": [650, 670, 520, 580],
                "variable_cost_usd_ha": [900, 920, 650, 720],
                "irrigation_cost_usd_ha": [0, 0, 0, 130],
                "seasonal_irrigation_mm": [0, 0, 0, 280],
                "irrigation_efficiency_fraction": [1.0, 1.0, 1.0, 0.85],
                "water_use_mm": [550, 530, 480, 600],
                "water_productivity_kg_per_mm": [20.0, 19.8, 9.0, 10.8],
                "irrigation_capacity_mm_per_day": [0.0, 0.0, 0.0, 4.0],
                "irrigation_capacity_mm_per_season": [0.0, 0.0, 0.0, 320.0],
                "pumping_rate_lps": [0.0, 0.0, 0.0, 25.0],
                "trigger_smad_percent": [None, None, None, 55],
                "trigger_eto_fraction": [None, None, None, 0.35],
                "irrigation_interval_days": [None, None, None, 7],
                "target_smad_after_irrigation": [None, None, None, 90],
                "allowed_deficit_mm": [None, None, None, 50],
                "water_shortfall_events": [0, 0, 0, 1],
                "n_applied_kg_ha": [180, 170, 120, 150],
                "p_applied_kg_ha": [40, 40, 35, 35],
                "k_applied_kg_ha": [30, 30, 25, 25],
                "cover_crop_fraction": [0.6, 0.8, 0.3, 0.5],
                "erosion_risk_index": [0.30, 0.25, 0.35, 0.40],
                "ghg_index": [0.55, 0.50, 0.60, 0.58],
            }
        )

    counties["fips"] = counties["fips"].astype(str).str.zfill(5)
    if "county_fips" in systems.columns:
        systems["county_fips"] = systems["county_fips"].astype(str).str.zfill(5)
    else:
        systems["county_fips"] = systems["county"].map(
            dict(zip(counties["county"], counties["fips"]))
        )

    # If region is missing for some counties, fill with "Unknown"
    if "region" not in counties.columns:
        counties["region"] = "Unknown"
    else:
        counties["region"] = counties["region"].fillna("Unknown")

    return counties, systems


def fetch_era5land_daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Fetch daily ERA5-Land climate from Open-Meteo archive.

    This version ensures the requested end_date does not exceed today's date,
    which would otherwise cause a 400 Bad Request error if you ask for future dates.
    """
    from datetime import date as _date, timedelta as _timedelta

    today = _date.today()
    # Ensure end date is not in the future
    if end > today:
        end = today
    # Ensure start is before end; if not, default to 365 days before end
    if start >= end:
        start = end - _timedelta(days=365)

    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": [
            "temperature_2m_mean",
            "precipitation_sum",
            "reference_evapotranspiration_sum",
        ],
        "timezone": "America/Chicago",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "daily" not in data:
            return pd.DataFrame()
        daily = data["daily"]
        df = pd.DataFrame(daily)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as exc:
        st.warning(f"ERA5-Land request failed: {exc}")
        return pd.DataFrame()

SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"

@st.cache_data(show_spinner=False)
def fetch_ssurgo_by_point(lat: float, lon: float) -> pd.DataFrame:
    d = 0.01
    polygon_wkt = (
        f"polygon(({lon-d} {lat-d},{lon-d} {lat+d},{lon+d} {lat+d},"
        f"{lon+d} {lat-d},{lon-d} {lat-d}))"
    )

    sql = f"""
    SELECT TOP 10
        mu.mukey,
        mu.musym,
        mu.muname,
        muagg.aws0150wta AS awc_0_150mm,
        muagg.drclassdcd AS drainage_class,
        muagg.hydgrpdcd AS hydrologic_group
    FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{polygon_wkt}') AS s
    INNER JOIN mapunit AS mu ON mu.mukey = s.mukey
    LEFT JOIN muaggatt AS muagg ON muagg.mukey = mu.mukey;
    """

    payload = {
        "SERVICE": "query",
        "REQUEST": "query",
        "QUERY": sql,
        "FORMAT": "JSON+COLUMNNAME",
    }

    try:
        r = requests.post(SDA_URL, data=payload, timeout=30)
        r.raise_for_status()
        rows = r.json()
    except Exception as exc:
        st.warning(f"SSURGO / SDA query failed: {exc}")
        return pd.DataFrame()

    if isinstance(rows, dict):
        if "Table" in rows and isinstance(rows["Table"], list):
            rows = rows["Table"]
        else:
            return pd.DataFrame()

    if not isinstance(rows, list) or len(rows) < 2:
        return pd.DataFrame()

    columns = rows[0]
    data = rows[1:]
    try:
        df = pd.DataFrame(data, columns=columns)
    except Exception:
        return pd.DataFrame()

    return df

@st.cache_data(show_spinner=False)
def load_us_counties_geojson():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

# -------------------------------------------------------------------
# Suitability scoring (with irrigation)
# -------------------------------------------------------------------
def compute_suitability_scores(df: pd.DataFrame, strategy: str, irrigation_profile: str) -> pd.DataFrame:
    """Compute composite scores based on chosen strategy + irrigation profile."""
    if df.empty:
        return df

    df = df.copy()


    def norm(col, invert=False):
        if col not in df.columns:
            return pd.Series(np.ones(len(df)), index=df.index)

        x = pd.to_numeric(df[col], errors="coerce")

        # If all values are NaN → neutral scores
        if x.isna().all():
            return pd.Series(np.ones(len(df)) * 0.5, index=df.index)

        xmin = x.min()
        xmax = x.max()

        if xmax - xmin < 1e-6:
            v = pd.Series(np.ones(len(x)), index=df.index)
        else:
            v = (x - xmin) / (xmax - xmin)

        if invert:
            v = 1.0 - v

        # Replace NaN with 0.5 (neutral score) and clip
        v = v.fillna(0.5).clip(0, 1)

        return v

    # Core dimensions
    score_profit = norm("net_return_usd_ha", invert=False)
    score_water = norm("water_use_mm", invert=True)
    score_yield = norm("baseline_yield_t_ha", invert=False)
    score_risk = norm("yield_cv", invert=True)
    score_ghg = norm("ghg_index", invert=True)
    score_cover = norm("cover_crop_fraction", invert=False)

    # Irrigation-specific dimensions
    score_irrig_eff = norm("irrigation_efficiency_fraction", invert=False)
    score_capacity = norm("irrigation_capacity_mm_per_day", invert=False)
    score_shortfall = norm("water_shortfall_events", invert=True)
    score_irrig_cost = norm("irrigation_cost_usd_ha", invert=True)

    # Base weights by strategy
    if strategy == "Maximum profit":
        w_profit, w_water, w_yield, w_risk, w_ghg, w_cover = 0.40, 0.10, 0.20, 0.10, 0.05, 0.15
    elif strategy == "Water saver":
        w_profit, w_water, w_yield, w_risk, w_ghg, w_cover = 0.20, 0.35, 0.10, 0.15, 0.05, 0.15
    elif strategy == "High-resilience rotation":
        w_profit, w_water, w_yield, w_risk, w_ghg, w_cover = 0.20, 0.20, 0.10, 0.25, 0.05, 0.20
    else:  # Balanced
        w_profit, w_water, w_yield, w_risk, w_ghg, w_cover = 0.25, 0.25, 0.15, 0.15, 0.05, 0.15

    # Irrigation weights by profile
    if irrigation_profile == "Full irrigation (max yield)":
        w_irrig_eff, w_capacity, w_shortfall, w_irrig_cost = 0.15, 0.20, 0.15, 0.10
    elif irrigation_profile == "Limited irrigation (high WP)":
        w_irrig_eff, w_capacity, w_shortfall, w_irrig_cost = 0.20, 0.10, 0.20, 0.15
    elif irrigation_profile == "Deficit / risk-resilient":
        w_irrig_eff, w_capacity, w_shortfall, w_irrig_cost = 0.20, 0.10, 0.25, 0.15
    elif irrigation_profile == "Dryland / supplemental":
        w_irrig_eff, w_capacity, w_shortfall, w_irrig_cost = 0.10, 0.05, 0.20, 0.15
    else:
        w_irrig_eff, w_capacity, w_shortfall, w_irrig_cost = 0.15, 0.15, 0.15, 0.10

    composite = (
        w_profit * score_profit
        + w_water * score_water
        + w_yield * score_yield
        + w_risk * score_risk
        + w_ghg * score_ghg
        + w_cover * score_cover
        + w_irrig_eff * score_irrig_eff
        + w_capacity * score_capacity
        + w_shortfall * score_shortfall
        + w_irrig_cost * score_irrig_cost
    )

    df["score_profit"] = score_profit
    df["score_water"] = score_water
    df["score_yield"] = score_yield
    df["score_risk"] = score_risk
    df["score_ghg"] = score_ghg
    df["score_cover"] = score_cover
    df["score_irrig_eff"] = score_irrig_eff
    df["score_capacity"] = score_capacity
    df["score_shortfall"] = score_shortfall
    df["score_irrig_cost"] = score_irrig_cost
    df["composite_score"] = composite

    tags = []
    for i, row in df.iterrows():
        tag_list = []
        if row["score_profit"] >= 0.7:
            tag_list.append("Income-strong")
        if row["score_water"] >= 0.7:
            tag_list.append("Water-efficient")
        if row["score_cover"] >= 0.7:
            tag_list.append("High cover crop use")
        if row["score_risk"] >= 0.7:
            tag_list.append("Stable yields")
        if row["score_irrig_eff"] >= 0.7:
            tag_list.append("Efficient irrigation")
        if row["score_shortfall"] >= 0.7:
            tag_list.append("Few water shortfalls")
        if not tag_list:
            tag_list.append("Balanced")
        tags.append(", ".join(tag_list))

    df["suitability_tags"] = tags
    return df

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("Kansas Cropping Systems Explorer – Irrigation v2")

strategy = st.sidebar.selectbox(
    "Suitability strategy",
    ["Balanced", "Maximum profit", "Water saver", "High-resilience rotation"],
)

irrigation_profile = st.sidebar.selectbox(
    "Irrigation profile",
    [
        "Full irrigation (max yield)",
        "Limited irrigation (high WP)",
        "Deficit / risk-resilient",
        "Dryland / supplemental",
    ],
    help="Profiles approximate different Kansas irrigation realities.",
)

counties, systems = load_kansas_data()

region_filter = st.sidebar.multiselect(
    "Region / sub-region (optional)",
    sorted(counties["region"].dropna().unique().tolist()),
)

irrigation_filter = st.sidebar.multiselect(
    "Irrigation system filter",
    sorted(systems["irrigation_system"].dropna().unique().tolist()),
)

crop_filter = st.sidebar.multiselect(
    "Crop filter (optional)",
    sorted(systems["crop"].dropna().unique().tolist()),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Use tabs to explore climate, soils, irrigation scenarios, and ML-ready trials."
)

# -------------------------------------------------------------------
# Filter systems
# -------------------------------------------------------------------
if region_filter:
    systems = systems[systems["region"].isin(region_filter)]

if irrigation_filter:
    systems = systems[systems["irrigation_system"].isin(irrigation_filter)]

if crop_filter:
    systems = systems[systems["crop"].isin(crop_filter)]

systems_scored = compute_suitability_scores(
    systems, strategy=strategy, irrigation_profile=irrigation_profile
)

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_overview, tab_climate, tab_soil, tab_systems, tab_profit, tab_ml, tab_about = st.tabs(
    [
        "KS Overview",
        "Climate & Water",
        "Soils & SSURGO",
        "Cropping Systems",
        "Profit & Risk",
        "ML & Trial Data",
        "About & Setup",
    ]
)

# -------------------------------------------------------------------
# Overview tab
# -------------------------------------------------------------------
with tab_overview:
    st.subheader("Kansas Overview – Dominant Cropping Systems (with irrigation)")

    col_top, col_stats = st.columns([2, 1])
    with col_top:
        st.markdown("### A. County map – composite or system cluster")

    st.markdown(
        f"**Strategy:** {strategy} &nbsp;&nbsp; | &nbsp;&nbsp; **Irrigation profile:** {irrigation_profile}"
    )

    color_mode = st.radio(
        "Choropleth coloring mode",
        [
            "Composite score (continuous)",
            "Dominant system cluster (categorical)",
            "Climate baseline (annual precip mm)",
        ],
        index=0,
        help="Switch between continuous scores and categorical system clusters.",
    )

    try:
        counties_geojson = load_us_counties_geojson()
    except Exception as exc:
        st.error(f"Could not load US counties GeoJSON: {exc}")
        counties_geojson = None

    if counties_geojson is None:
        st.info("Map cannot be displayed because county GeoJSON could not be loaded.")
    else:
        ks_fips = [f for f in counties["fips"].tolist() if f.startswith("20")]
        geo_features = [
            feat for feat in counties_geojson.get("features", [])
            if feat.get("id", "").startswith("20")
        ]
        ks_geojson = {"type": "FeatureCollection", "features": geo_features}

        df_map = counties.copy()
        df_map = df_map[df_map["fips"].isin(ks_fips)].copy()

        if not systems_scored.empty:
            sys_dom = (
                systems_scored.sort_values("composite_score", ascending=False)
                .drop_duplicates(subset=["county_fips"])
            )
            df_map = df_map.merge(
                sys_dom,
                left_on="fips",
                right_on="county_fips",
                how="left",
                suffixes=("", "_sys"),
            )

        color_col = None
        color_kwargs = {}
        label_name = ""

        if color_mode == "Composite score (continuous)" and "composite_score" in df_map.columns:
            color_col = "composite_score"
            label_name = "Composite score"
            color_kwargs["color_continuous_scale"] = "Viridis"
        elif color_mode == "Dominant system cluster (categorical)" and "system_name" in df_map.columns:
            unique_systems = sorted(df_map["system_name"].dropna().unique().tolist())
            cluster_map = {name: f"System {chr(ord('A') + i)}" for i, name in enumerate(unique_systems)}
            df_map["system_cluster_code"] = df_map["system_name"].map(cluster_map)
            color_col = "system_cluster_code"
            label_name = "System cluster"
        else:
            if "annual_precip_mm" in df_map.columns:
                color_col = "annual_precip_mm"
                label_name = "Annual precip (mm)"
                color_kwargs["color_continuous_scale"] = "Blues"

        if color_col is None:
            st.info("No suitable variable available for the map.")
        else:
            fig = px.choropleth(
                df_map,
                geojson=ks_geojson,
                locations="fips",
                color=color_col,
                hover_name="county",
                hover_data=[
                    "region",
                    "system_name" if "system_name" in df_map.columns else None,
                    "crop" if "crop" in df_map.columns else None,
                    "net_return_usd_ha" if "net_return_usd_ha" in df_map.columns else None,
                    "seasonal_irrigation_mm" if "seasonal_irrigation_mm" in df_map.columns else None,
                ],
                scope="usa",
                labels={color_col: label_name},
                **color_kwargs,
            )
            fig.update_geos(
                fitbounds="locations",
                visible=True,
                showcountries=True,
                showcoastlines=True,
                showland=True,
            )
            fig.update_layout(
                title="Kansas county map – dominant recommended system (prototype)",
                margin=dict(r=0, t=40, l=0, b=0),
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### B. Top systems under current strategy & irrigation profile")

    if systems_scored.empty:
        st.info("No systems available after filters. Adjust filters in the sidebar.")
    else:
        top_df = systems_scored.sort_values("composite_score", ascending=False).head(
            min(10, len(systems_scored))
        )
        fig_bar = px.bar(
            top_df,
            x="composite_score",
            y="system_name",
            orientation="h",
            hover_data=[
                "county",
                "region",
                "crop",
                "rotation",
                "irrigation_system",
                "seasonal_irrigation_mm",
                "irrigation_capacity_mm_per_day",
                "net_return_usd_ha",
                "water_use_mm",
                "suitability_tags",
            ],
            title="Top Kansas cropping systems (irrigation-aware composite score)",
        )
        fig_bar.update_layout(yaxis_title="", xaxis_title="Composite score (0–1)")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(
            top_df[
                [
                    "county",
                    "region",
                    "crop",
                    "system_name",
                    "rotation",
                    "irrigation_system",
                    "irrigation_capacity_mm_per_day",
                    "seasonal_irrigation_mm",
                    "baseline_yield_t_ha",
                    "yield_cv",
                    "net_return_usd_ha",
                    "irrigation_cost_usd_ha",
                    "water_use_mm",
                    "water_productivity_kg_per_mm",
                    "suitability_tags",
                    "composite_score",
                ]
            ],
            use_container_width=True,
        )

# -------------------------------------------------------------------
# Climate tab
# -------------------------------------------------------------------
with tab_climate:
    st.subheader("Climate & Water – ERA5-Land (Open-Meteo)")
    st.markdown(
        "Explore historical daily climate for a point in Kansas and relate it to crop water use."
    )

    col_loc, col_dates = st.columns([1, 1])

    with col_loc:
        county_choice = st.selectbox(
            "County (to auto-fill lat/lon)",
            sorted(counties["county"].unique().tolist()),
        )
        row_c = counties[counties["county"] == county_choice].iloc[0]
        lat_default = float(row_c["lat"])
        lon_default = float(row_c["lon"])

        lat = st.number_input("Latitude", value=lat_default, format="%.4f")
        lon = st.number_input("Longitude", value=lon_default, format="%.4f")

    with col_dates:
        today = date.today()
        start_date = st.date_input("Start date", value=today - timedelta(days=365))
        end_date = st.date_input("End date", value=today)

    if st.button("Fetch climate"):
        df_clim = fetch_era5land_daily(lat, lon, start_date, end_date)
        if df_clim.empty:
            st.info("No climate data returned for this period.")
        else:
            st.success(
                f"Loaded {len(df_clim)} days of ERA5-Land data for lat {lat:.2f}, lon {lon:.2f}."
            )
            st.line_chart(
                df_clim.set_index("time")[
                    [
                        "temperature_2m_mean",
                        "precipitation_sum",
                        "reference_evapotranspiration_sum",
                    ]
                ]
            )
            st.dataframe(df_clim.head(30), use_container_width=True)

# -------------------------------------------------------------------
# Soil tab
# -------------------------------------------------------------------
with tab_soil:
    st.subheader("Soils & SSURGO (SDA API)")
    st.markdown(
        "Enter a representative field location to pull nearby SSURGO map units using USDA Soil Data Access."
    )

    col_loc2, col_btn2 = st.columns([2, 1])

    with col_loc2:
        county_choice_s = st.selectbox(
            "County (optional helper for lat/lon)",
            sorted(counties["county"].unique().tolist()),
        )
        row_s = counties[counties["county"] == county_choice_s].iloc[0]
        lat_s = st.number_input("Latitude (soil point)", value=float(row_s["lat"]), format="%.4f")
        lon_s = st.number_input("Longitude (soil point)", value=float(row_s["lon"]), format="%.4f")

    with col_btn2:
        st.write(" ")
        st.write(" ")
        if st.button("Query SSURGO at this point"):
            soil_df = fetch_ssurgo_by_point(lat_s, lon_s)
            if soil_df.empty:
                st.info("No SSURGO / SDA records returned – try another location.")
            else:
                st.success(f"Returned {len(soil_df)} mapunit records.")
                st.dataframe(soil_df, use_container_width=True)

    st.markdown(
        "In a full pipeline, SSURGO-derived root-zone AWC, drainage, salinity, and organic matter "
        "would be pre-computed at county or field scale and merged into the cropping systems table."
    )

# -------------------------------------------------------------------
# Systems tab
# -------------------------------------------------------------------
with tab_systems:
    st.subheader("Kansas Cropping Systems – Irrigation & Management Details")

    if systems_scored.empty:
        st.info("No systems after filters; adjust filters in the sidebar.")
    else:
        st.markdown("Sort or filter this table to understand irrigation strategies and tradeoffs.")
        st.dataframe(systems_scored, use_container_width=True)

# -------------------------------------------------------------------
# Profit & Risk tab
# -------------------------------------------------------------------

with tab_profit:
    st.subheader("Profit & Risk – Irrigation-Aware View")

    if systems_scored.empty:
        st.info("No systems available for profit & risk analysis.")
    else:
        dfp = systems_scored.copy()
        dfp["gross_return_usd_ha"] = dfp["net_return_usd_ha"] + dfp["variable_cost_usd_ha"]
        dfp["profit_per_mm_usd"] = dfp["net_return_usd_ha"] / dfp["water_use_mm"]

        st.markdown("### Net return vs. seasonal irrigation")
        fig_scatter = px.scatter(
            dfp,
            x="seasonal_irrigation_mm",
            y="net_return_usd_ha",
            color="irrigation_system",
            hover_name="system_name",
            hover_data=["county", "crop", "rotation", "yield_cv"],
            labels={
                "seasonal_irrigation_mm": "Seasonal irrigation (mm)",
                "net_return_usd_ha": "Net return (USD/ha)",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Table – irrigation productivity and yield risk")
        st.dataframe(
            dfp[
                [
                    "county",
                    "region",
                    "crop",
                    "system_name",
                    "irrigation_system",
                    "irrigation_capacity_mm_per_day",
                    "seasonal_irrigation_mm",
                    "net_return_usd_ha",
                    "irrigation_cost_usd_ha",
                    "profit_per_mm_usd",
                    "yield_cv",
                    "water_shortfall_events",
                    "suitability_tags",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### Crop-level recommendations: water-saving vs profit")

        # Aggregate by crop
        crop_stats = (
            dfp.groupby("crop")
            .agg(
                mean_net_return_usd_ha=("net_return_usd_ha", "mean"),
                mean_water_use_mm=("water_use_mm", "mean"),
                mean_seasonal_irrigation_mm=("seasonal_irrigation_mm", "mean"),
                mean_profit_per_mm=("profit_per_mm_usd", "mean"),
            )
            .reset_index()
        )

        crop_stats["rank_profit"] = crop_stats["mean_net_return_usd_ha"].rank(ascending=False)
        crop_stats["rank_profit_per_mm"] = crop_stats["mean_profit_per_mm"].rank(ascending=False)

        crop_stats["w_profit"] = crop_stats["mean_net_return_usd_ha"].clip(lower=0)
        crop_stats["w_profit_per_mm"] = crop_stats["mean_profit_per_mm"].clip(lower=0)

        if crop_stats["w_profit"].sum() > 0:
            crop_stats["area_share_profit_focus"] = crop_stats["w_profit"] / crop_stats["w_profit"].sum()
        else:
            crop_stats["area_share_profit_focus"] = 1.0 / len(crop_stats)

        if crop_stats["w_profit_per_mm"].sum() > 0:
            crop_stats["area_share_water_focus"] = crop_stats["w_profit_per_mm"] / crop_stats["w_profit_per_mm"].sum()
        else:
            crop_stats["area_share_water_focus"] = 1.0 / len(crop_stats)

        st.markdown("#### A. Which crops save water and improve profit?")
        st.markdown(
            "Crops with **high mean profit per mm** and **moderate or low water use** "
            "are good candidates for expanding area under water-limited conditions."
        )

        st.dataframe(
            crop_stats.sort_values("mean_profit_per_mm", ascending=False)[
                [
                    "crop",
                    "mean_net_return_usd_ha",
                    "mean_water_use_mm",
                    "mean_seasonal_irrigation_mm",
                    "mean_profit_per_mm",
                    "rank_profit",
                    "rank_profit_per_mm",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("#### B. Suggested area shares by objective (normalized to 1.0)")
        st.dataframe(
            crop_stats.sort_values("area_share_water_focus", ascending=False)[
                [
                    "crop",
                    "area_share_water_focus",
                    "area_share_profit_focus",
                ]
            ],
            use_container_width=True,
        )

# -------------------------------------------------------------------
# ML & Trial Data tab
# -------------------------------------------------------------------
with tab_ml:
    st.subheader("ML & Trial Data – Standard Schema (Irrigation-Enabled)")

    st.markdown(
        "This tab connects Kansas APSIM/DSSAT or field trial outputs into an ML-ready schema, "
        "including irrigation management fields."
    )

    cols_schema = [
        "trial_id",
        "county",
        "county_fips",
        "lat",
        "lon",
        "year",
        "crop",
        "system_name",
        "rotation_code",
        "previous_crop",
        "tillage_type",
        "cover_crop",
        "planting_date",
        "harvest_date",
        "irrigation_system",
        "irrigation_capacity_mm_per_day",
        "seasonal_irrigation_mm",
        "trigger_smad_percent",
        "trigger_eto_fraction",
        "n_rate_kg_ha",
        "p_rate_kg_ha",
        "k_rate_kg_ha",
        "gdd_season_degC",
        "precip_season_mm",
        "et0_season_mm",
        "heat_days_gt35C",
        "frost_events_gs",
        "awc_rootzone_mm",
        "soil_texture_class",
        "grain_yield_t_ha",
        "net_return_usd_ha",
        "water_use_mm",
        "yield_cv",
    ]
    schema_df = pd.DataFrame({"column_name": cols_schema})
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download blank irrigation-aware trial template (CSV)",
        data=schema_df.to_csv(index=False),
        file_name="kansas_trial_template_irrigation_v2.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("#### Train a simple RandomForest on your trials (optional)")

    if RandomForestRegressor is None:
        st.info("scikit-learn is not installed; ML training is disabled.")
    else:
        uploaded = st.file_uploader("Upload trial CSV in the schema above", type=["csv"])
        target = st.selectbox(
            "Target variable to predict",
            ["grain_yield_t_ha", "net_return_usd_ha", "composite_score"],
        )
        if uploaded is not None:
            trials_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded trials:")
            st.dataframe(trials_df.head(), use_container_width=True)
            feature_cols = [
                c
                for c in trials_df.columns
                if c
                in [
                    "gdd_season_degC",
                    "precip_season_mm",
                    "et0_season_mm",
                    "awc_rootzone_mm",
                    "seasonal_irrigation_mm",
                    "irrigation_capacity_mm_per_day",
                    "n_rate_kg_ha",
                    "cover_crop",
                ]
            ]
            if target not in trials_df.columns:
                st.warning(f"Target column '{target}' not found in uploaded data.")
            elif not feature_cols:
                st.warning(
                    "No numeric feature columns found for training – please ensure climate, "
                    "soil, irrigation, and management columns are present."
                )
            else:
                df_ml = trials_df.dropna(subset=[target] + feature_cols)
                if len(df_ml) < 30:
                    st.warning("Not enough records to train a model (need at least ~30 rows).")
                else:
                    X = df_ml[feature_cols]
                    y = df_ml[target]
                    model = RandomForestRegressor(n_estimators=300, random_state=42)
                    model.fit(X, y)
                    st.success(
                        f"Trained RandomForest with {len(df_ml)} records to predict {target}."
                    )

# -------------------------------------------------------------------
# About & Setup tab
# -------------------------------------------------------------------
with tab_about:
    st.subheader("About & Setup")

    st.markdown(
        """
This dashboard is a **Kansas Cropping Systems Explorer – Irrigation v2 (prototype)**.

It mirrors the US-wide explorer but focuses on:

- County-level cropping systems in Kansas with explicit irrigation water management
- Climate signals and crop water use (ERA5-Land via Open-Meteo)
- Soils from SSURGO (USDA Soil Data Access) at point level
- Multi-objective suitability scoring (profit, water, risk, resilience, irrigation performance)
- A standard ML-ready trial schema for APSIM/DSSAT or field experiments with irrigation fields

To fully enable all integrations you typically need:

1. A stable internet connection (for Open-Meteo and SDA).
2. Python packages: `streamlit`, `pandas`, `numpy`, `plotly`, `requests`, `scikit-learn`.
3. Local CSVs populated with:
   - Kansas county centroids and climate/soils summaries
   - Kansas cropping systems and economic + irrigation metrics

You can replace the example CSVs in `data/` with your own outputs from APSIM/DSSAT + NASS + SSURGO pipelines.
        """
    )
    st.markdown("Version: Kansas Cropping Systems Explorer – Irrigation v2 (prototype)")

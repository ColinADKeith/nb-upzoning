"""
New Brunswick Upzoning Opportunity Analysis — Enhanced
=======================================================
Capstone Project: Identifying lots ripe for upzoning in Saint John, Moncton, Fredericton

Added layers: Property assessment, transit stops, schools, building permits

Requirements:
    pip install geopandas pandas numpy shapely folium matplotlib scipy
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import zipfile, os, io, warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
DATA_DIR = "./data"

# =======================================================================
# FILE PATHS — matched to your exact filenames
# =======================================================================

# Parcel data (your provincial shapefile exported as CSV, 90 MB)
PARCELS_SHP = os.path.join(DATA_DIR, "SHP_CSV.csv")

# Hospital data
HOSPITALS_CSV = os.path.join(DATA_DIR, "Hospitals.csv")

# Building Footprints
BUILDINGS = {
    "Saint John": os.path.join(DATA_DIR, "Building_Footprints-Saint John.geojson"),
}

# Zoning
ZONING = {
    "Saint John": os.path.join(DATA_DIR, "Zoning-Saint-john.geojson"),
}

# Schools (provincial, 130 KB)
SCHOOLS_GEOJSON = os.path.join(
    DATA_DIR,
    "New Brunswick Public Schools _ Écoles publiques du Nouveau-Brunswick.geojson"
)

# Property Assessment — Saint John city-level (73 MB, best option)
SJ_ASSESSMENT_GEOJSON = os.path.join(DATA_DIR, "Assessments-SaintJohn.geojson")
# Provincial fallbacks
ASSESSMENT_TSV_ZIP = os.path.join(DATA_DIR, "geonb_evan_tsv.zip")
ASSESSMENT_CSV = os.path.join(
    DATA_DIR,
    "Property_Assessment_Map__Carte_d__valuation_fonci_re.csv"
)

# Transit stops (43 KB text file — already extracted from GTFS)
TRANSIT_STOPS = {
    "Saint John": os.path.join(DATA_DIR, "stops-saint-john.txt"),
}

# Building Permits (52 MB GeoJSON)
PERMITS = {
    "Saint John": os.path.join(DATA_DIR, "Permits-SaintJohn.geojson"),
}

CITIES = {
    "Saint John":  {"lat": 45.2733, "lon": -66.0633, "county": "Saint John"},
    "Moncton":     {"lat": 46.0878, "lon": -64.7782, "county": "Westmorland"},
    "Fredericton": {"lat": 45.9636, "lon": -66.6431, "county": "York"},
}
CITY_RADIUS_M = 15000
CRS_PROJ = "EPSG:2953"
CRS_WGS84 = "EPSG:4326"


# =============================================================================
# 2. DATA LOADING
# =============================================================================
def load_parcels(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.shp', '.geojson', '.json']:
        parcels = gpd.read_file(path)
    elif ext == '.csv':
        df = pd.read_csv(path)
        geometry = [Point(x, y) for x, y in zip(df['X'], df['Y'])]
        parcels = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_PROJ)
        parcels = parcels.to_crs(CRS_WGS84)
    else:
        raise ValueError(f"Unsupported: {ext}")
    print(f"  Loaded {len(parcels)} parcels")
    return parcels


def load_hospitals(path):
    # Try comma first, then tab — auto-detect
    for sep in [',', '\t', ';']:
        try:
            df = pd.read_csv(path, sep=sep, encoding='latin-1')
            if len(df.columns) > 3:  # found the right delimiter
                break
        except Exception:
            continue
    # Find lat/lon columns flexibly
    lat_col = [c for c in df.columns if 'lat' in c.lower()][0]
    lon_col = [c for c in df.columns if 'lon' in c.lower()][0]
    geometry = [Point(float(lon), float(lat))
                for lat, lon in zip(df[lat_col], df[lon_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_WGS84)
    print(f"  Loaded {len(gdf)} hospitals")
    return gdf


def load_city_geodata(file_dict, label="features"):
    frames = []
    for city, path in file_dict.items():
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            gdf['city'] = city
            if gdf.crs is None:
                gdf = gdf.set_crs(CRS_WGS84)
            elif gdf.crs != CRS_WGS84:
                gdf = gdf.to_crs(CRS_WGS84)
            frames.append(gdf)
            print(f"    {city}: {len(gdf)} {label}")
        else:
            print(f"    {city}: FILE NOT FOUND — {path}")
    return pd.concat(frames, ignore_index=True) if frames else gpd.GeoDataFrame()


def load_schools(path):
    if not os.path.exists(path):
        print("  Schools file not found, skipping.")
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs != CRS_WGS84:
        gdf = gdf.set_crs(CRS_WGS84) if gdf.crs is None else gdf.to_crs(CRS_WGS84)
    print(f"  Loaded {len(gdf)} schools")
    return gdf


def load_transit_stops(stops_dict):
    """Load GTFS stops.txt files (CSV with stop_lat, stop_lon)."""
    frames = []
    for city, path in stops_dict.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            geom = [Point(lon, lat) for lat, lon
                     in zip(df['stop_lat'], df['stop_lon'])]
            gdf = gpd.GeoDataFrame(df, geometry=geom, crs=CRS_WGS84)
            gdf['city'] = city
            frames.append(gdf)
            print(f"    {city}: {len(gdf)} transit stops")
        else:
            print(f"    {city}: GTFS stops.txt NOT FOUND — {path}")
    return pd.concat(frames, ignore_index=True) if frames else None


def load_assessment_geojson(path):
    """Load city-level assessment GeoJSON (e.g. Saint John's 73MB file).
    This is the best option — has geometry + values in one file."""
    if not os.path.exists(path):
        print(f"  Assessment GeoJSON not found: {path}")
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(CRS_WGS84)
    elif gdf.crs != CRS_WGS84:
        gdf = gdf.to_crs(CRS_WGS84)
    print(f"  Loaded {len(gdf)} assessment records (GeoJSON)")
    print(f"  Columns: {list(gdf.columns[:15])}...")
    return gdf


def load_assessment_tsv_zip(zip_path):
    """Load provincial assessment from geonb_evan_tsv.zip."""
    if not os.path.exists(zip_path):
        print(f"  Assessment ZIP not found: {zip_path}")
        return None
    with zipfile.ZipFile(zip_path, 'r') as z:
        tsv_files = [f for f in z.namelist() if f.endswith('.tsv') or f.endswith('.txt')]
        if not tsv_files:
            print(f"  No TSV found inside zip. Contents: {z.namelist()[:5]}")
            return None
        with z.open(tsv_files[0]) as f:
            df = pd.read_csv(f, sep='\t', low_memory=False, encoding='latin-1')
    print(f"  Loaded {len(df)} assessment records (TSV from zip)")
    print(f"  Columns: {list(df.columns[:15])}...")
    return df


def load_assessment_csv(path):
    """Load the large provincial assessment CSV (424 MB)."""
    if not os.path.exists(path):
        print(f"  Assessment CSV not found: {path}")
        return None
    df = pd.read_csv(path, low_memory=False, encoding='latin-1')
    print(f"  Loaded {len(df)} assessment records (CSV)")
    print(f"  Columns: {list(df.columns[:15])}...")
    return df


# =============================================================================
# 3. SPATIAL FILTERING
# =============================================================================
def filter_parcels_by_city(parcels, city_name, city_info, radius_m=CITY_RADIUS_M):
    center = Point(city_info['lon'], city_info['lat'])
    center_gdf = gpd.GeoDataFrame(geometry=[center], crs=CRS_WGS84).to_crs(CRS_PROJ)
    parcels_proj = parcels.to_crs(CRS_PROJ)
    mask = parcels_proj.geometry.distance(center_gdf.geometry.iloc[0]) <= radius_m
    filtered = parcels[mask].copy()
    filtered['city'] = city_name
    print(f"    {city_name}: {len(filtered)} parcels within {radius_m/1000:.0f}km")
    return filtered


# =============================================================================
# 4. SPATIAL JOINS & ENRICHMENT
# =============================================================================
def join_zoning(parcels, zoning):
    if zoning is None or len(zoning) == 0:
        parcels['ZoningCode'] = 'Unknown'
        parcels['ZoningGroup'] = 'Unknown'
        return parcels
    centroids = parcels.copy().reset_index(drop=True)
    centroids['geometry'] = centroids.geometry.centroid
    zoning_clean = zoning[['ZoningCode', 'Description', 'ZoningGroup', 'geometry']].copy().reset_index(drop=True)
    joined = gpd.sjoin(centroids, zoning_clean, how='left', predicate='within')
    joined['ZoningCode'] = joined['ZoningCode'].fillna('Unzoned')
    joined['ZoningGroup'] = joined['ZoningGroup'].fillna('Unzoned')
    # Drop duplicate rows from overlapping zones — keep first match
    joined = joined[~joined.index.duplicated(keep='first')]
    # Clean up sjoin artifacts and restore original geometry
    joined = joined.drop(columns=['index_right'], errors='ignore')
    joined['geometry'] = parcels.geometry.values[:len(joined)]
    print(f"  Zoning join: {joined['ZoningCode'].nunique()} unique zones")
    return joined


def calculate_building_coverage(parcels, buildings):
    """Calculate building density around each parcel centroid.
    Since parcels are POINTS (centroids), we count/sum building footprints
    within an estimated parcel radius based on Shape_Area."""
    if buildings is None or len(buildings) == 0:
        parcels['building_coverage'] = 0.0
        parcels['num_buildings'] = 0
        parcels['lot_area_m2'] = parcels.to_crs(CRS_PROJ).geometry.area if \
            parcels.geometry.iloc[0].geom_type != 'Point' else 0
        if 'Shape_Area' in parcels.columns:
            parcels['lot_area_m2'] = pd.to_numeric(parcels['Shape_Area'], errors='coerce').fillna(0)
        return parcels

    # Clean up leftover join columns
    drop_cols = [c for c in parcels.columns if c.startswith('index_')]
    parcels = parcels.drop(columns=drop_cols, errors='ignore')

    parcels = parcels.copy()
    p_proj = parcels.to_crs(CRS_PROJ)
    b_proj = buildings.to_crs(CRS_PROJ)

    # Get lot area from Shape_Area column (from original shapefile) or geometry
    if 'Shape_Area' in parcels.columns:
        parcels['lot_area_m2'] = pd.to_numeric(parcels['Shape_Area'], errors='coerce').fillna(0)
    elif p_proj.geometry.iloc[0].geom_type != 'Point':
        parcels['lot_area_m2'] = p_proj.geometry.area
    else:
        parcels['lot_area_m2'] = 500  # default if no area info

    # Use KD-tree: for each parcel centroid, find buildings within ~50m
    parcel_coords = np.array([(g.x, g.y) for g in p_proj.geometry])
    bldg_centroids = np.array([(g.centroid.x, g.centroid.y) for g in b_proj.geometry])
    bldg_areas = b_proj['Shape__Area'].values if 'Shape__Area' in b_proj.columns else \
                 b_proj.geometry.area.values

    tree = cKDTree(bldg_centroids)
    # Find all buildings within 50m of each parcel centroid
    matches = tree.query_ball_point(parcel_coords, r=50)

    num_bldgs = np.array([len(m) for m in matches])
    total_bldg_area = np.array([sum(bldg_areas[i] for i in m) if m else 0
                                 for m in matches])

    parcels['num_buildings'] = num_bldgs
    parcels['total_bldg_area'] = total_bldg_area
    parcels['building_coverage'] = np.where(
        parcels['lot_area_m2'] > 0,
        total_bldg_area / parcels['lot_area_m2'].values, 0
    ).clip(0, 1)

    print(f"  Building coverage — mean: {parcels['building_coverage'].mean():.2%}, "
          f"parcels with buildings: {(num_bldgs > 0).sum()}")
    return parcels


# =============================================================================
# 5. PROXIMITY CALCULATIONS (hospitals, schools, transit, permits)
# =============================================================================
def _nearest_distance(parcels, points_gdf, label="point"):
    """Generic nearest-neighbor distance using KD-tree."""
    p_proj = parcels.to_crs(CRS_PROJ)
    pt_proj = points_gdf.to_crs(CRS_PROJ)
    pt_coords = np.array([(g.x, g.y) for g in pt_proj.geometry])
    parcel_coords = np.array([(g.centroid.x, g.centroid.y) for g in p_proj.geometry])
    tree = cKDTree(pt_coords)
    dists, idxs = tree.query(parcel_coords)
    return dists, idxs


def _decay_score(distances, near_m=400, far_m=2000):
    """1.0 if within near_m, decays linearly to 0 at far_m."""
    return np.clip(1 - (distances - near_m) / (far_m - near_m), 0, 1)


def enrich_hospital_proximity(parcels, hospitals):
    dists, idxs = _nearest_distance(parcels, hospitals)
    parcels = parcels.copy()
    parcels['dist_hospital_m'] = dists
    # Find name and beds columns flexibly
    name_col = next((c for c in hospitals.columns if 'name' in c.lower() or 'nom' in c.lower()), None)
    beds_col = next((c for c in hospitals.columns if 'bed' in c.lower() or 'lit' in c.lower()), None)
    if name_col:
        parcels['nearest_hospital'] = hospitals.iloc[idxs][name_col].values
    else:
        parcels['nearest_hospital'] = 'Unknown'
    if beds_col:
        parcels['hospital_beds'] = pd.to_numeric(hospitals.iloc[idxs][beds_col].values, errors='coerce')
    else:
        parcels['hospital_beds'] = 0
    parcels['score_hospital'] = _decay_score(dists, 500, 3000)
    print(f"  Hospital proximity — median: {np.median(dists):.0f}m")
    return parcels


def enrich_school_proximity(parcels, schools):
    if schools is None or len(schools) == 0:
        parcels['dist_school_m'] = 99999
        parcels['score_school'] = 0.0
        return parcels
    dists, idxs = _nearest_distance(parcels, schools)
    parcels = parcels.copy()
    parcels['dist_school_m'] = dists
    # Extract school name — column name varies, try common ones
    for col in ['school_name', 'SCHOOL_NAME', 'name', 'NAME', 'School_Name']:
        if col in schools.columns:
            parcels['nearest_school'] = schools.iloc[idxs][col].values
            break
    parcels['score_school'] = _decay_score(dists, 400, 2000)
    print(f"  School proximity — median: {np.median(dists):.0f}m")
    return parcels


def enrich_transit_proximity(parcels, transit_stops):
    if transit_stops is None or len(transit_stops) == 0:
        parcels['dist_transit_m'] = 99999
        parcels['score_transit'] = 0.0
        parcels['transit_stops_800m'] = 0
        return parcels

    parcels = parcels.copy()
    parcels['dist_transit_m'] = 99999.0
    parcels['transit_stops_800m'] = 0

    p_proj = parcels.to_crs(CRS_PROJ)
    t_proj = transit_stops.to_crs(CRS_PROJ)
    parcel_coords = np.array([(g.centroid.x, g.centroid.y) for g in p_proj.geometry])

    # Process per city so Moncton parcels don't measure to Saint John stops
    for city in parcels['city'].unique():
        city_mask = parcels['city'] == city
        city_coords = parcel_coords[city_mask]

        # Find stops for this city, fall back to all stops
        if 'city' in transit_stops.columns:
            city_stops = t_proj[transit_stops['city'] == city]
        else:
            city_stops = t_proj

        if len(city_stops) == 0:
            continue

        t_coords = np.array([(g.x, g.y) for g in city_stops.geometry])
        tree = cKDTree(t_coords)

        dists, _ = tree.query(city_coords)
        parcels.loc[city_mask, 'dist_transit_m'] = dists

        counts = tree.query_ball_point(city_coords, r=800)
        parcels.loc[city_mask, 'transit_stops_800m'] = [len(c) for c in counts]

    # Score: combination of nearest distance + stop density
    dist_score = _decay_score(parcels['dist_transit_m'].values, 200, 1000)
    density_score = np.clip(parcels['transit_stops_800m'] / 10, 0, 1)
    parcels['score_transit'] = 0.6 * dist_score + 0.4 * density_score

    valid = parcels[parcels['dist_transit_m'] < 99000]
    if len(valid) > 0:
        print(f"  Transit proximity — median: {valid['dist_transit_m'].median():.0f}m, "
              f"mean stops/800m: {valid['transit_stops_800m'].mean():.1f}")
    else:
        print("  Transit proximity — no stops matched to any city")
    return parcels


def enrich_development_pressure(parcels, permits):
    """Score based on recent building permit activity nearby."""
    if permits is None or len(permits) == 0:
        parcels['permits_1km'] = 0
        parcels['score_dev_pressure'] = 0.0
        return parcels

    p_proj = parcels.to_crs(CRS_PROJ)
    pm_proj = permits.to_crs(CRS_PROJ)
    pm_coords = np.array([(g.centroid.x, g.centroid.y) for g in pm_proj.geometry])
    tree = cKDTree(pm_coords)
    parcel_coords = np.array([(g.centroid.x, g.centroid.y) for g in p_proj.geometry])
    counts = tree.query_ball_point(parcel_coords, r=1000)

    parcels = parcels.copy()
    parcels['permits_1km'] = [len(c) for c in counts]
    # Normalize: more permits nearby = higher development pressure = higher score
    max_permits = parcels['permits_1km'].quantile(0.95) if parcels['permits_1km'].max() > 0 else 1
    parcels['score_dev_pressure'] = np.clip(parcels['permits_1km'] / max_permits, 0, 1)

    print(f"  Development pressure — mean permits within 1km: "
          f"{parcels['permits_1km'].mean():.1f}")
    return parcels


# =============================================================================
# 6. PROPERTY ASSESSMENT INTEGRATION
# =============================================================================
def enrich_assessment(parcels, assessment_data):
    """Join property assessment data."""
    if assessment_data is None:
        parcels['total_assessed'] = np.nan
        parcels['land_to_total_ratio'] = np.nan
        parcels['score_assessment'] = 0.0
        return parcels

    parcels = parcels.copy()
    cols = assessment_data.columns.tolist()
    print(f"  Assessment columns: {cols[:15]}")

    # --- Find the assessed value column ---
    val_col = None
    for col in cols:
        cl = col.upper().replace(' ', '_').replace('-', '_')
        if any(k in cl for k in ['CURR_ASSMT', 'ASSTVAL', 'VALÉVAL', 'VALEVAL',
                                  'ASSESSED', 'TOTAL_VAL', 'ASSESS_VAL']):
            val_col = col
            break

    if val_col is None:
        print(f"  WARNING: No assessment value column found in: {cols}")
        parcels['score_assessment'] = 0.0
        return parcels

    print(f"  Using value column: '{val_col}'")

    # --- Spatial join (if GeoDataFrame) or PID join ---
    if isinstance(assessment_data, gpd.GeoDataFrame) and 'geometry' in assessment_data.columns:
        print("  Spatial join for assessment...")
        # Build a simple DataFrame with just value + geometry to avoid recursion
        assess_simple = gpd.GeoDataFrame({
            'assess_value': pd.to_numeric(assessment_data[val_col], errors='coerce'),
        }, geometry=assessment_data.geometry.copy(), crs=assessment_data.crs)
        assess_simple = assess_simple.dropna(subset=['assess_value'])
        assess_simple = assess_simple.to_crs(CRS_WGS84)

        # Clean parcel data for join
        centroids = parcels[['geometry', 'city']].copy()
        centroids['geometry'] = centroids.geometry.centroid
        centroids = centroids.reset_index(drop=True)
        assess_simple = assess_simple.reset_index(drop=True)

        joined = gpd.sjoin(centroids, assess_simple, how='left', predicate='within')
        joined = joined[~joined.index.duplicated(keep='first')]

        parcels['total_assessed'] = joined['assess_value'].values
    else:
        print("  PID/PAN join for assessment...")
        pid_a = next((c for c in ['PAN-NCB', 'PAN', 'PID', 'pan'] if c in cols), None)
        pid_p = next((c for c in ['PID', 'PAN', 'PID_INT'] if c in parcels.columns), None)

        if pid_a and pid_p:
            assessment_data[pid_a] = assessment_data[pid_a].astype(str).str.strip()
            parcels[pid_p] = parcels[pid_p].astype(str).str.strip()
            merge_df = assessment_data[[pid_a, val_col]].drop_duplicates(subset=[pid_a])
            merge_df['assess_value'] = pd.to_numeric(merge_df[val_col], errors='coerce')
            parcels = parcels.merge(merge_df[[pid_a, 'assess_value']],
                                    left_on=pid_p, right_on=pid_a, how='left')
            parcels['total_assessed'] = parcels['assess_value']
            parcels = parcels.drop(columns=[pid_a, 'assess_value'], errors='ignore')
        else:
            print(f"  No matching ID column found.")
            parcels['total_assessed'] = np.nan

    # Score: normalize assessed value per m² (higher value = more development pressure)
    if 'lot_area_m2' in parcels.columns:
        parcels['assessed_per_m2'] = np.where(
            parcels['lot_area_m2'] > 0,
            parcels['total_assessed'] / parcels['lot_area_m2'], 0
        )
        valid = parcels['assessed_per_m2'].dropna()
        if len(valid[valid > 0]) > 0:
            p95 = valid[valid > 0].quantile(0.95)
            parcels['score_assessment'] = np.clip(parcels['assessed_per_m2'] / p95, 0, 1)
        else:
            parcels['score_assessment'] = 0.0
    else:
        parcels['score_assessment'] = 0.0

    parcels['land_to_total_ratio'] = np.nan  # Not available in this dataset

    n = parcels['total_assessed'].notna().sum()
    print(f"  Assessment matched: {n}/{len(parcels)} parcels")
    return parcels


# =============================================================================
# 7. COMPOSITE UPZONING SCORE
# =============================================================================
ZONING_INTENSITY = {
    'RS': 1, 'RL': 1, 'RM': 2, 'RH': 3,
    'CC': 4, 'CBP': 4, 'CG': 5, 'CD': 6,
    'MU': 7, 'UC': 8,
    'IL': 3, 'IH': 3,
    'Unzoned': 0, 'Unknown': 0,
}


def compute_upzoning_score(parcels):
    """
    Enhanced composite score (0-100) with 7 weighted components:

    1. Underutilization       25%  — low building coverage on lot
    2. Assessment ratio       20%  — high land value vs building value
    3. Transit proximity      15%  — near bus stops, high stop density
    4. Zoning gap             15%  — low-intensity zone = room to upzone
    5. School proximity       10%  — near schools (family-friendly development)
    6. Hospital proximity      5%  — near healthcare
    7. Development pressure   10%  — recent permit activity nearby
    """
    p = parcels.copy()

    # 1. Underutilization
    p['score_underutil'] = 1 - p['building_coverage']
    small = (p['building_coverage'] > 0) & (p['building_coverage'] < 0.1)
    p.loc[small, 'score_underutil'] *= 1.2
    p['score_underutil'] = p['score_underutil'].clip(0, 1)

    # 2. Assessment ratio — already computed as score_assessment

    # 3. Transit — already computed as score_transit

    # 4. Zoning gap
    p['zone_intensity'] = p['ZoningCode'].map(
        lambda x: ZONING_INTENSITY.get(x, ZONING_INTENSITY.get(x[:2] if len(str(x)) > 1 else x, 0))
    )
    max_i = max(ZONING_INTENSITY.values())
    p['score_zoning_gap'] = 1 - (p['zone_intensity'] / max_i)

    # 5. School — already computed as score_school

    # 6. Hospital — already computed as score_hospital

    # 7. Development pressure — already computed as score_dev_pressure

    # --- Lot size filter ---
    log_area = np.log1p(p['lot_area_m2'])
    p['score_lot_size'] = ((log_area - log_area.min()) /
                           (log_area.max() - log_area.min() + 1e-9))
    p.loc[p['lot_area_m2'] < 200, 'score_lot_size'] = 0
    p.loc[p['lot_area_m2'] > 50000, 'score_lot_size'] *= 0.5

    # --- Weighted composite ---
    W = {
        'underutil':     0.25,
        'assessment':    0.20,
        'transit':       0.15,
        'zoning_gap':    0.15,
        'school':        0.10,
        'hospital':      0.05,
        'dev_pressure':  0.10,
    }

    p['upzoning_score'] = (
        W['underutil']    * p['score_underutil'] +
        W['assessment']   * p['score_assessment'] +
        W['transit']      * p['score_transit'] +
        W['zoning_gap']   * p['score_zoning_gap'] +
        W['school']       * p['score_school'] +
        W['hospital']     * p['score_hospital'] +
        W['dev_pressure'] * p['score_dev_pressure']
    ) * 100

    # Bonus multiplier for lot size sweet spot (500-10000 m²)
    sweet_spot = (p['lot_area_m2'] >= 500) & (p['lot_area_m2'] <= 10000)
    p.loc[sweet_spot, 'upzoning_score'] *= 1.1
    p['upzoning_score'] = p['upzoning_score'].clip(0, 100).round(1)

    print(f"\n  Score distribution:")
    print(f"    Mean:  {p['upzoning_score'].mean():.1f}")
    print(f"    Median:{p['upzoning_score'].median():.1f}")
    print(f"    Top 10%: >= {p['upzoning_score'].quantile(0.9):.1f}")

    # Store weights for reference
    p.attrs['weights'] = W
    return p


# =============================================================================
# 8. VISUALIZATION
# =============================================================================
def create_opportunity_map(parcels, city_name, city_info, top_n=250):
    m = folium.Map(location=[city_info['lat'], city_info['lon']],
                   zoom_start=13, tiles='CartoDB positron')

    top = parcels.nlargest(top_n, 'upzoning_score')

    def color(s):
        if s >= 75: return '#d73027'
        elif s >= 55: return '#fc8d59'
        elif s >= 40: return '#fee090'
        else: return '#91bfdb'

    for _, r in top.iterrows():
        c = r.geometry.centroid
        if np.isnan(c.y) or np.isnan(c.x):
            continue
        html = f"""
        <b>Score: {r['upzoning_score']}/100</b><br>
        Zone: {r.get('ZoningCode','?')} ({r.get('ZoningGroup','?')})<br>
        Lot: {r.get('lot_area_m2', 0):.0f} m² | Coverage: {r.get('building_coverage', 0):.0%}<br>
        <hr>
        Hospital: {r.get('dist_hospital_m', 0):.0f}m | School: {r.get('dist_school_m', 0):.0f}m<br>
        Transit: {r.get('dist_transit_m', 0):.0f}m ({r.get('transit_stops_800m', 0)} stops/800m)<br>
        Permits nearby: {r.get('permits_1km', 0)}<br>
        Assessed: ${r.get('total_assessed', 0):,.0f}<br>
        """
        folium.CircleMarker(
            location=[c.y, c.x], radius=6,
            color=color(r['upzoning_score']),
            fill=True, fill_opacity=0.7,
            popup=folium.Popup(html, max_width=320)
        ).add_to(m)

    heat = [[r.geometry.centroid.y, r.geometry.centroid.x, r['upzoning_score']]
            for _, r in top.iterrows()
            if not np.isnan(r.geometry.centroid.y) and not np.isnan(r['upzoning_score'])]
    if heat:
        HeatMap(heat, radius=20, blur=15, name="Heatmap").add_to(m)

    legend = """
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;
         background:#fff;padding:10px;border-radius:5px;border:2px solid grey;">
    <b>Upzoning Score</b><br>
    <span style="color:#d73027;">●</span> 75-100 (High)<br>
    <span style="color:#fc8d59;">●</span> 55-75 (Medium-High)<br>
    <span style="color:#fee090;">●</span> 40-55 (Medium)<br>
    <span style="color:#91bfdb;">●</span> 0-40 (Lower)<br>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)

    out = os.path.join(DATA_DIR, f"upzoning_{city_name.replace(' ','_')}.html")
    m.save(out)
    print(f"  Map: {out}")
    return m


def plot_score_components(all_parcels):
    """Radar-style comparison of score components across cities."""
    components = ['score_underutil', 'score_assessment', 'score_transit',
                  'score_zoning_gap', 'score_school', 'score_hospital',
                  'score_dev_pressure']
    labels = ['Underutil.', 'Assessment', 'Transit', 'Zoning Gap',
              'Schools', 'Hospital', 'Dev Pressure']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (city, grp) in enumerate(all_parcels.groupby('city')):
        ax = axes[i]
        scores = grp['upzoning_score'].dropna()
        if len(scores) == 0:
            ax.set_title(f'{city} (no data)', fontweight='bold')
            continue
        thresh = scores.quantile(0.9)
        top = grp[grp['upzoning_score'] >= thresh]
        means = [top[c].mean() if c in top.columns else 0 for c in components]
        ax.barh(labels, means, color='steelblue', edgecolor='white')
        ax.set_xlim(0, 1)
        ax.set_title(f'{city} — Top 10% Parcels', fontweight='bold')
        ax.set_xlabel('Mean Component Score')
    plt.suptitle('What drives upzoning potential in each city?',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "component_breakdown.png"), dpi=150,
                bbox_inches='tight')
    plt.show()


def plot_distributions(all_parcels):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (city, grp) in enumerate(all_parcels.groupby('city')):
        ax = axes[i]
        scores = grp['upzoning_score'].dropna()
        if len(scores) == 0:
            ax.set_title(f'{city} (no data)', fontsize=14)
            continue
        ax.hist(scores, bins=50, color='steelblue',
                edgecolor='white', alpha=0.8)
        t90 = scores.quantile(0.9)
        ax.axvline(t90, color='red', ls='--',
                   label=f"Top 10% ({t90:.0f})")
        ax.set_title(city, fontsize=14, fontweight='bold')
        ax.set_xlabel('Upzoning Score')
        ax.set_ylabel('# Parcels')
        ax.legend()
    plt.suptitle('Score Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "distributions.png"), dpi=150,
                bbox_inches='tight')
    plt.show()


# =============================================================================
# 9. EXPORT
# =============================================================================
def export_results(all_parcels):
    export_cols = [
        'PID', 'city', 'ZoningCode', 'ZoningGroup', 'lot_area_m2',
        'building_coverage', 'num_buildings',
        'dist_hospital_m', 'nearest_hospital', 'hospital_beds',
        'dist_school_m', 'dist_transit_m', 'transit_stops_800m',
        'permits_1km', 'total_assessed', 'assessed_per_m2',
        'score_underutil', 'score_assessment', 'score_transit',
        'score_zoning_gap', 'score_school', 'score_hospital',
        'score_dev_pressure', 'upzoning_score'
    ]
    avail = [c for c in export_cols if c in all_parcels.columns]

    for city, grp in all_parcels.groupby('city'):
        safe = city.replace(' ', '_')
        top = grp.nlargest(200, 'upzoning_score')
        top[avail].to_csv(os.path.join(DATA_DIR, f"top200_{safe}.csv"), index=False)
        top.to_file(os.path.join(DATA_DIR, f"top200_{safe}.geojson"), driver='GeoJSON')
        print(f"  Exported top 200 for {city}")

    all_parcels.to_file(os.path.join(DATA_DIR, "all_scored.geojson"), driver='GeoJSON')
    print("  Exported full scored dataset")


import json

# =============================================================================
# 10. PUBLISHABLE INTERACTIVE MAP
# =============================================================================
def generate_publishable_map(gdf, output_path=None, top_n=500):
    """Generate a self-contained HTML map for Substack / web publishing."""
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "upzoning_interactive.html")

    features = []
    for city in ['Saint John', 'Moncton', 'Fredericton']:
        city_data = gdf[gdf['city'] == city].dropna(subset=['upzoning_score'])
        top = city_data.nlargest(top_n, 'upzoning_score')
        for _, r in top.iterrows():
            c = r.geometry.centroid
            if np.isnan(c.x) or np.isnan(c.y):
                continue
            features.append({
                'lat': round(c.y, 6), 'lng': round(c.x, 6),
                'score': round(r.get('upzoning_score', 0), 1),
                'city': city,
                'zone': str(r.get('ZoningCode', '?')),
                'zoneGroup': str(r.get('ZoningGroup', '?')),
                'lotArea': round(r.get('lot_area_m2', 0)),
                'coverage': round(r.get('building_coverage', 0) * 100, 1),
                'hospital': round(r.get('dist_hospital_m', 0)),
                'school': round(r.get('dist_school_m', 0)),
                'transit': round(r.get('dist_transit_m', 0)),
                'stops800': int(r.get('transit_stops_800m', 0)),
                'permits': int(r.get('permits_1km', 0)),
                'assessed': round(r.get('total_assessed', 0) if pd.notna(r.get('total_assessed')) else 0),
            })

    data_json = json.dumps(features)

    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NB Upzoning Opportunities</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0a}
  #map{width:100%;height:100vh}
  .header{position:fixed;top:0;left:0;right:0;z-index:1000;
    background:linear-gradient(135deg,#1a1a2e,#16213e);
    padding:12px 20px;display:flex;align-items:center;gap:16px;
    box-shadow:0 2px 20px rgba(0,0,0,0.4);border-bottom:1px solid rgba(255,255,255,0.1)}
  .header h1{color:#fff;font-size:16px;font-weight:600;white-space:nowrap}
  .header .sub{color:rgba(255,255,255,0.5);font-size:12px;margin-left:-8px}
  .cbtn{padding:6px 16px;border:1px solid rgba(255,255,255,0.2);border-radius:20px;
    background:transparent;color:rgba(255,255,255,0.7);cursor:pointer;font-size:13px;transition:all 0.2s}
  .cbtn:hover{border-color:rgba(255,255,255,0.5);color:#fff}
  .cbtn.active{background:rgba(99,179,237,0.2);border-color:#63b3ed;color:#63b3ed}
  .fg{display:flex;align-items:center;gap:8px;margin-left:auto;color:rgba(255,255,255,0.6);font-size:13px}
  .fg input{width:60px;padding:4px 8px;border:1px solid rgba(255,255,255,0.2);
    border-radius:6px;background:rgba(255,255,255,0.05);color:#fff;font-size:13px;text-align:center}
  .legend{position:fixed;bottom:24px;left:24px;z-index:1000;
    background:rgba(26,26,46,0.95);padding:16px;border-radius:12px;
    border:1px solid rgba(255,255,255,0.1);backdrop-filter:blur(10px)}
  .legend h3{color:#fff;font-size:13px;margin-bottom:10px;font-weight:600}
  .li{display:flex;align-items:center;gap:8px;margin:6px 0}
  .ld{width:14px;height:14px;border-radius:50%;border:2px solid rgba(255,255,255,0.2)}
  .ll{color:rgba(255,255,255,0.7);font-size:12px}
  .stats{position:fixed;bottom:24px;right:24px;z-index:1000;
    background:rgba(26,26,46,0.95);padding:16px;border-radius:12px;
    border:1px solid rgba(255,255,255,0.1);min-width:180px}
  .stats h3{color:#fff;font-size:13px;margin-bottom:10px;font-weight:600}
  .sr{display:flex;justify-content:space-between;margin:4px 0}
  .sl{color:rgba(255,255,255,0.5);font-size:12px}
  .sv{color:#63b3ed;font-size:12px;font-weight:600}
  .leaflet-popup-content-wrapper{background:rgba(26,26,46,0.97)!important;color:#fff!important;
    border-radius:12px!important;border:1px solid rgba(99,179,237,0.3)!important}
  .leaflet-popup-tip{background:rgba(26,26,46,0.97)!important}
  .ps{font-size:28px;font-weight:700;color:#63b3ed;border-bottom:1px solid rgba(255,255,255,0.1);
    padding-bottom:8px;margin-bottom:8px}
  .pr{display:flex;justify-content:space-between;padding:3px 0;font-size:12px}
  .pl{color:rgba(255,255,255,0.5)}.pv{color:rgba(255,255,255,0.9);font-weight:500}
  .pz{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;
    background:rgba(99,179,237,0.15);color:#63b3ed;margin-top:4px}
  @media(max-width:768px){.header{flex-wrap:wrap;gap:8px;padding:8px 12px}
    .header h1{font-size:14px}.fg{margin-left:0}.legend,.stats{display:none}}
</style>
</head>
<body>
<div class="header">
  <h1>NB Upzoning Opportunities</h1>
  <span class="sub">New Brunswick, Canada</span>
  <button class="cbtn active" onclick="flyTo('all',this)">All Cities</button>
  <button class="cbtn" onclick="flyTo('Saint John',this)">Saint John</button>
  <button class="cbtn" onclick="flyTo('Moncton',this)">Moncton</button>
  <button class="cbtn" onclick="flyTo('Fredericton',this)">Fredericton</button>
  <div class="fg">Min score:<input type="number" id="ms" value="0" min="0" max="100" onchange="render()"></div>
</div>
<div id="map"></div>
<div class="legend">
  <h3>Upzoning Score</h3>
  <div class="li"><div class="ld" style="background:#e53e3e"></div><span class="ll">75-100 High</span></div>
  <div class="li"><div class="ld" style="background:#ed8936"></div><span class="ll">55-75 Med-High</span></div>
  <div class="li"><div class="ld" style="background:#ecc94b"></div><span class="ll">40-55 Medium</span></div>
  <div class="li"><div class="ld" style="background:#4299e1"></div><span class="ll">0-40 Lower</span></div>
</div>
<div class="stats" id="st"></div>
<script>
const D=''' + data_json + ''';
const C={all:{lat:46,lng:-65.8,z:8},'Saint John':{lat:45.273,lng:-66.063,z:13},
  Moncton:{lat:46.088,lng:-64.778,z:13},Fredericton:{lat:45.964,lng:-66.643,z:13}};
const map=L.map('map',{zoomControl:false,attributionControl:false}).setView([46,-65.8],8);
L.control.zoom({position:'bottomleft'}).addTo(map);
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',{maxZoom:19}).addTo(map);
let mk=[],cc='all';
function sc(s){return s>=75?'#e53e3e':s>=55?'#ed8936':s>=40?'#ecc94b':'#4299e1'}
function sr(s){return s>=75?7:s>=55?6:5}
function fm(n){return n?n.toLocaleString():'0'}
function ph(d){return`<div style="min-width:200px"><div class="ps">${d.score}<span style="font-size:14px;color:rgba(255,255,255,0.4)">/100</span></div>
<span class="pz">${d.zone} — ${d.zoneGroup}</span><div style="margin-top:10px">
<div class="pr"><span class="pl">Lot area</span><span class="pv">${fm(d.lotArea)} m²</span></div>
<div class="pr"><span class="pl">Coverage</span><span class="pv">${d.coverage}%</span></div>
<div class="pr"><span class="pl">Hospital</span><span class="pv">${fm(d.hospital)}m</span></div>
<div class="pr"><span class="pl">School</span><span class="pv">${fm(d.school)}m</span></div>
<div class="pr"><span class="pl">Transit</span><span class="pv">${fm(d.transit)}m (${d.stops800} stops)</span></div>
<div class="pr"><span class="pl">Permits nearby</span><span class="pv">${fm(d.permits)}</span></div>
<div class="pr"><span class="pl">Assessed</span><span class="pv">${fm(d.assessed)}</span></div></div></div>`}
function render(){mk.forEach(m=>map.removeLayer(m));mk=[];
const mn=parseInt(document.getElementById('ms').value)||0;
let f=D.filter(d=>d.score>=mn);if(cc!=='all')f=f.filter(d=>d.city===cc);
f.forEach(d=>{const m=L.circleMarker([d.lat,d.lng],{radius:sr(d.score),color:sc(d.score),
fillColor:sc(d.score),fillOpacity:0.7,weight:1,opacity:0.9}).bindPopup(ph(d),{maxWidth:280});
m.addTo(map);mk.push(m)});
const ss=f.map(d=>d.score),avg=ss.length?(ss.reduce((a,b)=>a+b,0)/ss.length).toFixed(1):0,
mx=ss.length?Math.max(...ss):0,hi=ss.filter(s=>s>=75).length;
document.getElementById('st').innerHTML=`<h3>${cc==='all'?'All Cities':cc}</h3>
<div class="sr"><span class="sl">Parcels</span><span class="sv">${fm(f.length)}</span></div>
<div class="sr"><span class="sl">Avg score</span><span class="sv">${avg}</span></div>
<div class="sr"><span class="sl">Highest</span><span class="sv">${mx}</span></div>
<div class="sr"><span class="sl">High opportunity</span><span class="sv">${hi}</span></div>`}
function flyTo(c,el){cc=c;document.querySelectorAll('.cbtn').forEach(b=>b.classList.remove('active'));
el.classList.add('active');const v=C[c];map.flyTo([v.lat,v.lng],v.z,{duration:1.2});setTimeout(render,300)}
render();
</script></body></html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    sz = os.path.getsize(output_path) / 1024
    print(f"  Publishable map: {output_path} ({sz:.0f} KB, {len(features)} parcels)")
    print(f"  Host on GitHub Pages or Netlify, then embed on Substack:")
    print(f'    <iframe src="YOUR_URL" width="100%" height="600" frameborder="0"></iframe>')


# =============================================================================
# 11. MAIN PIPELINE
# =============================================================================
def main():
    print("=" * 65)
    print(" NB UPZONING OPPORTUNITY ANALYSIS — ENHANCED")
    print("=" * 65)

    # ---- Load ----
    print("\n[1/9] Loading core data...")
    parcels = load_parcels(PARCELS_SHP)
    hospitals = load_hospitals(HOSPITALS_CSV)

    print("\n[2/9] Loading building footprints...")
    buildings = load_city_geodata(BUILDINGS, "buildings")

    print("\n[3/9] Loading zoning...")
    zoning = load_city_geodata(ZONING, "zones")

    print("\n[4/9] Loading additional layers...")
    schools = load_schools(SCHOOLS_GEOJSON)

    print("  Transit stops:")
    transit = load_transit_stops(TRANSIT_STOPS)

    print("  Building permits:")
    permits = load_city_geodata(PERMITS, "permits")

    print("  Property assessment:")
    # Try city-level GeoJSON first (best for Saint John), fall back to provincial
    assessment = load_assessment_geojson(SJ_ASSESSMENT_GEOJSON)
    if assessment is None:
        assessment = load_assessment_tsv_zip(ASSESSMENT_TSV_ZIP)
    if assessment is None:
        assessment = load_assessment_csv(ASSESSMENT_CSV)

    # ---- Filter to cities ----
    print("\n[5/9] Filtering to target cities...")
    city_frames = []
    for name, info in CITIES.items():
        city_frames.append(filter_parcels_by_city(parcels, name, info))
    all_p = gpd.GeoDataFrame(pd.concat(city_frames, ignore_index=True),
                             geometry='geometry', crs=CRS_WGS84)
    print(f"  Total: {len(all_p)} parcels")

    # ---- Enrich ----
    print("\n[6/9] Spatial joins...")
    all_p = join_zoning(all_p, zoning)
    all_p = calculate_building_coverage(all_p, buildings)

    print("\n[7/9] Proximity & enrichment...")
    all_p = enrich_hospital_proximity(all_p, hospitals)
    all_p = enrich_school_proximity(all_p, schools)
    all_p = enrich_transit_proximity(all_p, transit)
    all_p = enrich_development_pressure(all_p, permits)
    all_p = enrich_assessment(all_p, assessment)

    # ---- Score ----
    print("\n[8/9] Computing upzoning scores...")
    all_p = compute_upzoning_score(all_p)

    # ---- Output ----
    print("\n[9/9] Generating outputs...")
    for name, info in CITIES.items():
        city_data = all_p[all_p['city'] == name]
        if len(city_data) > 0:
            create_opportunity_map(city_data, name, info)

    plot_distributions(all_p)
    plot_score_components(all_p)
    export_results(all_p)

    # ---- Summary ----
    print("\n" + "=" * 65)
    print(" SUMMARY")
    print("=" * 65)
    for city, grp in all_p.groupby('city'):
        t90 = grp['upzoning_score'].quantile(0.9)
        top = grp[grp['upzoning_score'] >= t90]
        print(f"\n  {city}:")
        print(f"    Parcels analyzed: {len(grp)}")
        print(f"    Mean score: {grp['upzoning_score'].mean():.1f}")
        print(f"    Top 10% threshold: {t90:.1f}")
        print(f"    Top zones:")
        for zone, cnt in (top.groupby('ZoningGroup').size()
                          .sort_values(ascending=False).head(5).items()):
            print(f"      {zone}: {cnt} parcels")
        if 'assessed_per_m2' in top.columns:
            apm = top['assessed_per_m2'].dropna()
            if len(apm) > 0:
                print(f"    Avg assessed $/m² (top 10%): ${apm.mean():.0f}")

    print(f"\n  All outputs saved to {DATA_DIR}/")

    # --- Generate publishable interactive map ---
    print("\n[BONUS] Generating publishable map...")
    generate_publishable_map(all_p)
    print("  Done!")


if __name__ == "__main__":
    main()
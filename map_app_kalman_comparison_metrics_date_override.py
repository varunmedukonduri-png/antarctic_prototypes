import streamlit as st
import folium
from streamlit_folium import st_folium
from navigation_with_date import a_star_route, hill_climb_optimize, d_star_lite, predict_iceberg_path
import pandas as pd
import numpy as np
import math
import pickle
from datetime import datetime

# Helper functions
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def route_distance_km(route):
    if not route or len(route) < 2:
        return 0.0
    total = 0.0
    for i in range(len(route)-1):
        total += haversine(route[i][0], route[i][1], route[i+1][0], route[i+1][1])
    return total

def count_turns(route):
    if len(route) < 3:
        return 0
    turns = 0
    for i in range(1, len(route)-1):
        v1 = (route[i][0]-route[i-1][0], route[i][1]-route[i-1][1])
        v2 = (route[i+1][0]-route[i][0], route[i+1][1]-route[i][1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 == 0 or mag2 == 0:
            continue
        cos_angle = dot / (mag1 * mag2)
        angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
        if angle > 30:
            turns += 1
    return turns

st.set_page_config(page_title="Antarctic Navigation with Excel Upload", layout="wide")
st.title("🧊 Antarctic Navigation with AI, Kalman & Excel Upload")
st.markdown("**Upload your Excel file → Select a row → Click 'Update Map' to see actual vs predicted tracks.**")

# ---- Check Models ----
try:
    with open('model_lat.pkl', 'rb') as f:
        pickle.load(f)
    with open('model_lon.pkl', 'rb') as f:
        pickle.load(f)
    models_ok = True
except:
    models_ok = False
    st.warning("⚠️ Models not found. Run 'python setup_and_train.py' first.")

# ---- SIDEBAR ----
st.sidebar.header("📂 1. Upload Excel File")
uploaded_file = st.sidebar.file_uploader("Upload iceberg_data.xlsx", type=["xlsx", "xls"])

# Default values
s_lat, s_lon = -62.0, 10.0
e_lat, e_lon = -58.0, 15.0
wind = 15.0
month, day_of_year, year = 12, 350, 2024
use_excel = False

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success(f"✅ Loaded {len(df)} rows")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    st.sidebar.subheader("🗺️ Map Columns")
    lat_col = st.sidebar.selectbox("Start Latitude", numeric_cols, index=0)
    lon_col = st.sidebar.selectbox("Start Longitude", numeric_cols, index=min(1, len(numeric_cols)-1))
    wind_col = st.sidebar.selectbox("Wind Speed", numeric_cols, index=min(2, len(numeric_cols)-1))
    end_lat_col = st.sidebar.selectbox("End Latitude (Historical)", numeric_cols, index=min(3, len(numeric_cols)-1))
    end_lon_col = st.sidebar.selectbox("End Longitude (Historical)", numeric_cols, index=min(4, len(numeric_cols)-1))
    
    date_candidates = [col for col in all_cols if 'date' in col.lower() or 'time' in col.lower()]
    date_col = st.sidebar.selectbox("📅 Date Column", date_candidates, index=0) if date_candidates else None
    
    row_idx = st.sidebar.slider("Select Data Row", 0, len(df)-1, 0)
    row = df.iloc[row_idx]
    
    excel_s_lat = float(row[lat_col])
    excel_s_lon = float(row[lon_col])
    excel_wind = float(row[wind_col]) if wind_col in df.columns else 15.0
    excel_e_lat = float(row[end_lat_col]) if end_lat_col in df.columns else excel_s_lat - 2.0
    excel_e_lon = float(row[end_lon_col]) if end_lon_col in df.columns else excel_s_lon + 5.0
    
    if date_col and pd.notna(row[date_col]):
        dt_excel = pd.to_datetime(row[date_col])
        excel_month, excel_day, excel_year = dt_excel.month, dt_excel.dayofyear, dt_excel.year
    else:
        excel_month, excel_day, excel_year = 12, 350, 2024
    
    use_excel = st.sidebar.checkbox("Use Excel data for Start/End", value=True)
    
    if use_excel:
        s_lat, s_lon = excel_s_lat, excel_s_lon
        e_lat, e_lon = excel_e_lat, excel_e_lon
        wind = excel_wind
        month, day_of_year, year = excel_month, excel_day, excel_year
        st.sidebar.info(f"Using Excel row {row_idx}: Start ({s_lat:.2f}, {s_lon:.2f}) → End ({e_lat:.2f}, {e_lon:.2f})")

# ---- Manual Inputs ----
st.sidebar.header("📍 2. Manual Start/End")
manual_s_lat = st.sidebar.number_input("Manual Start Lat", value=s_lat, step=0.5, format="%.2f")
manual_s_lon = st.sidebar.number_input("Manual Start Lon", value=s_lon, step=0.5, format="%.2f")
manual_e_lat = st.sidebar.number_input("Manual End Lat", value=e_lat, step=0.5, format="%.2f")
manual_e_lon = st.sidebar.number_input("Manual End Lon", value=e_lon, step=0.5, format="%.2f")

if uploaded_file is None or not use_excel:
    s_lat, s_lon = manual_s_lat, manual_s_lon
    e_lat, e_lon = manual_e_lat, manual_e_lon

st.sidebar.header("📅 3. Date Control")
use_custom_date = st.sidebar.checkbox("Custom Date", value=False)
if use_custom_date:
    dt = st.sidebar.date_input("Pick a date", datetime(2024, 12, 1))
    month, day_of_year, year = dt.month, dt.timetuple().tm_yday, dt.year
else:
    if uploaded_file is not None and use_excel:
        pass
    else:
        month, day_of_year, year = 12, 350, 2024

st.sidebar.header("⚙️ 4. Settings")
use_kalman = st.sidebar.checkbox("Apply Kalman Filter", value=True)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 5, 30, int(wind) if wind else 15)
vessel_speed = st.sidebar.slider("Vessel Speed (knots)", 5, 25, 12, 1)
priority = st.sidebar.selectbox("Your Priority", ["⚡ Speed", "⛽ Fuel", "🛡️ Safety"])

st.sidebar.write(f"**Start:** ({s_lat:.2f}, {s_lon:.2f})")
st.sidebar.write(f"**End:** ({e_lat:.2f}, {e_lon:.2f})")

update = st.sidebar.button("🔄 Update Map", type="primary")

# ---- MAIN EXECUTION ----
# Initialize session state for map data if not exists
if 'map_data' not in st.session_state:
    st.session_state['map_data'] = None

# When update is clicked, compute new routes and store in session state
if update and models_ok:
    try:
        raw, filtered = predict_iceberg_path(
            s_lat - 1.5, s_lon + 1.0,
            wind_speed, 15, 10, 0.3, 1000, 20, 10,
            month, day_of_year, year,
            use_kalman=use_kalman
        )
        danger_path = filtered if use_kalman and filtered else raw

        route_astar = a_star_route((s_lat, s_lon), (e_lat, e_lon), danger_path) if danger_path else None
        route_dstar = d_star_lite((s_lat, s_lon), (e_lat, e_lon), danger_path) if danger_path else None
        route_hill = hill_climb_optimize(route_astar) if route_astar else None

        # Build the map object and store its HTML representation
        m = folium.Map(tiles='OpenTopoMap', zoom_start=2)
        folium.TileLayer('OpenStreetMap').add_to(m)

        # Historical Track
        folium.PolyLine(locations=[(s_lat, s_lon), (e_lat, e_lon)], color='cyan', weight=3, dash_array='10', popup='📜 Historical Track').add_to(m)

        if raw:
            folium.PolyLine(locations=raw, color='red', weight=4, dash_array='8', popup='🔴 Raw AI').add_to(m)
        if use_kalman and filtered:
            folium.PolyLine(locations=filtered, color='gold', weight=6, popup='🟡 Kalman').add_to(m)
        if route_astar:
            folium.PolyLine(locations=route_astar, color='blue', weight=4, popup='🔵 A*').add_to(m)
        if route_dstar:
            folium.PolyLine(locations=route_dstar, color='orange', weight=5, dash_array='4', popup='🟠 D*').add_to(m)
        if route_hill:
            folium.PolyLine(locations=route_hill, color='green', weight=6, popup='🟢 Hill Climbing').add_to(m)

        folium.Marker([s_lat, s_lon], popup='🚢 Start', icon=folium.Icon(color='blue', icon='ship', prefix='fa')).add_to(m)
        folium.Marker([e_lat, e_lon], popup='🏁 End', icon=folium.Icon(color='green', icon='flag-checkered', prefix='fa')).add_to(m)
        if raw:
            folium.Marker(raw[0], popup='🧊 Iceberg', icon=folium.Icon(color='red', icon='warning', prefix='fa')).add_to(m)

        # Fit bounds
        bounds = [[s_lat, s_lon], [e_lat, e_lon]]
        for path in [raw, route_astar, route_dstar, route_hill]:
            if path:
                for pt in path:
                    bounds.append([pt[0], pt[1]])
        if bounds:
            lats = [b[0] for b in bounds]
            lons = [b[1] for b in bounds]
            if lats and lons:
                try:
                    m.fit_bounds([[min(lats)-2, min(lons)-2], [max(lats)+2, max(lons)+2]])
                except:
                    pass

        folium.LayerControl().add_to(m)

        # =====================================================
        # ✅ ADDED: Saves the map as index.html (local only!)
        # =====================================================
        m.save("index.html")   # This creates a file in the current folder

        # Store the map HTML and other data in session state
        st.session_state['map_data'] = {
            'html': m._repr_html_(),
            'routes': {
                'A*': route_astar,
                'D*': route_dstar,
                'Hill Climbing': route_hill
            },
            'danger_path': danger_path,
            'raw_path': raw,
            'vessel_speed': vessel_speed,
            'priority': priority
        }

    except Exception as e:
        st.error(f"❌ Computation error: {e}")

# ---- Display the map from session state ----
if models_ok and st.session_state['map_data'] is not None:
    st.subheader("🗺️ Navigation Map")
    # Use a fixed key so the component doesn't reinitialize
    st.components.v1.html(st.session_state['map_data']['html'], height=600)

    # Download button for the map
    st.download_button(
        "📥 Download Map as HTML",
        st.session_state['map_data']['html'],
        "map.html",
        "text/html"
    )

    # ---- Comparison Table ----
    st.subheader("📊 Algorithm Comparison")
    data = st.session_state['map_data']
    danger_path = data['danger_path']
    routes = data['routes']

    def calc_safety(route, danger_path):
        if not route: return "N/A", "N/A"
        min_d = float('inf')
        total = 0
        cnt = 0
        for r in route:
            for d in danger_path:
                dist = math.sqrt((r[0]-d[0])**2 + (r[1]-d[1])**2)
                if dist < min_d: min_d = dist
                total += dist; cnt += 1
        return min_d, (total / cnt) if cnt > 0 else 0

    table_data = []
    for name, route in routes.items():
        if route is None:
            continue
        min_d, avg_d = calc_safety(route, danger_path)
        dist_km = route_distance_km(route)
        time_h = dist_km / (data['vessel_speed'] * 1.852)
        turns = count_turns(route)
        table_data.append({
            "Algorithm": name,
            "Steps": len(route),
            "Distance (km)": f"{dist_km:.1f}",
            "Time (h)": f"{time_h:.1f}",
            "Turns": turns,
            "Min Safety (deg)": f"{min_d:.3f}" if min_d != float('inf') else "N/A",
            "Avg Safety (deg)": f"{avg_d:.3f}" if avg_d > 0 else "N/A"
        })

    if table_data:
        df_compare = pd.DataFrame(table_data)
        min_steps = min(d["Steps"] for d in table_data)
        min_turns = min(d["Turns"] for d in table_data)
        for d in table_data:
            score = (min_steps / d["Steps"]) * 50 + (min_turns / max(1, d["Turns"])) * 50
            d["Fuel Score"] = f"{score:.1f}%"
        st.dataframe(df_compare, use_container_width=True)

        # Recommendation
        st.subheader("💡 Recommendation")
        priority = data['priority']
        if priority == "⚡ Speed":
            best = min(table_data, key=lambda x: float(x["Time (h)"]))
            st.success(f"**{best['Algorithm']}** is fastest ({best['Time (h)']} hours).")
        elif priority == "⛽ Fuel":
            best = max(table_data, key=lambda x: float(x["Fuel Score"].replace('%','')))
            st.success(f"**{best['Algorithm']}** is most fuel efficient (Score {best['Fuel Score']}).")
        else:  # Safety
            best = max(table_data, key=lambda x: float(x["Avg Safety (deg)"]) if x["Avg Safety (deg)"] != "N/A" else 0)
            st.success(f"**{best['Algorithm']}** is safest (Avg distance {best['Avg Safety (deg)']} deg from ice).")
    else:
        st.warning("No routes found. Try different start/end points.")

    # Accuracy if Excel was used
    if uploaded_file is not None and use_excel and data['raw_path']:
        st.subheader("📐 AI Prediction Accuracy vs Historical End")
        raw_end = data['raw_path'][-1]
        actual = (e_lat, e_lon)
        raw_err = haversine(raw_end[0], raw_end[1], actual[0], actual[1])
        st.metric("Raw AI Error", f"{raw_err:.1f} km")

elif models_ok and st.session_state['map_data'] is None:
    st.info("👈 Upload an Excel file or set coordinates, then click **'Update Map'** to see the route.")
else:
    st.error("⚠️ Models not loaded. Run 'python setup_and_train.py' first.")
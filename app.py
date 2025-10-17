import streamlit as st
import pandas as pd
import numpy as np
import pickle
try:
    import plotly.express as px
except ModuleNotFoundError:
    px = None
try:
    import trimesh
except ModuleNotFoundError:
    trimesh = None
from io import BytesIO
try:
    import joblib
except ModuleNotFoundError:
    joblib = None

DATA_PATH = 'materials_data.csv'
MODEL_PATH = 'model.pkl'

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            saved_data = pickle.load(f)
        return saved_data['model'], saved_data['feature_cols'], saved_data['norm_factor']
    except Exception as e:
        st.warning(f"Error loading model: {e}. Falling back to heuristic recommendations.")
        return None, [], 1.0

def process_stl_file(uploaded_file):
    if trimesh is None:
        st.warning("trimesh not available; using default geometry estimates.")
        return 1.0, 50.0, None
    try:
        stl_data = uploaded_file.read()
        stl_bytes = BytesIO(stl_data)
        mesh = trimesh.load(file_obj=stl_bytes, file_type='stl')
        volume_raw = mesh.volume / 1_000_000
        surface_area_cm2 = mesh.area / 100
        if show_debug:
            st.write(f"Debug: Raw volume: {volume_raw:.3f} cm³, Is watertight: {mesh.is_watertight}, Is volume: {mesh.is_volume}")
        if volume_raw <= 0 or not mesh.is_watertight or not mesh.is_volume:
            st.warning("STL mesh is not watertight, has zero volume, or is 2D. Attempting repair and estimation.")
            mesh.fill_holes()
            mesh.fix_normals()
            repaired_volume = mesh.volume / 1_000_000
            if show_debug:
                st.write(f"Debug: Repaired volume: {repaired_volume:.3f} cm³")
            if repaired_volume <= 0:
                st.warning("STL remains non-volume or 2D. Estimating volume using bounding box.")
                bbox = mesh.bounding_box.extents
                if show_debug:
                    st.write(f"Debug: Bounding box (mm): {bbox}")
                if not all(bbox) or any(x < 0.1 for x in bbox):
                    st.error("Bounding box is degenerate. Using default volume (1 cm³).")
                    volume_cm3 = 1.0
                else:
                    thickness = max(bbox[2], 1.0) if len(bbox) == 3 else 1.0
                    volume_cm3 = (bbox[0] * bbox[1] * thickness) / 1_000_000
                    if show_debug:
                        st.write(f"Debug: Estimated volume: {volume_cm3:.3f} cm³, Thickness: {thickness:.2f} mm")
            else:
                volume_cm3 = repaired_volume
        else:
            volume_cm3 = volume_raw
        if volume_cm3 <= 0:
            st.warning(f"STL volume ({volume_cm3:.3f} cm³) is zero or negative. Using default (1 cm³).")
            volume_cm3 = 1.0
        if surface_area_cm2 > 10_000 or surface_area_cm2 < 1:
            st.warning(f"STL surface area ({surface_area_cm2:.2f} cm²) is unrealistic. Using default (50 cm²).")
            surface_area_cm2 = 50.0
        if show_debug:
            st.write(f"Debug: Final volume: {volume_cm3:.3f} cm³, Final surface area: {surface_area_cm2:.2f} cm²")
        return volume_cm3, surface_area_cm2, stl_bytes
    except Exception as e:
        st.error(f"Error processing STL file: {e}. Using defaults.")
        return 1.0, 50.0, None

def display_stl(stl_bytes):
    if trimesh is None:
        st.info("3D preview disabled: trimesh not available.")
        return
    if stl_bytes is not None:
        try:
            mesh = trimesh.load(file_obj=stl_bytes, file_type='stl')
            scene = trimesh.Scene(mesh)
            img_bytes = scene.save_image(resolution=(800, 600))
            st.image(img_bytes)
        except Exception as e:
            st.error(f"Error rendering 3D model: {e}. Preview skipped on server.")

def estimate_print_time(volume_cm3, surface_area_cm2, material_density=1.25):
    density_factor = material_density / 1.25
    print_time_hours = (volume_cm3 / (5.0 / density_factor)) + (surface_area_cm2 / 100.0)
    return max(print_time_hours, 0.1)

def compute_sustainability_score(row, carbon_w, recyclability_w, norm_factor):
    inv_cf = 1.0 / np.clip(row['carbon_footprint'], 0.1, None)
    score = (
        row['recyclability'] * recyclability_w +
        inv_cf * carbon_w +
        row['biodegradability'] * 0.30 +
        (1.0 - row['toxicity']) * 0.25 +
        row['energy_efficiency'] * 0.20
    )
    if show_debug:
        st.write(f"Debug: Raw score for {row['material_name']}: {score:.3f}, norm_factor: {norm_factor:.3f}")
    return (score / norm_factor).clip(0, 1)

def get_recommendations(strength, flexibility, max_temp, budget, volume_cm3, surface_area_cm2, carbon_w, recyclability_w):
    volume_cm3 = max(volume_cm3, 1.0)
    surface_area_cm2 = max(surface_area_cm2, 10.0)
    temp_threshold = max_temp * 0.9 if volume_cm3 > 5 else max_temp * 0.5
    flex_min, flex_max = flexibility * 0.7, flexibility * 1.3
    filtered_data = data[
        (data['tensile_strength'] >= strength * 0.8) &
        (data['max_temp'] >= temp_threshold) &
        (data['cost_per_kg'] <= budget * 1.2) &
        (data['flexibility'].between(flex_min, flex_max)) &
        (data['volume_cm3'] <= max(volume_cm3 * 2.0, 10.0)) &
        (data['surface_area_cm2'] <= max(surface_area_cm2 * 2.0, 50.0))
    ]
    if filtered_data.empty:
        st.warning("No materials match criteria. Relaxing filters.")
        filtered_data = data[
            (data['tensile_strength'] >= strength * 0.8) &
            (data['max_temp'] >= max_temp * 0.5) &
            (data['cost_per_kg'] <= budget * 1.2) &
            (data['flexibility'].between(flex_min, flex_max))
        ]
    if filtered_data.empty:
        return None, None
    filtered_data = filtered_data.copy()
    filtered_data['volume_cm3'] = volume_cm3
    filtered_data['surface_area_cm2'] = surface_area_cm2
    if model is not None and feature_cols:
        X = filtered_data[feature_cols]
        try:
            filtered_data['suitability_score'] = model.predict(X)
        except ValueError as ve:
            st.error(f"Model prediction failed: {ve}. Check feature columns.")
            return None, None
    else:
        # Fallback: use a neutral suitability score when the ML model isn't available
        filtered_data['suitability_score'] = 0.5
    filtered_data['sustainability_score'] = filtered_data.apply(
        lambda row: compute_sustainability_score(row, carbon_w, recyclability_w, norm_factor), axis=1)
    if model is not None and feature_cols:
        filtered_data['combined_score'] = 0.6 * filtered_data['suitability_score'] + 0.4 * filtered_data['sustainability_score']
    else:
        filtered_data['combined_score'] = filtered_data['sustainability_score']
    top_materials = filtered_data.sort_values('combined_score', ascending=False).head(5)
    if show_debug:
        st.write(f"Debug: Filtered data columns: {list(filtered_data.columns)}")
    return top_materials, filtered_data

st.set_page_config(page_title="EcoPrint AI", page_icon="leaf", layout="wide")
st.title("EcoPrint AI - Sustainable 3D Printing Material Recommendations")

try:
    data = load_data()
    model, feature_cols, norm_factor = load_model()
except FileNotFoundError as e:
    st.error(f"Required files missing: {e}. Ensure materials_data.csv and model.pkl in repo root.")
    st.stop()

st.sidebar.header("Project Requirements")
strength = st.sidebar.slider("Tensile Strength (MPa)", 20, 100, 50)
flexibility = st.sidebar.slider("Flexibility (0-1)", 0.0, 1.0, 0.5, step=0.05)
max_temp = st.sidebar.slider("Max Temperature (°C)", 50, 150, 80)
budget = st.sidebar.slider("Budget ($/kg)", 15, 70, 30)
st.sidebar.header("STL File Upload")
uploaded_file = st.sidebar.file_uploader("Upload STL File", type=["stl"])
st.sidebar.header("Sustainability Preferences")
carbon_weight = st.sidebar.slider("Carbon Footprint Importance", 0.0, 1.0, 0.3)
recyclability_weight = st.sidebar.slider("Recyclability Importance", 0.0, 1.0, 0.25)
st.sidebar.header("Debug Options")
show_debug = st.sidebar.checkbox("Show Debug Output", value=False)

if uploaded_file:
    if uploaded_file.size > 10_000_000:
        st.error("File too large (max 10MB).")
        volume_cm3, surface_area_cm2, stl_bytes = 1.0, 50.0, None
    else:
        volume_cm3, surface_area_cm2, stl_bytes = process_stl_file(uploaded_file)
else:
    volume_cm3, surface_area_cm2, stl_bytes = 1.0, 50.0, None
    st.sidebar.info("No STL uploaded. Using defaults.")

st.header("Recommendations")
top_materials, filtered_data = get_recommendations(strength, flexibility, max_temp, budget, volume_cm3, surface_area_cm2, carbon_weight, recyclability_weight)

st.sidebar.write(f"STL Volume: {volume_cm3:.2f} cm³")
st.sidebar.write(f"STL Surface Area: {surface_area_cm2:.2f} cm²")
material_density = top_materials['density'].iloc[0] if top_materials is not None and not top_materials.empty and 'density' in top_materials.columns else 1.25
print_time = estimate_print_time(volume_cm3, surface_area_cm2, material_density)
st.sidebar.write(f"Estimated Print Time: {print_time:.2f} hours")

if top_materials is None or top_materials.empty:
    st.warning("No materials match. Adjust inputs.")
else:
    display_cols = ['material_name', 'type', 'tensile_strength', 'flexibility', 'max_temp', 'cost_per_kg', 'recyclability', 'carbon_footprint', 'sustainability_score', 'volume_cm3', 'surface_area_cm2']
    available_cols = [col for col in display_cols if col in top_materials.columns]
    top_display = top_materials[available_cols]
    st.dataframe(top_display, use_container_width=True, column_config={col: st.column_config.NumberColumn(format="%.2f") for col in available_cols if col not in ['material_name', 'type']})
    
    st.write("### Sustainability Breakdown")
    metrics = ['recyclability', 'biodegradability', 'energy_efficiency']
    avail_metrics = [m for m in metrics if m in top_materials.columns]
    if avail_metrics and px is not None:
        fig = px.bar(top_materials, x='material_name', y=avail_metrics, barmode='group', title="Sustainability Metrics")
        st.plotly_chart(fig, use_container_width=True)
    elif avail_metrics and px is None:
        st.info("Plotly not available; charts are disabled.")
    
    st.write("### Carbon vs Cost")
    if 'cost_per_kg' in top_materials.columns and 'carbon_footprint' in top_materials.columns:
        if px is not None:
            fig2 = px.scatter(top_materials, x='cost_per_kg', y='carbon_footprint', color='sustainability_score', size='sustainability_score', text='material_name', title="Carbon Footprint vs Cost")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Plotly not available; charts are disabled.")
    
    st.write("### Export")
    csv = top_display.to_csv(index=False)
    st.download_button("Download CSV", csv, "recommendations.csv", "text/csv")

if uploaded_file and stl_bytes:
    st.header("3D Model Preview")
    display_stl(stl_bytes)

st.header("Learn More")
st.write("EcoPrint AI uses RF model on 25 materials for sustainable recs. Upload STL for tailored advice.")
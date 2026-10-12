import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from util import severity_label, log_transform1
import joblib

st.set_page_config(layout="wide", page_title="Earthquake Impact Prediction")
st.title("Earthquake Prediction Dashboard")

df = pd.read_csv('earthquake_1995-2023.csv')
df['severity_label'] = df.apply(lambda x : severity_label(x['magnitude'], x['sig']), axis = 1)
st.subheader("Earthquake Severity Map")
map_center = [df['latitude'].mean(), df['longitude'].mean()+90]
m = folium.Map(location=map_center, zoom_start=1)
source_layers = {}
for severity_type in df['severity_label'].unique():
    source_layers[severity_type] = folium.FeatureGroup(name=severity_type)

severity_colors = {
    "Catastrophic": 'red',
    "Extreme":'orange',      
    "Major": 'yellow',
    "Moderate": 'green'
}
for _, row in df.iterrows():
    dot_color = severity_colors.get(row['severity_label'], "blue")
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3,
        color=dot_color,
        fill_color=dot_color,
        popup=f"<b>{row['severity_label']}</b>, Magnitude: {row['magnitude']}, Location: {row['location']}",
        fill=True
    ).add_to(source_layers[row['severity_label']])

for layer in source_layers.values():
    layer.add_to(m)

folium.LayerControl(position='topright', collapsed=True).add_to(m)

c1, c2, c3 = st.columns([1, 4, 1])
with c2:
    st_folium(m, width=900, height=500)

st.divider()
st.subheader("Data Preview")
st.dataframe(df.head(), use_container_width=True)
st.divider()
st.subheader("Data Plot")

numeric_cols = df.select_dtypes(['float64', 'int64']).columns
c1_choice = st.sidebar.selectbox("X axis", numeric_cols, index=0)
c2_choice = st.sidebar.selectbox("Y axis", numeric_cols, index=1)

if c1_choice and c2_choice:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots()
        ax.scatter(df[c1_choice], df[c2_choice], s=5, alpha=0.5, color='teal')
        ax.set_xlabel(c1_choice)
        ax.set_ylabel(c2_choice)
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

st.divider()
st.subheader("Predict Earthquake Severity")
st.write("Input earthquake parameters")
st.markdown("## **Input Features**")
col1, col2, col3 = st.columns(3)
with col1:
    latitude = st.slider("Latitude", -90.0, 90.0, 0.0)
    longitude = st.slider("Longitude", -180.0, 180.0, 0.0)
    cdi = st.slider("cdi", 0, 10, 1)
    mmi = st.slider("mmi", 1, 12, 1)
    alert = st.select_slider("Select Alert Level",options=["green", "yellow", "orange", "red"],value="green")
with col2:
    sig = st.slider("sig", 0, 3000, 50)
    net = st.select_slider("Select Data Contributor ID", options=['us', 'at', 'pt', 'ak', 'nn', 'ci', 'hv', 'nc', 'official', 'duputel', 'uw'],value='us')
    nst = st.slider("nst", 0, 1000, 10)
    dmin = st.slider("dmin", 0.0, 10.0, 1.0)
    gap = st.slider("Azimuthal Gap", 0.0, 360.0, 90.0)
with col3:
    magType = st.select_slider("Select Algorithm used", options=['mww', 'Mi', 'mwc', 'mwb', 'ml', 'mw', 'ms', 'mb', 'md'], value='mww')
    depth = st.slider("depth", 0.0, 700.0, 30.0)
    daytime = st.select_slider("Select daytime", options=['Morning', 'Afternoon', 'Evening', 'Night'],value='Morning')
    season = st.select_slider("Select season", options=['Winter',"Spring",'Summer','Autumn'], value='Winter')
    tsunami = st.slider("tsunami", 0, 1, 0)

@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

model = load_model()
st.divider()
st.subheader("Predict")
if st.button("Severity Analysis", type="primary", use_container_width=False):
    
    input_values = [
        cdi, mmi, alert, tsunami, sig, net,
        nst, dmin, gap, magType, depth, latitude,
        longitude, daytime, season
    ]
    
    feature_order = [
        'cdi', 'mmi', 'alert', 'tsunami', 'sig', 'net', 
        'nst', 'dmin', 'gap', 'magType', 'depth', 'latitude', 
        'longitude', 'daytime', 'season'
    ]

    input_df = pd.DataFrame([input_values], columns=feature_order)

    try:
        prediction = model.predict(input_df)
        # st.balloons()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Earthquake Magnitude", value=str(prediction[0]))
        with res_col2:
            st.metric(label="Earthquake Severity Status", value=severity_label(prediction[0], input_values[4]))
    except Exception as e:
        st.error(f"Prediction Failed")
        st.write(f"Error Details:{e}")
st.divider()
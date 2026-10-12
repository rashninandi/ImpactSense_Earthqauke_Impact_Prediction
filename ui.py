import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from streamlit_folium import st_folium
from util import severity_label, log_transform1
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="ImpactSense - Eathquake Impact Prediction",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
# All prediction results live here — they survive widget interactions / reruns
if "pred_mag"     not in st.session_state: st.session_state.pred_mag     = None
if "pred_sev"     not in st.session_state: st.session_state.pred_sev     = None
if "pred_depth"   not in st.session_state: st.session_state.pred_depth   = None
if "pred_tsunami" not in st.session_state: st.session_state.pred_tsunami = None
if "pred_sig"     not in st.session_state: st.session_state.pred_sig     = None
if "pred_lat"     not in st.session_state: st.session_state.pred_lat     = None
if "pred_lon"     not in st.session_state: st.session_state.pred_lon     = None
if "pred_alert"   not in st.session_state: st.session_state.pred_alert   = None
if "pred_error"   not in st.session_state: st.session_state.pred_error   = None

# ── Load data & model ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('earthquake_1995-2023.csv')
    df['severity_label'] = df.apply(
        lambda x: severity_label(x['magnitude'], x['sig']), axis=1
    )
    return df

@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

df    = load_data()
model = load_model()

SEVERITY_COLORS = {
    "Catastrophic": "red",
    "Extreme":       "orange",
    "Major":         "blue",
    "Moderate":      "green",
}
SEVERITY_ORDER = ["Moderate", "Major", "Extreme", "Catastrophic"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ImpactSense")
st.sidebar.caption("Earthquake Impact Prediction · 1995–2023")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview & Map", "Data Explorer", "Analytics", "Predict Severity"],
)

st.sidebar.divider()
st.sidebar.subheader("Dataset Filters")

sev_filter = st.sidebar.multiselect(
    "Severity levels",
    options=SEVERITY_ORDER,
    default=SEVERITY_ORDER,
)

mag_min, mag_max = float(df.magnitude.min()), float(df.magnitude.max())
mag_range = st.sidebar.slider(
    "Magnitude range", mag_min, mag_max,
    (mag_min, mag_max), step=0.1,
)

year_col = next((c for c in ['year', 'Year'] if c in df.columns), None)
if year_col:
    yr_min, yr_max = int(df[year_col].min()), int(df[year_col].max())
    year_range = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
else:
    year_range = None

st.sidebar.divider()
st.sidebar.metric("Total records", f"{len(df):,}")

# Apply filters
dff = df[df['severity_label'].isin(sev_filter) & df['magnitude'].between(*mag_range)]
if year_col and year_range:
    dff = dff[dff[year_col].between(*year_range)]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — OVERVIEW & MAP                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
if page == "Overview & Map":

    st.title("Earthquake Impact Prediction")
    st.caption("Global seismic activity · 1995–2023 · ML-powered severity prediction")
    st.divider()

    total   = len(dff)
    avg_mag = dff['magnitude'].mean()
    max_mag = dff['magnitude'].max()
    catast  = (dff['severity_label'] == 'Catastrophic').sum()
    with_ts = int(dff['tsunami'].sum()) if 'tsunami' in dff.columns else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Events",         f"{total:,}")
    k2.metric("Avg Magnitude",         f"{avg_mag:.2f}")
    k3.metric("Max Magnitude",         f"{max_mag:.1f}")
    k4.metric("Catastrophic Events",   f"{catast:,}")
    k5.metric("Tsunami Triggered",     f"{with_ts:,}")

    st.divider()
    st.subheader("Severity Breakdown")
    sev_counts = dff['severity_label'].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    for col_w, sev in zip(st.columns(4), SEVERITY_ORDER):
        count = sev_counts.get(sev, 0)
        pct   = count / total * 100 if total else 0
        col_w.metric(sev, f"{count:,}", f"{pct:.1f}% of total")

    st.divider()
    st.subheader("Global Seismic Activity Map")

    map_style = st.radio(
        "Map style",
        ["OpenStreetMap", "CartoDB dark_matter", "CartoDB positron"],
        horizontal=True,
    )

    m = folium.Map(location=[dff['latitude'].mean(), dff['longitude'].mean()],
                    zoom_start=2, tiles=map_style)
    fgs = {sev: folium.FeatureGroup(name=sev, show=True) for sev in SEVERITY_ORDER}
    for _, row in dff.iterrows():
        sev   = row['severity_label']
        color = SEVERITY_COLORS.get(sev, "blue")
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4, color=color, fill_color=color,
            fill_opacity=0.7, weight=0.5,
            popup=folium.Popup(
                f"<b>{sev}</b><br>Magnitude: <b>{row['magnitude']}</b><br>"
                f"Location: {row.get('location','N/A')}<br>Depth: {row.get('depth','N/A')} km",
                max_width=220),
        ).add_to(fgs[sev])
    for fg in fgs.values(): fg.add_to(m)
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    _, mc, _ = st.columns([1, 10, 1])
    with mc:
        st_folium(m, width=None, height=520, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — DATA EXPLORER                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
elif page == "Data Explorer":

    st.title("Data Explorer")
    st.caption(f"Showing **{len(dff):,}** filtered records · {len(dff.columns)} columns")
    st.divider()

    search = st.text_input("Search by location", placeholder="e.g. Japan, Chile, Indonesia…")
    view_df = dff[dff['location'].str.contains(search, case=False, na=False)] \
              if (search and 'location' in dff.columns) else dff
    if search:
        st.caption(f"Found **{len(view_df):,}** matching records")

    col_left, col_right = st.columns([3, 1])
    rows_pp   = col_left.select_slider("Rows to display", options=[10, 25, 50, 100], value=25)
    highlight = col_right.checkbox("Highlight magnitude", value=True)

    if highlight:
        st.dataframe(
            view_df.head(rows_pp).style.background_gradient(subset=['magnitude'], cmap='YlOrRd'),
            use_container_width=True, height=460)
    else:
        st.dataframe(view_df.head(rows_pp), use_container_width=True, height=460)

    st.divider()
    st.subheader("Summary Statistics")
    num_cols = dff.select_dtypes(['float64','int64']).columns.tolist()
    sel_cols = st.multiselect("Select columns to summarise", num_cols, default=num_cols[:6])
    if sel_cols:
        st.dataframe(dff[sel_cols].describe().round(3), use_container_width=True)

    st.divider()
    st.subheader("Missing Values")
    missing = dff.isnull().sum().reset_index()
    missing.columns = ['Column','Missing Count']
    missing['Missing %'] = (missing['Missing Count'] / len(dff) * 100).round(2)
    missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if missing.empty:
        st.success("No missing values in the filtered dataset.")
    else:
        st.dataframe(missing, use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — ANALYTICS                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
elif page == "Analytics":

    st.title("Seismic Analytics")
    st.divider()

    numeric_cols = dff.select_dtypes(['float64','int64']).columns.tolist()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Scatter Plot", "Distribution", "Correlation Heatmap", "Severity Over Time"]
    )

    with tab1:
        st.subheader("Scatter Plot")
        c1, c2, c3 = st.columns(3)
        x_col    = c1.selectbox("X axis", numeric_cols, index=0)
        y_col    = c2.selectbox("Y axis", numeric_cols, index=1)
        color_by = c3.selectbox("Colour by", ["Severity","Uniform"]) == "Severity"

        fig, ax = plt.subplots(figsize=(9, 5))
        if color_by:
            for sev in SEVERITY_ORDER:
                sub = dff[dff['severity_label'] == sev]
                ax.scatter(sub[x_col], sub[y_col], s=8, alpha=0.55,
                           color=SEVERITY_COLORS[sev], label=sev)
            ax.legend(fontsize=8)
        else:
            ax.scatter(dff[x_col], dff[y_col], s=8, alpha=0.5, color='steelblue')
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        ax.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig)

    with tab2:
        st.subheader("Distribution Explorer")
        dist_col = st.selectbox("Column to plot", numeric_cols, key="dist_col")
        bins     = st.slider("Bins", 10, 120, 50)

        fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(dff[dist_col].dropna(), bins=bins,
                     color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.4)
        axes[0].set_xlabel(dist_col); axes[0].set_ylabel('Count')
        axes[0].set_title(f'Histogram — {dist_col}')
        axes[0].grid(True, linestyle='--', alpha=0.4)

        sev_data = [dff[dff['severity_label']==s][dist_col].dropna().values for s in SEVERITY_ORDER]
        bp = axes[1].boxplot(sev_data, patch_artist=True,
                             medianprops=dict(color='black', linewidth=1.5))
        for patch, color in zip(bp['boxes'], ['#22c55e','#3b82f6','#f97316','#ef4444']):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        axes[1].set_xticks(range(1, 5))
        axes[1].set_xticklabels(SEVERITY_ORDER, rotation=15, fontsize=8)
        axes[1].set_title(f'{dist_col} by Severity')
        axes[1].grid(True, linestyle='--', alpha=0.4)
        fig2.tight_layout(pad=2)
        st.pyplot(fig2)

    with tab3:
        st.subheader("Feature Correlation Heatmap")
        top_n     = st.slider("Number of columns", 5, min(20, len(numeric_cols)), 10)
        corr_cols = numeric_cols[:top_n]
        corr      = dff[corr_cols].corr()

        fig3, ax3 = plt.subplots(figsize=(10, 8))
        im = ax3.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.04)
        ax3.set_xticks(range(len(corr_cols))); ax3.set_yticks(range(len(corr_cols)))
        ax3.set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=8)
        ax3.set_yticklabels(corr_cols, fontsize=8)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr.values[i,j]
                ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=7, color='white' if abs(val)>0.5 else 'black')
        ax3.set_title('Feature Correlation Matrix', fontsize=12, pad=12)
        fig3.tight_layout()
        st.pyplot(fig3)

    with tab4:
        st.subheader("Severity Counts Over Time")
        if year_col:
            yearly = (dff.groupby([year_col,'severity_label'])
                        .size().unstack(fill_value=0)
                        .reindex(columns=SEVERITY_ORDER, fill_value=0))
            fig4, ax4 = plt.subplots(figsize=(12, 5))
            for sev, color in zip(SEVERITY_ORDER, ['#22c55e','#3b82f6','#f97316','#ef4444']):
                if sev in yearly.columns:
                    ax4.plot(yearly.index, yearly[sev], marker='o', markersize=3,
                             linewidth=1.8, label=sev, color=color)
            ax4.set_xlabel("Year"); ax4.set_ylabel("Events")
            ax4.set_title("Events by Severity per Year")
            ax4.legend(); ax4.grid(True, linestyle='--', alpha=0.4)
            fig4.tight_layout(); st.pyplot(fig4)

            st.divider()
            st.subheader("Stacked Area Chart")
            fig5, ax5 = plt.subplots(figsize=(12, 5))
            ax5.stackplot(yearly.index,
                          [yearly.get(s, pd.Series(0, index=yearly.index)) for s in SEVERITY_ORDER],
                          labels=SEVERITY_ORDER,
                          colors=['#22c55e','#3b82f6','#f97316','#ef4444'], alpha=0.75)
            ax5.set_xlabel("Year"); ax5.set_ylabel("Events")
            ax5.set_title("Stacked Severity Distribution Over Time")
            ax5.legend(loc='upper left', fontsize=8)
            ax5.grid(True, linestyle='--', alpha=0.3)
            fig5.tight_layout(); st.pyplot(fig5)
        else:
            st.info("No year column found in the dataset.")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — PREDICT SEVERITY                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
elif page == "Predict Severity":

    st.title("Earthquake Severity Prediction")
    st.caption("Enter seismic parameters to predict magnitude and severity class.")
    st.divider()

    st.subheader("Input Parameters")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Location & Intensity**")
        latitude  = st.slider("Latitude",  -90.0,  90.0,   0.0)
        longitude = st.slider("Longitude", -180.0, 180.0,  0.0)
        cdi       = st.slider("CDI (Community Intensity)", 0, 10, 1)
        mmi       = st.slider("MMI (Instrumental Intensity)", 1, 12, 1)
        alert     = st.select_slider("Alert Level",
                        options=["green","yellow","orange","red"], value="green")

    with col_b:
        st.markdown("**Seismic Network**")
        sig  = st.slider("Significance (sig)", 0, 3000, 50)
        net  = st.select_slider("Data Contributor",
                   options=['us','at','pt','ak','nn','ci','hv','nc','official','duputel','uw'],
                   value='us')
        nst  = st.slider("Station Count (nst)", 0, 1000, 10)
        dmin = st.slider("Min Station Distance (dmin)", 0.0, 10.0, 1.0)
        gap  = st.slider("Azimuthal Gap (degrees)", 0.0, 360.0, 90.0)

    with col_c:
        st.markdown("**Event Properties**")
        magType = st.select_slider("Magnitude Algorithm",
                    options=['mww','Mi','mwc','mwb','ml','mw','ms','mb','md'], value='mww')
        depth   = st.slider("Depth (km)", 0.0, 700.0, 30.0)
        daytime = st.select_slider("Time of Day",
                    options=['Morning','Afternoon','Evening','Night'], value='Morning')
        season  = st.select_slider("Season",
                    options=['Winter','Spring','Summer','Autumn'], value='Winter')
        tsunami = st.slider("Tsunami Trigger (0 = No, 1 = Yes)", 0, 1, 0)

    st.divider()

    with st.expander("Review your inputs before predicting", expanded=False):
        summary = pd.DataFrame({
            "Parameter": ["Latitude","Longitude","CDI","MMI","Alert","Sig","Net",
                          "NST","Dmin","Gap","MagType","Depth","Daytime","Season","Tsunami"],
            "Value":      [latitude, longitude, cdi, mmi, alert, sig, net,
                          nst, dmin, gap, magType, depth, daytime, season, tsunami],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    run_col, clear_col, _ = st.columns([2, 1, 4])
    with run_col:
        run = st.button("Run Severity Analysis", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear Result", use_container_width=True):
            st.session_state.pred_mag     = None
            st.session_state.pred_sev     = None
            st.session_state.pred_error   = None

    # ── Run prediction — save everything into session_state ───────────────
    if run:
        feature_order = ['cdi','mmi','alert','tsunami','sig','net',
                         'nst','dmin','gap','magType','depth',
                         'latitude','longitude','daytime','season']
        input_values  = [cdi, mmi, alert, tsunami, sig, net,
                         nst, dmin, gap, magType, depth,
                         latitude, longitude, daytime, season]
        input_df = pd.DataFrame([input_values], columns=feature_order)

        try:
            with st.spinner("Running model inference…"):
                prediction = model.predict(input_df)

            # ← Store in session_state so results survive any rerun
            st.session_state.pred_mag     = float(prediction[0])
            st.session_state.pred_sev     = severity_label(prediction[0], sig)
            st.session_state.pred_depth   = depth
            st.session_state.pred_tsunami = tsunami
            st.session_state.pred_sig     = sig
            st.session_state.pred_lat     = latitude
            st.session_state.pred_lon     = longitude
            st.session_state.pred_alert   = alert
            st.session_state.pred_error   = None

        except Exception as e:
            st.session_state.pred_error = str(e)
            st.session_state.pred_mag   = None

    # ── Display results — reads from session_state, always visible ────────
    if st.session_state.pred_error:
        st.error(f"Prediction failed: {st.session_state.pred_error}")

    if st.session_state.pred_mag is not None:

        pred_mag  = st.session_state.pred_mag
        pred_sev  = st.session_state.pred_sev
        depth_    = st.session_state.pred_depth
        tsunami_  = st.session_state.pred_tsunami
        sig_      = st.session_state.pred_sig
        lat_      = st.session_state.pred_lat
        lon_      = st.session_state.pred_lon
        alert_    = st.session_state.pred_alert

        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Magnitude", f"{pred_mag:.2f}")
        r2.metric("Severity Class",       pred_sev)
        r3.metric("Depth",                f"{depth_} km")
        r4.metric("Tsunami",              "Yes" if tsunami_ else "No")

        st.divider()

        # Magnitude gauge
        st.subheader("Magnitude Scale")
        gauge_fig, gauge_ax = plt.subplots(figsize=(9, 1.4))
        cmap_gauge = plt.cm.get_cmap('RdYlGn_r')
        for val in np.linspace(0, 10, 200):
            gauge_ax.barh(0, 0.05, left=val, height=0.6,
                          color=cmap_gauge(val / 10), edgecolor='none')
        gauge_ax.axvline(pred_mag, color='black', linewidth=2.5,
                          label=f'M {pred_mag:.2f}')
        gauge_ax.set_xlim(0, 10)
        gauge_ax.set_ylim(-0.5, 0.5)
        gauge_ax.set_xlabel("Magnitude")
        gauge_ax.set_yticks([])
        gauge_ax.legend(loc='upper right', fontsize=9)
        gauge_ax.set_title("Predicted Magnitude on Scale", fontsize=10)
        gauge_ax.grid(False)
        gauge_fig.tight_layout()
        _, gc, _ = st.columns([1, 4, 1])
        with gc:
            st.pyplot(gauge_fig)

        st.divider()

        # Mini location map
        st.subheader("Event Location Preview")
        sev_color = SEVERITY_COLORS.get(pred_sev, "blue")
        mini_m = folium.Map(location=[lat_, lon_], zoom_start=5,
                            tiles='CartoDB positron')
        folium.CircleMarker(
            location=[lat_, lon_], radius=14,
            color=sev_color, fill_color=sev_color, fill_opacity=0.55,
            popup=f"{pred_sev} · M{pred_mag:.2f}",
            tooltip=f"M{pred_mag:.2f} — {pred_sev}",
        ).add_to(mini_m)
        _, mc2, _ = st.columns([1, 4, 1])
        with mc2:
            st_folium(mini_m, width=None, height=320, use_container_width=True)

        st.divider()

        # Historical comparison
        st.subheader("How does this compare to historical data?")
        comp_fig, comp_axes = plt.subplots(1, 2, figsize=(11, 4))

        comp_axes[0].hist(df['magnitude'].dropna(), bins=60,
                          color='steelblue', alpha=0.7,
                          edgecolor='white', linewidth=0.3)
        comp_axes[0].axvline(pred_mag, color='red', linewidth=2,
                              label=f'Your prediction: M{pred_mag:.2f}')
        pct_below = (df['magnitude'] < pred_mag).mean() * 100
        comp_axes[0].set_xlabel("Magnitude"); comp_axes[0].set_ylabel("Count")
        comp_axes[0].set_title(f"M{pred_mag:.2f} is higher than {pct_below:.1f}% of all events")
        comp_axes[0].legend(fontsize=8)
        comp_axes[0].grid(True, linestyle='--', alpha=0.4)

        sev_bar_counts = df['severity_label'].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
        bars = comp_axes[1].bar(SEVERITY_ORDER, sev_bar_counts.values,
                                color=['#22c55e','#3b82f6','#f97316','#ef4444'],
                                alpha=0.75, edgecolor='white', linewidth=0.5)
        for bar, sev in zip(bars, SEVERITY_ORDER):
            if sev == pred_sev:
                bar.set_edgecolor('black'); bar.set_linewidth(2.5); bar.set_alpha(1.0)
        comp_axes[1].set_xlabel("Severity Class"); comp_axes[1].set_ylabel("Count")
        comp_axes[1].set_title(f"Your prediction falls in: {pred_sev}")
        comp_axes[1].grid(True, linestyle='--', alpha=0.4, axis='y')

        comp_fig.tight_layout(pad=2)
        st.pyplot(comp_fig)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("ImpactSense · Built with Streamlit · Earthquake data 1995–2023")
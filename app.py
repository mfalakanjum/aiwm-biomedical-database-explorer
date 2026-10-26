"""AI-WM Lab unified database explorer launcher."""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="AI-WM Lab | Biomedical Database Explorer", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

def home():
    a, b = st.columns([1, 6], vertical_alignment="center")
    with a:
        logo = ROOT / "mimic3" / "logo.jpg"
        if logo.exists():
            st.image(str(logo), width=125)
    with b:
        st.title("AI-WM Lab | Biomedical Database Explorer")
        st.caption("AI in Wearables and Medicine")
    st.success("2 databases found and available for exploration")
    st.write("Select a database to explore its records, clinical information, and physiological signals.")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("MIMIC-III")
            st.write("Critical care and ICU clinical records, waveform availability, diagnoses, admissions, and patient-level metadata.")
            if st.button("Explore MIMIC-III", use_container_width=True, type="primary"):
                st.switch_page(mimic_page)
    with right:
        with st.container(border=True):
            st.subheader("VitalDB")
            st.write("Surgical cases, physiological signal inventories, laboratory measurements, waveform visualization, and exports.")
            if st.button("Explore VitalDB", use_container_width=True, type="primary"):
                st.switch_page(vital_page)
    st.divider()
    st.caption("The explorers retain independent data backends. MIMIC-III uses a local read-only DuckDB; VitalDB fetches remote signals only on request.")

home_page = st.Page(home, title="Choose Database", icon="🏠", default=True)
mimic_page = st.Page(str(ROOT / "mimic3" / "src" / "13_app.py"), title="MIMIC-III Explorer", icon="🫀")
vital_page = st.Page(str(ROOT / "vitaldb" / "src" / "13_explorer_ui.py"), title="VitalDB Explorer", icon="📈")
nav = st.navigation([home_page, mimic_page, vital_page], position="sidebar", expanded=True)
nav.run()

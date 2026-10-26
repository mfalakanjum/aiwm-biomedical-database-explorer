"""AI-WM Lab unified biomedical database explorer — dashboard homepage."""
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="AI-WM Lab | Biomedical Database Explorer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --navy:#15304d; --teal:#087f83; --muted:#63758a; --bg:#f3f6fa; }
.stApp { background:var(--bg); }
.main .block-container { max-width:1380px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e5eaf0; }
[data-testid="stSidebarNav"] a { border-radius:9px; }
.hero { background:linear-gradient(112deg,#123450 0%,#17636d 100%);
  border-radius:18px; padding:30px 34px; color:white; margin-bottom:18px; }
.hero .eyebrow { color:#a8e4e2; font-size:12px; letter-spacing:2px; font-weight:750; }
.hero h1 { color:white; font-size:clamp(27px,3vw,39px); line-height:1.18; margin:10px 0; }
.hero p { color:#dcecf1; font-size:15px; margin:0; }
.stat { background:white; border:1px solid #e3eaf1; border-radius:13px;
 padding:19px 22px; box-shadow:0 3px 13px rgba(20,47,76,.035); min-height:114px; }
.stat .label { color:#62758a; font-size:13px; font-weight:600; }
.stat .value { color:#15304d; font-size:29px; font-weight:800; margin:5px 0 0; }
.stat .detail { color:#6a7d8e; font-size:12px; }
.section-title { color:#15304d; font-size:21px; font-weight:800; margin:23px 0 5px; }
.section-sub { color:#687a8e; font-size:14px; margin:0 0 16px; }
.db-card { background:white; border:1px solid #e0e8ef; border-radius:14px;
 padding:23px 24px 12px; min-height:260px; box-shadow:0 4px 16px rgba(17,48,76,.045); }
.db-tag { display:inline-block; background:#e8f5f4; color:#087f83; border-radius:30px;
 padding:5px 11px; font-size:11px; font-weight:800; letter-spacing:.6px; }
.db-card h3 { color:#15304d; font-size:25px; margin:15px 0 9px; }
.db-card p { color:#64768a; font-size:14px; line-height:1.7; min-height:70px; }
.db-divider { border-top:1px solid #edf1f5; margin:12px 0; }
.db-foot { color:#526b7d; font-size:12px; margin-bottom:10px; }
.feature { background:white; border:1px solid #e5eaf1; border-radius:12px;
 padding:17px 18px; min-height:118px; }
.feature b { color:#173652; font-size:14px; }
.feature p { color:#687a8e; font-size:12px; margin:9px 0 0; line-height:1.6; }
.note { background:#e8f3f5; border:1px solid #cde7e8; color:#365b66;
 padding:15px 19px; border-radius:11px; font-size:13px; line-height:1.65; margin-top:20px; }
div.stButton > button[kind="primary"] { background:#087f83; border:1px solid #087f83;
 border-radius:9px; font-weight:700; min-height:44px; }
div.stButton > button[kind="primary"]:hover { background:#06666a; border-color:#06666a; }
@media(max-width:760px) { .hero { padding:24px 20px; } .db-card { min-height:0; } }
</style>
""", unsafe_allow_html=True)

def home():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">AI-WM LAB · RESEARCH DATA PLATFORM</div>
      <h1>Biomedical Database Explorer</h1>
      <p>Explore clinical information, physiological recordings, and research-ready
      metadata through one unified interface.</p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        st.markdown('<div class="stat"><div class="label">Available databases</div><div class="value">02</div><div class="detail">MIMIC-III · VitalDB</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="stat"><div class="label">Explorer modules</div><div class="value">02</div><div class="detail">Independent research interfaces</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="stat"><div class="label">Platform status</div><div class="value" style="font-size:24px;color:#087f83;">Ready</div><div class="detail">Select a database below</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Select a database</div><div class="section-sub">Choose a research dataset to open its dedicated explorer.</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("""
        <div class="db-card">
          <span class="db-tag">CRITICAL CARE</span>
          <h3>MIMIC-III Explorer</h3>
          <p>Investigate ICU records, clinical admissions, diagnoses,
          subject-level metadata, and physiological waveform availability.</p>
          <div class="db-divider"></div>
          <div class="db-foot">Clinical records &nbsp;·&nbsp; ICU stays &nbsp;·&nbsp; Signal catalog</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore MIMIC-III  →", type="primary", use_container_width=True, key="open_mimic"):
            st.switch_page(mimic_page)
    with right:
        st.markdown("""
        <div class="db-card">
          <span class="db-tag">PERIOPERATIVE CARE</span>
          <h3>VitalDB Explorer</h3>
          <p>Explore surgical cases, laboratory measurements, monitoring tracks,
          physiological waveforms, and selected signal exports.</p>
          <div class="db-divider"></div>
          <div class="db-foot">Surgical cases &nbsp;·&nbsp; Laboratory data &nbsp;·&nbsp; Signal viewer</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore VitalDB  →", type="primary", use_container_width=True, key="open_vital"):
            st.switch_page(vital_page)

    st.markdown('<div class="section-title">Research capabilities</div><div class="section-sub">Tools available across the database-specific explorers.</div>', unsafe_allow_html=True)
    features = [
        ("Clinical data exploration", "Browse patient, admission, and case metadata."),
        ("Physiological signals", "Inspect signal inventories and recording availability."),
        ("Cohort discovery", "Apply database-specific filters to identify relevant records."),
        ("Research exports", "Export supported metadata and selected signal data."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (title, desc) in zip(cols, features):
        with col:
            st.markdown(f'<div class="feature"><b>{title}</b><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="note"><b>Data access and scientific scope</b><br>
    MIMIC-III uses a local read-only DuckDB containing prepared metadata.
    VitalDB uses local metadata and retrieves remote signal samples only when explicitly requested.
    Signal availability does not necessarily establish sample-level signal quality or continuity.
    </div>
    """, unsafe_allow_html=True)
    st.caption("AI-WM Lab · AI in Wearables and Medicine · Biomedical Database Explorer")

home_page = st.Page(home, title="Choose Database", icon="🏠", default=True)
mimic_page = st.Page(str(ROOT / "mimic3" / "src" / "13_app.py"), title="MIMIC-III Explorer", icon="🫀")
vital_page = st.Page(str(ROOT / "vitaldb" / "src" / "13_explorer_ui.py"), title="VitalDB Explorer", icon="📈")
nav = st.navigation([home_page, mimic_page, vital_page], position="sidebar", expanded=True)
nav.run()

# streamlit_app.py
# Streamlit web interface for PDC Drill Bit Designer.
import os
os.environ['OCC_DISABLE_GL'] = '1'
import streamlit as st
from iadc_mapper import decode_iadc, IADC_BODY, IADC_FORMATION, IADC_CUTTER, IADC_PROFILE
from main import run_design
from pathlib import Path

st.set_page_config(page_title="PDC Drill Bit Designer", layout="wide")
st.title("🛢️ PDC Drill Bit Designer – IADC Classification")

col1, col2 = st.columns(2)

with col1:
    body = st.selectbox("Body Material", [f"{k} – {v['name']}" for k, v in IADC_BODY.items()])
    formation = st.selectbox("Formation Hardness", [f"{k} – {v['name']}" for k, v in IADC_FORMATION.items()])
    cutter = st.selectbox("Cutter Size", [f"{k} – {v['label']} ({v['size_mm']}mm)" for k, v in IADC_CUTTER.items()])
    profile = st.selectbox("Bit Profile", [f"{k} – {v['name']}" for k, v in IADC_PROFILE.items()])

code = f"{body[0]}{formation[0]}{cutter[0]}{profile[0]}"
st.info(f"**IADC Code:** `{code}`")

with col2:
    st.subheader("Decoded Parameters")
    params = decode_iadc(code)
    st.json(params)

if st.button("🚀 Generate PDC Bit", type="primary"):
    with st.spinner("Generating bit..."):
        components = run_design(params, visualize=False, export_fea=True)
        st.success("Bit generated successfully!")
        output_dir = Path("Output")
        stl_file = output_dir / "pdc_bit_surface.stl"
        if stl_file.exists():
            with open(stl_file, "rb") as f:
                st.download_button("Download STL mesh", f, file_name="pdc_bit.stl")
        else:
            st.error("STL file not found.")

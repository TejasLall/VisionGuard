"""VisionGuard — Industrial Anomaly Inspection (Streamlit entry point).

Run with:  streamlit run app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.app_support import init_state  # noqa: E402

st.set_page_config(page_title="VisionGuard — Industrial Anomaly Inspection", page_icon=":material/policy:",
                   layout="wide")
init_state()

page = st.navigation(
    [
        st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True),
        st.Page("app_pages/inspect.py", title="Inspect an image", icon=":material/image_search:"),
        st.Page("app_pages/compare.py", title="Compare methods", icon=":material/compare:"),
        st.Page("app_pages/training.py", title="Dataset & training", icon=":material/model_training:"),
        st.Page("app_pages/evaluation.py", title="Evaluation", icon=":material/analytics:"),
    ],
    position="top",
)

with st.sidebar:
    st.markdown("### :material/policy: VisionGuard")
    st.caption("Industrial anomaly inspection · PCA normal-subspace detection with illumination-aware, "
               "multi-scale and adaptive-threshold enhancements.")
    st.caption(f"Dataset folder: `{st.session_state.dataset_root}`")

st.title(page.title, anchor=False)
page.run()

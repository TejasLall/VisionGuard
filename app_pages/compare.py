import pandas as pd
import streamlit as st

from src.app_support import SCORE_NOTE, fmt, gt_mask_for, image_source, load_model, model_pairs
from src.config import ThresholdConfig
from src.data.image_io import ImageLoadError
from src.experiments.runner import STANDARD_VARIANTS
from src.visualization.plots import render_heatmap, render_overlay

st.caption("The same image through all five variants. Each enhancement is switched on alone (B, C, D) and all "
           "together (E). Models are identical apart from the stated differences.")

pairs = {k: v for k, v in model_pairs().items() if {"raw", "illum"} <= set(v)}
if not pairs:
    st.info("Comparison needs both the `__raw` and `__illum` model of a category. Fit them on the "
            "**Dataset & training** page (option *Fit both preprocessing variants*).", title="Models missing")
    st.stop()

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    choice = c1.selectbox("Category / backend", list(pairs), key="cmp_model")
    alpha = c2.slider("α for multi-scale variants (C, E)", 0.05, 0.95, 0.5, 0.05, key="cmp_alpha")
    k = c3.slider("k for adaptive variants (D, E)", 0.0, 6.0, 3.0, 0.25, key="cmp_k",
                  help="T = mean + k·std of held-out normal calibration image scores")
    fixed = c4.number_input("Fixed T for A, B, C (z-units)", 0.0, 50.0, 3.0, 0.5, key="cmp_fixed")
    try:
        rgb, img_name, mask_path, true_label = image_source("cmp", st.session_state.dataset_root)
    except ImageLoadError as exc:
        st.error(str(exc), title="Invalid image")
        rgb = None

rows = []
for v in STANDARD_VARIANTS:
    a = alpha if v.multiscale else 1.0
    rows.append({"variant": v.key, "method": v.name, "preprocessing": "on" if v.preprocess else "off",
                 "scales": f"fine + coarse (α={a:.2f})" if v.multiscale else "fine only",
                 "threshold": f"mean + {k:g}·std" if v.threshold == "mean_std" else f"fixed {fixed:g}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Parameters of the five compared method variants")

if rgb is None:
    st.info("Choose an image to compare the variants.", icon=":material/upload:")
    st.stop()

models = {"raw": load_model(pairs[choice]["raw"]), "illum": load_model(pairs[choice]["illum"])}
with st.spinner("Running five variants…"):
    analyses = {key: m.analyze(rgb, ("fine", "coarse")) for key, m in models.items()}
    preds = []
    for v in STANDARD_VARIANTS:
        m = models["illum" if v.preprocess else "raw"]
        thr = ThresholdConfig(method=v.threshold, fixed_value=fixed, k=k)
        preds.append(m.predict(rgb, alpha=alpha if v.multiscale else 1.0, thr_cfg=thr,
                               analysis=analyses["illum" if v.preprocess else "raw"]))

gt = gt_mask_for(mask_path, models["raw"].size)
if true_label is not None:
    st.caption(f"Ground truth (evaluation only): **{'defective' if true_label else 'normal'}**")
cols = st.columns(5, border=True)
for col, v, p in zip(cols, STANDARD_VARIANTS, preds):
    with col:
        st.markdown(f"**{v.key} · {v.name}**")
        st.image(render_overlay(p.image, p.heatmap, p.mask, p.threshold), width="stretch",
                 alt=f"Overlay for variant {v.key}")
        st.image(render_heatmap(p.heatmap, p.threshold), width="stretch", alt=f"Heat map for variant {v.key}")
        verdict = ":red[**ANOMALOUS**]" if p.is_anomalous else ":green[**normal**]"
        st.markdown(f"{verdict}  \nscore {fmt(p.image_score, 2)} · T {fmt(p.threshold, 2)}  \n"
                    f"flagged {100 * p.mask.mean():.2f}%")
        if gt is not None and gt.any():
            inter = (p.mask & gt).sum()
            st.caption(f"pixel recall {inter / gt.sum():.2f} · precision "
                       f"{(inter / p.mask.sum()) if p.mask.any() else float('nan'):.2f}")
if gt is not None:
    st.image(gt.astype("uint8") * 255, caption="Ground-truth mask (evaluation only)", width=220,
             alt="Ground-truth defect mask")
st.caption(SCORE_NOTE + " A single image says little about a method; use the *Evaluation* page for dataset-level "
           "comparisons.")

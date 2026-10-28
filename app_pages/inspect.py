import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.app_support import (SCORE_NOTE, fmt, gt_mask_for, image_source, load_model, model_pairs, png_bytes,
                             threshold_controls)
from src.data.image_io import ImageLoadError
from src.preprocessing.illumination import ImagePreprocessor
from src.visualization.plots import render_heatmap, render_legend, render_mask, render_overlay

pairs = model_pairs()
if not pairs:
    st.info("No fitted model found. Open **Dataset & training**, choose a dataset folder and fit a model on its "
            "normal images (or run `python scripts/train.py --dataset <folder> --category <name>`).",
            title="No trained model yet", icon=":material/info:")
    st.stop()

left, right = st.columns([1, 2.6], gap="large")
with left:
    with st.container(border=True):
        st.markdown("**Model**")
        choice = st.selectbox("Product category / model", list(pairs), key="insp_model",
                              help="Models are fitted on normal training images of one category.")
        avail = pairs[choice]
        pre = st.toggle("Illumination-aware preprocessing (A)", value="illum" in avail, disabled=len(avail) < 2,
                        key="insp_pre", help="Flat-field + luminance normalisation + CLAHE. Needs the '__illum' model.")
        name = avail["illum" if pre and "illum" in avail else "raw" if "raw" in avail else "illum"]
        scale = st.segmented_control("Detection scale (B)", ["Single-scale", "Multi-scale"], default="Multi-scale",
                                     required=True, key="insp_scale",
                                     help="Single-scale uses the fine 9 px features only; multi-scale fuses fine "
                                          "and coarse (33 px) maps.")
        alpha = st.slider("α (weight of the fine scale)", 0.0, 1.0, 0.5, 0.05, key="insp_alpha",
                          disabled=scale == "Single-scale",
                          help="S = α·S_fine + (1 − α)·S_coarse")
    with st.container(border=True):
        st.markdown("**Decision threshold (C)**")
        thr_cfg = threshold_controls("insp")
    with st.container(border=True):
        st.markdown("**Image**")
        try:
            rgb, img_name, mask_path, true_label = image_source("insp", st.session_state.dataset_root)
        except ImageLoadError as exc:
            st.error(str(exc), title="Invalid image")
            rgb = None

with right:
    if rgb is None:
        st.info("Upload an image or pick a dataset test image to run the inspection.", icon=":material/upload:")
        st.stop()
    try:
        pipe = load_model(name)
    except Exception as exc:  # corrupt or incompatible model folder
        st.error(f"Could not load model '{name}': {exc}", title="Model error")
        st.stop()

    a = alpha if scale == "Multi-scale" else 1.0
    with st.spinner("Running inspection…"):
        pred = pipe.predict(rgb, alpha=a, thr_cfg=thr_cfg)
    heat_rgb = render_heatmap(pred.heatmap, pred.threshold)
    mask_img = render_mask(pred.mask)
    overlay = render_overlay(pred.image, pred.heatmap, pred.mask, pred.threshold)

    with st.container(horizontal=True):
        st.metric("Verdict", "ANOMALOUS" if pred.is_anomalous else "Normal", border=True)
        st.metric("Max anomaly score", fmt(pred.max_score, 2), border=True)
        st.metric("Mean anomaly score", fmt(pred.mean_score, 2), border=True)
        st.metric("Threshold T", fmt(pred.threshold, 2), border=True)
        st.metric("Flagged area", f"{100 * pred.mask.mean():.2f}%", border=True)
    if pred.is_anomalous:
        st.error(f"Image score {pred.image_score:.2f} ≥ T = {pred.threshold:.2f}.", title="Defect suspected",
                 icon=":material/report:")
    else:
        st.success(f"Image score {pred.image_score:.2f} < T = {pred.threshold:.2f}.", title="No defect detected",
                   icon=":material/verified:")
    if true_label is not None:
        st.caption(f"Ground-truth label of this dataset image (for evaluation only): "
                   f"**{'defective' if true_label else 'normal'}**")
    st.caption(f"{SCORE_NOTE} Threshold: {pred.threshold_description}. Backend: {pipe.extractor.label}. "
               f"Model `{name}`.")

    r1 = st.columns(3)
    r1[0].image(pred.image, caption="Original (resized)", width="stretch", alt="Original input image")
    r1[1].image(pred.processed, caption="Preprocessed" + ("" if pre else " (preprocessing off = original)"),
                width="stretch", alt="Image after illumination-aware preprocessing")
    r1[2].image(heat_rgb, caption="Anomaly heat map", width="stretch", alt="Anomaly heat map")
    r2 = st.columns(3)
    gt = gt_mask_for(mask_path, pipe.size)
    r2[0].image(mask_img, caption="Predicted binary mask", width="stretch", alt="Predicted defect mask")
    r2[1].image(overlay, caption="Overlay (cyan = predicted defect outline)", width="stretch",
                alt="Heat map and predicted defect outline over the image")
    if gt is not None:
        r2[2].image(gt.astype(np.uint8) * 255, caption="Ground-truth mask (evaluation only)", width="stretch",
                    alt="Ground-truth defect mask")
    else:
        r2[2].image(render_legend(pred.threshold, pred.heatmap), caption="Legend: heat-map colours",
                    width="stretch", alt="Colour legend from normal to anomalous with threshold marker")
    if gt is not None:
        st.image(render_legend(pred.threshold, pred.heatmap), caption="Legend: dark = normal, bright = anomalous; "
                 "marker = threshold T", width=360, alt="Colour legend from normal to anomalous")

    stem = (img_name or "image").rsplit(".", 1)[0]
    with st.container(horizontal=True):
        st.download_button("Heat map", png_bytes(heat_rgb), f"{stem}_heatmap.png", "image/png",
                           icon=":material/download:")
        st.download_button("Binary mask", png_bytes(mask_img), f"{stem}_mask.png", "image/png",
                           icon=":material/download:")
        st.download_button("Overlay", png_bytes(overlay), f"{stem}_overlay.png", "image/png",
                           icon=":material/download:")

    with st.expander("Illumination diagnostics and per-scale maps"):
        d0 = ImagePreprocessor.diagnostics(pred.image)
        d1 = ImagePreprocessor.diagnostics(pred.processed)
        st.dataframe(pd.DataFrame({"original": d0, "preprocessed": d1}).T, alt="Luminance statistics before and "
                     "after preprocessing", column_config={k: st.column_config.NumberColumn(format="%.3f") for k in d0})
        g0 = cv2.calcHist([cv2.cvtColor(pred.image, cv2.COLOR_RGB2GRAY)], [0], None, [32], [0, 256]).ravel()
        g1 = cv2.calcHist([cv2.cvtColor(pred.processed, cv2.COLOR_RGB2GRAY)], [0], None, [32], [0, 256]).ravel()
        hist = pd.DataFrame({"grey level": np.arange(32) * 8, "original": g0, "preprocessed": g1})
        st.bar_chart(hist, x="grey level", y=["original", "preprocessed"], stack=False, x_label="grey level (0–255)",
                     y_label="pixel count", alt="Luminance histograms of the original and preprocessed image")
        cols = st.columns(len(pred.z_maps))
        for c, (s, z) in zip(cols, pred.z_maps.items()):
            c.image(render_heatmap(z, pred.threshold), caption=f"{s} scale (z-normalised)", width="stretch",
                    alt=f"{s}-scale anomaly map")
        t = pred.timings
        st.caption("Timing (ms): " + " · ".join(f"{k} {1000 * v:.0f}" for k, v in t.items()))

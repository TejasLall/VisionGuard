import streamlit as st

from src.app_support import (cached_validation, deep_backend_installed, experiment_dirs, fmt, load_experiment,
                             model_table)
from src.config import PipelineConfig
from src.experiments.runner import summarize
from src.utils.paths import DEFAULT_CONFIG_PATH

st.caption("Adaptive multi-scale subspace anomaly detection for robust industrial defect inspection")

with st.container(border=True):
    st.markdown(
        "VisionGuard learns what **normal** products look like from defect-free images only, then flags and localises "
        "anything that deviates. Patch features are modelled with a **PCA normal subspace**; the anomaly score of a patch "
        "is its **reconstruction residual** `‖x − x̂‖`, with `x̂ = μ + V_k V_kᵀ(x − μ)`.\n\n"
        "Three enhancements can be switched on independently and are compared experimentally: "
        "**A** illumination-aware preprocessing · **B** multi-scale anomaly maps · **C** adaptive statistical thresholds."
    )
    st.caption("Research inspiration: SubspaceAD (Lendering, Akdag, Bondarev, CVPR 2026). This is an independent, "
               "simplified implementation inspired by that work — not a reproduction of the paper.")

root = st.session_state.dataset_root
rep = cached_validation(root)
models = model_table()
cfg = PipelineConfig.from_yaml(DEFAULT_CONFIG_PATH)

n_train = sum(c["n_train"] for c in rep.categories.values())
with st.container(horizontal=True):
    st.metric("Dataset status", "Valid" if rep.ok else "Not ready", border=True)
    st.metric("Categories", len(rep.categories), border=True)
    st.metric("Normal training images", n_train, border=True)
    st.metric("Saved models", len(models), border=True)

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True, height="stretch"):
        st.subheader("Dataset", anchor=False)
        st.caption(f"`{root}` · layout: **{rep.layout}**")
        if rep.categories:
            st.dataframe(
                [{"category": k, "train (normal)": v["n_train"], "test normal": v["n_test_normal"],
                  "test defective": v["n_test_defective"], "masks": "yes" if v["has_masks"] else "no"}
                 for k, v in rep.categories.items()],
                hide_index=True, alt="Categories in the selected dataset with image counts")
        for e in rep.errors[:3]:
            st.error(e)
        if "synthetic" in str(root).lower():
            st.warning("This is the synthetic smoke-test dataset: useful for checking the software, "
                       "meaningless as a performance benchmark.", icon=":material/science:")

with col2:
    with st.container(border=True, height="stretch"):
        st.subheader("Feature backend & configuration", anchor=False)
        st.markdown(f"**Default backend:** `{cfg.features.backend}` — colour, gradient and LBP texture statistics, CPU only.")
        st.markdown("**Optional DINOv2 backend:** " + ("installed (PyTorch + transformers found)" if deep_backend_installed()
                                                       else "not installed — see README, section 5"))
        st.markdown(
            f"- Image size **{cfg.features.image_size} px** · PCA **{cfg.detector.n_components}** "
            f"{'explained-variance ratio' if isinstance(cfg.detector.n_components, float) else 'components'}\n"
            f"- Fine scale: {cfg.features.fine_window} px window / stride {cfg.features.fine_stride} · "
            f"coarse: {cfg.features.coarse_window} px / stride {cfg.features.coarse_stride} · default α = {cfg.detector.alpha}\n"
            f"- Calibration: **{cfg.detector.calibration}** ({cfg.detector.calib_folds} folds, normal training images only)\n"
            f"- Thresholds: fixed T = {cfg.threshold.fixed_value}, mean + {cfg.threshold.k}·std, "
            f"{cfg.threshold.percentile}th percentile")

with st.container(border=True):
    st.subheader("Latest evaluation results", anchor=False)
    dirs = experiment_dirs()
    if not dirs:
        st.info("No experiment has been run yet. Use the *Evaluation* page or `python scripts/run_experiments.py`.",
                title="Not run")
    else:
        res, _, meta = load_experiment(dirs[0])
        st.caption(f"`{dirs[0].name}` · dataset `{meta.get('dataset_root', '?')}` · "
                   f"{res['category'].nunique()} categor{'y' if res['category'].nunique() == 1 else 'ies'}")
        summ = summarize(res).reset_index()
        st.dataframe(
            summ[["variant", "method", "image_auroc", "pixel_auroc", "image_f1", "image_fpr", "pixel_f1",
                  "mean_inference_ms"]],
            hide_index=True, alt="Mean metrics per method variant in the latest experiment",
            column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                           ["image_auroc", "pixel_auroc", "image_f1", "image_fpr", "pixel_f1"]}
            | {"mean_inference_ms": st.column_config.NumberColumn("ms / image", format="%.0f")})
        if "synthetic" in str(meta.get("dataset_root", "")).lower():
            st.caption("These numbers come from the synthetic smoke-test data and do not indicate real performance.")
        best = summ.loc[summ["image_auroc"].idxmax()] if summ["image_auroc"].notna().any() else None
        if best is not None:
            st.caption(f"Highest mean image AUROC in this run: variant {best['variant']} ({fmt(best['image_auroc'])}).")

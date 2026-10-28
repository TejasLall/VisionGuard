import time
from pathlib import Path

import streamlit as st

from src.app_support import cached_category, cached_validation, deep_backend_installed, manager, model_table
from src.config import PipelineConfig
from src.data.dataset_validator import MVTEC_HELP
from src.data.image_io import load_image_rgb, resize_square
from src.data.synthetic import create_synthetic_dataset
from src.experiments.runner import select_train_paths
from src.features.base import BackendUnavailableError
from src.models.model_manager import model_name
from src.models.pipeline import AnomalyPipeline
from src.utils.paths import DATA_DIR, DEFAULT_CONFIG_PATH

st.caption("Models are fitted on NORMAL training images only. Test images and ground-truth masks are never used "
           "here — they are reserved for evaluation.")

# ------------------------------------------------------------------ dataset
with st.container(border=True):
    st.subheader("1 · Dataset folder", anchor=False)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        root = st.text_input("Dataset directory", st.session_state.dataset_root, width=520,
                             help="MVTec-style root (folder containing bottle/, cable/, …), a single category folder, "
                                  "or a custom dataset (normal/, test/normal/, test/defective/).")
        if st.button("Validate", icon=":material/fact_check:"):
            cached_validation.clear()
        if st.button("Create synthetic demo data", icon=":material/science:",
                     help="Writes data/synthetic — for software testing only, not a benchmark."):
            create_synthetic_dataset(DATA_DIR / "synthetic")
            root = str(DATA_DIR / "synthetic")
            cached_validation.clear()
    st.session_state.dataset_root = root
    rep = cached_validation(root)
    if rep.ok:
        st.success(f"{len(rep.categories)} usable categor{'y' if len(rep.categories) == 1 else 'ies'} "
                   f"({rep.layout} layout).", title="Dataset valid")
    else:
        st.error("\n\n".join(rep.errors) or "Unknown problem", title="Dataset not usable")
        with st.expander("How to get MVTec AD"):
            st.markdown(MVTEC_HELP)
            st.code("python scripts/prepare_dataset.py extract <archive.tar.xz> --out data/mvtec_ad\n"
                    "python scripts/prepare_dataset.py validate data/mvtec_ad", language="bash")
    for w in rep.warnings:
        st.warning(w)
    if rep.categories:
        st.dataframe(
            [{"category": k, "normal train": v["n_train"], "test normal": v["n_test_normal"],
              "test defective": v["n_test_defective"], "defect types": ", ".join(v["defect_types"]),
              "masks": "yes" if v["has_masks"] else "no"} for k, v in rep.categories.items()],
            hide_index=True, alt="Discovered categories and image counts")

# ------------------------------------------------------------------ fit
with st.container(border=True):
    st.subheader("2 · Fit models", anchor=False)
    if not rep.ok:
        st.info("Select a valid dataset first.")
    else:
        base = PipelineConfig.from_yaml(DEFAULT_CONFIG_PATH)
        cats = st.multiselect("Categories", list(rep.categories), default=list(rep.categories)[:1])
        c1, c2, c3 = st.columns(3)
        with c1:
            backend = st.segmented_control(
                "Feature backend", ["handcrafted", "dinov2"], default="handcrafted", required=True,
                help="Handcrafted: CPU colour/gradient/LBP features. DINOv2: pretrained ViT patch tokens "
                     "(optional install, slower on CPU).")
            if backend == "dinov2" and not deep_backend_installed():
                st.warning("PyTorch/transformers are not installed; DINOv2 cannot be used.")
            size = st.select_slider("Analysis resolution (px)", [128, 192, 256, 320], value=base.features.image_size)
        with c2:
            n_train = st.number_input("Normal images to use (0 = all)", 0, 1000, 20, 1,
                                      help="A seeded random subset — the few-shot setting.")
            pca_mode = st.segmented_control("PCA size", ["Explained variance", "Components"],
                                            default="Explained variance", required=True)
            if pca_mode == "Explained variance":
                n_comp = st.slider("Explained-variance ratio", 0.5, 0.995, 0.95, 0.005)
            else:
                n_comp = int(st.number_input("Number of components", 1, 1000, 10))
        with c3:
            folds = st.number_input("Calibration folds (cross-fitting)", 2, 10, base.detector.calib_folds,
                                    help="Held-out NORMAL training images provide the calibration scores.")
            both = st.toggle("Fit both preprocessing variants", value=True,
                             help="Saves <cat>__<backend>__raw and __illum, needed for 'Compare methods'.")
            pre_only = None if both else st.segmented_control("Preprocessing", ["off", "on"], default="off",
                                                               required=True)
        if st.button("Fit and save", type="primary", icon=":material/play_arrow:", disabled=not cats):
            base.features.backend, base.features.image_size = backend, int(size)
            base.detector.n_components = n_comp
            base.detector.calib_folds = int(folds)
            base.detector.multiscale = True
            variants = [False, True] if both else [pre_only == "on"]
            bar = st.progress(0.0, "Starting…")
            log, extractor, step, total = [], None, 0, len(cats) * len(variants)
            try:
                for cat in cats:
                    data = cached_category(root, None if rep.layout == "custom" else cat)
                    paths = select_train_paths(data.train_paths, n_train or None, base.seed)
                    imgs = [resize_square(load_image_rgb(p), int(size)) for p in paths]
                    for pre in variants:
                        bar.progress(step / total, f"Fitting {cat} (preprocessing {'on' if pre else 'off'})…")
                        cfg = base.copy()
                        cfg.preprocess.enabled = pre
                        t0 = time.perf_counter()
                        pipe = AnomalyPipeline(cfg, extractor=extractor)
                        extractor = pipe.extractor
                        info = pipe.fit(imgs)
                        name = model_name(data.name, pipe.config.features.backend, pre)
                        manager().save(pipe, name, {"category": data.name, "dataset": str(root),
                                                    "train_files": [Path(p).name for p in paths]})
                        log.append({"model": name, "normal images": info["n_train_images"],
                                    "PCA components": str(info["pca_components"]),
                                    "calibration": info["calibration_mode"], "seconds": round(time.perf_counter() - t0, 1)})
                        step += 1
                bar.progress(1.0, "Done")
                st.success(f"Saved {len(log)} model(s) to the models/ folder.", title="Fitting complete")
                st.dataframe(log, hide_index=True, alt="Fitted models")
            except BackendUnavailableError as exc:
                st.error(str(exc), title="Backend unavailable")
            except ValueError as exc:
                st.error(str(exc), title="Fitting failed")

# ------------------------------------------------------------------ saved models
with st.container(border=True):
    st.subheader("3 · Saved models", anchor=False)
    table = model_table()
    if table.empty:
        st.caption("No saved models yet.")
    else:
        st.dataframe(table, hide_index=True, alt="Saved models and their configuration")
    if st.button("Reload saved models from disk", icon=":material/refresh:"):
        st.cache_resource.clear()
        st.toast("Model cache cleared — models will be re-read from disk.")

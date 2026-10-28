import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.app_support import (cached_category, cached_validation, experiment_dirs, fmt, load_experiment, load_model,
                             manager, threshold_controls)
from src.config import PipelineConfig
from src.evaluation import metrics as M
from src.experiments.robustness import add_deltas
from src.experiments.runner import ExperimentRunner, summarize
from src.experiments.single_eval import evaluate_pipeline
from src.utils.paths import DEFAULT_CONFIG_PATH

METRIC_HELP = """
| Metric | Needs | Meaning |
|---|---|---|
| Image AUROC / AP | image labels | ranking quality of image scores, threshold-free |
| Image precision · recall · F1 · FPR | image labels + threshold | decision quality at the calibrated threshold |
| Pixel AUROC / AP | **ground-truth masks** | ranking quality of pixel scores |
| Pixel precision · recall · F1 | **ground-truth masks** + threshold | segmentation quality at the threshold |
| Oracle F1 | labels | best F1 over all thresholds using test labels — an upper bound, **not deployable** |

Accuracy is not shown: test sets are imbalanced, so always read metrics together with the class counts.
"""
NUM = {c: st.column_config.NumberColumn(format="%.3f") for c in
       ["image_auroc", "image_ap", "image_precision", "image_recall", "image_f1", "image_fpr", "image_f1_oracle",
        "pixel_auroc", "pixel_ap", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_f1_oracle",
        "threshold_value", "delta_image_auroc", "delta_pixel_auroc", "image_recall", "pixel_recall"]}

with st.expander("Which metrics need ground truth?", icon=":material/help:"):
    st.markdown(METRIC_HELP)

mode = st.segmented_control("View", ["Saved experiment results", "Evaluate a model now", "Run a new experiment"],
                            default="Saved experiment results", required=True, key="ev_mode")


def roc_pr_charts(labels, scores, title):
    if len(np.unique(labels)) < 2:
        st.info("ROC/PR curves need both normal and defective images.")
        return
    fpr, tpr = M.roc_points(labels, scores)
    p, r = M.pr_points(labels, scores)
    c1, c2 = st.columns(2)
    roc = alt.Chart(pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})).mark_line().encode(
        x="False positive rate", y="True positive rate").properties(title=f"ROC — {title}", height=300)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
    c1.altair_chart(roc + diag, alt=f"ROC curve, {title}")
    pr = alt.Chart(pd.DataFrame({"Recall": r, "Precision": p})).mark_line().encode(
        x="Recall", y=alt.Y("Precision", scale=alt.Scale(domain=[0, 1]))).properties(title=f"Precision–recall — {title}", height=300)
    c2.altair_chart(pr, alt=f"Precision-recall curve, {title}")


def grouped_bars(df, x, metric, title):
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x}:N", title=None), xOffset="variant:N", y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ")),
        color=alt.Color("variant:N", legend=alt.Legend(title="variant")),
        tooltip=["variant", "method", x, alt.Tooltip(metric, format=".3f")]).properties(title=title, height=300)
    st.altair_chart(chart, alt=title)


# ======================================================================= saved results
if mode == "Saved experiment results":
    dirs = experiment_dirs()
    if not dirs:
        st.info("No experiment results in `outputs/` yet. Run one under *Run a new experiment* or with "
                "`python scripts/run_experiments.py`.", title="Not run")
        st.stop()
    d = st.selectbox("Experiment folder", dirs, format_func=lambda p: p.name)
    res, size, meta = load_experiment(d)
    if "synthetic" in str(meta.get("dataset_root", "")).lower():
        st.warning("Synthetic smoke-test data: verifies the software, says nothing about real performance.",
                   icon=":material/science:")
    st.caption(f"Dataset `{meta.get('dataset_root')}` · seed {meta.get('seed')} · n_train {meta.get('n_train_requested') or 'all'}"
               f" · backend {res['feature_backend'].iloc[0]}")
    ident = res[res["transform"] == "identity"]
    dist = ident.drop_duplicates("category")[["category", "n_train", "n_calibration", "n_test_normal", "n_test_defective"]]
    st.markdown("**Class distribution of the test sets**")
    st.dataframe(dist, hide_index=True, alt="Number of training and test images per category")

    st.markdown("**Mean over categories**")
    summ = summarize(res).reset_index()
    st.dataframe(summ, hide_index=True, column_config=NUM | {"mean_inference_ms": st.column_config.NumberColumn(
        "ms / image", format="%.0f")}, alt="Mean metrics per variant")
    long = summ.melt(id_vars=["variant", "method"], value_vars=["image_auroc", "pixel_auroc", "image_f1", "pixel_f1"],
                     var_name="metric", value_name="value").dropna()
    if not long.empty:
        grouped_bars(long, "metric", "value", "Method comparison (mean over categories)")
    st.markdown("**Per category**")
    st.dataframe(ident[["category", "variant", "method", "image_auroc", "image_precision", "image_recall", "image_f1",
                        "image_fpr", "pixel_auroc", "pixel_f1", "threshold_value", "mean_inference_ms"]],
                 hide_index=True, column_config=NUM, alt="Metrics per category and variant")

    if res["transform"].nunique() > 1:
        st.subheader("Robustness to synthetic illumination changes", anchor=False)
        st.caption("Test images were perturbed in software after fitting on clean images. Synthetic perturbations "
                   "are not real lighting conditions.")
        dd = add_deltas(res)
        metric = st.segmented_control("Metric", ["image_auroc", "pixel_auroc", "image_f1", "image_fpr"],
                                      default="image_auroc", required=True, key="ev_rob_metric")
        agg = dd.groupby(["transform", "variant", "method"], sort=False)[metric].mean().reset_index()
        grouped_bars(agg, "transform", metric, f"{metric.replace('_', ' ')} under each transform")
        st.dataframe(dd.groupby(["transform", "variant"], sort=False)[["image_auroc", "pixel_auroc", "image_f1", "image_fpr",
                                                                        "delta_image_auroc", "delta_pixel_auroc"]].mean().reset_index(),
                     hide_index=True, column_config=NUM, alt="Robustness metrics and changes versus the clean images")

    if not size.empty:
        st.subheader("Small versus large defects", anchor=False)
        st.caption("Bins by ground-truth area: small < 1 %, medium 1–5 %, large ≥ 5 % of the image. Each bin is "
                   "evaluated against all normal test images.")
        s = size[size["transform"] == "identity"]
        g = s.groupby(["size_bin", "variant", "method"])[["image_recall", "pixel_recall", "image_auroc"]].mean().reset_index()
        n = s.groupby("size_bin")["n_defective"].sum() / max(s["variant"].nunique(), 1)
        st.caption("Defective images per bin: " + ", ".join(f"{b} {int(n.get(b, 0))}" for b in ["small", "medium", "large"]))
        sm = st.segmented_control("Metric", ["pixel_recall", "image_recall", "image_auroc"], default="pixel_recall",
                                  required=True, key="ev_size_metric")
        grouped_bars(g, "size_bin", sm, f"{sm.replace('_', ' ')} by defect size")

    pngs = sorted(d.glob("*.png"))
    if pngs:
        with st.expander(f"Saved figures ({len(pngs)})"):
            for p in pngs:
                st.image(str(p), caption=p.name, alt=p.stem.replace("_", " "))

# ======================================================================= evaluate one model
elif mode == "Evaluate a model now":
    names = manager().list_models()
    root = st.session_state.dataset_root
    rep = cached_validation(root)
    if not names or not rep.ok:
        st.info("Needs a saved model and a valid dataset (see *Dataset & training*).")
        st.stop()
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        name = c1.selectbox("Model", names)
        default_cat = name.split("__")[0]
        cats = list(rep.categories)
        cat = c2.selectbox("Test category", cats, index=cats.index(default_cat) if default_cat in cats else 0)
        multi = c3.toggle("Multi-scale", value=True)
        alpha = c3.slider("α", 0.0, 1.0, 0.5, 0.05, disabled=not multi)
        thr = threshold_controls("ev")
        go = st.button("Evaluate on the test split", type="primary", icon=":material/play_arrow:")
    if go:
        pipe = load_model(name)
        data = cached_category(root, None if rep.layout == "custom" else cat)
        with st.spinner(f"Scoring {len(data.test_samples)} test images…"):
            out = evaluate_pipeline(pipe, data, alpha if multi else 1.0, thr)
        m = out.metrics
        st.caption(f"{m['n_test']} test images: {m['image_n_normal']} normal / {m['image_n_anomalous']} defective "
                   f"(anomalous fraction {fmt(m['image_anomalous_fraction'], 2)}) · {m['threshold_description']}")
        with st.container(horizontal=True):
            for k in ["image_auroc", "image_precision", "image_recall", "image_f1", "image_fpr"]:
                st.metric(k.replace("_", " "), fmt(m[k]), border=True)
        if m["pixel_metrics_available"]:
            with st.container(horizontal=True):
                for k in ["pixel_auroc", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_f1_oracle"]:
                    st.metric(k.replace("_", " "), fmt(m[k]), border=True)
        else:
            st.info("Pixel-level metrics are not reported: ground-truth masks are missing for this dataset.")
        st.caption(f"Mean inference {out.mean_inference_ms:.0f} ms / image on this machine · "
                   f"model size {sum(f.stat().st_size for f in manager().path_for(name).iterdir()) / 1e6:.2f} MB on disk")
        roc_pr_charts(out.labels, out.scores, "image level")
        hist = pd.DataFrame({"score": out.scores, "label": np.where(out.labels == 1, "defective", "normal")})
        h = alt.Chart(hist).mark_bar(opacity=0.6).encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=30), title="image anomaly score"), y=alt.Y("count()", stack=None),
            color="label:N")
        rule = alt.Chart(pd.DataFrame({"T": [m["threshold"]]})).mark_rule(strokeDash=[4, 4]).encode(x="T:Q")
        st.altair_chart((h + rule).properties(title="Image scores and threshold", height=260),
                        alt="Histogram of image anomaly scores for normal and defective images with threshold")
        if out.size_rows:
            st.dataframe(pd.DataFrame(out.size_rows), hide_index=True, column_config=NUM, alt="Metrics by defect size")

# ======================================================================= new experiment
else:
    root = st.session_state.dataset_root
    rep = cached_validation(root)
    if not rep.ok:
        st.info("Select a valid dataset on *Dataset & training* first.")
        st.stop()
    st.caption("Runs the five variants A–E with identical data, features and evaluation; results are saved to "
               "`outputs/` and appear under *Saved experiment results*. This may take minutes on a CPU.")
    with st.form("ev_run"):
        cats = st.multiselect("Categories", list(rep.categories), default=list(rep.categories)[:1])
        n_train = st.number_input("Normal training images per category (0 = all)", 0, 1000, 20)
        name = st.text_input("Experiment name", "dashboard_experiment")
        submitted = st.form_submit_button("Run experiment", type="primary", icon=":material/play_arrow:")
    if submitted and cats:
        cfg = PipelineConfig.from_yaml(DEFAULT_CONFIG_PATH)
        bar = st.progress(0.0, "Starting…")
        runner = ExperimentRunner(cfg, n_train=n_train or None, experiment_name=name)
        result = runner.run(root, None if rep.layout == "custom" else cats,
                            progress=lambda msg, f: bar.progress(f, msg))
        bar.progress(1.0, "Done")
        from src.experiments.reporting import make_experiment_plots, write_summary_markdown
        make_experiment_plots(result.results, result.size_results, result.out_dir)
        write_summary_markdown(result.results, result.size_results, result.out_dir, f"Experiment {name}")
        st.success(f"Saved to `{result.out_dir}`", title="Experiment finished")
        st.dataframe(summarize(result.results).reset_index(), hide_index=True, column_config=NUM,
                     alt="Mean metrics per variant for the new experiment")

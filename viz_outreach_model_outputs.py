# viz_outreach_model_outputs.py
# Visualizations for the outputs produced by your training/eval script.
#
# Assumptions:
# - Your existing script produces these in-memory objects:
#     fold_aucs (list[float])
#     y_all (np.ndarray)
#     proba_all (np.ndarray)
#     pred_all (np.ndarray)  # optional; recomputed if missing
#     THRESHOLD (float)
#     results (pd.DataFrame) with columns:
#         - proba_win, pred_win
#         - viability_score_model, viability_bucket_model, viability_bucket_num
#         - viability_score_mean, viability_bucket_orig, viability_bucket_orig_num
#         - bucket_distance
#     imp (pd.DataFrame) index=feature, columns:
#         mean_coef, std_coef, mean_abs_coef, sign_consistency
#
# If you prefer file-based I/O instead, see the "EXPORT/IMPORT" notes at bottom.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

# -----------------------
# Small utilities
# -----------------------

def _safe_pred_from_proba(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)

def _confusion_rates(cm: np.ndarray):
    # cm = [[tn, fp],[fn, tp]]
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan  # recall
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan  # specificity
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    return {"TPR": tpr, "FPR": fpr, "TNR": tnr, "FNR": fnr}

def _maybe_int_series(s):
    # Helpful if bucket labels are categorical and you want stable ordering
    return s.astype("Int64") if pd.api.types.is_integer_dtype(s) or str(s.dtype) == "Int64" else s

import re

def pretty_feature_name(name: str) -> str:
    s = str(name)

    # common one-hot prefixes
    s = s.replace("state_usps_", "State: ")
    s = s.replace("region_", "Region: ")
    s = s.replace("office_level_clean_", "Office level: ")
    s = s.replace("office_type_", "Office type: ")
    s = s.replace("partisan_type_", "Partisan: ")
    s = s.replace("incumbency_status_", "Incumbency: ")
    s = s.replace("election_dow_", "Election DOW: ")

    # your engineered names
    s = s.replace("candidates - available seats", "Competitiveness (candidates − seats)")
    s = s.replace("days_between_outreach_and_election", "Days between outreach & election")
    s = s.replace("n_outreach_rows", "# outreach rows")

    # cleanup
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s
# -----------------------
# Plotting functions
# -----------------------

US_STATE_CODES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI',
    'MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT',
    'VT','VA','WA','WV','WI','WY','DC'
}
US_TERRITORY_CODES = {'AS','GU','MP','PR','VI','PW'}
US_CODES_ALL = US_STATE_CODES | US_TERRITORY_CODES

STATE_NAME_TO_USPS = {
    "ALABAMA":"AL","ALASKA":"AK","ARIZONA":"AZ","ARKANSAS":"AR","CALIFORNIA":"CA","COLORADO":"CO","CONNECTICUT":"CT",
    "DELAWARE":"DE","DISTRICT OF COLUMBIA":"DC","FLORIDA":"FL","GEORGIA":"GA","HAWAII":"HI","IDAHO":"ID","ILLINOIS":"IL",
    "INDIANA":"IN","IOWA":"IA","KANSAS":"KS","KENTUCKY":"KY","LOUISIANA":"LA","MAINE":"ME","MARYLAND":"MD",
    "MASSACHUSETTS":"MA","MICHIGAN":"MI","MINNESOTA":"MN","MISSISSIPPI":"MS","MISSOURI":"MO","MONTANA":"MT",
    "NEBRASKA":"NE","NEVADA":"NV","NEW HAMPSHIRE":"NH","NEW JERSEY":"NJ","NEW MEXICO":"NM","NEW YORK":"NY",
    "NORTH CAROLINA":"NC","NORTH DAKOTA":"ND","OHIO":"OH","OKLAHOMA":"OK","OREGON":"OR","PENNSYLVANIA":"PA",
    "RHODE ISLAND":"RI","SOUTH CAROLINA":"SC","SOUTH DAKOTA":"SD","TENNESSEE":"TN","TEXAS":"TX","UTAH":"UT",
    "VERMONT":"VT","VIRGINIA":"VA","WASHINGTON":"WA","WEST VIRGINIA":"WV","WISCONSIN":"WI","WYOMING":"WY",
    # Territories (optional)
    "AMERICAN SAMOA":"AS","GUAM":"GU","NORTHERN MARIANA ISLANDS":"MP","PUERTO RICO":"PR",
    "VIRGIN ISLANDS":"VI","U.S. VIRGIN ISLANDS":"VI","PALAU":"PW",
}

STATE_ALIASES = {
    "DELEWARE": "DE",
    "WASHINGTON DC": "DC",
    "WASHINGTON D.C.": "DC",
    "WASHINGTON D C": "DC",
    "DISTRICT OF COLUMBIA": "DC",
}
def plot_diverging_top_features(imp: pd.DataFrame, top_each=10, savepath=None):
    """
    One chart: top positive + top negative mean_coef.
    Sorted by absolute magnitude.
    """
    df = imp.copy()
    df = df[df["mean_coef"].notna()]

    pos = df.sort_values("mean_coef", ascending=False).head(top_each)
    neg = df.sort_values("mean_coef", ascending=True).head(top_each)
    d = pd.concat([neg, pos], axis=0)

    # sort by absolute magnitude (small to large, so largest at bottom for barh)
    d = d.reindex(d["mean_coef"].abs().sort_values().index)

    labels = [pretty_feature_name(i) for i in d.index]
    vals = d["mean_coef"].values

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(d))
    ax.barh(y, vals)
    ax.axvline(0, linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean coefficient (log-odds)")
    ax.set_title(f"Top drivers (mean coefficients): {top_each} negative + {top_each} positive")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig

def plot_odds_ratio_dotplot(imp: pd.DataFrame, top_n=20, savepath=None):
    """
    Plot odds ratios = exp(mean_coef) with a reference line at OR=1.
    Shows the biggest effects by mean_abs_coef.
    """
    df = imp.copy().dropna(subset=["mean_coef"])
    df = df.sort_values("mean_abs_coef", ascending=False).head(top_n).copy()

    df["odds_ratio"] = np.exp(df["mean_coef"])
    # optional CI-ish band using std of coef across folds
    if "std_coef" in df.columns:
        df["or_lo"] = np.exp(df["mean_coef"] - df["std_coef"])
        df["or_hi"] = np.exp(df["mean_coef"] + df["std_coef"])

    df = df.iloc[::-1]  # so top is at top in barh/dot plot
    y = np.arange(len(df))
    labels = [pretty_feature_name(i) for i in df.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axvline(1.0, linewidth=1, linestyle="--")

    ax.plot(df["odds_ratio"].values, y, marker="o", linestyle="None")

    if "or_lo" in df.columns and "or_hi" in df.columns:
        for i, (lo, hi) in enumerate(zip(df["or_lo"], df["or_hi"])):
            ax.hlines(i, lo, hi)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (log scale)  —  >1 increases odds of Win")
    ax.set_title(f"Top {top_n} effects as odds ratios (mean ± std across folds)")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def feature_domain(name: str) -> str:
    s = str(name)
    if s.startswith("state_usps_") or s.startswith("region_"):
        return "Geography"
    if s.startswith("office_level_clean_") or s.startswith("office_type_"):
        return "Office"
    if s.startswith("incumbency_status_") or "seats" in s or "opponents" in s:
        return "Race structure"
    if "election_" in s or "midterm" in s or "presidential" in s or "normal_election" in s:
        return "Election timing"
    if "outreach" in s:
        return "Outreach"
    return "Election Type"

def plot_grouped_small_multiples(imp: pd.DataFrame, per_group=10, savepath=None):
    """
    Small multiples by feature domain, all with the same x-axis scale.
    """
    df = imp.copy().dropna(subset=["mean_coef"])
    df["domain"] = [feature_domain(i) for i in df.index]
    df = df.sort_values("mean_abs_coef", ascending=False)

    domains = [d for d in ["Race structure","Office","Geography","Election timing","Outreach","Election Type"]
               if d in df["domain"].unique()]

    # Collect the specific rows that will be plotted (so scaling reflects what's on the figure)
    plotted = []
    for dom in domains:
        sub = df[df["domain"] == dom].head(per_group).copy()
        plotted.append(sub)
    plotted_df = pd.concat(plotted) if plotted else df.head(per_group)

    # Global symmetric limit (same for all panels)
    m = float(plotted_df["mean_coef"].abs().max())
    m = max(m, 1e-6)  # avoid zero-width axis
    pad = 0.10
    xlim = (-m * (1 + pad), m * (1 + pad))

    # Build subplots with shared x
    n = len(domains)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.0*n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, dom in zip(axes, domains):
        sub = df[df["domain"] == dom].head(per_group).copy()
        sub = sub.reindex(sub["mean_abs_coef"].sort_values().index)

        y = np.arange(len(sub))
        ax.barh(y, sub["mean_coef"].values)
        ax.axvline(0, linewidth=1)

        ax.set_yticks(y)
        ax.set_yticklabels([pretty_feature_name(i) for i in sub.index])
        ax.set_title(dom)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Mean coefficient")

    fig.suptitle("Top effects by feature domain (common x-axis scale)", y=1.01, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_top_features_stability(imp: pd.DataFrame, top_n=20, savepath=None):
    """
    Error bars: mean_coef ± std_coef, sorted by mean_abs_coef.
    Shows sign_consistency as text.
    """
    df = imp.copy().dropna(subset=["mean_coef"])
    df = df.sort_values("mean_abs_coef", ascending=False).head(top_n).copy()
    df = df.iloc[::-1]

    y = np.arange(len(df))
    labels = [pretty_feature_name(i) for i in df.index]
    means = df["mean_coef"].values
    stds = df["std_coef"].values if "std_coef" in df.columns else np.zeros_like(means)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(means, y, xerr=stds, fmt="o")
    ax.axvline(0, linewidth=1, linestyle="--")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean coefficient (± 1 std across folds)")
    ax.set_title(f"Top {top_n} features — effect size + stability")

    # annotate sign consistency if present
    if "sign_consistency" in df.columns:
        for i, sc in enumerate(df["sign_consistency"].values):
            ax.text(ax.get_xlim()[0] + 0.02*(ax.get_xlim()[1]-ax.get_xlim()[0]), i,
                    f"sign={sc:.2f}", va="center", fontsize=8, color="#444444")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig

def plot_structural_vs_state_effects(imp: pd.DataFrame, top_struct=12, top_states=8, savepath=None):
    """
    Two-panel: structural features vs state one-hots.
    """
    df = imp.copy().dropna(subset=["mean_coef"])

    states = df[df.index.to_series().astype(str).str.startswith("state_usps_")].copy()
    non_states = df[~df.index.to_series().astype(str).str.startswith("state_usps_")].copy()

    # structural: take biggest abs effects excluding states
    struct = non_states.sort_values("mean_abs_coef", ascending=False).head(top_struct).copy()
    struct = struct.reindex(struct["mean_abs_coef"].sort_values().index)

    # states: show top +/- by abs coef
    st = states.sort_values("mean_abs_coef", ascending=False).head(top_states).copy()
    st = st.reindex(st["mean_abs_coef"].sort_values().index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

    # left: structural
    ax = axes[0]
    y = np.arange(len(struct))
    ax.barh(y, struct["mean_coef"].values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_feature_name(i) for i in struct.index])
    ax.set_title("Structural drivers (non-state)")
    ax.set_xlabel("Mean coefficient")

    # right: states
    ax = axes[1]
    y = np.arange(len(st))
    ax.barh(y, st["mean_coef"].values)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty_feature_name(i) for i in st.index])
    ax.set_title("State effects (top by |coef|)")
    ax.set_xlabel("Mean coefficient")

    fig.suptitle("Separate structural effects from geography (reduces over-interpretation of states)", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def _norm_state(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    s = re.sub(r"\s+", " ", s)
    s = s.replace(".", "")
    return s.upper()

def add_state_usps(results: pd.DataFrame, state_col: str = "state") -> pd.DataFrame:
    """
    Adds results['state_usps'] mapped from results[state_col].
    Keeps only US states/DC (drops territories for the map by default).
    """
    out = results.copy()
    if "state_usps" in out.columns:
        return out

    s = out[state_col].map(_norm_state) if state_col in out.columns else pd.Series([""] * len(out), index=out.index)

    def _map_one(v: str):
        if v == "" or v in {"NULL", "N/A", "NA", "UNDEFINED"}:
            return np.nan
        if v in STATE_ALIASES:
            v = STATE_ALIASES[v]
        # already code
        if len(v) == 2 and v.isalpha():
            return v if v in US_STATE_CODES else np.nan  # keep states/DC only for map
        # full name
        if v in STATE_NAME_TO_USPS:
            code = STATE_NAME_TO_USPS[v]
            return code if code in US_STATE_CODES else np.nan
        return np.nan

    out["state_usps"] = s.map(_map_one)
    return out

def state_win_rate_table(results: pd.DataFrame, win_col: str = "Win") -> pd.DataFrame:
    """
    Returns per-state counts and win rate (%), using results['state_usps'].
    """
    df = results[["state_usps", win_col]].copy()
    df[win_col] = pd.to_numeric(df[win_col], errors="coerce")
    df = df.dropna(subset=["state_usps"])

    tbl = (
        df.groupby("state_usps", dropna=False)[win_col]
          .agg(n_rows="count", win_rate="mean")
          .reset_index()
    )
    tbl["win_pct"] = 100.0 * tbl["win_rate"]
    return tbl.sort_values("n_rows", ascending=False)

def plot_state_win_rate_map(results: pd.DataFrame, outdir: Path, min_n: int = 50):
    """
    Saves:
      - state_win_rate_map.html (always)
      - state_win_rate_map.png (if kaleido is installed)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import plotly.express as px
    except ImportError:
        print("[viz] plotly not installed; skipping state map.")
        return None

    if "state_usps" not in results.columns:
        raise ValueError("results must contain 'state_usps' (call add_state_usps first).")
    if "Win" not in results.columns:
        raise ValueError("results must contain 'Win' (0/1).")

    tbl = state_win_rate_table(results, win_col="Win")
    tbl = tbl[tbl["n_rows"] >= min_n].copy()

    fig = px.choropleth(
        tbl,
        locations="state_usps",
        locationmode="USA-states",
        color="win_pct",
        scope="usa",
        hover_data={"state_usps": True, "n_rows": True, "win_pct": ":.1f"},
        labels={"win_pct": "Win rate (%)"},
        title=f"Observed Win Rate by State (n ≥ {min_n})"
    )

    # save HTML (works everywhere)
    html_path = outdir / "state_win_rate_map.html"
    fig.write_html(str(html_path))

    # try to save PNG (requires kaleido)
    png_path = outdir / "state_win_rate_map.png"
    try:
        fig.write_image(str(png_path), scale=2)
    except Exception:
        print("[viz] PNG export requires kaleido. HTML saved at:", html_path)

    return fig


def plot_score_hist_by_class(y_true, proba, threshold, savepath=None):
    fig = plt.figure()
    proba = np.asarray(proba)
    y_true = np.asarray(y_true)

    plt.hist(proba[y_true == 0], bins=40, alpha=0.7, label="Win=0")
    plt.hist(proba[y_true == 1], bins=40, alpha=0.7, label="Win=1")
    plt.axvline(threshold, linestyle="--", label=f"threshold={threshold}")
    plt.xlabel("Predicted P(Win=1)")
    plt.ylabel("Count")
    plt.title("Predicted probability distribution by true class")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_confusion_matrix(y_true, proba, threshold, normalize=False, savepath=None):
    fig = plt.figure()
    pred = _safe_pred_from_proba(proba, threshold)
    cm = confusion_matrix(y_true, pred)
    cm_display = cm.astype(float)

    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm_display, row_sums, out=np.zeros_like(cm_display), where=row_sums != 0)

    plt.imshow(cm_display, aspect="auto")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))

    # annotate
    for (i, j), v in np.ndenumerate(cm_display):
        txt = f"{v:.2f}" if normalize else f"{int(cm[i, j])}"
        plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)

    return fig, cm, _confusion_rates(cm)

def plot_threshold_sweep(y_true, proba, savepath=None):
    # plots precision/recall/f1 and TPR/FPR across thresholds
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s, tprs, fprs = [], [], [], [], []

    from sklearn.metrics import precision_score, recall_score, f1_score

    for t in thresholds:
        pred = _safe_pred_from_proba(proba, t)
        precisions.append(precision_score(y_true, pred, zero_division=0))
        recalls.append(recall_score(y_true, pred, zero_division=0))
        f1s.append(f1_score(y_true, pred, zero_division=0))
        cm = confusion_matrix(y_true, pred)
        rates = _confusion_rates(cm)
        tprs.append(rates["TPR"])
        fprs.append(rates["FPR"])

    fig = plt.figure()
    plt.plot(thresholds, precisions, label="precision")
    plt.plot(thresholds, recalls, label="recall")
    plt.plot(thresholds, f1s, label="f1")
    plt.plot(thresholds, tprs, label="TPR (recall)", linestyle="--")
    plt.plot(thresholds, fprs, label="FPR", linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold sweep (pooled)")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_viability_bucket_distributions(results: pd.DataFrame, savepath=None):
    fig = plt.figure()

    order = ["No Chance", "Unlikely to Win", "Has a Chance", "Likely to Win", "Frontrunner"]

    orig = results["viability_bucket_orig"].astype(str).value_counts()
    model = results["viability_bucket_model"].astype(str).value_counts()

    orig = orig.reindex(order, fill_value=0)
    model = model.reindex(order, fill_value=0)

    x = np.arange(len(order))
    width = 0.4
    plt.bar(x - width/2, orig.values, width=width, label="Original")
    plt.bar(x + width/2, model.values, width=width, label="Model")
    plt.xticks(x, order, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Viability bucket distribution: Original vs Model")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_bucket_distance(results: pd.DataFrame, savepath=None):
    fig = plt.figure()
    dist = results["bucket_distance"].value_counts(dropna=False).sort_index()
    plt.bar(dist.index.astype(str), dist.values)
    plt.xlabel("Bucket distance (0=match)")
    plt.ylabel("Count")
    plt.title("Bucket distance distribution")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_bucket_crosstab_heatmap(results: pd.DataFrame, savepath=None):
    fig, ax = plt.subplots(figsize=(7.8, 5.8))

    ct = pd.crosstab(
        results["viability_bucket_orig"],
        results["viability_bucket_model"],
        dropna=False
    )

    # row-normalize (% within each original bucket)
    row_sums = ct.sum(axis=1).replace(0, np.nan)
    pct = ct.div(row_sums, axis=0)  # 0..1

    vmax = float(np.nanmax(pct.values)) if pct.size else 1.0
    im = ax.imshow(pct.values, aspect="auto", cmap="RdBu", vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(pct.shape[1]))
    ax.set_xticklabels(pct.columns.astype(str), rotation=30, ha="right")
    ax.set_yticks(np.arange(pct.shape[0]))
    ax.set_yticklabels(pct.index.astype(str))

    ax.set_xlabel("Model bucket")
    ax.set_ylabel("Original bucket")
    ax.set_title("Row-normalized crosstab (% of each original bucket)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share of original bucket")
    # optional: show colorbar tick labels as %
    ticks = np.linspace(0, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t*100:.0f}%" for t in ticks])

    # annotate with %
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            val = pct.iat[i, j]
            label = "" if pd.isna(val) else f"{val*100:.1f}%"
            ax.text(j, i, label, ha="center", va="center", color="black")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
    return fig

def plot_top_features(imp: pd.DataFrame, top_n=25, which="mean_abs_coef", savepath=None):
    # which in {"mean_abs_coef","mean_coef","std_coef","sign_consistency"}
    fig = plt.figure()

    s = imp[which].sort_values(ascending=False).head(top_n)
    # horizontal bar is easier to read
    plt.barh(np.arange(len(s)), s.values)
    plt.yticks(np.arange(len(s)), s.index)
    plt.gca().invert_yaxis()
    plt.xlabel(which)
    plt.title(f"Top {top_n} features by {which}")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

def plot_top_pos_neg_features(imp: pd.DataFrame, top_n=20, savepath_pos=None, savepath_neg=None):
    # two plots: most positive / most negative mean_coef
    fig1 = plt.figure()
    pos = imp["mean_coef"].sort_values(ascending=False).head(top_n)
    plt.barh(np.arange(len(pos)), pos.values)
    plt.yticks(np.arange(len(pos)), pos.index)
    plt.gca().invert_yaxis()
    plt.xlabel("mean_coef")
    plt.title(f"Top {top_n} positive mean coefficients (toward Win=1)")
    plt.tight_layout()
    if savepath_pos:
        plt.savefig(savepath_pos, dpi=200)

    fig2 = plt.figure()
    neg = imp["mean_coef"].sort_values(ascending=True).head(top_n)
    plt.barh(np.arange(len(neg)), neg.values)
    plt.yticks(np.arange(len(neg)), neg.index)
    plt.gca().invert_yaxis()
    plt.xlabel("mean_coef")
    plt.title(f"Top {top_n} negative mean coefficients (toward Win=0)")
    plt.tight_layout()
    if savepath_neg:
        plt.savefig(savepath_neg, dpi=200)

    return fig1, fig2
def plot_metrics_table(y_true: np.ndarray, proba: np.ndarray, threshold: float, outdir: Path) -> None:
    outdir = Path(outdir)            
    outdir.mkdir(parents=True, exist_ok=True)

    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else np.nan  # sensitivity
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan  # specificity
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    # Build a compact display table
    rows = [
        ["ROC-AUC", f"{auc:.3f}"],
        ["Avg Precision (PR-AUC)", f"{ap:.3f}"],
        ["Threshold", f"{threshold:.3f}"],
        ["TPR (Recall)", f"{tpr:.3f}"],
        ["FPR", f"{fpr:.3f}"],
        ["TNR (Specificity)", f"{tnr:.3f}"],
        ["FNR", f"{fnr:.3f}"],
        ["TP", f"{tp:d}"],
        ["FP", f"{fp:d}"],
        ["TN", f"{tn:d}"],
        ["FN", f"{fn:d}"],
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axis("off")
    ax.set_title("Pooled Metrics Summary", pad=12)

    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )

    # “Pretty” formatting
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.35)

    # header style
    for j in range(2):
        cell = tbl[0, j]
        cell.set_text_props(weight="bold")
        cell.set_height(cell.get_height() * 1.15)

    # subtle row striping (keeps default matplotlib colors)
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in range(2):
                tbl[i, j].set_alpha(0.85)

    fig.tight_layout()
    fig.savefig(outdir / "metrics_table.png", dpi=200)
    plt.close(fig)

def bucket_summary_table(
    results: pd.DataFrame,
    bucket_col: str,
    win_col: str = "Win",
    proba_col: str = "proba_win",
) -> pd.DataFrame:
    """
    Per-bucket:
      - n_rows
      - observed win rate (%)
      - avg predicted probability (%)
    """
    df = results[[bucket_col, win_col, proba_col]].copy()
    df[win_col] = pd.to_numeric(df[win_col], errors="coerce")
    df[proba_col] = pd.to_numeric(df[proba_col], errors="coerce")

    out = (
        df.groupby(bucket_col, dropna=False)
          .agg(
              n_rows=(win_col, "count"),
              win_rate=(win_col, "mean"),
              avg_proba=(proba_col, "mean"),
          )
          .reset_index()
    )
    out["win_pct"] = 100.0 * out["win_rate"]
    out["avg_proba_pct"] = 100.0 * out["avg_proba"]
    return out.sort_values(bucket_col)

def plot_bucket_calibration(results: pd.DataFrame, bucket_col: str, savepath=None):
    fig = plt.figure()
    tbl = bucket_summary_table(results, bucket_col)

    x = np.arange(len(tbl))
    w = 0.4
    plt.bar(x - w/2, tbl["win_pct"].values, width=w, label="Observed win %")
    plt.bar(x + w/2, tbl["avg_proba_pct"].values, width=w, label="Avg predicted %")
    plt.xticks(x, tbl[bucket_col].astype(str).values, rotation=30, ha="right")
    plt.ylabel("Percent")
    plt.title(f"Bucket calibration: {bucket_col}")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    return fig

# -----------------------
# Main entry point
# -----------------------

def make_all_plots(
    fold_aucs,
    y_all,
    proba_all,
    threshold,
    results: pd.DataFrame,
    imp: pd.DataFrame,
    outdir="viz_outputs",
):
    import os
    os.makedirs(outdir, exist_ok=True)

    # Ensure arrays
    y_all = np.asarray(y_all)
    proba_all = np.asarray(proba_all)


    # 3) Score distributions
    plot_score_hist_by_class(y_all, proba_all, threshold, savepath=f"{outdir}/score_hist_by_class.png")

    # 4) Confusion matrices
    _, cm, rates = plot_confusion_matrix(y_all, proba_all, threshold, normalize=False, savepath=f"{outdir}/confusion_matrix.png")
    plot_confusion_matrix(y_all, proba_all, threshold, normalize=True, savepath=f"{outdir}/confusion_matrix_normalized.png")

    # Save a small text summary for CM rates
    with open(f"{outdir}/confusion_rates.txt", "w") as f:
        f.write("Confusion matrix (tn fp / fn tp):\n")
        f.write(str(cm) + "\n\n")
        f.write("Rates:\n")
        for k, v in rates.items():
            f.write(f"{k}: {v:.4f}\n")

    # 5) Threshold sweep
    plot_threshold_sweep(y_all, proba_all, savepath=f"{outdir}/threshold_sweep.png")

    # 6) Viability bucket comparisons
    required_cols = {"viability_bucket_orig", "viability_bucket_model", "bucket_distance"}
    if required_cols.issubset(set(results.columns)):
        plot_viability_bucket_distributions(results, savepath=f"{outdir}/bucket_distribution.png")
        plot_bucket_crosstab_heatmap(results, savepath=f"{outdir}/bucket_crosstab_heatmap.png")
        plot_bucket_distance(results, savepath=f"{outdir}/bucket_distance.png")

    # 7) Feature importance plots
    if imp is not None and len(imp) > 0:
        plot_top_features(imp, top_n=25, which="mean_abs_coef", savepath=f"{outdir}/top_features_mean_abs_coef.png")
        plot_top_features(imp, top_n=25, which="sign_consistency", savepath=f"{outdir}/top_features_sign_consistency.png")
        plot_top_pos_neg_features(
            imp,
            top_n=20,
            savepath_pos=f"{outdir}/top_positive_mean_coef.png",
            savepath_neg=f"{outdir}/top_negative_mean_coef.png"
        )
    
    # 8) Metrics table
    plot_metrics_table(y_all, proba_all, threshold, outdir)

    if "Win" in results.columns and "proba_win" in results.columns:
        if "viability_bucket_model" in results.columns:
            tbl_model = bucket_summary_table(results, "viability_bucket_model")
            tbl_model.to_csv(f"{outdir}/bucket_summary_model.csv", index=False)

        if "viability_bucket_orig" in results.columns:
            tbl_orig = bucket_summary_table(results, "viability_bucket_orig")
            tbl_orig.to_csv(f"{outdir}/bucket_summary_orig.csv", index=False)
    
    plot_bucket_calibration(results, "viability_bucket_model", savepath=f"{outdir}/bucket_calibration_model.png")
    
        # 9) Win rate by state map
    if "state" in results.columns and "Win" in results.columns:
        results_with_state = add_state_usps(results, state_col="state")
        plot_state_win_rate_map(results_with_state, Path(outdir), min_n=50)


        # NEW: better coefficient visuals
    plot_diverging_top_features(imp, top_each=10, savepath=f"{outdir}/coef_diverging_top10_each.png")
    plot_odds_ratio_dotplot(imp, top_n=20, savepath=f"{outdir}/coef_odds_ratio_top20.png")
    plot_grouped_small_multiples(imp, per_group=10, savepath=f"{outdir}/coef_small_multiples_by_domain.png")
    plot_top_features_stability(imp, top_n=20, savepath=f"{outdir}/coef_stability_top20.png")
    plot_structural_vs_state_effects(imp, top_struct=12, top_states=8, savepath=f"{outdir}/coef_structural_vs_states.png")
    # Close figures to avoid memory accumulation in notebooks
    plt.close("all")


if __name__ == "__main__":
    # -----------------------
    # HOW TO USE
    # -----------------------
    # Option 1 (recommended): import this file from your training script AFTER it finishes,
    # then call make_all_plots(fold_aucs, y_all, proba_all, THRESHOLD, results, imp).
    #
    # Example at the very bottom of your training file:
    #   from viz_outreach_model_outputs import make_all_plots
    #   make_all_plots(fold_aucs, y_all, proba_all, THRESHOLD, results, imp, outdir="viz_outputs")
    #
    # Option 2: If you export artifacts to disk (results.csv, imp.csv, arrays),
    # you can load them here and run. Uncomment below and point to your files.

    # ---- EXPORT/IMPORT option (uncomment if you want standalone) ----
    # import os
    # ARTIFACT_DIR = "artifacts"
    # fold_aucs = np.load(f"{ARTIFACT_DIR}/fold_aucs.npy").tolist()
    # y_all = np.load(f"{ARTIFACT_DIR}/y_all.npy")
    # proba_all = np.load(f"{ARTIFACT_DIR}/proba_all.npy")
    # THRESHOLD = float(open(f"{ARTIFACT_DIR}/threshold.txt").read().strip())
    # results = pd.read_csv(f"{ARTIFACT_DIR}/results.csv")
    # imp = pd.read_csv(f"{ARTIFACT_DIR}/imp.csv", index_col=0)
    #
    # make_all_plots(fold_aucs, y_all, proba_all, THRESHOLD, results, imp, outdir="viz_outputs")

    pass

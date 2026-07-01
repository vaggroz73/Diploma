# ================================================================
# Football Player Performance Predictor — Streamlit App
# Διπλωματική Εργασία: ML στο Ποδόσφαιρο
# Εκτέλεση: streamlit run app.py
# ================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Σταθερές ────────────────────────────────────────────────────
C_LSTM  = "#0984e3"
C_GRU   = "#a29bfe"
C_ACT   = "#00b894"
C_RED   = "#d63031"
C_GOLD  = "#f9ca24"

st.set_page_config(
    page_title="Football ML Predictor",
    layout="wide",
)

st.markdown("""
    <style>
    [data-testid="stStatusWidget"] { display: none; }
    </style>
""", unsafe_allow_html=True)

# ================================================================
# HELPERS
# ================================================================

def interpret(v):
    if v > 1.5:    return "Σημαντική βελτίωση"
    elif v > 0.3:  return " Μικρή βελτίωση"
    elif v >= -0.3:return "Σταθερή απόδοση"
    elif v >= -1.5:return " Μικρή πτώση"
    else:          return "Σημαντική πτώση"

def load_csv(f):
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    return df

def calc_metrics(y, p):
    return {
        "MAE":      mean_absolute_error(y, p),
        "RMSE":     np.sqrt(mean_squared_error(y, p)),
        "R²":       r2_score(y, p),
        "Dir.Acc":  np.mean((p > 0) == (y > 0)),
    }

def mpl_defaults():
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#444",
        "text.color":       "white",
        "axes.labelcolor":  "#b2bec3",
        "xtick.color":      "#b2bec3",
        "ytick.color":      "#b2bec3",
        "grid.color":       "#333",
        "grid.alpha":       0.5,
    })

# ================================================================
# SESSION STATE — αποθήκευση DataFrames μεταξύ re-runs
# ================================================================
for key in ["lstm_df", "gru_df", "fi_lstm", "fi_gru", "names", "selected_pid"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ================================================================
# SIDEBAR — φόρτωση αρχείων
# ================================================================
with st.sidebar:
    st.title("Football ML")
    st.caption("LSTM vs GRU · Διπλωματική")
    st.divider()

    st.subheader("Φόρτωση αρχείων")

    f = st.file_uploader("LSTM Predictions CSV", type="csv", key="up_lstm")
    if f:
        df = load_csv(f)
        if "lstm_pred" not in df.columns and "predicted" in df.columns:
            df = df.rename(columns={"predicted": "lstm_pred"})
        df["err_lstm"] = np.abs(df["y_true"] - df["lstm_pred"])
        st.session_state.lstm_df = df
        st.success(f"LSTM: {len(df)} παίκτες")

    f = st.file_uploader("GRU Predictions CSV", type="csv", key="up_gru")
    if f:
        df = load_csv(f)
        if "gru_pred" not in df.columns and "predicted" in df.columns:
            df = df.rename(columns={"predicted": "gru_pred"})
        df["err_gru"] = np.abs(df["y_true"] - df["gru_pred"])
        st.session_state.gru_df = df
        st.success(f"GRU: {len(df)} παίκτες")

    f = st.file_uploader("Feature Importance LSTM CSV", type="csv", key="up_fi_lstm")
    if f:
        st.session_state.fi_lstm = load_csv(f)
        st.success("Feature Imp. LSTM ✓")

    f = st.file_uploader("Feature Importance GRU CSV", type="csv", key="up_fi_gru")
    if f:
        st.session_state.fi_gru = load_csv(f)
        st.success("Feature Imp. GRU ✓")

    f = st.file_uploader("Dataset Excel (για ονόματα)", type=["xlsx","xls"], key="up_xls")
    if f:
        xdf = pd.read_excel(f)
        if "id" in xdf.columns and "Player" in xdf.columns:
            st.session_state.names = (xdf[["id","Player"]]
                                      .drop_duplicates("id")
                                      .set_index("id")["Player"]
                                      .to_dict())
            st.success("Ονόματα παικτών ✓")

# ── Shorthand refs ───────────────────────────────────────────────
lstm_df = st.session_state.lstm_df
gru_df  = st.session_state.gru_df
fi_lstm = st.session_state.fi_lstm
fi_gru  = st.session_state.fi_gru
names   = st.session_state.names or {}

# ================================================================
# MAIN
# ================================================================
st.title("Football Player Performance Predictor")
st.caption("LSTM vs GRU Διπλωματική Εργασία")

if lstm_df is None and gru_df is None:
    st.info("Φορτώστε τουλάχιστον ένα CSV από την πλαϊνή μπάρα για να ξεκινήσετε.")
    st.stop()

mpl_defaults()

# Tabs
tab_player, tab_wfall, tab_compare, tab_feat = st.tabs([
    "Παίκτης",
    "Waterfall",
    "LSTM vs GRU",
    "Feature Importance",
])

# ================================================================
# TAB 1 — ΠΑΙΚΤΗΣ
# ================================================================
with tab_player:
    src = lstm_df if lstm_df is not None else gru_df
    all_ids = src["id"].unique().tolist()

    # Επιλογή παίκτη
    def fmt_id(pid):
        n = names.get(pid, "")
        return f"{pid} — {n}" if n else str(pid)

    sel_label = st.selectbox("Επιλέξτε παίκτη", options=all_ids,
                             format_func=fmt_id, key="selected_pid")
    pid = sel_label

    # Ανάκτηση δεδομένων παίκτη
    lstm_row = (lstm_df[lstm_df["id"] == pid].iloc[-1]
                if lstm_df is not None and pid in lstm_df["id"].values else None)
    gru_row  = (gru_df[gru_df["id"] == pid].iloc[-1]
                if gru_df  is not None and pid in gru_df["id"].values  else None)

    if lstm_row is None and gru_row is None:
        st.warning("Δεν βρέθηκαν δεδομένα για αυτόν τον παίκτη.")
        st.stop()

    y_true    = (lstm_row["y_true"] if lstm_row is not None else gru_row["y_true"])
    lstm_pred = lstm_row["lstm_pred"] if lstm_row is not None else None
    gru_pred  = gru_row["gru_pred"]   if gru_row  is not None else None
    lstm_err  = lstm_row["err_lstm"]   if lstm_row is not None else None
    gru_err   = gru_row["err_gru"]     if gru_row  is not None else None
    pname     = names.get(pid, f"ID {pid}")

    # ── Metrics cards ─────────────────────────────────────────────
    st.subheader(f"{pname}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual Δoverall",   f"{y_true:+.3f}")
    c2.metric("LSTM Prediction",   f"{lstm_pred:+.3f}" if lstm_pred is not None else "—",
              delta=f"err {lstm_err:.3f}" if lstm_err is not None else None)
    c3.metric("GRU Prediction",    f"{gru_pred:+.3f}"  if gru_pred  is not None else "—",
              delta=f"err {gru_err:.3f}"  if gru_err  is not None else None)
    ref_pred = lstm_pred if lstm_pred is not None else gru_pred
    c4.metric("Ερμηνεία", interpret(ref_pred))

    st.divider()

    # ── Γραφήματα ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Γράφημα 1: bar actual / lstm / gru
    ax = axes[0]
    labels, vals, colors = ["Actual"], [y_true], [C_ACT]
    if lstm_pred is not None:
        labels += ["LSTM"]; vals += [lstm_pred]; colors += [C_LSTM]
    if gru_pred is not None:
        labels += ["GRU"];  vals += [gru_pred];  colors += [C_GRU]
    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="none")
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_title("Actual vs Predicted Δoverall", pad=8)
    ax.set_ylabel("Δoverall")
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + (0.03 if v >= 0 else -0.11),
                f"{v:+.3f}", ha="center", color="white", fontsize=9, fontweight="bold")

    # Γράφημα 2: error distribution + player marker
    ax2 = axes[1]
    if lstm_df is not None:
        ax2.hist(lstm_df["err_lstm"], bins=30, color=C_LSTM, alpha=0.55, label="LSTM test set")
    if gru_df is not None:
        ax2.hist(gru_df["err_gru"],   bins=30, color=C_GRU,  alpha=0.55, label="GRU test set")
    if lstm_err is not None:
        ax2.axvline(lstm_err, color=C_LSTM, lw=2, ls="--", label=f"Παίκτης LSTM {lstm_err:.3f}")
    if gru_err is not None:
        ax2.axvline(gru_err,  color=C_GRU,  lw=2, ls=":",  label=f"Παίκτης GRU  {gru_err:.3f}")
    ax2.set_title("Κατανομή Απόλυτων Σφαλμάτων", pad=8)
    ax2.set_xlabel("Absolute Error")
    ax2.spines[:].set_visible(False)
    ax2.grid(axis="y", alpha=0.4)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ================================================================
# TAB 2 — WATERFALL
# ================================================================
with tab_wfall:
    has_lstm_fi = fi_lstm is not None and lstm_df is not None
    has_gru_fi  = fi_gru  is not None and gru_df  is not None

    if not has_lstm_fi and not has_gru_fi:
        st.info("Φορτώστε Feature Importance CSV (LSTM ή/και GRU) για να εμφανιστεί το Waterfall.")
        st.stop()

    # Χρήση του ίδιου παίκτη που επιλέχθηκε στο Tab 1
    src2    = lstm_df if lstm_df is not None else gru_df
    all_ids2 = src2["id"].unique().tolist()

    if st.session_state.selected_pid in all_ids2:
        pid2 = st.session_state.selected_pid
    else:
        pid2 = all_ids2[0]

    st.caption(f"Παίκτης: {fmt_id(pid2)}  (επιλέγεται από την καρτέλα «Παίκτης»)")

    lr2 = (lstm_df[lstm_df["id"]==pid2].iloc[-1]
           if lstm_df is not None and pid2 in lstm_df["id"].values else None)
    gr2 = (gru_df[gru_df["id"]==pid2].iloc[-1]
           if gru_df  is not None and pid2 in gru_df["id"].values  else None)
    y2  = lr2["y_true"] if lr2 is not None else gr2["y_true"]

    n_plots = (1 if has_lstm_fi else 0) + (1 if has_gru_fi else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    idx = 0
    for fi_df, pred_val, model, col in [
        (fi_lstm, lr2["lstm_pred"] if lr2 is not None else None, "LSTM", C_LSTM),
        (fi_gru,  gr2["gru_pred"]  if gr2 is not None else None, "GRU",  C_GRU),
    ]:
        if fi_df is None or pred_val is None:
            continue
        ax = axes[idx]; idx += 1

        feat_col = "feature"     if "feature"    in fi_df.columns else fi_df.columns[0]
        imp_col  = "importance"  if "importance" in fi_df.columns else fi_df.columns[1]
        top      = fi_df.nlargest(12, imp_col).reset_index(drop=True)
        src_df   = lstm_df if model == "LSTM" else gru_df
        baseline = src_df["y_true"].mean()
        diff     = pred_val - baseline
        total    = top[imp_col].sum()

        # Signed SHAP-style contributions:
        # Direction = sign((player_value - feature_mean) * corr_with_y_true)
        # A feature pushes prediction UP if the player is above average on a
        # positively-correlated feature, and DOWN otherwise.
        if total:
            weights = (top[imp_col] / total).values
            signs   = np.ones(len(top))
            for i, fname in enumerate(top[feat_col]):
                if fname in src_df.columns:
                    player_rows = src_df.loc[src_df["id"] == pid2, fname]
                    if player_rows.empty:
                        continue
                    feat_mean  = src_df[fname].mean()
                    feat_std   = src_df[fname].std()
                    deviation  = (player_rows.iloc[-1] - feat_mean) / (feat_std if feat_std else 1)
                    corr       = src_df[[fname, "y_true"]].corr().iloc[0, 1]
                    if not np.isnan(corr) and not np.isnan(deviation):
                        signs[i] = np.sign(deviation * corr) if deviation * corr != 0 else 1
            raw   = weights * abs(diff) * signs
            c_sum = raw.sum()
            # Rescale so contributions sum exactly to (pred - baseline)
            contribs = raw / c_sum * diff if c_sum != 0 else np.zeros(len(top))
        else:
            contribs = np.zeros(len(top))

        # Ταξινόμηση: αρνητικές συνεισφορές πρώτα, μετά θετικές
        sort_idx   = np.argsort(contribs)
        contribs   = contribs[sort_idx]
        feat_names = np.array(top[feat_col].tolist())[sort_idx].tolist()

        feats = feat_names + ["Prediction"]
        starts, heights, bcolors = [], [], []
        run = baseline
        for c in contribs:
            starts.append(run); heights.append(c)
            bcolors.append(C_ACT if c >= 0 else C_RED)
            run += c
        starts.append(0); heights.append(pred_val); bcolors.append(col)

        yp = np.arange(len(feats))
        ax.barh(yp, heights, left=starts, color=bcolors, height=0.55, edgecolor="none")
        for i in range(len(contribs) - 1):
            end = starts[i] + heights[i]
            ax.plot([end, end], [yp[i]-0.45, yp[i+1]+0.45], color="#555", lw=0.8, ls="--")
        ax.axvline(baseline, color=C_GOLD,  lw=1.2, ls=":", label=f"Baseline {baseline:.2f}")
        ax.axvline(pred_val, color=col,      lw=1.5, ls="--", label=f"Prediction {pred_val:.3f}")
        for i, (s, h) in enumerate(zip(starts, heights)):
            sign = "+" if h >= 0 else ""
            ax.text(s+h + (0.005 if h>=0 else -0.005), yp[i],
                    f"{sign}{h:.2f}", va="center",
                    ha="left" if h>=0 else "right", color="white", fontsize=7)
        ax.set_yticks(yp); ax.set_yticklabels(feats, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Waterfall — {model}\nActual: {y2:+.3f}  Pred: {pred_val:+.3f}", pad=8)
        ax.set_xlabel("Συμβολή στη Δoverall")
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ================================================================
# TAB 3 — LSTM vs GRU
# ================================================================
with tab_compare:
    if lstm_df is None or gru_df is None:
        st.info("Φορτώστε και τα δύο CSV (LSTM + GRU) για σύγκριση.")
        st.stop()

    merged = lstm_df[["id","y_true","lstm_pred","err_lstm"]].merge(
             gru_df[["id","gru_pred","err_gru"]], on="id", how="inner")

    if merged.empty:
        st.warning("Δεν υπάρχουν κοινοί παίκτες.")
        st.stop()

    m_lstm = calc_metrics(merged["y_true"], merged["lstm_pred"])
    m_gru  = calc_metrics(merged["y_true"], merged["gru_pred"])

    # Metrics row
    st.subheader(f"Κοινοί παίκτες: {len(merged)}")
    cols = st.columns(4)
    for i, key in enumerate(["MAE","RMSE","R²","Dir.Acc"]):
        delta = m_gru[key] - m_lstm[key]
        better = "GRU" if (m_gru[key] < m_lstm[key] and key in ["MAE","RMSE"]) or \
                          (m_gru[key] > m_lstm[key] and key in ["R²","Dir.Acc"]) else "LSTM"
        fmt = ".1%" if key == "Dir.Acc" else ".4f"
        cols[i].metric(key,
                       f"LSTM {m_lstm[key]:{fmt}}",
                       f"GRU {m_gru[key]:{fmt}}  ({'↑' if better=='GRU' else '↓'})")

    st.divider()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor("none")

    # (0,0) Scatter actual vs pred
    ax = axes[0,0]
    ax.scatter(merged["y_true"], merged["lstm_pred"], s=14, alpha=0.6,
               color=C_LSTM, label="LSTM", edgecolors="none")
    ax.scatter(merged["y_true"], merged["gru_pred"],  s=14, alpha=0.6,
               color=C_GRU,  label="GRU",  edgecolors="none", marker="^")
    lim = [min(merged["y_true"].min(), merged[["lstm_pred","gru_pred"]].min().min()) - 0.3,
           max(merged["y_true"].max(), merged[["lstm_pred","gru_pred"]].max().max()) + 0.3]
    ax.plot(lim, lim, "--", color="#888", lw=0.9)
    ax.set_title("Actual vs Predicted"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.spines[:].set_visible(False); ax.grid(alpha=0.4); ax.legend(fontsize=8)

    # (0,1) Per-player error scatter
    ax = axes[0,1]
    better_gru = merged["err_gru"] < merged["err_lstm"]
    ax.scatter(merged.loc[~better_gru,"err_lstm"], merged.loc[~better_gru,"err_gru"],
               s=14, alpha=0.65, color=C_LSTM, label=f"LSTM καλύτερο ({(~better_gru).sum()})")
    ax.scatter(merged.loc[better_gru,"err_lstm"],  merged.loc[better_gru,"err_gru"],
               s=14, alpha=0.65, color=C_GRU,  label=f"GRU καλύτερο ({better_gru.sum()})",
               marker="^")
    mx = max(merged["err_lstm"].max(), merged["err_gru"].max()) + 0.1
    ax.plot([0,mx],[0,mx],"--",color="#888",lw=0.9)
    ax.set_title("Per-Player Error"); ax.set_xlabel("LSTM err"); ax.set_ylabel("GRU err")
    ax.spines[:].set_visible(False); ax.grid(alpha=0.4); ax.legend(fontsize=8)

    # (1,0) Metrics bar
    ax = axes[1,0]
    keys = ["MAE","RMSE","Dir.Acc"]
    x = np.arange(len(keys)); w = 0.32
    ax.bar(x-w/2, [m_lstm[k] for k in keys], w, color=C_LSTM, label="LSTM", edgecolor="none")
    ax.bar(x+w/2, [m_gru[k]  for k in keys], w, color=C_GRU,  label="GRU",  edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_title("Μετρικές Σύγκρισης")
    ax.spines[:].set_visible(False); ax.grid(axis="y", alpha=0.4); ax.legend(fontsize=8)
    for bar, v in zip(ax.patches, [m_lstm[k] for k in keys] + [m_gru[k] for k in keys]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", color="white", fontsize=7)

    # (1,1) Error diff histogram
    ax = axes[1,1]
    diff = merged["err_lstm"] - merged["err_gru"]
    ax.hist(diff[diff < 0],  bins=20, color=C_LSTM, alpha=0.75, label=f"LSTM καλύτερο ({(diff<0).sum()})")
    ax.hist(diff[diff >= 0], bins=20, color=C_GRU,  alpha=0.75, label=f"GRU καλύτερο ({(diff>=0).sum()})")
    ax.axvline(0, color="#888", lw=1, ls="--")
    ax.axvline(diff.mean(), color=C_GOLD, lw=1.5, ls=":", label=f"Mean {diff.mean():+.3f}")
    ax.set_title("Διαφορά Σφάλματος (LSTM − GRU)")
    ax.set_xlabel("LSTM_err − GRU_err")
    ax.spines[:].set_visible(False); ax.grid(axis="y", alpha=0.4); ax.legend(fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    winner = "GRU" if m_gru["MAE"] < m_lstm["MAE"] else "LSTM"
    st.success(f"**Νικητής βάσει MAE: {winner}** — LSTM {m_lstm['MAE']:.4f}  |  GRU {m_gru['MAE']:.4f}")

# ================================================================
# TAB 4 — GLOBAL FEATURE IMPORTANCE
# ================================================================
with tab_feat:
    if fi_lstm is None and fi_gru is None:
        st.info("Φορτώστε τουλάχιστον ένα Feature Importance CSV (LSTM ή GRU).")
        st.stop()

    # ── Slider για top-N features ────────────────────────────────
    top_n = st.slider("Αριθμός top features", min_value=5, max_value=30, value=15, step=1)
    st.divider()

    def get_top(fi_df, n):
        feat_col = "feature"    if "feature"    in fi_df.columns else fi_df.columns[0]
        imp_col  = "importance" if "importance" in fi_df.columns else fi_df.columns[1]
        top = fi_df.nlargest(n, imp_col)[[feat_col, imp_col]].reset_index(drop=True)
        top.columns = ["feature", "importance"]
        # Normalise to % of total
        top["pct"] = top["importance"] / fi_df[imp_col].sum() * 100
        return top

    both = fi_lstm is not None and fi_gru is not None
    n_cols = 2 if both else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6.5 * n_cols, max(top_n * 0.38, 4.5)))

    if not both:
        axes = [axes]

    plot_data = []
    if fi_lstm is not None:
        plot_data.append((fi_lstm, "LSTM", C_LSTM))
    if fi_gru is not None:
        plot_data.append((fi_gru,  "GRU",  C_GRU))

    for ax, (fi_df, model, color) in zip(axes, plot_data):
        top = get_top(fi_df, top_n)
        bars = ax.barh(top["feature"], top["pct"], color=color,
                       edgecolor="none", height=0.65, alpha=0.88)
        # Value labels
        for bar, pct in zip(bars, top["pct"]):
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", color="white", fontsize=8)
        ax.set_title(f"Global Feature Importance — {model}\n"
                     f"Top {top_n} features  (% of total importance)",
                     pad=10)
        ax.set_xlabel("Σχετική Σημασία (%)")
        ax.invert_yaxis()
        ax.spines[:].set_visible(False)
        ax.grid(axis="x", alpha=0.4)
        ax.tick_params(labelsize=8)

    fig.tight_layout(pad=2.5)
    st.pyplot(fig)
    plt.close(fig)

    # ── Side-by-side comparison table (αν και τα δύο φορτωθούν) ─
    if both:
        st.divider()
        st.subheader("Σύγκριση Feature Importance LSTM vs GRU")

        top_l = get_top(fi_lstm, top_n).rename(columns={"pct": "LSTM %"})
        top_g = get_top(fi_gru,  top_n).rename(columns={"pct": "GRU %"})
        merged_fi = top_l[["feature","LSTM %"]].merge(
                    top_g[["feature","GRU %"]], on="feature", how="outer").fillna(0)
        merged_fi = merged_fi.sort_values("LSTM %", ascending=False).reset_index(drop=True)
        merged_fi["Διαφορά (LSTM−GRU)"] = merged_fi["LSTM %"] - merged_fi["GRU %"]
        merged_fi["LSTM %"]             = merged_fi["LSTM %"].map("{:.2f}%".format)
        merged_fi["GRU %"]              = merged_fi["GRU %"].map("{:.2f}%".format)
        merged_fi["Διαφορά (LSTM−GRU)"] = merged_fi["Διαφορά (LSTM−GRU)"].map("{:+.2f}%".format)
        st.dataframe(merged_fi, use_container_width=True, hide_index=True)
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================

DEFAULT_PRICE = "normal"  # change as needed


# ============================================================
# Load solutions (local folder)
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))  # .../NestedBenders/Solutions_LP

def load_solutions_lp(price_setting=DEFAULT_PRICE, filename=None):
    """
    Loads: <HERE>/<price_setting>_solutions.npy
    Expected saved structure (new):
      sol["Rolling → SDDiP"], sol["2-SP → SDDiP"], sol["3-SP → SDDiP"]
      sol["PSDDiP"]["approx"] and sol["PSDDiP"]["exact"]
      sol["meta"]["K_list"]
    """
    if filename is None:
        filename = f"{price_setting}_solutions.npy"
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing solution file: {path}")
    return np.load(path, allow_pickle=True).item()


# ============================================================
# Helpers: mode routing (approx/exact) for PSDDiP
# ============================================================

def _mode_key(approx_mode: bool) -> str:
    return "approx" if approx_mode else "exact"

def _get_psddip_block(sol: dict, approx_mode: bool) -> dict:
    psd = sol.get("PSDDiP", {})
    mk = _mode_key(approx_mode)
    if mk not in psd:
        raise KeyError(
            f"Missing PSDDiP['{mk}'] in saved file. "
            f"Available PSDDiP keys: {list(psd.keys())}"
        )
    return psd[mk]


# ============================================================
# Helpers: means and label parsing
# ============================================================

def _mean_curve(mat2d):
    """mat2d: (K_eval, T) or (K_eval, T+1) -> mean over eval scenarios."""
    arr = np.asarray(mat2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr.mean(axis=0)

def _parse_psddip_label(label: str):
    """
    Returns K (int) if label is PSDDiP(K=...), else None.
    Accepts "PSDDiP (K=10)" or "PSDDiP(K=10)".
    """
    s = label.replace(" ", "")
    if s.startswith("PSDDiP(K=") and s.endswith(")"):
        return int(s[len("PSDDiP(K="):-1])
    return None

def _normalize_eval_label(label: str) -> str:
    """
    Normalize common variants to saved dict keys.
    """
    s = label.strip().lower().replace(" ", "")
    s = s.replace("->", "→")
    if "rolling" in s:
        return "Rolling → SDDiP"
    if s.startswith("2sp") or s.startswith("2-sp") or "2sp" in s:
        return "2-SP → SDDiP"
    if s.startswith("3sp") or s.startswith("3-sp") or "3sp" in s:
        return "3-SP → SDDiP"
    return label


# ============================================================
# Plot: single curve helpers (per algorithm)
# ============================================================

def plot_q_da_single(name, q_da, xlabel="Hour", ylabel="q_DA",
                     ylim=(-1000, 32000), figsize=(7, 4), pause=False):
    y = np.asarray(q_da, dtype=float).reshape(-1)
    x = np.arange(len(y))

    plt.figure(figsize=figsize)
    plt.plot(x, y, marker="o", linewidth=1.2)
    plt.title(name, fontsize=11)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if pause:
        input("Press Enter for next plot...")


def plot_q_ID_density_single(
    name,
    q_ID_matrix,
    xlabel="Hour",
    ylabel="q_ID",
    ylim=None,
    figsize=(8, 4),
    alpha=0.25,
    pause=False,
    overlay_mean=False
):
    q_ID_arr = np.asarray(q_ID_matrix, dtype=float)
    if q_ID_arr.ndim != 2:
        raise ValueError(f"{name}: q_ID must be 2D (K_eval, T). Got {q_ID_arr.shape}")

    K_eval, T = q_ID_arr.shape
    hours = np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)
    for k in range(K_eval):
        ax.plot(hours, q_ID_arr[k, :], color="black", alpha=alpha)

    if overlay_mean:
        mean_line = q_ID_arr.mean(axis=0)
        ax.plot(hours, mean_line, linewidth=2.0, label="mean")
        ax.legend()

    ax.set_title(name, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if pause:
        input("Press Enter for next plot...")


def plot_S_density_single(
    name,
    S_matrix,
    S_cap,
    xlabel="Hour",
    ylabel="S",
    ylim=None,
    figsize=(8, 4),
    alpha=0.25,
    pause=False,
    overlay_mean=False
):
    S_arr = np.asarray(S_matrix, dtype=float)
    if S_arr.ndim != 2:
        raise ValueError(f"{name}: S must be 2D (K_eval, T+1). Got {S_arr.shape}")

    K_eval, Tp1 = S_arr.shape
    hours = np.arange(Tp1)

    fig, ax = plt.subplots(figsize=figsize)
    for k in range(K_eval):
        ax.plot(hours, S_arr[k, :], color="black", alpha=alpha)

    if overlay_mean:
        mean_line = S_arr.mean(axis=0)
        ax.plot(hours, mean_line, linewidth=2.0, label="mean")
        ax.legend()

    ax.set_title(name, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylim is None:
        ax.set_ylim(0, S_cap)
    else:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if pause:
        input("Press Enter for next plot...")


# ============================================================
# Plot: ALL algorithms in ONE figure (overlay)
#   - q_DA: overlay directly
#   - q_ID: overlay MEAN curves (recommended)
#   - S:    overlay MEAN curves (recommended)
# ============================================================

def plot_overlay_q_da(sol, price_setting, approx_mode=True, K_pick=30,
                      ylim=(-1000, 32000), figsize=(9, 4.5)):
    mode_tag = _mode_key(approx_mode)
    x = np.arange(24)  # assumes T=24 for q_DA

    plt.figure(figsize=figsize)

    # baselines
    for nm in ["Rolling → SDDiP", "2-SP → SDDiP", "3-SP → SDDiP"]:
        y = np.asarray(sol[nm]["q_da"], dtype=float).reshape(-1)
        plt.plot(x, y, linewidth=2, label=nm)

    # PSDDiP
    psd = _get_psddip_block(sol, approx_mode)
    K_list = list(sol["meta"]["K_list"])

    if K_pick is None:
        for k_idx, K in enumerate(K_list):
            y = np.asarray(psd["q_da"][k_idx], dtype=float).reshape(-1)
            plt.plot(x, y, linewidth=2, label=f"PSDDiP-{mode_tag} (K={K})")
    else:
        if K_pick not in K_list:
            raise ValueError(f"K_pick={K_pick} not in saved K_list={K_list}")
        k_idx = K_list.index(K_pick)
        y = np.asarray(psd["q_da"][k_idx], dtype=float).reshape(-1)
        plt.plot(x, y, linewidth=2, label=f"PSDDiP-{mode_tag} (K={K_pick})")

    plt.title(f"q_DA overlay ({price_setting}, {mode_tag})")
    plt.xlabel("Hour")
    plt.ylabel("q_DA")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlay_q_ID_mean(sol, price_setting, approx_mode=True, K_pick=30,
                           ylim=None, figsize=(9, 4.5)):
    mode_tag = _mode_key(approx_mode)
    x = np.arange(24)  # assumes T=24

    plt.figure(figsize=figsize)

    # baselines
    for nm in ["Rolling → SDDiP", "2-SP → SDDiP", "3-SP → SDDiP"]:
        y = _mean_curve(sol[nm]["q_ID"])
        plt.plot(x, y, linewidth=2, label=f"{nm} (mean)")

    # PSDDiP
    psd = _get_psddip_block(sol, approx_mode)
    K_list = list(sol["meta"]["K_list"])

    if K_pick is None:
        for k_idx, K in enumerate(K_list):
            y = _mean_curve(psd["q_ID"][k_idx])
            plt.plot(x, y, linewidth=2, label=f"PSDDiP-{mode_tag} (K={K}) mean")
    else:
        if K_pick not in K_list:
            raise ValueError(f"K_pick={K_pick} not in saved K_list={K_list}")
        k_idx = K_list.index(K_pick)
        y = _mean_curve(psd["q_ID"][k_idx])
        plt.plot(x, y, linewidth=2, label=f"PSDDiP-{mode_tag} (K={K_pick}) mean")

    plt.title(f"q_ID mean overlay ({price_setting}, {mode_tag})")
    plt.xlabel("Hour")
    plt.ylabel("q_ID (mean over eval scenarios)")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlay_S_mean(sol, price_setting, approx_mode=True, K_pick=30,
                        S_cap=21022.1, ylim=None, figsize=(9, 4.5)):
    mode_tag = _mode_key(approx_mode)

    # infer length T+1 from baseline
    S_arr0 = np.asarray(sol["Rolling → SDDiP"]["S"], dtype=float)
    if S_arr0.ndim != 2:
        raise ValueError(f"Baseline Rolling → SDDiP S must be 2D; got {S_arr0.shape}")
    Tp1 = S_arr0.shape[1]
    x = np.arange(Tp1)

    plt.figure(figsize=figsize)

    # baselines
    for nm in ["Rolling → SDDiP", "2-SP → SDDiP", "3-SP → SDDiP"]:
        y = _mean_curve(sol[nm]["S"])
        plt.plot(x, y, linewidth=2, label=f"{nm} (mean)")

    # PSDDiP
    psd = _get_psddip_block(sol, approx_mode)
    K_list = list(sol["meta"]["K_list"])

    if K_pick is None:
        for k_idx, K in enumerate(K_list):
            y = _mean_curve(psd["S"][k_idx])
            plt.plot(x, y, linewidth=2, label=f"PSDDiP-{mode_tag} (K={K}) mean")
    else:
        if K_pick not in K_list:
            raise ValueError(f"K_pick={K_pick} not in saved K_list={K_list}")
        k_idx = K_list.index(K_pick)
        y = _mean_curve(psd["S"][k_idx])
        plt.plot(x, y, linewidth=2, label=f"PSDDiP-{mode_tag} (K={K_pick}) mean")

    plt.title(f"SoC S mean overlay ({price_setting}, {mode_tag})")
    plt.xlabel("Hour")
    plt.ylabel("S (mean over eval scenarios)")

    if ylim is not None:
        plt.ylim(*ylim)
    else:
        plt.ylim(0, S_cap)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlays_all(price_setting="sunny", approx_mode=True, K_pick=30,
                      da_ylim=(-1000, 32000), qid_ylim=None,
                      S_cap=21022.1, S_ylim=None):
    """
    What you asked:
      - q_DA at once (all algorithms overlaid)
      - q_ID at once (mean curve over eval scenarios, overlaid)
      - SoC S at once (mean curve over eval scenarios, overlaid)
    """
    sol = load_solutions_lp(price_setting=price_setting)
    plot_overlay_q_da(sol, price_setting, approx_mode=approx_mode, K_pick=K_pick, ylim=da_ylim)
    plot_overlay_q_ID_mean(sol, price_setting, approx_mode=approx_mode, K_pick=K_pick, ylim=qid_ylim)
    plot_overlay_S_mean(sol, price_setting, approx_mode=approx_mode, K_pick=K_pick, S_cap=S_cap, ylim=S_ylim)


# ============================================================
# Profit curve plots (selected labels) with approx_mode
# ============================================================

def _get_series_from_label(sol, label, field, approx_mode: bool):
    """
    field: "f_P" or "f_Im"
    label:
      - baselines: "Rolling → SDDiP", "2-SP → SDDiP", "3-SP → SDDiP"
      - psddip:    "PSDDiP (K=10)"
    returns: 1D mean series length T
    """
    label_clean = label.replace(" ", "")

    # Baselines
    if label in sol:
        data = sol[label].get(field, None)
        if data is None:
            raise KeyError(f"{label} has no field '{field}'.")
        arr = np.asarray(data, dtype=float)
        return _mean_curve(arr) if arr.ndim == 2 else arr.reshape(-1)

    # PSDDiP by K (mode)
    if label_clean.startswith("PSDDiP(K=") and label_clean.endswith(")"):
        K = int(label_clean[len("PSDDiP(K="):-1])
        K_list = sol["meta"]["K_list"]
        if K not in K_list:
            raise ValueError(f"K={K} not found. Available K_list: {K_list}")
        k_idx = K_list.index(K)

        psd = _get_psddip_block(sol, approx_mode)
        data_list = psd.get(field, None)
        if data_list is None:
            raise KeyError(f"PSDDiP[{_mode_key(approx_mode)}] has no field '{field}'.")
        mat = data_list[k_idx]
        return _mean_curve(mat)

    raise ValueError(f"Unknown label '{label}'.")


def plot_profit_mean_selected(
    price_setting="normal",
    approx_mode=True,
    field="f_P",
    labels=None,
    title=None,
    xlabel="Hour",
    ylabel=None,
    ylim=None,
    figsize=(9, 4.5)
):
    if labels is None or len(labels) == 0:
        raise ValueError("labels must be non-empty.")

    sol = load_solutions_lp(price_setting=price_setting)

    curves = []
    for lab in labels:
        y = _get_series_from_label(sol, lab, field=field, approx_mode=approx_mode)
        curves.append((lab, np.asarray(y).reshape(-1)))

    T = len(curves[0][1])
    x = np.arange(T)

    plt.figure(figsize=figsize)
    for lab, y in curves:
        if len(y) != T:
            raise ValueError(f"Length mismatch for '{lab}': got {len(y)}, expected {T}")
        plt.plot(x, y, linewidth=2, label=lab)

    mode_tag = _mode_key(approx_mode)
    if title is None:
        title = f"{field} mean ({price_setting}, {mode_tag})"
    if ylabel is None:
        ylabel = field

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Evaluation vs K plot with approx_mode
# ============================================================

def plot_evaluation_vs_K(
    price_setting="normal",
    approx_mode=True,
    baselines=None,
    approx_ks=None,
    title=None,
    xlabel="K",
    ylabel="Evaluation",
    figsize=(8, 4.5),
    show_points=True,
    marker="o",
    linewidth=1.8,
    legend=True
):
    sol = load_solutions_lp(price_setting=price_setting)
    mode_tag = _mode_key(approx_mode)

    psd = _get_psddip_block(sol, approx_mode)
    K_list_all = list(sol["meta"]["K_list"])
    eval_p_all = list(psd["eval"])

    if approx_ks is None:
        K_show = K_list_all
    else:
        K_show = list(approx_ks)

    eval_show = []
    for K in K_show:
        if K not in K_list_all:
            raise ValueError(f"K={K} not found in saved K_list={K_list_all}")
        idx = K_list_all.index(K)
        eval_show.append(eval_p_all[idx])

    x_pos = np.arange(len(K_show))
    plt.figure(figsize=figsize)

    baseline_colors = ["red", "blue", "green"]
    if baselines is None:
        baselines = []

    for i, b in enumerate(baselines[:3]):
        key = _normalize_eval_label(b)
        if key not in sol:
            raise KeyError(f"Baseline '{b}' -> '{key}' not found.")
        yb = sol[key].get("eval", None)
        if yb is None:
            raise KeyError(f"Baseline '{key}' has no 'eval'.")
        plt.axhline(y=yb, linewidth=2.2, color=baseline_colors[i], label=f"{key}")

    plt.plot(
        x_pos,
        eval_show,
        color="black",
        linewidth=linewidth,
        marker=(marker if show_points else None),
        label=f"PSDDiP-{mode_tag}"
    )

    if title is None:
        title = f"Evaluation vs K ({price_setting}, {mode_tag})"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(x_pos, [str(k) for k in K_show])
    all_vals = list(eval_show)
    for b in baselines:
        all_vals.append(sol[_normalize_eval_label(b)]["eval"])
    y_min, y_max = min(all_vals), max(all_vals)
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    plt.ylim(y_min - pad, y_max + pad)

    plt.grid(True, alpha=0.3)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Printing: evaluation values with approx_mode
# ============================================================

def get_eval_values(sol, baselines, approx_mode=True, approx_ks=None):
    baseline_eval = []
    for b in (baselines or []):
        key = _normalize_eval_label(b)
        if key not in sol:
            raise KeyError(f"Baseline '{b}' -> '{key}' not found.")
        baseline_eval.append((key, float(sol[key]["eval"])))

    psd = _get_psddip_block(sol, approx_mode)
    K_all = list(sol["meta"]["K_list"])
    eval_all = list(psd["eval"])

    if approx_ks is None:
        K_show = K_all
    else:
        K_show = list(approx_ks)

    psddip_eval = []
    for K in K_show:
        if K not in K_all:
            raise ValueError(f"K={K} not found. Available: {K_all}")
        idx = K_all.index(K)
        psddip_eval.append((K, float(eval_all[idx])))

    return baseline_eval, psddip_eval


def print_eval_table(price_setting, baselines, approx_mode=True, approx_ks=None):
    sol = load_solutions_lp(price_setting=price_setting)
    mode_tag = _mode_key(approx_mode)
    baseline_eval, psddip_eval = get_eval_values(sol, baselines, approx_mode=approx_mode, approx_ks=approx_ks)

    print("\n==============================")
    print(f"Evaluation values | price={price_setting} | mode={mode_tag}")
    print("==============================")

    if baseline_eval:
        print("\n[Baselines]")
        for name, val in baseline_eval:
            print(f"{name:18s} : {val:.6f}")

    print(f"\n[PSDDiP-{mode_tag}]")
    for K, val in psddip_eval:
        print(f"K={K:<3d} : {val:.6f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    price_setting = "cloudy"     # change as needed
    approx_mode = False          # True=approx, False=exact
    K_pick = 50                 # PSDDiP curve to compare. Use None to plot ALL K curves.

    # ----------------------------
    # 1) Overlay plots you asked for
    #    q_DA at once, q_ID at once, SoC at once
    # ----------------------------
    plot_overlays_all(
        price_setting=price_setting,
        approx_mode=approx_mode,
        K_pick=K_pick,
        da_ylim=(-1000, 32000),
        qid_ylim=None,
        S_cap=21022.1,
        S_ylim=None
    )

    # ----------------------------
    # 2) Profit mean curve comparisons (selected labels)
    # ----------------------------
    SELECTED = [
        "2-SP → SDDiP",
        "3-SP → SDDiP",
        f"PSDDiP (K={K_pick})",
    ]

    plot_profit_mean_selected(
        price_setting=price_setting,
        approx_mode=approx_mode,
        field="f_P",
        labels=SELECTED,
        title=f"Mean f_P comparison ({price_setting}, {_mode_key(approx_mode)})"
    )

    plot_profit_mean_selected(
        price_setting=price_setting,
        approx_mode=approx_mode,
        field="f_Im",
        labels=SELECTED,
        title=f"Mean f_Im comparison ({price_setting}, {_mode_key(approx_mode)})"
    )

    # ----------------------------
    # 3) Evaluation vs K
    # ----------------------------
    plot_evaluation_vs_K(
        price_setting=price_setting,
        approx_mode=approx_mode,
        baselines=["Rolling → SDDiP", "2SP → SDDiP", "3SP → SDDiP"],
        approx_ks=None
    )

    # ----------------------------
    # 4) Print evaluation table
    # ----------------------------
    print_eval_table(
        price_setting=price_setting,
        approx_mode=approx_mode,
        baselines=["Rolling → SDDiP", "2SP → SDDiP", "3SP → SDDiP"],
        approx_ks=None
    )
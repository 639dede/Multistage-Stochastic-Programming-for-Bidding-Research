import os
import numpy as np
import matplotlib.pyplot as plt


K_list = [1, 3, 6, 10, 15, 30]

# ----------------------------
# Load solutions (local folder)
# ----------------------------

HERE = os.path.dirname(os.path.abspath(__file__))  # .../NestedBenders/Solutions_LP
DEFAULT_PRICE = "cloudy"  # change as needed

def load_solutions_lp(price_setting=DEFAULT_PRICE, filename=None):
    if filename is None:
        filename = f"{price_setting}_solutions.npy"
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing solution file: {path}")
    return np.load(path, allow_pickle=True).item()

# 1. Plot helpers

# ----------------------------
# Plot helpers (DA / q_ID / S)
# ----------------------------

def plot_q_da_single(name, q_da, xlabel="Hour", ylabel="Value",
                     ylim=(-1000, 32000), figsize=(7, 4), pause=False):
    y = np.asarray(q_da).reshape(-1)
    x = np.arange(len(y))

    plt.figure(figsize=figsize)
    plt.plot(x, y, marker="o", linewidth=1.2)
    plt.title(name, fontsize=11)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
    q_ID_arr = np.asarray(q_ID_matrix)
    if q_ID_arr.ndim != 2:
        raise ValueError(f"{name}: q_ID must be 2D (K_eval, T). Got shape {q_ID_arr.shape}")

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
    S_arr = np.asarray(S_matrix)
    if S_arr.ndim != 2:
        raise ValueError(f"{name}: S must be 2D (K_eval, T+1). Got shape {S_arr.shape}")

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

    # default ylim: [0, S_cap] if not provided
    if ylim is None:
        ax.set_ylim(0, S_cap)
    else:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    if pause:
        input("Press Enter for next plot...")


# ----------------------------
# Plot helpers (mean-profit curves)
# ----------------------------

def mean_over_scenarios(mat_2d):
    """
    mat_2d: shape (K_eval, T)
    returns: shape (T,)
    """
    arr = np.asarray(mat_2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (K_eval, T), got shape {arr.shape}")
    return arr.mean(axis=0)

def _get_series_from_label(sol, label, field):
    """
    field: "f_P" or "f_Im"
    label:
      - "Rolling → SDDiP"
      - "2-SP → SDDiP"
      - "3-SP → SDDiP"
      - "PSDDiP (K=10)" or "PSDDiP(K=10)"
    returns: 1D mean series length T
    """
    label_clean = label.replace(" ", "")

    # Baselines
    if label in sol:
        data = sol[label].get(field, None)
        if data is None:
            raise KeyError(f"{label} has no field '{field}' in saved file.")
        data_arr = np.asarray(data)
        if data_arr.ndim == 2:
            return mean_over_scenarios(data_arr)
        return data_arr.reshape(-1)

    # PSDDiP by K
    if label_clean.startswith("PSDDiP(K=") and label_clean.endswith(")"):
        K_str = label_clean[len("PSDDiP(K="):-1]
        K = int(K_str)

        K_list = sol["meta"]["K_list"]
        if K not in K_list:
            raise ValueError(f"K={K} not found. Available K_list: {K_list}")

        k_idx = K_list.index(K)
        data = sol["PSDDiP"].get(field, None)
        if data is None:
            raise KeyError(f"PSDDiP has no field '{field}' in saved file.")

        mat = data[k_idx]  # (K_eval, T)
        return mean_over_scenarios(mat)

    raise ValueError(
        f"Unknown label '{label}'. Use baselines like 'Rolling → SDDiP', '2-SP → SDDiP', "
        f"or 'PSDDiP (K=...)'."
    )

def plot_profit_mean_selected(
    price_setting="normal",
    field="f_P",                    # "f_P" or "f_Im"
    labels=None,                    # list of curve labels
    title=None,
    xlabel="Hour",
    ylabel=None,
    ylim=None,
    figsize=(9, 4.5)
):
    if labels is None or len(labels) == 0:
        raise ValueError("labels must be a non-empty list.")

    sol = load_solutions_lp(price_setting=price_setting)

    curves = []
    for lab in labels:
        y = _get_series_from_label(sol, lab, field=field)
        curves.append((lab, np.asarray(y).reshape(-1)))

    T = len(curves[0][1])
    x = np.arange(T)

    plt.figure(figsize=figsize)
    for lab, y in curves:
        if len(y) != T:
            raise ValueError(f"Length mismatch for '{lab}': got {len(y)}, expected {T}")
        plt.plot(x, y, linewidth=2, label=lab)

    if title is None:
        title = f"{field} mean over scenarios ({price_setting})"
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


# ----------------------------
# Main plotting routine
# ----------------------------

def plot_all_from_file(price_setting="normal", pause=False,
                       da_ylim=(-1000, 32000), id_ylim=None, overlay_mean=False):
    sol = load_solutions_lp(price_setting=price_setting)

    K_list = sol["meta"]["K_list"]

    # You must set SOC capacity here (or load it from elsewhere)
    # If your Evaluation_LP.py uses S=C etc., just copy the same scalar here:
    S_cap = 21022.1  # <-- CHANGE if needed (must match your model)

    # Baselines
    base_names = ["Rolling → SDDiP", "2-SP → SDDiP", "3-SP → SDDiP"]

    # ---- DA ----
    for nm in base_names:
        plot_q_da_single(nm, sol[nm]["q_da"], ylim=da_ylim, pause=pause)

    for k_idx, K in enumerate(K_list):
        plot_q_da_single(f"PSDDiP (K={K})", sol["PSDDiP"]["q_da"][k_idx], ylim=da_ylim, pause=pause)

    # ---- q_ID density ----
    for nm in base_names:
        plot_q_ID_density_single(nm, sol[nm]["q_ID"], ylim=id_ylim, pause=pause, overlay_mean=overlay_mean)

    for k_idx, K in enumerate(K_list):
        plot_q_ID_density_single(
            f"PSDDiP (K={K})",
            sol["PSDDiP"]["q_ID"][k_idx],
            ylim=id_ylim,
            pause=pause,
            overlay_mean=overlay_mean
        )

    # ---- S density ----
    for nm in base_names:
        plot_S_density_single(nm, sol[nm]["S"], S_cap=S_cap, pause=pause, overlay_mean=overlay_mean)

    for k_idx, K in enumerate(K_list):
        plot_S_density_single(
            f"PSDDiP (K={K})",
            sol["PSDDiP"]["S"][k_idx],
            S_cap=S_cap,
            pause=pause,
            overlay_mean=overlay_mean
        )

def _normalize_eval_label(label: str) -> str:
    """
    Allow user inputs like:
      "2SP -> SDDiP" / "2-SP → SDDiP" / "2-SP -> SDDiP"
    and normalize to the exact dict keys:
      "Rolling → SDDiP", "2-SP → SDDiP", "3-SP → SDDiP"
    """
    s = label.strip().lower().replace(" ", "")
    s = s.replace("->", "→")  # unify arrow

    if "rolling" in s:
        return "Rolling → SDDiP"
    if s.startswith("2sp") or s.startswith("2-sp") or "2sp" in s:
        return "2-SP → SDDiP"
    if s.startswith("3sp") or s.startswith("3-sp") or "3sp" in s:
        return "3-SP → SDDiP"

    # if user already passed exact key
    return label

def plot_evaluation_vs_K(
    price_setting="normal",
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
    """
    1) Plot selected baseline evaluations as horizontal lines with fixed colors:
       - baseline #1: red
       - baseline #2: blue
       - baseline #3: green
    2) Plot PSDDiP evaluations vs K in BLACK.
    3) X-axis uses EVEN SPACING for chosen K ticks (categorical positions),
       while tick labels show actual K values (e.g., 1,3,6,10,15,30).
    4) Y-axis range fixed to [0, 8e6].

    baselines: list of baseline names, e.g. ["2SP -> SDDiP", "3SP -> SDDiP"]
    approx_ks: list of K's to include. If None, include all in file's K_list.
              If you want even spacing including 30, pass approx_ks=[1,3,6,10,15,30].
    """
    sol = load_solutions_lp(price_setting=price_setting)

    # --- PSDDiP curve data ---
    K_list_all = list(sol["meta"]["K_list"])
    eval_p_all = list(sol["PSDDiP"]["eval"])

    if len(K_list_all) != len(eval_p_all):
        raise ValueError(f"Mismatch: len(K_list)={len(K_list_all)} vs len(eval_p)={len(eval_p_all)}")

    # choose which K's to show
    if approx_ks is None:
        K_show = K_list_all
    else:
        K_show = list(approx_ks)

    # pull evals in the same order as K_show
    eval_show = []
    for K in K_show:
        if K not in K_list_all:
            raise ValueError(f"K={K} not found in saved K_list={K_list_all}")
        idx = K_list_all.index(K)
        eval_show.append(eval_p_all[idx])

    # EVEN-SPACED x positions (categorical)
    x_pos = np.arange(len(K_show))

    # --- Plot ---
    plt.figure(figsize=figsize)

    # Fixed baseline colors by input order
    baseline_colors = ["red", "blue", "green"]

    if baselines is None:
        baselines = []

    for i, b in enumerate(baselines[:3]):  # support up to 3 colored baselines
        key = _normalize_eval_label(b)
        if key not in sol:
            raise KeyError(
                f"Baseline '{b}' normalized to '{key}', but not found in file. "
                f"Available baselines: {[k for k in sol.keys() if '→' in k]}"
            )
        yb = sol[key].get("eval", None)
        if yb is None:
            raise KeyError(f"Baseline '{key}' exists but has no 'eval' field in saved file.")

        color = baseline_colors[i]
        plt.axhline(
            y=yb,
            linewidth=2.2,
            linestyle="-",
            color=color,
            label=f"{key} (eval={yb:.4g})"
        )

    # PSDDiP curve (BLACK)
    plt.plot(
        x_pos,
        eval_show,
        color="black",
        linewidth=linewidth,
        marker=(marker if show_points else None),
        label="PSDDiP"
    )

    # Title / labels
    if title is None:
        base_txt = ", ".join([_normalize_eval_label(b) for b in baselines]) if baselines else "None"
        title = f"Evaluation vs K ({price_setting}) | baselines: {base_txt}"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # X ticks: even spacing but show actual K values
    plt.xticks(x_pos, [str(k) for k in K_show])

    # Y range: 0 ~ 8e6
    all_vals = []

    # PSDDiP values
    all_vals.extend(eval_show)

    # Baseline values
    for b in baselines:
        key = _normalize_eval_label(b)
        all_vals.append(sol[key]["eval"])

    y_min = min(all_vals)
    y_max = max(all_vals)

    pad = 0.05 * (y_max - y_min)  # 5% padding
    plt.ylim(y_min - pad, y_max + pad)

    plt.grid(True, alpha=0.3)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


# 2. Value printing helpers

def _parse_psddip_label(label: str):
    """
    Returns K (int) if label is PSDDiP (K=...), else None.
    Accepts "PSDDiP (K=10)" or "PSDDiP(K=10)" with spaces.
    """
    s = label.replace(" ", "")
    if s.startswith("PSDDiP(K=") and s.endswith(")"):
        return int(s[len("PSDDiP(K="):-1])
    return None

def get_eval_values(sol, baselines, approx_ks=None):
    """
    Returns:
      baseline_eval: list of (baseline_key, eval_value)
      psddip_eval:   list of (K, eval_value)
    """
    # baselines
    baseline_eval = []
    for b in (baselines or []):
        key = _normalize_eval_label(b)
        if key not in sol:
            raise KeyError(f"Baseline '{b}' -> '{key}' not found in saved file.")
        baseline_eval.append((key, float(sol[key]["eval"])))

    # PSDDiP
    K_all = list(sol["meta"]["K_list"])
    eval_all = list(sol["PSDDiP"]["eval"])

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

def get_solution_means(sol, labels, fields=("q_ID", "S", "f_P", "f_Im")):
    """
    For each label, compute mean-over-scenarios curve for requested fields
    (only for 2D fields). For q_da (1D) just return directly if requested.

    Returns dict:
      out[label][field] = 1D numpy array
    """
    out = {}

    for label in labels:
        out[label] = {}
        K = _parse_psddip_label(label)

        # ----- Baselines -----
        if (K is None) and (label in sol):
            for field in fields:
                data = sol[label].get(field, None)
                if data is None:
                    continue
                arr = np.asarray(data, dtype=float)

                # q_ID, f_P, f_Im expected (K_eval, T), S expected (K_eval, T+1)
                if arr.ndim == 2:
                    out[label][field] = arr.mean(axis=0)
                else:
                    out[label][field] = arr.reshape(-1)

            # optional q_da too (1D)
            if "q_da" in sol[label]:
                out[label]["q_da"] = np.asarray(sol[label]["q_da"], dtype=float).reshape(-1)

            continue

        # ----- PSDDiP by K -----
        if K is not None:
            K_list = sol["meta"]["K_list"]
            if K not in K_list:
                raise ValueError(f"{label}: K={K} not found in file. Available: {K_list}")
            k_idx = K_list.index(K)

            for field in fields:
                data_list = sol["PSDDiP"].get(field, None)
                if data_list is None:
                    continue
                arr = np.asarray(data_list[k_idx], dtype=float)
                if arr.ndim == 2:
                    out[label][field] = arr.mean(axis=0)
                else:
                    out[label][field] = arr.reshape(-1)

            # optional q_da (1D)
            out[label]["q_da"] = np.asarray(sol["PSDDiP"]["q_da"][k_idx], dtype=float).reshape(-1)

            continue

        raise ValueError(f"Unknown label '{label}' (not baseline key and not PSDDiP(K=...)).")

    return out

def print_eval_table(price_setting, baselines, approx_ks=None):
    sol = load_solutions_lp(price_setting=price_setting)
    baseline_eval, psddip_eval = get_eval_values(sol, baselines, approx_ks=approx_ks)

    print("\n==============================")
    print(f"Evaluation values | price={price_setting}")
    print("==============================")

    if baseline_eval:
        print("\n[Baselines]")
        for name, val in baseline_eval:
            print(f"{name:18s} : {val:.6f}")

    print("\n[PSDDiP]")
    for K, val in psddip_eval:
        print(f"K={K:<3d} : {val:.6f}")

def print_solution_means(price_setting, labels, fields=("q_da","q_ID","S","f_P","f_Im"), decimals=4):
    sol = load_solutions_lp(price_setting=price_setting)
    means = get_solution_means(sol, labels, fields=tuple(f for f in fields if f != "q_da"))

    print("\n=======================================")
    print(f"Mean solution curves | price={price_setting}")
    print("=======================================")
    print("(Each curve printed as a Python list for copy/paste.)")

    for label in labels:
        print("\n---------------------------------------")
        print(f"[{label}]")

        # q_da (1D)
        if "q_da" in fields:
            if _parse_psddip_label(label) is None:
                q_da = np.asarray(sol[label]["q_da"], dtype=float).reshape(-1)
            else:
                K = _parse_psddip_label(label)
                k_idx = sol["meta"]["K_list"].index(K)
                q_da = np.asarray(sol["PSDDiP"]["q_da"][k_idx], dtype=float).reshape(-1)

            print("q_da =")
            print(np.round(q_da, decimals).tolist())

        # mean-over-scenarios fields
        for field in fields:
            if field == "q_da":
                continue
            if field in means[label]:
                print(f"\n{field} mean =")
                print(np.round(means[label][field], decimals).tolist())


if __name__ == "__main__":
    
    
    price_setting = "sunny"  # change as needed
    
    # ----------------------------
    # 1. Plotting solutions
    # ----------------------------
    
    plot_all_from_file(
        price_setting=price_setting,
        pause=False,
        da_ylim=(-1000, 32000),
        id_ylim=None,
        overlay_mean=False
    )

    SELECTED = [
        #"Rolling → SDDiP",
        "2-SP → SDDiP",
        "3-SP → SDDiP",
        #"PSDDiP (K=1)",
        "PSDDiP (K=30)",
    ]


    # ----------------------------
    # 2. Plotting profit mean curves
    # ----------------------------

    # 1) Mean f_P curves (power profit)
    plot_profit_mean_selected(
        price_setting=price_setting,
        field="f_P",
        labels=SELECTED,
        title=f"Mean f_P comparison ({price_setting})"
    )

    # 2) Mean f_Im curves (imbalance profit)  <-- your "f_ID"
    plot_profit_mean_selected(
        price_setting=price_setting,
        field="f_Im",
        labels=SELECTED,
        title=f"Mean f_Im comparison ({price_setting})"
    )
    
    # Example:
    #  - Baselines: 2SP and 3SP as horizontal lines
    #  - PSDDiP: all K in the saved file (K_list)
    
    plot_evaluation_vs_K(
        price_setting=price_setting,
        baselines=["Rolling → SDDiP", "2SP → SDDiP", "3SP → SDDiP"],   # red, blue
        approx_ks=K_list,              # even spacing with these labels
    )
    
    # ----------------------------
    # 3. Printing actual values
    # ----------------------------

    # (A) Evaluation values
    print_eval_table(
        price_setting=price_setting,
        baselines=["Rolling → SDDiP", "2SP → SDDiP", "3SP → SDDiP"],
        approx_ks=K_list
    )

    # (B) Mean curves (copy/paste)
    SELECTED = [
        "Rolling → SDDiP",
        "2-SP → SDDiP",
        "3-SP → SDDiP",
        "PSDDiP (K=1)",
        "PSDDiP (K=30)",
    ]
    """
    print_solution_means(
        price_setting=price_setting,
        labels=SELECTED,
        fields=("q_da","q_ID","S","f_P","f_Im"),
        decimals=4
    )"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================

DEFAULT_BIN_NUM = 1
HERE = os.path.dirname(os.path.abspath(__file__))   # .../NestedBenders/Solutions_LP


# ============================================================
# Load all solution files automatically
# ============================================================

def _extract_bin_num_from_filename(filename: str):
    m = re.fullmatch(r"(\d+)_solutions\.npy", filename)
    return None if m is None else int(m.group(1))


def load_all_solutions_lp(folder=HERE):
    """
    Load every <bin_num>_solutions.npy in folder.

    Returns
    -------
    all_sol : dict
        all_sol[bin_num] = loaded payload dict
    """
    all_sol = {}

    for fn in os.listdir(folder):
        bin_num = _extract_bin_num_from_filename(fn)
        if bin_num is None:
            continue

        path = os.path.join(folder, fn)
        all_sol[bin_num] = np.load(path, allow_pickle=True).item()

    if not all_sol:
        raise FileNotFoundError(f"No '*_solutions.npy' files found in {folder}")

    return dict(sorted(all_sol.items(), key=lambda kv: kv[0]))


def load_solutions_lp(bin_num=DEFAULT_BIN_NUM, folder=HERE):
    """
    Load one selected bin_num from automatically discovered files.
    """
    all_sol = load_all_solutions_lp(folder=folder)
    if bin_num not in all_sol:
        raise KeyError(f"bin_num={bin_num} not found. Available: {list(all_sol.keys())}")
    return all_sol[bin_num]


def get_available_bin_nums(folder=HERE):
    return list(load_all_solutions_lp(folder=folder).keys())


# ============================================================
# Helpers
# ============================================================

def _mode_key(approx_mode: bool) -> str:
    return "approx" if approx_mode else "exact"


def _mean_curve(mat2d):
    arr = np.asarray(mat2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr.mean(axis=0)


def _pct_diff(reference_value, target_value):
    """
    Percentage difference of target relative to reference:

        100 * (target - reference) / reference
    """
    ref = float(reference_value)
    tgt = float(target_value)

    if abs(ref) < 1e-12:
        raise ZeroDivisionError("Reference value is zero; cannot compute percentage difference.")

    return 100.0 * (tgt - ref) / ref


def _get_psddip_root(sol: dict) -> dict:
    key = "PSDDiP -> SDDiP"
    if key not in sol:
        raise KeyError(f"Missing '{key}' in saved file. Available keys: {list(sol.keys())}")
    return sol[key]


def _get_psddip_block(sol: dict, approx_mode: bool) -> dict:
    psd_root = _get_psddip_root(sol)
    mk = _mode_key(approx_mode)
    if mk not in psd_root:
        raise KeyError(
            f"Missing '{mk}' in '{list(sol.keys())}'. "
            f"Available PSDDiP keys: {list(psd_root.keys())}"
        )
    return psd_root[mk]


def _get_k_list(sol: dict):
    if "meta" not in sol or "K_list" not in sol["meta"]:
        raise KeyError("Missing sol['meta']['K_list']")
    return list(sol["meta"]["K_list"])


def _normalize_label(label: str) -> str:
    """
    Normalize user-facing labels.

    Supported:
      - "Rolling"
      - "2SP", "2-SP"
      - "SDDP"
      - "PSDDP(K=30)", "PSDDiP(K=30)", "PSDDP [K=30]"
    """
    s = label.strip()
    s2 = s.lower().replace(" ", "")

    if s2 in {"rolling", "rolling->rolling"}:
        return "Rolling"

    if s2 in {"2sp", "2-sp", "2sp->rolling", "2-sp->rolling"}:
        return "2SP"

    if s2 == "sddp":
        return "SDDP"

    s2 = s2.replace("[", "(").replace("]", ")")
    s2 = s2.replace("psddp", "psddip")

    m = re.fullmatch(r"psddip\(k=(\d+)\)", s2)
    if m:
        return f"PSDDiP(K={int(m.group(1))})"

    return s


def _parse_psddip_k(label: str):
    lab = _normalize_label(label)
    m = re.fullmatch(r"PSDDiP\(K=(\d+)\)", lab)
    return int(m.group(1)) if m else None


def _get_baseline_block(sol: dict, label: str) -> dict:
    lab = _normalize_label(label)

    if lab == "Rolling":
        key = "Rolling -> Rolling"
    elif lab == "2SP":
        key = "2-SP -> Rolling"
    else:
        raise ValueError(f"Unknown baseline label: {label}")

    if key not in sol:
        raise KeyError(f"Missing '{key}' in saved file. Available keys: {list(sol.keys())}")
    return sol[key]


def _get_sddp_block(sol: dict) -> dict:
    """
    Treat PSDDiP approx K=1 as SDDP.
    """
    psd = _get_psddip_block(sol, approx_mode=True)
    K_list = _get_k_list(sol)

    if 1 not in K_list:
        raise ValueError(f"K=1 not found in K_list={K_list}; cannot construct SDDP from approx K=1")

    k_idx = K_list.index(1)

    return {
        "q_da": psd["q_da"][k_idx],
        "q_ID": psd["q_ID"][k_idx],
        "S": psd["S"][k_idx],
        "f_P": psd["f_P"][k_idx],
        "f_Im": psd["f_Im"][k_idx],
        "eval": psd["eval"][k_idx],
    }


def _get_psddip_k_block(sol: dict, K: int, approx_mode=True) -> dict:
    psd = _get_psddip_block(sol, approx_mode=approx_mode)
    K_list = _get_k_list(sol)

    if K not in K_list:
        raise ValueError(f"K={K} not found in K_list={K_list}")

    k_idx = K_list.index(K)

    return {
        "q_da": psd["q_da"][k_idx],
        "q_ID": psd["q_ID"][k_idx],
        "S": psd["S"][k_idx],
        "f_P": psd["f_P"][k_idx],
        "f_Im": psd["f_Im"][k_idx],
        "eval": psd["eval"][k_idx],
    }


def _get_algorithm_block(sol: dict, label: str, approx_mode=True) -> dict:
    """
    Supported labels:
      - Rolling
      - 2SP
      - SDDP
      - PSDDiP(K=30)
    """
    lab = _normalize_label(label)

    if lab in {"Rolling", "2SP"}:
        return _get_baseline_block(sol, lab)

    if lab == "SDDP":
        return _get_sddp_block(sol)

    K = _parse_psddip_k(lab)
    if K is not None:
        return _get_psddip_k_block(sol, K=K, approx_mode=approx_mode)

    raise ValueError(f"Unknown label: {label}")


def _default_solution_labels():
    return ["Rolling", "2SP", "SDDP", "PSDDiP(K=30)"]


def _get_runtime_series(sol, approx_mode: bool):
    psd = _get_psddip_block(sol, approx_mode=approx_mode)
    if "runtime" not in psd:
        raise KeyError(f"Missing 'runtime' in PSDDiP {_mode_key(approx_mode)} block")
    return list(psd["runtime"])


# ============================================================
# Overlay solution plots for one selected bin_num
# ============================================================

def plot_overlay_q_da(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    approx_mode=True,
    ylim=(-1000, 32000),
    figsize=(9, 4.5),
):
    sol = load_solutions_lp(bin_num=bin_num)
    labels = _default_solution_labels() if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y = np.asarray(_get_algorithm_block(sol, label, approx_mode=approx_mode)["q_da"], dtype=float).reshape(-1)
        x = np.arange(len(y))
        plt.plot(x, y, linewidth=2, label=_normalize_label(label))

    plt.title(f"q_DA overlay (bin_num={bin_num})")
    plt.xlabel("Hour")
    plt.ylabel("q_DA")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlay_q_ID_mean(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    approx_mode=True,
    ylim=None,
    figsize=(9, 4.5),
):
    sol = load_solutions_lp(bin_num=bin_num)
    labels = _default_solution_labels() if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y = _mean_curve(_get_algorithm_block(sol, label, approx_mode=approx_mode)["q_ID"])
        x = np.arange(len(y))
        plt.plot(x, y, linewidth=2, label=_normalize_label(label))

    plt.title(f"q_ID mean overlay (bin_num={bin_num})")
    plt.xlabel("Hour")
    plt.ylabel("q_ID (mean over eval scenarios)")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlay_S_mean(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    approx_mode=True,
    S_cap=21022.1,
    ylim=None,
    figsize=(9, 4.5),
):
    sol = load_solutions_lp(bin_num=bin_num)
    labels = _default_solution_labels() if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y = _mean_curve(_get_algorithm_block(sol, label, approx_mode=approx_mode)["S"])
        x = np.arange(len(y))
        plt.plot(x, y, linewidth=2, label=_normalize_label(label))

    plt.title(f"SoC S mean overlay (bin_num={bin_num})")
    plt.xlabel("Hour")
    plt.ylabel("S (mean over eval scenarios)")
    plt.ylim(*(ylim if ylim is not None else (0, S_cap)))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlays_all(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    approx_mode=True,
    da_ylim=(-1000, 32000),
    qid_ylim=None,
    S_cap=21022.1,
    S_ylim=None,
):
    plot_overlay_q_da(bin_num=bin_num, labels=labels, approx_mode=approx_mode, ylim=da_ylim)
    plot_overlay_q_ID_mean(bin_num=bin_num, labels=labels, approx_mode=approx_mode, ylim=qid_ylim)
    plot_overlay_S_mean(bin_num=bin_num, labels=labels, approx_mode=approx_mode, S_cap=S_cap, ylim=S_ylim)


# ============================================================
# Profit comparison for one selected bin_num
# ============================================================

def _get_series_from_label(sol, label, field, approx_mode=True):
    data = _get_algorithm_block(sol, label, approx_mode=approx_mode)[field]
    arr = np.asarray(data, dtype=float)
    return _mean_curve(arr) if arr.ndim == 2 else arr.reshape(-1)


def plot_profit_mean_selected(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    approx_mode=True,
    field="f_P",
    title=None,
    xlabel="Hour",
    ylabel=None,
    ylim=None,
    figsize=(9, 4.5),
):
    labels = _default_solution_labels() if labels is None else labels
    sol = load_solutions_lp(bin_num=bin_num)

    curves = [
        (_normalize_label(lab), np.asarray(_get_series_from_label(sol, lab, field=field, approx_mode=approx_mode)).reshape(-1))
        for lab in labels
    ]

    x = np.arange(len(curves[0][1]))

    plt.figure(figsize=figsize)
    for lab, y in curves:
        plt.plot(x, y, linewidth=2, label=lab)

    plt.title(title if title is not None else f"{field} mean (bin_num={bin_num})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel is not None else field)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Evaluation vs K on SAME axis: approx and exact together
# ============================================================

def plot_evaluation_vs_K_both_modes(
    bin_num=DEFAULT_BIN_NUM,
    include_baselines=True,
    figsize=(8, 4.5),
    show_points=True,
    marker="o",
    linewidth=1.8,
    legend=True,
):
    sol = load_solutions_lp(bin_num=bin_num)

    K_list = _get_k_list(sol)
    psd_root = _get_psddip_root(sol)

    eval_approx = list(psd_root["approx"]["eval"])
    eval_exact = list(psd_root["exact"]["eval"])
    x_pos = np.arange(len(K_list))

    plt.figure(figsize=figsize)

    if include_baselines:
        plt.axhline(float(sol["Rolling -> Rolling"]["eval"]), linewidth=2.0, linestyle="--", color="tab:green", label="Rolling")
        plt.axhline(float(sol["2-SP -> Rolling"]["eval"]), linewidth=2.0, linestyle="--", color="tab:orange", label="2SP")
        plt.axhline(float(_get_sddp_block(sol)["eval"]), linewidth=2.0, linestyle="--", color="tab:red", label="SDDP")

    plt.plot(
        x_pos, eval_approx,
        linewidth=linewidth,
        marker=(marker if show_points else None),
        color="tab:blue",
        label="PSDDiP approx",
    )
    plt.plot(
        x_pos, eval_exact,
        linewidth=linewidth,
        marker=(marker if show_points else None),
        color="tab:purple",
        label="PSDDiP exact",
    )

    plt.title(f"Evaluation vs K (bin_num={bin_num})")
    plt.xlabel("K")
    plt.ylabel("Evaluation")
    plt.xticks(x_pos, [str(k) for k in K_list])
    plt.grid(True, alpha=0.3)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_runtime_vs_K_both_modes(
    bin_num=DEFAULT_BIN_NUM,
    figsize=(8, 4.5),
    show_points=True,
    marker="o",
    linewidth=1.8,
    legend=True,
):
    sol = load_solutions_lp(bin_num=bin_num)

    K_list = _get_k_list(sol)
    runtime_approx = _get_runtime_series(sol, approx_mode=True)
    runtime_exact = _get_runtime_series(sol, approx_mode=False)
    x_pos = np.arange(len(K_list))

    plt.figure(figsize=figsize)

    plt.plot(
        x_pos, runtime_approx,
        linewidth=linewidth,
        marker=(marker if show_points else None),
        color="tab:blue",
        label="PSDDiP approx runtime",
    )
    plt.plot(
        x_pos, runtime_exact,
        linewidth=linewidth,
        marker=(marker if show_points else None),
        color="tab:purple",
        label="PSDDiP exact runtime",
    )

    plt.title(f"Running time vs K (bin_num={bin_num})")
    plt.xlabel("K")
    plt.ylabel("Running time (sec)")
    plt.xticks(x_pos, [str(k) for k in K_list])
    plt.grid(True, alpha=0.3)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Evaluation vs bin_num
# ============================================================

def plot_evaluation_vs_bin_num(
    labels=None,
    figsize=(8.5, 4.8),
    show_points=True,
    marker="o",
    linewidth=1.8,
    legend=True,
):
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())
    labels = ["SDDP", "PSDDiP(K=5)", "PSDDiP(K=30)"] if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y_vals = [
            float(_get_algorithm_block(all_sol[bin_num], label, approx_mode=True)["eval"])
            for bin_num in bin_num_list
        ]

        plt.plot(
            bin_num_list,
            y_vals,
            linewidth=linewidth,
            marker=(marker if show_points else None),
            label=_normalize_label(label),
        )

    plt.title("Evaluation vs bin_num")
    plt.xlabel("bin_num")
    plt.ylabel("Evaluation")
    plt.xticks(bin_num_list, [str(b) for b in bin_num_list])
    plt.grid(True, alpha=0.3)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_evaluation_diff_pct_vs_bin_num(
    labels=None,
    baseline_label="SDDP",
    figsize=(8.5, 4.8),
    show_points=True,
    marker="o",
    linewidth=1.8,
    legend=True,
):
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())
    labels = ["SDDP", "PSDDiP(K=5)", "PSDDiP(K=30)"] if labels is None else labels

    if _normalize_label(baseline_label) not in [_normalize_label(lab) for lab in labels]:
        labels = [baseline_label] + labels

    plt.figure(figsize=figsize)

    for label in labels:
        y_vals = []
        for bin_num in bin_num_list:
            sol = all_sol[bin_num]
            baseline_eval = float(_get_algorithm_block(sol, baseline_label, approx_mode=True)["eval"])
            method_eval = float(_get_algorithm_block(sol, label, approx_mode=True)["eval"])
            y_vals.append(_pct_diff(baseline_eval, method_eval))

        plt.plot(
            bin_num_list,
            y_vals,
            linewidth=linewidth,
            marker=(marker if show_points else None),
            label=_normalize_label(label),
        )

    plt.axhline(0.0, linewidth=1.5, linestyle="--", color="black")
    plt.title(f"Evaluation difference vs bin_num (baseline = {_normalize_label(baseline_label)})")
    plt.xlabel("bin_num")
    plt.ylabel("Difference from baseline (%)")
    plt.xticks(bin_num_list, [str(b) for b in bin_num_list])
    plt.grid(True, alpha=0.3)
    if legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Printing helpers
# ============================================================

def print_available_files():
    all_sol = load_all_solutions_lp()
    print("\nAvailable bin_num files:")
    for bin_num, sol in all_sol.items():
        print(f"  bin_num={bin_num} | K_list={_get_k_list(sol)}")


def print_eval_table_one_bin(bin_num=DEFAULT_BIN_NUM):
    sol = load_solutions_lp(bin_num=bin_num)
    K_list = _get_k_list(sol)

    print("\n====================================")
    print(f"Evaluation table | bin_num={bin_num}")
    print("====================================")
    print(f"Rolling : {float(sol['Rolling -> Rolling']['eval']):.6f}")
    print(f"2SP     : {float(sol['2-SP -> Rolling']['eval']):.6f}")
    print(f"SDDP    : {float(_get_sddp_block(sol)['eval']):.6f}")

    approx_eval = _get_psddip_block(sol, True)["eval"]
    exact_eval = _get_psddip_block(sol, False)["eval"]

    print("\nK | approx | exact")
    for K, va, ve in zip(K_list, approx_eval, exact_eval):
        print(f"{K:<3d} | {float(va):.6f} | {float(ve):.6f}")


def print_eval_table_across_bin_num(labels=None, baseline_label="SDDP"):
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())

    if labels is None:
        labels = ["SDDP", "PSDDiP(K=5)", "PSDDiP(K=30)"]

    baseline_norm = _normalize_label(baseline_label)

    print("\n==============================================================")
    print("Evaluation across bin_num (with % vs SDDP)")
    print("==============================================================")

    header = (
        "label".ljust(18)
        + " | "
        + " | ".join([f"bin={b}".rjust(15) for b in bin_num_list])
        + " | "
        + " | ".join([f"% vs {baseline_norm}".rjust(12) for _ in bin_num_list])
    )
    print(header)
    print("-" * len(header))

    for label in labels:
        vals = []
        pct_vals = []

        for bin_num in bin_num_list:
            sol = all_sol[bin_num]

            method_eval = float(_get_algorithm_block(sol, label, approx_mode=True)["eval"])
            baseline_eval = float(_get_algorithm_block(sol, baseline_label, approx_mode=True)["eval"])

            vals.append(method_eval)

            if _normalize_label(label) == baseline_norm:
                pct_vals.append(0.0)
            else:
                pct_vals.append(_pct_diff(baseline_eval, method_eval))

        val_str = " | ".join([f"{v:15.6f}" for v in vals])
        pct_str = " | ".join([f"{p:11.4f}%" for p in pct_vals])

        print(f"{_normalize_label(label).ljust(18)} | {val_str} | {pct_str}")


def print_eval_diff_pct_table_across_bin_num(
    labels=None,
    baseline_label="Rolling",
):
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())
    labels = ["Rolling", "2SP", "SDDP", "PSDDiP(K=5)", "PSDDiP(K=30)"] if labels is None else labels

    if _normalize_label(baseline_label) not in [_normalize_label(lab) for lab in labels]:
        labels = [baseline_label] + labels

    print("\n==============================================================")
    print(f"Evaluation percentage difference across bin_num | baseline={_normalize_label(baseline_label)}")
    print("==============================================================")

    header = "label".ljust(18) + " | " + " | ".join([f"bin={b}".rjust(12) for b in bin_num_list])
    print(header)
    print("-" * len(header))

    for label in labels:
        vals = []
        for bin_num in bin_num_list:
            sol = all_sol[bin_num]
            baseline_eval = float(_get_algorithm_block(sol, baseline_label, approx_mode=True)["eval"])
            method_eval = float(_get_algorithm_block(sol, label, approx_mode=True)["eval"])
            vals.append(_pct_diff(baseline_eval, method_eval))

        row = _normalize_label(label).ljust(18) + " | " + " | ".join([f"{v:11.4f}%" for v in vals])
        print(row)


def print_summary_table_one_bin(bin_num=DEFAULT_BIN_NUM):
    """
    Compact summary for one selected W=bin_num.

    Shows:
      - K=30 Running Time
      - SDDP vs PSDDiP(K=...) (%) for all available K
        at the fixed W = bin_num
    """
    sol = load_solutions_lp(bin_num=bin_num)
    K_list = _get_k_list(sol)

    print("\n==============================================")
    print(f"Compact summary | W={bin_num}")
    print("==============================================")

    if 30 in K_list:
        k30_idx = K_list.index(30)
        runtime_30 = float(_get_runtime_series(sol, approx_mode=True)[k30_idx])
        print(f"K=30 Running Time                 : {runtime_30:.4f}")
    else:
        print("K=30 Running Time                 : N/A (K=30 not available)")

    eval_sddp = float(_get_sddp_block(sol)["eval"])

    print("")
    for K in K_list:
        eval_psddip = float(_get_psddip_k_block(sol, K=K, approx_mode=True)["eval"])
        pct = _pct_diff(eval_sddp, eval_psddip)
        print(f"SDDP vs PSDDP(K={K}) (%) (W={bin_num}) : {pct: .4f}%")


def print_runtime_table_across_bin_num():
    """
    Print PSDDiP runtime table across all bin_num and all K.

    Rows: bin_num (W)
    Columns: K values
    """
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())

    # assume all share same K_list
    first_sol = next(iter(all_sol.values()))
    K_list = _get_k_list(first_sol)

    print("\n==============================================================")
    print("PSDDiP Runtime across bin_num")
    print("==============================================================")

    # header (no "K=")
    header = "bin_num".ljust(10) + " | " + " | ".join([f"{k}".rjust(10) for k in K_list])
    print(header)
    print("-" * len(header))

    # rows
    for bin_num in bin_num_list:
        sol = all_sol[bin_num]
        runtime = _get_runtime_series(sol, approx_mode=True)

        row_vals = " | ".join([f"{float(rt):10.2f}" for rt in runtime])
        print(f"{str(bin_num).ljust(10)} | {row_vals}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    bin_num = 6

    SOLUTION_LABELS = ["Rolling", "2SP", "SDDP", "PSDDiP(K=50)"]
    COMPARE_LABELS_ACROSS_BIN = ["SDDP", "PSDDiP(K=50)"]

    print_available_files()

    plot_overlays_all(
        bin_num=bin_num,
        labels=SOLUTION_LABELS,
        approx_mode=True,
        da_ylim=(-1000, 32000),
        qid_ylim=None,
        S_cap=21022.1,
        S_ylim=None,
    )

    plot_profit_mean_selected(
        bin_num=bin_num,
        labels=SOLUTION_LABELS,
        approx_mode=True,
        field="f_P",
        title=f"Mean f_P comparison (bin_num={bin_num})",
    )

    plot_profit_mean_selected(
        bin_num=bin_num,
        labels=SOLUTION_LABELS,
        approx_mode=True,
        field="f_Im",
        title=f"Mean f_Im comparison (bin_num={bin_num})",
    )

    plot_evaluation_vs_K_both_modes(
        bin_num=bin_num,
        include_baselines=True,
    )

    plot_evaluation_vs_bin_num(
        labels=COMPARE_LABELS_ACROSS_BIN,
    )

    plot_evaluation_diff_pct_vs_bin_num(
        labels=COMPARE_LABELS_ACROSS_BIN,
        baseline_label="SDDP",
    )

    print_eval_table_one_bin(bin_num=bin_num)
    print_eval_table_across_bin_num(labels=COMPARE_LABELS_ACROSS_BIN)

    print_eval_diff_pct_table_across_bin_num(
        labels=["Rolling", "2SP", "SDDP", "PSDDiP(K=5)", "PSDDiP(K=50)"],
        baseline_label="Rolling",
    )

    print_summary_table_one_bin(
        bin_num=bin_num,
    )

    plot_runtime_vs_K_both_modes(
        bin_num=bin_num,
    )

    print_runtime_table_across_bin_num()
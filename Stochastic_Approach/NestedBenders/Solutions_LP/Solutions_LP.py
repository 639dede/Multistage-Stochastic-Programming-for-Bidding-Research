import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Global plotting style
# ============================================================

FONT_SCALE = 2.0

plt.rcParams.update({
    "font.size": 10 * FONT_SCALE,
    "axes.titlesize": 10 * FONT_SCALE,
    "axes.labelsize": 9 * FONT_SCALE,
    "xtick.labelsize": 8 * FONT_SCALE,
    "ytick.labelsize": 8 * FONT_SCALE,
    "legend.fontsize": 7.5 * FONT_SCALE,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
})

# ============================================================
# Config
# ============================================================

DEFAULT_BIN_NUM = 1
HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# Load all solution files automatically
# ============================================================

def _extract_bin_num_from_filename(filename: str):
    m = re.fullmatch(r"(\d+)_solutions\.npy", filename)
    return None if m is None else int(m.group(1))


def _labels_to_k_list(labels):
    k_vals = []
    for label in labels:
        lab = _normalize_label(label)

        if lab == "SDDP":
            continue

        if lab == "Hybrid NBD/SDDP":
            k_vals.append(50)
            continue

        K = _parse_psddip_k(label)
        if K is not None:
            k_vals.append(K)

    return k_vals


def _extract_curve_with_optional_path(data, scenario_num=None):
    """
    data:
      - mean mode: nested lists or arrays
      - path mode: data[n][path_idx][t]

    If scenario_num is None:
        return mean curve across all available paths from all DA nodes.

    If scenario_num is an integer:
        flatten all paths across all DA nodes, then return that one path.
    """
    if scenario_num is None:
        all_curves = []
        for data_n in data:
            arr_n = np.asarray(data_n, dtype=float)

            if arr_n.ndim == 1:
                all_curves.append(arr_n)
            elif arr_n.ndim == 2:
                for row in arr_n:
                    all_curves.append(np.asarray(row, dtype=float))
            else:
                raise ValueError(f"Unsupported shape in mean extraction: {arr_n.shape}")

        if not all_curves:
            raise ValueError("No curves found.")

        return np.mean(np.vstack(all_curves), axis=0)

    flat_paths = []
    for data_n in data:
        arr_n = np.asarray(data_n, dtype=float)

        if arr_n.ndim == 1:
            flat_paths.append(arr_n)
        elif arr_n.ndim == 2:
            for p in range(arr_n.shape[0]):
                flat_paths.append(arr_n[p])
        else:
            raise ValueError(f"Unsupported shape in path extraction: {arr_n.shape}")

    if not flat_paths:
        raise ValueError("No pathwise data found.")

    if scenario_num < 0 or scenario_num >= len(flat_paths):
        raise IndexError(
            f"scenario_num={scenario_num} is out of range; "
            f"available global paths: 0..{len(flat_paths) - 1}"
        )

    return np.asarray(flat_paths[scenario_num], dtype=float)


def load_all_solutions_lp(folder=HERE):
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
    all_sol = load_all_solutions_lp(folder=folder)
    if bin_num not in all_sol:
        raise KeyError(f"bin_num={bin_num} not found. Available: {list(all_sol.keys())}")
    return all_sol[bin_num]


def get_available_bin_nums(folder=HERE):
    return list(load_all_solutions_lp(folder=folder).keys())


# ============================================================
# Helpers
# ============================================================

def _pct_diff(reference_value, target_value):
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


def _get_psddip_block(sol: dict, mode="approx") -> dict:
    psd_root = _get_psddip_root(sol)
    if mode not in psd_root:
        raise KeyError(
            f"Missing '{mode}' in '{list(sol.keys())}'. "
            f"Available PSDDiP keys: {list(psd_root.keys())}"
        )
    return psd_root[mode]


def _get_k_list(sol: dict):
    if "meta" not in sol or "K_list" not in sol["meta"]:
        raise KeyError("Missing sol['meta']['K_list']")
    return list(sol["meta"]["K_list"])


def _normalize_label(label: str) -> str:
    s = label.strip()
    s2 = s.lower().replace(" ", "")

    if s2 in {"rolling", "rolling->rolling"}:
        return "Rolling"

    if s2 in {"2sp", "2-sp", "2sp->rolling", "2-sp->rolling"}:
        return "2SP"

    if s2 == "sddp":
        return "SDDP"

    if s2 in {"psddp", "psddip", "hybridnbd/sddp"}:
        return "Hybrid NBD/SDDP"

    s2 = s2.replace("[", "(").replace("]", ")")
    s2 = s2.replace("psddp", "psddip")

    m = re.fullmatch(r"psddip\(k=(\d+)\)", s2)
    if m:
        K = int(m.group(1))
        return "Hybrid NBD/SDDP" if K == 50 else f"Hybrid NBD/SDDP(K={K})"

    m = re.fullmatch(r"hybridnbd/sddp\(k=(\d+)\)", s2)
    if m:
        K = int(m.group(1))
        return "Hybrid NBD/SDDP" if K == 50 else f"Hybrid NBD/SDDP(K={K})"

    return s


def _parse_psddip_k(label: str):
    raw = label.strip()
    lowered = raw.lower().replace(" ", "")
    lowered = lowered.replace("psddp", "psddip")

    if lowered in {"psddp", "psddip", "hybridnbd/sddp"}:
        return 50

    lowered = lowered.replace("[", "(").replace("]", ")")
    m = re.fullmatch(r"psddip\(k=(\d+)\)", lowered)
    if m:
        return int(m.group(1))

    m = re.fullmatch(r"hybridnbd/sddp\(k=(\d+)\)", lowered)
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


def _attach_optional_keys(out, src, idx=None):
    keys = [
        "f_DA", "f_DA_paths",
        "q_ID_paths", "S_paths", "f_P_paths", "f_Im_paths", "eval_paths",
        "P_DA_paths", "P_ID_paths", "delta_E_paths", "delta_C_paths"
    ]
    for key in keys:
        if key in src:
            out[key] = src[key] if idx is None else src[key][idx]
    return out


def _get_sddp_block(sol: dict) -> dict:
    psd = _get_psddip_block(sol, mode="approx")
    K_list = _get_k_list(sol)

    if 1 not in K_list:
        raise ValueError(f"K=1 not found in K_list={K_list}; cannot construct SDDP from approx K=1")

    k_idx = K_list.index(1)

    out = {
        "q_da": psd["q_da"][k_idx],
        "q_ID": psd["q_ID"][k_idx],
        "S": psd["S"][k_idx],
        "f_P": psd["f_P"][k_idx],
        "f_Im": psd["f_Im"][k_idx],
        "eval": psd["eval"][k_idx],
    }

    if "f_DA" in psd:
        out["f_DA"] = psd["f_DA"][k_idx]

    return _attach_optional_keys(out, psd, idx=k_idx)


def _get_psddip_k_block(sol: dict, K: int, mode="approx") -> dict:
    psd = _get_psddip_block(sol, mode=mode)
    K_list = _get_k_list(sol)

    if K not in K_list:
        raise ValueError(f"K={K} not found in K_list={K_list}")

    k_idx = K_list.index(K)

    out = {
        "q_da": psd["q_da"][k_idx],
        "q_ID": psd["q_ID"][k_idx],
        "S": psd["S"][k_idx],
        "f_P": psd["f_P"][k_idx],
        "f_Im": psd["f_Im"][k_idx],
        "eval": psd["eval"][k_idx],
    }

    if "f_DA" in psd:
        out["f_DA"] = psd["f_DA"][k_idx]

    return _attach_optional_keys(out, psd, idx=k_idx)


def _get_algorithm_block(sol: dict, label: str) -> dict:
    lab = _normalize_label(label)

    if lab in {"Rolling", "2SP"}:
        return _get_baseline_block(sol, lab)

    if lab == "SDDP":
        return _get_sddp_block(sol)

    K = _parse_psddip_k(label)
    if K is not None:
        return _get_psddip_k_block(sol, K=K, mode="approx")

    raise ValueError(f"Unknown label: {label}")


def _get_series_from_label(sol, label, field, scenario_num=None):
    block = _get_algorithm_block(sol, label)

    if field == "q_da":
        return np.asarray(block["q_da"], dtype=float).reshape(-1)

    path_field_map = {
        "q_ID": "q_ID_paths",
        "S": "S_paths",
        "f_P": "f_P_paths",
        "f_Im": "f_Im_paths",
        "P_DA": "P_DA_paths",
        "P_ID": "P_ID_paths",
        "delta_E": "delta_E_paths",
        "delta_C": "delta_C_paths",
    }

    if field in path_field_map:
        pf = path_field_map[field]
        if pf not in block:
            raise KeyError(f"Missing '{pf}' for label '{label}'")
        return _extract_curve_with_optional_path(block[pf], scenario_num=scenario_num)

    data = block[field]
    arr = np.asarray(data, dtype=float)
    return arr.mean(axis=0) if arr.ndim == 2 else arr.reshape(-1)


def _mean_total_profit_from_block(block, field):
    if field == "f_DA":
        if "f_DA" not in block:
            raise KeyError(
                "Missing 'f_DA' in saved solution file. "
                "Please re-run evaluation_with_DA_profit.py and re-save the .npy files."
            )
        values = np.asarray(block["f_DA"], dtype=float).reshape(-1)
        return float(values.mean())

    path_field_map = {
        "f_P": "f_P_paths",
        "f_Im": "f_Im_paths",
    }

    if field not in path_field_map:
        raise ValueError(f"Unsupported profit field: {field}")

    pf = path_field_map[field]
    if pf not in block:
        data = np.asarray(block[field], dtype=float)
        if data.ndim == 1:
            return float(data.mean())
        return float(data.sum(axis=1).mean())

    totals = []
    for data_n in block[pf]:
        arr_n = np.asarray(data_n, dtype=float)
        if arr_n.ndim == 1:
            totals.append(float(arr_n.sum()))
        elif arr_n.ndim == 2:
            totals.extend(arr_n.sum(axis=1).astype(float).tolist())
        else:
            raise ValueError(f"Unsupported shape in profit aggregation: {arr_n.shape}")

    if not totals:
        raise ValueError(f"No pathwise data found for {field}")

    return float(np.mean(totals))


def get_profit_breakdown_means(bin_num=DEFAULT_BIN_NUM, labels=None):
    sol = load_solutions_lp(bin_num=bin_num)
    labels = _default_solution_labels() if labels is None else labels

    normalized_labels = []
    da_vals = []
    id_vals = []
    im_vals = []

    for label in labels:
        block = _get_algorithm_block(sol, label)
        normalized_labels.append(_normalize_label(label))
        da_vals.append(_mean_total_profit_from_block(block, "f_DA"))
        id_vals.append(_mean_total_profit_from_block(block, "f_P"))
        im_vals.append(_mean_total_profit_from_block(block, "f_Im"))

    return {
        "labels": normalized_labels,
        "DA profit": da_vals,
        "ID profit": id_vals,
        "Im penalty": im_vals,
    }


def _default_solution_labels():
    return ["2SP", "SDDP", "Hybrid NBD/SDDP"]


def _default_eval_labels(sol):
    k_list = _get_k_list(sol)
    labels = ["SDDP"]
    for K in k_list:
        if K == 1:
            continue
        labels.append("Hybrid NBD/SDDP" if K == 50 else f"Hybrid NBD/SDDP(K={K})")
    return labels


def _get_runtime_series(sol):
    psd = _get_psddip_block(sol, mode="approx")
    if "runtime" not in psd:
        raise KeyError("Missing 'runtime' in PSDDiP approx block")
    return list(psd["runtime"])


# ============================================================
# Plot styles
# ============================================================

def _style_from_solution_label(label: str):
    lab = _normalize_label(label)

    style_map = {
        "2SP": {
            "linestyle": "--",
            "marker": "s",
            "markevery": 2,
            "linewidth": 1.6,
        },
        "SDDP": {
            "linestyle": "-.",
            "marker": "^",
            "markevery": 2,
            "linewidth": 1.6,
        },
        "Hybrid NBD/SDDP": {
            "linestyle": "-",
            "marker": "o",
            "markevery": 2,
            "linewidth": 1.8,
        },
        "Hybrid NBD/SDDP(K=5)": {
            "linestyle": "-",
            "marker": "o",
            "markevery": 2,
            "linewidth": 1.8,
        },
        "Hybrid NBD/SDDP(K=10)": {
            "linestyle": "-",
            "marker": "D",
            "markevery": 2,
            "linewidth": 1.8,
        },
        "Hybrid NBD/SDDP(K=20)": {
            "linestyle": "-",
            "marker": "v",
            "markevery": 2,
            "linewidth": 1.8,
        },
        "Hybrid NBD/SDDP(K=30)": {
            "linestyle": "-",
            "marker": "P",
            "markevery": 2,
            "linewidth": 1.8,
        },
    }

    return style_map.get(lab, {
        "linestyle": "-",
        "marker": "o",
        "markevery": 2,
        "linewidth": 1.6,
    })


def _style_from_k_label(label: str):
    lab = _normalize_label(label)

    style_map = {
        "SDDP": {"linestyle": "-", "marker": "o", "markevery": 1, "linewidth": 1.8},
        "Hybrid NBD/SDDP(K=5)": {"linestyle": "--", "marker": "s", "markevery": 1, "linewidth": 1.6},
        "Hybrid NBD/SDDP(K=10)": {"linestyle": "-.", "marker": "^", "markevery": 1, "linewidth": 1.6},
        "Hybrid NBD/SDDP(K=20)": {"linestyle": ":", "marker": "D", "markevery": 1, "linewidth": 1.8},
        "Hybrid NBD/SDDP(K=30)": {"linestyle": "--", "marker": "v", "markevery": 1, "linewidth": 1.6},
        "Hybrid NBD/SDDP": {"linestyle": "-.", "marker": "P", "markevery": 1, "linewidth": 1.8},
    }

    return style_map.get(lab, {
        "linestyle": "-",
        "marker": "o",
        "markevery": 1,
        "linewidth": 1.6,
    })


def _style_from_baseline_name(label: str):
    lab = _normalize_label(label)

    style_map = {
        "Rolling": {"linestyle": "--", "linewidth": 1.8, "color": "tab:green"},
        "2SP": {"linestyle": "-.", "linewidth": 1.8, "color": "tab:orange"},
        "SDDP": {"linestyle": ":", "linewidth": 2.0, "color": "tab:red"},
    }

    return style_map.get(lab, {"linestyle": "--", "linewidth": 1.8})


# ============================================================
# Overlay solution plots
# ============================================================

def plot_overlay_q_da(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    ylim=(0, 20000),
    figsize=(9, 4.5),
):
    sol = load_solutions_lp(bin_num=bin_num)
    labels = _default_solution_labels() if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y = np.asarray(_get_algorithm_block(sol, label)["q_da"], dtype=float).reshape(-1)
        x = np.arange(len(y))
        style = _style_from_solution_label(label)

        plt.plot(
            x, y,
            label=_normalize_label(label),
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )

    plt.title("Day Ahead Bidding")
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlay_q_ID(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    scenario_num=None,
    ylim=(0, 20000),
    figsize=(9, 4.5),
):
    sol = load_solutions_lp(bin_num=bin_num)
    labels = _default_solution_labels() if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y = _get_series_from_label(sol, label, field="q_ID", scenario_num=scenario_num)
        x = np.arange(len(y))
        style = _style_from_solution_label(label)

        plt.plot(
            x, y,
            label=_normalize_label(label),
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )

    title = "Intraday Bidding" if scenario_num is None else f"Intraday Bidding for scenario_num={scenario_num}"

    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlay_S(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    scenario_num=None,
    S_cap=21022.1,
    ylim=None,
    figsize=(9, 4.5),
):
    sol = load_solutions_lp(bin_num=bin_num)
    labels = _default_solution_labels() if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y = _get_series_from_label(sol, label, field="S", scenario_num=scenario_num)
        x = np.arange(len(y))
        style = _style_from_solution_label(label)

        plt.plot(
            x, y,
            label=_normalize_label(label),
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )

    title = "Stage of Charge" if scenario_num is None else f"Stage of Charge for scenario_num={scenario_num}"

    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.ylim(*(ylim if ylim is not None else (0, S_cap)))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlays_all(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    scenario_num=None,
    da_ylim=(0, 20000),
    qid_ylim=(0, 20000),
    S_cap=21022.1,
    S_ylim=None,
):
    plot_overlay_q_da(bin_num=bin_num, labels=labels, ylim=da_ylim)
    plot_overlay_q_ID(bin_num=bin_num, labels=labels, scenario_num=scenario_num, ylim=qid_ylim)
    plot_overlay_S(bin_num=bin_num, labels=labels, scenario_num=scenario_num, S_cap=S_cap, ylim=S_ylim)


# ============================================================
# Profit comparison
# ============================================================

def plot_profit_selected(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    field="f_P",
    scenario_num=None,
    title=None,
    xlabel="Hour",
    ylabel=None,
    ylim=None,
    figsize=(9, 4.5),
):
    labels = _default_solution_labels() if labels is None else labels
    sol = load_solutions_lp(bin_num=bin_num)

    curves = []
    for lab in labels:
        block = _get_algorithm_block(sol, lab)

        if field == "f_DA":
            if "f_DA" not in block:
                raise KeyError(
                    "Missing 'f_DA' in saved solution file. "
                    "Please re-run evaluation_with_DA_profit.py and re-save the .npy files."
                )
            y = np.asarray(block["f_DA"], dtype=float).reshape(-1)
        else:
            y = np.asarray(
                _get_series_from_label(sol, lab, field=field, scenario_num=scenario_num)
            ).reshape(-1)

        curves.append((_normalize_label(lab), y))

    x = np.arange(len(curves[0][1]))

    plt.figure(figsize=figsize)
    for lab, y in curves:
        style = _style_from_solution_label(lab)
        plt.plot(
            x, y,
            label=lab,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )

    if title is None:
        if field == "f_DA":
            title = "Day Ahead Settlement Profit" if scenario_num is None else f"Day Ahead Settlement Profit for scenario_num={scenario_num}"
        elif field == "f_P":
            title = "Intraday Settlement Profit" if scenario_num is None else f"Intraday Settlement Profit for scenario_num={scenario_num}"
        elif field == "f_Im":
            title = "Imbalance Penalty" if scenario_num is None else f"Imbalance Penalty for scenario_num={scenario_num}"
        else:
            title = field if scenario_num is None else f"{field} for scenario_num={scenario_num}"

    if ylabel is None:
        if field in {"f_P", "f_Im"}:
            ylabel = "KRW"
        else:
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
# Stochastic-parameter plots
# ============================================================

def plot_stochastic_path_selected(
    bin_num=DEFAULT_BIN_NUM,
    label="Hybrid NBD/SDDP",
    scenario_num=0,
    fields=("P_DA", "P_ID", "delta_E", "delta_C"),
    figsize=(9, 4.5),
):
    sol = load_solutions_lp(bin_num=bin_num)

    for field in fields:
        y = np.asarray(
            _get_series_from_label(sol, label, field=field, scenario_num=scenario_num),
            dtype=float
        ).reshape(-1)

        if field == "delta_E":
            y = y[:-1]

        x = np.arange(len(y))

        plt.figure(figsize=figsize)
        plt.plot(x, y, linewidth=1.8, marker="o", markevery=2)
        plt.title(f"{field} for {_normalize_label(label)}, scenario_num={scenario_num}")
        plt.xlabel("Hour")
        plt.ylabel(field)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ============================================================
# Evaluation plots across DA-ID dependency levels
# ============================================================

def _pretty_eval_label(label):
    name = _normalize_label(label)

    if name == "SDDP":
        return "SDDP"

    if name == "Hybrid NBD/SDDP":
        return "Hybrid NBD/SDDP"

    m = re.match(r"Hybrid NBD/SDDP\(K=(\d+)\)", name)
    if m:
        K = m.group(1)
        return f"Approx (K={K})"

    return name

def plot_evaluation_across_bin_num(
    labels=None,
    figsize=(8, 4.8),
):
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())

    first_sol = next(iter(all_sol.values()))
    labels = _default_eval_labels(first_sol) if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y = []
        for bin_num in bin_num_list:
            sol = all_sol[bin_num]
            y.append(float(_get_algorithm_block(sol, label)["eval"]))

        style = _style_from_k_label(label)
        plt.plot(
            bin_num_list, y,
            label=_pretty_eval_label(label),
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )

    plt.title("Evaluation across DA-ID dependency levels")
    plt.xlabel("DA-ID dependency level")
    plt.ylabel("Evaluation")
    plt.xticks(bin_num_list)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eval_diff_across_bin_num(
    labels=None,
    baseline_label="SDDP",
    figsize=(8, 4.8),
):
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())

    first_sol = next(iter(all_sol.values()))
    labels = _default_eval_labels(first_sol) if labels is None else labels

    plt.figure(figsize=figsize)

    for label in labels:
        y = []
        for bin_num in bin_num_list:
            sol = all_sol[bin_num]
            baseline_eval = float(_get_algorithm_block(sol, baseline_label)["eval"])
            method_eval = float(_get_algorithm_block(sol, label)["eval"])
            y.append(_pct_diff(baseline_eval, method_eval))

        style = _style_from_k_label(label)
        plt.plot(
            bin_num_list, y,
            label=_pretty_eval_label(label),
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=style["markevery"],
            linewidth=style["linewidth"],
        )

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    plt.title("Evaluation difference across DA-ID dependency levels")
    plt.xlabel("DA-ID dependency level")
    plt.ylabel("Difference from baseline (%)")
    plt.xticks(bin_num_list)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Evaluation and runtime vs K
# ============================================================

def plot_evaluation_vs_K(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    figsize=(8, 4.8),
):
    sol = load_solutions_lp(bin_num=bin_num)

    K_list = _get_k_list(sol)
    psd = _get_psddip_block(sol, mode="approx")
    eval_hybrid_all = [float(v) for v in psd["eval"]]

    rolling_eval = float(_get_baseline_block(sol, "Rolling")["eval"])
    sp2_eval = float(_get_baseline_block(sol, "2SP")["eval"])
    sddp_eval = float(_get_sddp_block(sol)["eval"])

    if labels is None:
        K_plot = K_list
    else:
        selected_K = _labels_to_k_list(labels)
        K_plot = [K for K in K_list if K in selected_K]

    eval_plot = [eval_hybrid_all[K_list.index(K)] for K in K_plot]

    plt.figure(figsize=figsize)

    plt.axhline(
        rolling_eval,
        label="Det",
        **_style_from_baseline_name("Det")
    )
    plt.axhline(
        sp2_eval,
        label="2SP",
        **_style_from_baseline_name("2SP")
    )
    plt.axhline(
        sddp_eval,
        label="SDDP",
        **_style_from_baseline_name("SDDP")
    )

    plt.plot(
        K_plot, eval_plot,
        label="Hybrid NBD/SDDP",
        linestyle="-",
        marker="o",
        linewidth=1.8,
        color="tab:blue",
    )

    plt.title("Evaluation vs K")
    plt.xlabel("K")
    plt.ylabel("Evaluation")
    plt.xticks(K_plot)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_runtime_vs_K(
    bin_num=DEFAULT_BIN_NUM,
    figsize=(8, 4.8),
):
    sol = load_solutions_lp(bin_num=bin_num)
    K_list = _get_k_list(sol)
    runtime = [float(v) for v in _get_runtime_series(sol)]

    plt.figure(figsize=figsize)
    plt.plot(K_list, runtime, linestyle="-", marker="o", linewidth=1.8, label="Hybrid NBD/SDDP runtime")
    plt.title("Running Time vs K")
    plt.xlabel("K")
    plt.ylabel("Running Time (sec)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_evaluation_and_runtime_vs_K(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    figsize=(16, 5.2),
):
    sol = load_solutions_lp(bin_num=bin_num)

    K_list = _get_k_list(sol)
    psd = _get_psddip_block(sol, mode="approx")
    eval_hybrid_all = [float(v) for v in psd["eval"]]
    runtime_all = [float(v) for v in _get_runtime_series(sol)]

    rolling_eval = float(_get_baseline_block(sol, "Rolling")["eval"])
    sp2_eval = float(_get_baseline_block(sol, "2SP")["eval"])
    sddp_eval = float(_get_sddp_block(sol)["eval"])

    if labels is None:
        K_plot = K_list
    else:
        selected_K = _labels_to_k_list(labels)
        K_plot = [K for K in K_list if K in selected_K]

    eval_plot = [eval_hybrid_all[K_list.index(K)] for K in K_plot]
    runtime_plot = [runtime_all[K_list.index(K)] for K in K_plot]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.axhline(
        rolling_eval,
        label="Det",
        **_style_from_baseline_name("Rolling")
    )
    ax.axhline(
        sp2_eval,
        label="2SP",
        **_style_from_baseline_name("2SP")
    )
    ax.axhline(
        sddp_eval,
        label="SDDP",
        **_style_from_baseline_name("SDDP")
    )
    ax.plot(
        K_plot, eval_plot,
        linestyle="-",
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label="Hybrid NBD/SDDP"
    )
    ax.set_title("Evaluation vs K")
    ax.set_xlabel("K")
    ax.set_ylabel("Evaluation")
    ax.set_xticks(K_plot)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(
        K_plot, runtime_plot,
        linestyle="-",
        marker="o",
        linewidth=1.8,
        color="tab:blue",
        label="Hybrid NBD/SDDP runtime"
    )
    ax.set_title("Running Time vs K")
    ax.set_xlabel("K")
    ax.set_ylabel("Running Time (sec)")
    ax.set_xticks(K_plot)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# Mean profit breakdown plot
# ============================================================

def plot_mean_profit_breakdown(
    bin_num=DEFAULT_BIN_NUM,
    labels=None,
    figsize=(12, 7),
):
    sol = load_solutions_lp(bin_num=bin_num)
    labels = _default_solution_labels() if labels is None else labels

    alg_labels = []
    for l in labels:
        name = _normalize_label(l)
        if name == "Hybrid NBD/SDDP":
            name = "Hybrid"
        alg_labels.append(name)

    DA_vals = []
    ID_vals = []
    Im_vals = []

    for lab in labels:
        block = _get_algorithm_block(sol, lab)

        DA_mean = _mean_total_profit_from_block(block, "f_DA")

        ID_total = np.sum(np.asarray(block["f_P_paths"], dtype=float), axis=2)
        Im_total = np.sum(np.asarray(block["f_Im_paths"], dtype=float), axis=2)

        ID_mean = float(np.mean(ID_total))
        Im_mean = float(np.mean(Im_total))

        DA_vals.append(DA_mean)
        ID_vals.append(ID_mean)
        Im_vals.append(Im_mean)

    DA_vals = np.array(DA_vals) / 1e6
    ID_vals = np.array(ID_vals) / 1e6
    Im_vals = np.array(Im_vals) / 1e6

    Profit_vals = DA_vals + ID_vals
    Total_vals = DA_vals + ID_vals + Im_vals

    color_map = {
        "2SP": "tab:blue",
        "SDDP": "tab:orange",
        "Hybrid NBD/SDDP": "tab:purple",
    }
    colors = [color_map[_normalize_label(l)] for l in labels]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    plots = [
        ("DA profit", DA_vals),
        ("ID profit", ID_vals),
        ("Profit", Profit_vals),
        ("Im penalty", Im_vals),
        ("Total profit", Total_vals),
    ]

    for i, (title, vals) in enumerate(plots):
        axes[i].bar(alg_labels, vals, color=colors)
        axes[i].set_title(title)
        axes[i].set_ylabel("Mean value (*1e6KRW)")
        axes[i].grid(True, alpha=0.3)

    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


# ============================================================
# Printing helpers
# ============================================================

def print_available_files(folder=HERE):
    all_sol = load_all_solutions_lp(folder=folder)

    print("Available bin_num files:")
    for bin_num, sol in all_sol.items():
        k_list = _get_k_list(sol)
        print(f"  bin_num={bin_num} | K_list={k_list}")


def print_eval_table_across_bin_num(
    labels=None,
    baseline_label="SDDP",
):
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())

    first_sol = next(iter(all_sol.values()))
    labels = _default_eval_labels(first_sol) if labels is None else labels

    print("\n==============================================================")
    print(f"Evaluation comparison across DA-ID dependency levels | baseline={baseline_label}")
    print("==============================================================")

    header = "Method".ljust(24) + " | " + " | ".join([str(b).rjust(12) for b in bin_num_list])
    print(header)
    print("-" * len(header))

    for label in labels:
        vals = []
        for bin_num in bin_num_list:
            sol = all_sol[bin_num]
            baseline_eval = float(_get_algorithm_block(sol, baseline_label)["eval"])
            method_eval = float(_get_algorithm_block(sol, label)["eval"])
            vals.append(_pct_diff(baseline_eval, method_eval))

        row = _normalize_label(label).ljust(24) + " | " + " | ".join([f"{v:11.4f}%" for v in vals])
        print(row)


def print_summary_table_one_bin(bin_num=DEFAULT_BIN_NUM):
    sol = load_solutions_lp(bin_num=bin_num)
    K_list = _get_k_list(sol)

    print("\n==============================================")
    print(f"Compact summary | W={bin_num}")
    print("==============================================")

    if 30 in K_list:
        k30_idx = K_list.index(30)
        runtime_30 = float(_get_runtime_series(sol)[k30_idx])
        print(f"K=30 Running Time                           : {runtime_30:.4f}")
    else:
        print("K=30 Running Time                           : N/A (K=30 not available)")

    eval_sddp = float(_get_sddp_block(sol)["eval"])

    print("")
    for K in K_list:
        eval_hybrid = float(_get_psddip_k_block(sol, K=K)["eval"])
        pct = _pct_diff(eval_sddp, eval_hybrid)
        shown = "Hybrid NBD/SDDP" if K == 50 else f"Hybrid NBD/SDDP(K={K})"
        print(f"SDDP vs {shown} (%) (W={bin_num}) : {pct: .4f}%")


def print_runtime_table_across_bin_num():
    all_sol = load_all_solutions_lp()
    bin_num_list = list(all_sol.keys())

    first_sol = next(iter(all_sol.values()))
    K_list = _get_k_list(first_sol)

    print("\n==============================================================")
    print("Hybrid NBD/SDDP Runtime across DA-ID dependency levels")
    print("==============================================================")

    header = "bin_num".ljust(10) + " | " + " | ".join([f"{k}".rjust(10) for k in K_list])
    print(header)
    print("-" * len(header))

    for bin_num in bin_num_list:
        sol = all_sol[bin_num]
        runtime = _get_runtime_series(sol)
        row_vals = " | ".join([f"{float(rt):10.2f}" for rt in runtime])
        print(f"{str(bin_num).ljust(10)} | {row_vals}")


def print_evaluation_table_one_bin(bin_num=DEFAULT_BIN_NUM):
    sol = load_solutions_lp(bin_num=bin_num)

    roll_block = _get_baseline_block(sol, "Rolling")
    sp2_block = _get_baseline_block(sol, "2SP")
    sddp_block = _get_sddp_block(sol)
    hybrid_block = _get_psddip_k_block(sol, K=50)

    def _aggregate_total_from_paths(block, field):
        path_map = {
            "f_P": "f_P_paths",
            "f_Im": "f_Im_paths",
        }

        if field == "f_DA":
            if "f_DA" not in block:
                return 0.0
            arr = np.asarray(block["f_DA"], dtype=float).reshape(-1)
            return float(arr.mean())

        pf = path_map[field]
        if pf not in block:
            arr = np.asarray(block[field], dtype=float)
            if arr.ndim == 1:
                return float(arr.sum())
            return float(arr.sum(axis=1).mean())

        totals = []
        for data_n in block[pf]:
            arr_n = np.asarray(data_n, dtype=float)
            if arr_n.ndim == 1:
                totals.append(float(arr_n.sum()))
            elif arr_n.ndim == 2:
                totals.extend(arr_n.sum(axis=1).astype(float).tolist())
            else:
                raise ValueError(f"Unsupported shape in aggregation: {arr_n.shape}")

        if not totals:
            raise ValueError(f"No pathwise data found for {field}")

        return float(np.mean(totals))

    data = [
        ("Rolling", roll_block),
        ("2SP", sp2_block),
        ("SDDP", sddp_block),
        ("Hybrid NBD/SDDP", hybrid_block),
    ]

    rows = []
    for name, blk in data:
        f_DA_total = _aggregate_total_from_paths(blk, "f_DA")
        f_P_total = _aggregate_total_from_paths(blk, "f_P")
        f_Im_total = _aggregate_total_from_paths(blk, "f_Im")
        settlement = f_DA_total + f_P_total
        total = float(blk["eval"])
        rows.append((name, total, settlement, f_Im_total))

    eval_hybrid = float(hybrid_block["eval"])

    print("\n==========================================================================================================")
    print(f"Evaluation summary with settlement decomposition | W={bin_num}")
    print("==========================================================================================================")

    header = (
        "Method".ljust(24)
        + " | " + "Evaluation".rjust(14)
        + " | " + "Rel. (%)".rjust(10)
        + " | " + "Settlement".rjust(14)
        + " | " + "Im Penalty".rjust(14)
    )
    print(header)
    print("-" * len(header))

    for name, total, settlement, f_Im_total in rows:
        rel = 100.0 * (eval_hybrid - total) / eval_hybrid
        print(
            name.ljust(24)
            + " | " + f"{total:14.2f}"
            + " | " + f"{rel:9.2f}"
            + " | " + f"{settlement:14.2f}"
            + " | " + f"{f_Im_total:14.2f}"
        )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":

    full_node_num = 50
    bin_num = 5
    scenario_num = 190

    SOLUTION_LABELS = ["2SP", "SDDP", "Hybrid NBD/SDDP"]
    EVAL_LABELS = [
        "SDDP",
        "Hybrid NBD/SDDP(K=1)",
        "Hybrid NBD/SDDP(K=5)",
        "Hybrid NBD/SDDP(K=10)",
        "Hybrid NBD/SDDP(K=20)",
        "Hybrid NBD/SDDP",
    ]

    print_available_files()

    plot_overlays_all(
        bin_num=bin_num,
        labels=SOLUTION_LABELS,
        scenario_num=scenario_num,
        da_ylim=(-2000, 20000),
        qid_ylim=(-2000, 20000),
        S_cap=21022.1,
        S_ylim=None,
    )

    #plot_profit_selected(
    #    bin_num=bin_num,
    #    labels=SOLUTION_LABELS,
    #    field="f_P",
    #    scenario_num=scenario_num,
    #)

    plot_profit_selected(
        bin_num=bin_num,
        labels=SOLUTION_LABELS,
        field="f_Im",
        scenario_num=scenario_num,
    )

    plot_mean_profit_breakdown(
        bin_num=bin_num,
        labels=SOLUTION_LABELS,
    )

    plot_stochastic_path_selected(
        bin_num=bin_num,
        label="Hybrid NBD/SDDP",
        scenario_num=scenario_num,
        fields=("P_DA", "P_ID", "delta_E", "delta_C"),
    )

    plot_evaluation_across_bin_num(labels=EVAL_LABELS)
    plot_eval_diff_across_bin_num(labels=EVAL_LABELS, baseline_label="SDDP")
    plot_evaluation_vs_K(bin_num=bin_num, labels=EVAL_LABELS)
    plot_evaluation_and_runtime_vs_K(bin_num=bin_num, labels=EVAL_LABELS)

    print_summary_table_one_bin(bin_num=bin_num)
    print_runtime_table_across_bin_num()
    print_eval_table_across_bin_num(labels=EVAL_LABELS, baseline_label="SDDP")
    print_evaluation_table_one_bin(bin_num=bin_num)
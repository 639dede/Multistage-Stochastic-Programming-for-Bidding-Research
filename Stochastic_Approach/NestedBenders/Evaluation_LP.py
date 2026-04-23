import os
import signal, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
import random
import time
import math
import re
import warnings
import logging
import multiprocessing as mp
from joblib import Parallel, delayed
from pathlib import Path

from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.opt import TerminationCondition, SolverStatus

warnings.filterwarnings("ignore")
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

solver = 'gurobi'
SOLVER = pyo.SolverFactory(solver)

SOLVER.options['TimeLimit'] = 7200
SOLVER.options['MIPGap'] = 1e-3

assert SOLVER.available(), f"Solver {solver} is available."


price_setting = 'normal'  # 'cloudy', 'normal', 'sunny'

from NestedBenders.PSDDiP_LP import (
    fw_da,
    fw_rt_init,
    fw_rt,
    fw_rt_last,
    rolling_da,
    rolling_rt_init,
    rolling_rt,
    rolling_rt_last,
    two_stage_da,
    two_stage_rt_init,
    two_stage_rt,
    two_stage_rt_beforelast,
    two_stage_rt_last,
    three_stage_da,
    K_list,
)

# ============================================================
# 1. Parameters & Computational settings
# ============================================================

E_0_path_cloudy = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_cloudy.csv'
E_0_path_normal = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_normal.csv'
E_0_path_sunny  = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_sunny.csv'

np.set_printoptions(suppress=True, precision=4)

E_0_cloudy = np.loadtxt(E_0_path_cloudy, delimiter=',')
E_0_normal = np.loadtxt(E_0_path_normal, delimiter=',')
E_0_sunny  = np.loadtxt(E_0_path_sunny, delimiter=',')

if price_setting == 'cloudy':
    E_0 = E_0_cloudy
elif price_setting == 'normal':
    E_0 = E_0_normal
else:
    E_0 = E_0_sunny

P_r = 80
P_max = 200

C = 20000
S = C
B = C / 3

S_min = 0.1 * S
S_max = 0.9 * S

_price_re = re.compile(r'^K(\d+)\.csv$')
_tree_re  = re.compile(r'^scenario_(\d+)\.csv$')

bin_num = 5
T = 24


# ============================================================
# Load reduced DA prices / scenario trees
# ============================================================

def load_clustered_P_da(directory_path):
    """
    directory_path: './Stochastic_Approach/Scenarios/Reduced_data/P_da_<bin_num>'
    Returns:
      Reduced_P_da  -> list of (K,24) lists
      Reduced_Probs -> list of (K,) lists
    """
    names = [n for n in os.listdir(directory_path)
             if _price_re.match(n) and not n.endswith('.probs.csv')]
    names.sort(key=lambda n: int(_price_re.match(n).group(1)))

    Reduced_P_da, Reduced_Probs = [], []

    for name in names:
        price_path = os.path.join(directory_path, name)
        P = np.loadtxt(price_path, delimiter=',')

        if P.ndim == 1:
            P = P.reshape(1, -1)
        elif P.shape[1] != 24 and P.shape[0] == 24:
            P = P.T

        assert P.shape[1] == 24, f"{price_path} has shape {P.shape}; expected (K,24) or (24,K)"

        K = P.shape[0]

        probs_path = os.path.join(directory_path, name.replace('.csv', '.probs.csv'))
        if os.path.exists(probs_path):
            q = np.loadtxt(probs_path, delimiter=',').astype(float)
            q = np.atleast_1d(q).ravel()
            if q.size != K or not np.isfinite(q).all() or q.sum() <= 0:
                q = np.full(K, 1.0 / K, dtype=float)
            else:
                q = q / q.sum()
        else:
            q = np.full(K, 1.0 / K, dtype=float)

        Reduced_P_da.append(P.tolist())
        Reduced_Probs.append(q.tolist())

    return Reduced_P_da, Reduced_Probs


def load_scenario_trees(base_dir):
    """
    base_dir: './Stochastic_Approach/Scenarios/Reduced_data/scenario_trees_<bin_num>'
    Returns:
      Reduced_scenario_trees -> list over K (ascending), each is a list of trees
    """
    k_dirs = [d for d in os.listdir(base_dir)
              if d.startswith('K') and d[1:].isdigit()
              and os.path.isdir(os.path.join(base_dir, d))]
    k_dirs.sort(key=lambda d: int(d[1:]))

    Reduced_scenario_trees = []

    for kdir in k_dirs:
        k_path = os.path.join(base_dir, kdir)

        scen_files = [n for n in os.listdir(k_path)
                      if _tree_re.match(n) and n.endswith('.csv')]
        scen_files.sort(key=lambda n: int(_tree_re.match(n).group(1)))

        trees = []
        for fname in scen_files:
            fpath = os.path.join(k_path, fname)
            data = np.loadtxt(fpath, delimiter=',')

            if data.ndim == 1:
                data = data.reshape(1, -1)

            tree = [[] for _ in range(T)]
            for row in data:
                t = int(row[0])
                branch = row[2:].tolist()
                tree[t].append(branch)
            trees.append(tree)

        Reduced_scenario_trees.append(trees)

    return Reduced_scenario_trees


cluster_dir = f'./Stochastic_Approach/Scenarios/Reduced_data/P_da_{bin_num}'
Reduced_P_da, Reduced_Probs = load_clustered_P_da(cluster_dir)

clustered_tree_dir = f'./Stochastic_Approach/Scenarios/Reduced_data/scenario_trees_{bin_num}'
Reduced_scenario_trees = load_scenario_trees(clustered_tree_dir)

E_0_partial = E_0

P_da_eval = Reduced_P_da[-1]
Probs_eval = Reduced_Probs[-1]
Scenario_tree_eval = Reduced_scenario_trees[-1]

exp_P_da = Reduced_P_da[0][0]
scenario_exp = Reduced_scenario_trees[0]

K_eval = len(P_da_eval)


def expectation_P_rt():
    exp_P_rt_list = []
    scenario_tree_rt = scenario_exp[0]

    for t in range(T):
        branches_t = scenario_tree_rt[t]
        exp_P_rt = 0.0
        for b in branches_t:
            exp_P_rt += b[1] / len(branches_t)
        exp_P_rt_list.append(exp_P_rt)

    return exp_P_rt_list


def exp_P_rt_given_P_da(n, Scenario_tree_params):
    exp_P_rt_list = []
    scenario_tree_rt = Scenario_tree_params[n]

    for t in range(T):
        branches_t = scenario_tree_rt[t]
        exp_P_rt = 0.0
        for b in branches_t:
            exp_P_rt += b[1] / len(branches_t)
        exp_P_rt_list.append(exp_P_rt)

    return exp_P_rt_list


exp_P_rt_glob = expectation_P_rt()


# ============================================================
# 2. Load evaluation/test scenario paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SCEN_ROOT = BASE_DIR / "scenario_paths" / f"{bin_num}"

scenarios_for_eval = np.load(
    SCEN_ROOT / "scenarios_eval.npy",
    allow_pickle=True
).tolist()

scenarios_for_SP = np.load(
    SCEN_ROOT / "scenarios_SP.npy",
    allow_pickle=True
).tolist()


# ============================================================
# 3. Load ECTG functions for DA stage
# ============================================================

PSI_DA_DIR = BASE_DIR / "psi_DA_LP" / f"{bin_num}"

psi_DA_exact_list = []
psi_DA_approx_list = []

for K in K_list:
    psi_DA_exact_path = PSI_DA_DIR / 'exact' / f"psi_DA_{K}.npy"
    if not psi_DA_exact_path.exists():
        raise FileNotFoundError(f"Missing psi_DA_exact file: {psi_DA_exact_path}")
    psi_DA_exact_list.append(np.load(psi_DA_exact_path, allow_pickle=True).tolist())

    psi_DA_approx_path = PSI_DA_DIR / 'approx' / f"psi_DA_{K}.npy"
    if not psi_DA_approx_path.exists():
        raise FileNotFoundError(f"Missing psi_DA_approx file: {psi_DA_approx_path}")
    psi_DA_approx_list.append(np.load(psi_DA_approx_path, allow_pickle=True).tolist())


def load_psddip_runtime_lp(bin_num, K_list):
    """
    Returns:
        {
            "approx": [runtime for each K in K_list],
            "exact":  [runtime for each K in K_list],
        }
    """
    base_dir = Path(__file__).resolve().parent
    psi_root = base_dir / "psi_DA_LP" / f"{bin_num}"

    runtime_dict = {"approx": [], "exact": []}

    for mode in ["approx", "exact"]:
        mode_dir = psi_root / mode

        for K in K_list:
            meta_path = mode_dir / f"meta_{K}.npy"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing runtime meta file: {meta_path}")

            meta = np.load(meta_path, allow_pickle=True).item()
            runtime_dict[mode].append(float(meta["running_time"]))

    return runtime_dict


runtime_psddip = load_psddip_runtime_lp(bin_num=bin_num, K_list=K_list)
runtime_p_approx = runtime_psddip["approx"]
runtime_p_exact = runtime_psddip["exact"]


# ============================================================
# 4. Load ECTG functions for ID stages
# ============================================================

PSI_FULL_DIR = BASE_DIR / "psi_full_LP"
state_path = PSI_FULL_DIR / f"{bin_num}_state.npy"

if not state_path.exists():
    raise FileNotFoundError(f"Missing full checkpoint: {state_path}")

state = np.load(state_path, allow_pickle=True).item()
psi_ID = state["psi_ID"]


# ============================================================
# Helper: stochastic-parameter extraction
# ============================================================

def extract_stage_random_values(stage_data):
    """
    Extract realized stochastic parameters from one stage scenario[t].

    Expected return:
        P_ID_t, delta_E_t, delta_C_t
    """

    if isinstance(stage_data, dict):
        return (
            float(stage_data["P_ID"]),
            float(stage_data["delta_E"]),
            float(stage_data["delta_C"]),
        )

    if isinstance(stage_data, (list, tuple, np.ndarray)):
        if len(stage_data) == 3:
            # Most likely structure from your printed sample:
            # [delta_E, P_ID, delta_C]
            delta_E_t = float(stage_data[0])
            P_ID_t    = float(stage_data[1])
            delta_C_t = float(stage_data[2])
            return P_ID_t, delta_E_t, delta_C_t

        raise ValueError(
            f"Unsupported stage_data length={len(stage_data)}. "
            f"stage_data={stage_data}"
        )

    raise TypeError(
        f"Unsupported stage_data type: {type(stage_data)}. "
        f"stage_data={stage_data}"
    )


def build_pathwise_mean(nested_paths):
    """
    nested_paths[n][path_idx][t]  -> mean over path_idx for each n
    returns:
        mean_curves[n][t]
    """
    out = []
    for n_data in nested_paths:
        arr = np.asarray(n_data, dtype=float)
        if arr.ndim == 1:
            out.append(arr.tolist())
        else:
            out.append(arr.mean(axis=0).tolist())
    return out


def build_scalar_mean(values):
    """
    values[n][path_idx] or values[n] -> mean over path_idx for each n
    returns:
        mean_values[n]
    """
    out = []
    for n_data in values:
        arr = np.asarray(n_data, dtype=float)
        if arr.ndim == 0:
            out.append(float(arr))
        else:
            out.append(float(arr.mean()))
    return out


# ============================================================
# 5. Evaluate
# ============================================================

def evaluation_rolling_rolling(scenarios):
    da_subp = rolling_da(exp_P_da, exp_P_rt_glob)
    da_state = da_subp.get_state_solutions()
    q_da = da_state[0]

    f = []

    f_DA = []

    q_ID_paths = []
    S_paths = []
    f_P_paths = []
    f_Im_paths = []
    eval_paths = []

    P_DA_paths = []
    P_ID_paths = []
    delta_E_paths = []
    delta_C_paths = []

    for n, scenarios_n in enumerate(scenarios):
        P_da = P_da_eval[n]
        exp_P_rt = exp_P_rt_given_P_da(n, Scenario_tree_eval)

        rt_init_subp = rolling_rt_init(da_state, P_da, exp_P_rt)
        rt_init_state = rt_init_subp.get_state_solutions()
        fcn_value = rt_init_subp.get_settlement_fcn_value()
        f_DA.append(float(fcn_value))

        q_ID_paths_n = []
        S_paths_n = []
        f_P_paths_n = []
        f_Im_paths_n = []
        eval_paths_n = []

        P_DA_paths_n = []
        P_ID_paths_n = []
        delta_E_paths_n = []
        delta_C_paths_n = []

        for scenario in scenarios_n:
            state = rt_init_state
            f_scenario = fcn_value

            q_ID_one = [0.0] * T
            S_one = [0.5 * S] + [0.0] * T
            f_P_one = [0.0] * T
            f_Im_one = [0.0] * T

            P_DA_one = list(P_da)
            P_ID_one = [0.0] * T
            delta_E_one = [0.0] * T
            delta_C_one = [0.0] * T

            for t in range(T - 1):
                P_ID_t, delta_E_t, delta_C_t = extract_stage_random_values(scenario[t])

                P_ID_one[t] = P_ID_t
                delta_E_one[t] = delta_E_t
                delta_C_one[t] = delta_C_t

                rt_subp = rolling_rt(t, state, P_da, exp_P_rt, scenario[t])

                q_ID_one[t] = rt_subp.get_ID_solution()
                S_one[t + 1] = rt_subp.get_S_solution()
                f_P_one[t] = rt_subp.get_P_profit()
                f_Im_one[t] = rt_subp.get_Im_profit()

                state = rt_subp.get_state_solutions()
                f_scenario += rt_subp.get_settlement_fcn_value()

            P_ID_t, delta_E_t, delta_C_t = extract_stage_random_values(scenario[T - 1])
            P_ID_one[T - 1] = P_ID_t
            delta_E_one[T - 1] = delta_E_t
            delta_C_one[T - 1] = delta_C_t

            rt_last_subp = rolling_rt_last(state, P_da, scenario[T - 1])

            q_ID_one[T - 1] = rt_last_subp.get_ID_solution()
            S_one[T] = rt_last_subp.get_S_solution()
            f_P_one[T - 1] = rt_last_subp.get_P_profit()
            f_Im_one[T - 1] = rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()

            q_ID_paths_n.append(q_ID_one)
            S_paths_n.append(S_one)
            f_P_paths_n.append(f_P_one)
            f_Im_paths_n.append(f_Im_one)
            eval_paths_n.append(float(f_scenario))

            P_DA_paths_n.append(P_DA_one)
            P_ID_paths_n.append(P_ID_one)
            delta_E_paths_n.append(delta_E_one)
            delta_C_paths_n.append(delta_C_one)

            f.append(f_scenario)

        q_ID_paths.append(q_ID_paths_n)
        S_paths.append(S_paths_n)
        f_P_paths.append(f_P_paths_n)
        f_Im_paths.append(f_Im_paths_n)
        eval_paths.append(eval_paths_n)

        P_DA_paths.append(P_DA_paths_n)
        P_ID_paths.append(P_ID_paths_n)
        delta_E_paths.append(delta_E_paths_n)
        delta_C_paths.append(delta_C_paths_n)

    eval_mean = float(np.mean(f))

    f_DA_mean = build_scalar_mean(f_DA)
    q_ID_mean = build_pathwise_mean(q_ID_paths)
    S_mean    = build_pathwise_mean(S_paths)
    f_P_mean  = build_pathwise_mean(f_P_paths)
    f_Im_mean = build_pathwise_mean(f_Im_paths)

    print(f"\nRolling -> Rolling for bin_num = {bin_num}")
    print(f"Evaluation : {eval_mean}")

    return {
        "q_da": q_da,
        "f_DA": f_DA_mean,
        "q_ID": q_ID_mean,
        "S": S_mean,
        "f_P": f_P_mean,
        "f_Im": f_Im_mean,
        "eval": eval_mean,
        "f_DA_paths": f_DA,
        "q_ID_paths": q_ID_paths,
        "S_paths": S_paths,
        "f_P_paths": f_P_paths,
        "f_Im_paths": f_Im_paths,
        "eval_paths": eval_paths,
        "P_DA_paths": P_DA_paths,
        "P_ID_paths": P_ID_paths,
        "delta_E_paths": delta_E_paths,
        "delta_C_paths": delta_C_paths,
    }


def evaluation_2SP_rolling(scenarios, scenarios_SP):
    exp_P_rt_each = [exp_P_rt_given_P_da(n, Scenario_tree_eval) for n in range(K_eval)]

    da_subp = two_stage_da(P_da_eval, exp_P_rt_each)
    da_state = da_subp.get_state_solutions()
    q_da = da_state[0]

    f = []

    f_DA = []

    q_ID_paths = []
    S_paths = []
    f_P_paths = []
    f_Im_paths = []
    eval_paths = []

    P_DA_paths = []
    P_ID_paths = []
    delta_E_paths = []
    delta_C_paths = []

    for n, scenarios_n in enumerate(scenarios):
        P_da = P_da_eval[n]
        scenario_paths = scenarios_SP[n]
        exp_P_rt = exp_P_rt_given_P_da(n, Scenario_tree_eval)

        ID_params_list = [[] for _ in range(T)]
        for scenario in scenario_paths:
            for t in range(T):
                ID_params_list[t].append(scenario[t])

        rt_init_subp = two_stage_rt_init(da_state, P_da, ID_params_list[0], exp_P_rt)
        rt_init_state = rt_init_subp.get_state_solutions()
        fcn_value = rt_init_subp.get_settlement_fcn_value()
        f_DA.append(float(fcn_value))

        q_ID_paths_n = []
        S_paths_n = []
        f_P_paths_n = []
        f_Im_paths_n = []
        eval_paths_n = []

        P_DA_paths_n = []
        P_ID_paths_n = []
        delta_E_paths_n = []
        delta_C_paths_n = []

        for scenario in scenarios_n:
            state = rt_init_state
            f_scenario = fcn_value

            q_ID_one = [0.0] * T
            S_one = [0.5 * S] + [0.0] * T
            f_P_one = [0.0] * T
            f_Im_one = [0.0] * T

            P_DA_one = list(P_da)
            P_ID_one = [0.0] * T
            delta_E_one = [0.0] * T
            delta_C_one = [0.0] * T

            for t in range(T - 2):
                P_ID_t, delta_E_t, delta_C_t = extract_stage_random_values(scenario[t])

                P_ID_one[t] = P_ID_t
                delta_E_one[t] = delta_E_t
                delta_C_one[t] = delta_C_t

                rt_subp = two_stage_rt(
                    t, state, P_da, ID_params_list[t + 1], exp_P_rt, scenario[t]
                )

                q_ID_one[t] = rt_subp.get_ID_solution()
                S_one[t + 1] = rt_subp.get_S_solution()
                f_P_one[t] = rt_subp.get_P_profit()
                f_Im_one[t] = rt_subp.get_Im_profit()

                state = rt_subp.get_state_solutions()
                f_scenario += rt_subp.get_settlement_fcn_value()

            P_ID_t, delta_E_t, delta_C_t = extract_stage_random_values(scenario[T - 2])
            P_ID_one[T - 2] = P_ID_t
            delta_E_one[T - 2] = delta_E_t
            delta_C_one[T - 2] = delta_C_t

            rt_beforelast_subp = two_stage_rt_beforelast(
                state, P_da, ID_params_list[T - 1], scenario[T - 2]
            )

            q_ID_one[T - 2] = rt_beforelast_subp.get_ID_solution()
            S_one[T - 1] = rt_beforelast_subp.get_S_solution()
            f_P_one[T - 2] = rt_beforelast_subp.get_P_profit()
            f_Im_one[T - 2] = rt_beforelast_subp.get_Im_profit()

            state = rt_beforelast_subp.get_state_solutions()
            f_scenario += rt_beforelast_subp.get_settlement_fcn_value()

            P_ID_t, delta_E_t, delta_C_t = extract_stage_random_values(scenario[T - 1])
            P_ID_one[T - 1] = P_ID_t
            delta_E_one[T - 1] = delta_E_t
            delta_C_one[T - 1] = delta_C_t

            rt_last_subp = two_stage_rt_last(state, P_da, scenario[T - 1])

            q_ID_one[T - 1] = rt_last_subp.get_ID_solution()
            S_one[T] = rt_last_subp.get_S_solution()
            f_P_one[T - 1] = rt_last_subp.get_P_profit()
            f_Im_one[T - 1] = rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()

            q_ID_paths_n.append(q_ID_one)
            S_paths_n.append(S_one)
            f_P_paths_n.append(f_P_one)
            f_Im_paths_n.append(f_Im_one)
            eval_paths_n.append(float(f_scenario))

            P_DA_paths_n.append(P_DA_one)
            P_ID_paths_n.append(P_ID_one)
            delta_E_paths_n.append(delta_E_one)
            delta_C_paths_n.append(delta_C_one)

            f.append(f_scenario)

        q_ID_paths.append(q_ID_paths_n)
        S_paths.append(S_paths_n)
        f_P_paths.append(f_P_paths_n)
        f_Im_paths.append(f_Im_paths_n)
        eval_paths.append(eval_paths_n)

        P_DA_paths.append(P_DA_paths_n)
        P_ID_paths.append(P_ID_paths_n)
        delta_E_paths.append(delta_E_paths_n)
        delta_C_paths.append(delta_C_paths_n)

    eval_mean = float(np.mean(f))

    f_DA_mean = build_scalar_mean(f_DA)
    q_ID_mean = build_pathwise_mean(q_ID_paths)
    S_mean    = build_pathwise_mean(S_paths)
    f_P_mean  = build_pathwise_mean(f_P_paths)
    f_Im_mean = build_pathwise_mean(f_Im_paths)

    print(f"\n2-SP -> Rolling for bin_num = {bin_num}")
    print(f"Evaluation : {eval_mean}")

    return {
        "q_da": q_da,
        "f_DA": f_DA_mean,
        "q_ID": q_ID_mean,
        "S": S_mean,
        "f_P": f_P_mean,
        "f_Im": f_Im_mean,
        "eval": eval_mean,
        "f_DA_paths": f_DA,
        "q_ID_paths": q_ID_paths,
        "S_paths": S_paths,
        "f_P_paths": f_P_paths,
        "f_Im_paths": f_Im_paths,
        "eval_paths": eval_paths,
        "P_DA_paths": P_DA_paths,
        "P_ID_paths": P_ID_paths,
        "delta_E_paths": delta_E_paths,
        "delta_C_paths": delta_C_paths,
    }


def evaluation_psddip_sddip(K, scenarios, approx_mode):
    k_idx = K_list.index(K)
    psi_DA_list = psi_DA_approx_list if approx_mode else psi_DA_exact_list

    da_subp = fw_da(psi_DA_list[k_idx])
    da_state = da_subp.get_state_solutions()
    q_da = da_state[0]

    f = []

    f_DA = []

    q_ID_paths = []
    S_paths = []
    f_P_paths = []
    f_Im_paths = []
    eval_paths = []

    P_DA_paths = []
    P_ID_paths = []
    delta_E_paths = []
    delta_C_paths = []

    for n, scenarios_n in enumerate(scenarios):
        P_da = P_da_eval[n]

        rt_init_subp = fw_rt_init(da_state, psi_ID[n][0], P_da)
        rt_init_state = rt_init_subp.get_state_solutions()
        fcn_value = rt_init_subp.get_settlement_fcn_value()
        f_DA.append(float(fcn_value))

        q_ID_paths_n = []
        S_paths_n = []
        f_P_paths_n = []
        f_Im_paths_n = []
        eval_paths_n = []

        P_DA_paths_n = []
        P_ID_paths_n = []
        delta_E_paths_n = []
        delta_C_paths_n = []

        for scenario in scenarios_n:
            state = rt_init_state
            f_scenario = fcn_value

            q_ID_one = [0.0] * T
            S_one = [0.5 * S] + [0.0] * T
            f_P_one = [0.0] * T
            f_Im_one = [0.0] * T

            P_DA_one = list(P_da)
            P_ID_one = [0.0] * T
            delta_E_one = [0.0] * T
            delta_C_one = [0.0] * T

            for t in range(T - 1):
                P_ID_t, delta_E_t, delta_C_t = extract_stage_random_values(scenario[t])

                P_ID_one[t] = P_ID_t
                delta_E_one[t] = delta_E_t
                delta_C_one[t] = delta_C_t

                rt_subp = fw_rt(t, state, psi_ID[n][t + 1], P_da, scenario[t])

                q_ID_one[t] = rt_subp.get_ID_solution()
                S_one[t + 1] = rt_subp.get_S_solution()
                f_P_one[t] = rt_subp.get_P_profit()
                f_Im_one[t] = rt_subp.get_Im_profit()

                state = rt_subp.get_state_solutions()
                f_scenario += rt_subp.get_settlement_fcn_value()

            P_ID_t, delta_E_t, delta_C_t = extract_stage_random_values(scenario[T - 1])
            P_ID_one[T - 1] = P_ID_t
            delta_E_one[T - 1] = delta_E_t
            delta_C_one[T - 1] = delta_C_t

            rt_last_subp = fw_rt_last(state, P_da, scenario[T - 1])

            q_ID_one[T - 1] = rt_last_subp.get_ID_solution()
            S_one[T] = rt_last_subp.get_S_solution()
            f_P_one[T - 1] = rt_last_subp.get_P_profit()
            f_Im_one[T - 1] = rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()

            q_ID_paths_n.append(q_ID_one)
            S_paths_n.append(S_one)
            f_P_paths_n.append(f_P_one)
            f_Im_paths_n.append(f_Im_one)
            eval_paths_n.append(float(f_scenario))

            P_DA_paths_n.append(P_DA_one)
            P_ID_paths_n.append(P_ID_one)
            delta_E_paths_n.append(delta_E_one)
            delta_C_paths_n.append(delta_C_one)

            f.append(f_scenario)

        q_ID_paths.append(q_ID_paths_n)
        S_paths.append(S_paths_n)
        f_P_paths.append(f_P_paths_n)
        f_Im_paths.append(f_Im_paths_n)
        eval_paths.append(eval_paths_n)

        P_DA_paths.append(P_DA_paths_n)
        P_ID_paths.append(P_ID_paths_n)
        delta_E_paths.append(delta_E_paths_n)
        delta_C_paths.append(delta_C_paths_n)

    eval_mean = float(np.mean(f))

    f_DA_mean = build_scalar_mean(f_DA)
    q_ID_mean = build_pathwise_mean(q_ID_paths)
    S_mean    = build_pathwise_mean(S_paths)
    f_P_mean  = build_pathwise_mean(f_P_paths)
    f_Im_mean = build_pathwise_mean(f_Im_paths)

    mode_name = "approx" if approx_mode else "exact"
    print(f"\nPSDDiP ({mode_name}, K={K}) -> SDDiP for bin_num = {bin_num}")
    print(f"Evaluation : {eval_mean}")

    return {
        "q_da": q_da,
        "f_DA": f_DA_mean,
        "q_ID": q_ID_mean,
        "S": S_mean,
        "f_P": f_P_mean,
        "f_Im": f_Im_mean,
        "eval": eval_mean,
        "f_DA_paths": f_DA,
        "q_ID_paths": q_ID_paths,
        "S_paths": S_paths,
        "f_P_paths": f_P_paths,
        "f_Im_paths": f_Im_paths,
        "eval_paths": eval_paths,
        "P_DA_paths": P_DA_paths,
        "P_ID_paths": P_ID_paths,
        "delta_E_paths": delta_E_paths,
        "delta_C_paths": delta_C_paths,
    }


# ============================================================
# 6. Run evaluations
# ============================================================

scenarios = scenarios_for_eval

sol_r = evaluation_rolling_rolling(scenarios)
sol_s2 = evaluation_2SP_rolling(scenarios, scenarios_for_SP)

sol_p_approx = []
sol_p_exact = []

for K in K_list:
    sol_p_approx.append(evaluation_psddip_sddip(K, scenarios, approx_mode=True))

for K in K_list:
    sol_p_exact.append(evaluation_psddip_sddip(K, scenarios, approx_mode=False))


# ============================================================
# 7. Save solutions
# ============================================================

def save_solutions_lp(
    bin_num,
    K_list,
    sol_r,
    sol_s2,
    sol_p_approx,
    runtime_p_approx,
    sol_p_exact,
    runtime_p_exact,
    filename=None
):
    SOL_DIR = os.path.join(os.path.dirname(__file__), "Solutions_LP")
    os.makedirs(SOL_DIR, exist_ok=True)

    if filename is None:
        filename = f"{bin_num}_solutions.npy"

    payload = {
        "meta": {
            "bin_num": bin_num,
            "K_list": list(K_list),
        },
        "Rolling -> Rolling": sol_r,
        "2-SP -> Rolling": sol_s2,
        "PSDDiP -> SDDiP": {
            "approx": {
                "q_da": [d["q_da"] for d in sol_p_approx],
                "f_DA": [d["f_DA"] for d in sol_p_approx],
                "q_ID": [d["q_ID"] for d in sol_p_approx],
                "S": [d["S"] for d in sol_p_approx],
                "f_P": [d["f_P"] for d in sol_p_approx],
                "f_Im": [d["f_Im"] for d in sol_p_approx],
                "eval": [d["eval"] for d in sol_p_approx],
                "f_DA_paths": [d["f_DA_paths"] for d in sol_p_approx],
                "q_ID_paths": [d["q_ID_paths"] for d in sol_p_approx],
                "S_paths": [d["S_paths"] for d in sol_p_approx],
                "f_P_paths": [d["f_P_paths"] for d in sol_p_approx],
                "f_Im_paths": [d["f_Im_paths"] for d in sol_p_approx],
                "eval_paths": [d["eval_paths"] for d in sol_p_approx],
                "P_DA_paths": [d["P_DA_paths"] for d in sol_p_approx],
                "P_ID_paths": [d["P_ID_paths"] for d in sol_p_approx],
                "delta_E_paths": [d["delta_E_paths"] for d in sol_p_approx],
                "delta_C_paths": [d["delta_C_paths"] for d in sol_p_approx],
                "runtime": runtime_p_approx,
            },
            "exact": {
                "q_da": [d["q_da"] for d in sol_p_exact],
                "f_DA": [d["f_DA"] for d in sol_p_exact],
                "q_ID": [d["q_ID"] for d in sol_p_exact],
                "S": [d["S"] for d in sol_p_exact],
                "f_P": [d["f_P"] for d in sol_p_exact],
                "f_Im": [d["f_Im"] for d in sol_p_exact],
                "eval": [d["eval"] for d in sol_p_exact],
                "f_DA_paths": [d["f_DA_paths"] for d in sol_p_exact],
                "q_ID_paths": [d["q_ID_paths"] for d in sol_p_exact],
                "S_paths": [d["S_paths"] for d in sol_p_exact],
                "f_P_paths": [d["f_P_paths"] for d in sol_p_exact],
                "f_Im_paths": [d["f_Im_paths"] for d in sol_p_exact],
                "eval_paths": [d["eval_paths"] for d in sol_p_exact],
                "P_DA_paths": [d["P_DA_paths"] for d in sol_p_exact],
                "P_ID_paths": [d["P_ID_paths"] for d in sol_p_exact],
                "delta_E_paths": [d["delta_E_paths"] for d in sol_p_exact],
                "delta_C_paths": [d["delta_C_paths"] for d in sol_p_exact],
                "runtime": runtime_p_exact,
            },
        },
    }

    save_path = os.path.join(SOL_DIR, filename)
    np.save(save_path, payload, allow_pickle=True)
    print(f"✅ Saved LP solutions to: {save_path}")


save_solutions_lp(
    bin_num=bin_num,
    K_list=K_list,
    sol_r=sol_r,
    sol_s2=sol_s2,
    sol_p_approx=sol_p_approx,
    runtime_p_approx=runtime_p_approx,
    sol_p_exact=sol_p_exact,
    runtime_p_exact=runtime_p_exact,
    filename=f"{bin_num}_solutions.npy"
)


# ============================================================
# 8. Notify done
# ============================================================

def notify_done_via_plot(title="✅ Evaluation finished", subtitle=None):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    ax.text(0.5, 0.65, title, ha="center", va="center",
            fontsize=20, weight="bold")

    if subtitle is not None:
        ax.text(0.5, 0.40, subtitle, ha="center", va="center",
                fontsize=12)

    try:
        fig.canvas.manager.set_window_title("Evaluation — Done")
    except Exception:
        pass

    plt.show()


notify_done_via_plot()
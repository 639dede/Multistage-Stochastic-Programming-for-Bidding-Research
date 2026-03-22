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

solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

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

# 1. Parameters & Computational settings

## Load Energy Forecast list

E_0_path_cloudy = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_cloudy.csv'
np.set_printoptions(suppress=True, precision=4)

E_0_cloudy = np.loadtxt(E_0_path_cloudy, delimiter=',')


E_0_path_normal = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_normal.csv'
np.set_printoptions(suppress=True, precision=4)

E_0_normal = np.loadtxt(E_0_path_normal, delimiter=',')


E_0_path_sunny = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_sunny.csv'
np.set_printoptions(suppress=True, precision=4)

E_0_sunny = np.loadtxt(E_0_path_sunny, delimiter=',')


if price_setting == 'cloudy':
    E_0 = E_0_cloudy    

elif price_setting == 'normal':
    E_0 = E_0_normal
    
else:
    E_0 = E_0_sunny


## 0. Load Price and Scenario csv files

P_r = 80
P_max = 200

E_0 = E_0

C = 20000
S = C
B = C/3

S_min = 0.1*S
S_max = 0.9*S

_price_re = re.compile(r'^K(\d+)\.csv$')        # matches K6.csv, K500.csv
_tree_re  = re.compile(r'^scenario_(\d+)\.csv$')# matches scenario_0.csv ...


bin_num = 50


def load_clustered_P_da(directory_path):
    """
    directory_path: './Stochastic_Approach/Scenarios/Reduced_data/P_da_<mode>'
    Returns:
      Reduced_P_da  -> list of (K,24) lists
      Reduced_Probs -> list of (K,) lists
    """
    # only price files (exclude *.probs.csv)
    names = [n for n in os.listdir(directory_path)
             if _price_re.match(n) and not n.endswith('.probs.csv')]
    # sort by K
    names.sort(key=lambda n: int(_price_re.match(n).group(1)))

    Reduced_P_da, Reduced_Probs = [], []

    for name in names:
        price_path = os.path.join(directory_path, name)
        P = np.loadtxt(price_path, delimiter=',')

        # shape to (K,24)
        if P.ndim == 1:
            P = P.reshape(1, -1)
        elif P.shape[1] != 24 and P.shape[0] == 24:
            P = P.T
        assert P.shape[1] == 24, f"{price_path} has shape {P.shape}; expected (K,24) or (24,K)"

        K = P.shape[0]

        # matching probs file: Kx.probs.csv
        probs_path = os.path.join(
            directory_path, name.replace('.csv', '.probs.csv')
        )
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
    base_dir: './Stochastic_Approach/Scenarios/Reduced_data/scenario_trees_<mode>'
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

            T = 24
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

T = 24

E_0_partial = E_0

P_da_eval = Reduced_P_da[-1]
Probs_eval = Reduced_Probs[-1]
Scenario_tree_eval = Reduced_scenario_trees[-1]

exp_P_da = Reduced_P_da[0][0]
scenario_exp = Reduced_scenario_trees[0]

def expectation_P_rt():
    
    exp_P_rt_list = []
    
    scenario_tree_rt = scenario_exp[0]
    
    for t in range(T):
        
        branches_t = scenario_tree_rt[t]
        
        exp_P_rt = 0
        
        for b in branches_t:   
            
            exp_P_rt += b[1]/len(branches_t)
        
        exp_P_rt_list.append(exp_P_rt)
                        
    return exp_P_rt_list 

def exp_P_rt_given_P_da(n, Scenario_tree_params):

    exp_P_rt_list = []
    
    scenario_tree_rt = Scenario_tree_params[n]
    
    for t in range(T):
        
        branches_t = scenario_tree_rt[t]
        
        exp_P_rt = 0
        
        for b in branches_t:   
            
            exp_P_rt += b[1]/len(branches_t)
        
        exp_P_rt_list.append(exp_P_rt)
                        
    return exp_P_rt_list

exp_P_rt_glob = expectation_P_rt()

## 1. Load evaluation/test scenario paths

K_eval = len(P_da_eval)

evaluation_num = 1

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


## 2. Load ECTG functions for DA stage

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


## 3. Load ECTG functions for ID stages

PSI_FULL_DIR = BASE_DIR / "psi_full_LP"
state_path = PSI_FULL_DIR / f"{bin_num}_state.npy"

if not state_path.exists():
    raise FileNotFoundError(f"Missing full checkpoint: {state_path}")

state = np.load(state_path, allow_pickle=True).item()

psi_ID = state["psi_ID"]        # this is the list you saved as model.psi



## 4. Evaluate

# Evaluation

def evaluation_rolling_rolling(scenarios):
    
    da_subp = rolling_da(exp_P_da, exp_P_rt_glob)
    da_state = da_subp.get_state_solutions()
                
    q_da = da_state[0]
                        
    f = []
    
    q_ID_list = [[0] * T for _ in range(K_eval)]
    S_list = [[0.5 * S] + [0] * T for _ in range(K_eval)]
    
    f_P  = [[0] * T for _ in range(K_eval)]
    f_Im = [[0] * T for _ in range(K_eval)]
    
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_eval[n]  
        exp_P_rt = exp_P_rt_given_P_da(n, Scenario_tree_eval)
                    
        rt_init_subp = rolling_rt_init(da_state, P_da, exp_P_rt)
        rt_init_state = rt_init_subp.get_state_solutions()       
        
        fcn_value = rt_init_subp.get_settlement_fcn_value()
        
        j = True
        
        for scenario in scenarios_n:
            
            state = rt_init_state
            f_scenario = fcn_value
            
            for t in range(T - 1):   # t = 0, ..., T-2
                                                    
                rt_subp = rolling_rt(t, state, P_da, exp_P_rt, scenario[t])
                
                if j:
                    q_ID_list[n][t] = rt_subp.get_ID_solution()
                    S_list[n][t + 1] = rt_subp.get_S_solution()
                    f_P[n][t] = rt_subp.get_P_profit()
                    f_Im[n][t] = rt_subp.get_Im_profit()
                
                state = rt_subp.get_state_solutions()
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            # t = T-1
            rt_last_subp = rolling_rt_last(state, P_da, scenario[T - 1])

            if j:
                q_ID_list[n][T - 1] = rt_last_subp.get_ID_solution()
                S_list[n][T] = rt_last_subp.get_S_solution()
                f_P[n][T - 1] = rt_last_subp.get_P_profit()
                f_Im[n][T - 1] = rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()
            f.append(f_scenario)
            
            j = False
        
    mu_hat = np.mean(f)
    eval = mu_hat

    print(f"\nRolling -> Rolling for bin_num = {bin_num}")
    print(f"Evaluation : {eval}")
    
    return q_da, q_ID_list, S_list, f_P, f_Im, eval


def evaluation_2SP_rolling(scenarios, scenarios_SP):

    exp_P_rt_each = [exp_P_rt_given_P_da(n, Scenario_tree_eval) for n in range(K_eval)]
    
    da_subp = two_stage_da(P_da_eval, exp_P_rt_each)
    da_state = da_subp.get_state_solutions()
                
    q_da = da_state[0]
                        
    f = []
    
    q_ID_list = [[0] * T for _ in range(K_eval)]
    S_list = [[0.5 * S] + [0] * T for _ in range(K_eval)]
    
    f_P  = [[0] * T for _ in range(K_eval)]
    f_Im = [[0] * T for _ in range(K_eval)]
    
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
        
        j = True
        
        for scenario in scenarios_n:
            
            state = rt_init_state
            f_scenario = fcn_value
            
            for t in range(T - 2):   # t = 0, ..., T-3
                                                    
                rt_subp = two_stage_rt(
                    t, state, P_da, ID_params_list[t + 1], exp_P_rt, scenario[t]
                )
                
                if j:
                    q_ID_list[n][t] = rt_subp.get_ID_solution()
                    S_list[n][t + 1] = rt_subp.get_S_solution()
                    f_P[n][t] = rt_subp.get_P_profit()
                    f_Im[n][t] = rt_subp.get_Im_profit()
                
                state = rt_subp.get_state_solutions()
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            # t = T-2
            rt_beforelast_subp = two_stage_rt_beforelast(
                state, P_da, ID_params_list[T - 1], scenario[T - 2]
            )
                
            if j:
                q_ID_list[n][T - 2] = rt_beforelast_subp.get_ID_solution()
                S_list[n][T - 1] = rt_beforelast_subp.get_S_solution()
                f_P[n][T - 2] = rt_beforelast_subp.get_P_profit()
                f_Im[n][T - 2] = rt_beforelast_subp.get_Im_profit()

            state = rt_beforelast_subp.get_state_solutions()
            f_scenario += rt_beforelast_subp.get_settlement_fcn_value()
            
            # t = T-1
            rt_last_subp = two_stage_rt_last(
                state, P_da, scenario[T - 1]
            )

            if j:
                q_ID_list[n][T - 1] = rt_last_subp.get_ID_solution()
                S_list[n][T] = rt_last_subp.get_S_solution()
                f_P[n][T - 1] = rt_last_subp.get_P_profit()
                f_Im[n][T - 1] = rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()
            f.append(f_scenario)
            
            j = False
        
    mu_hat = np.mean(f)
    eval = mu_hat

    print(f"\n2-SP -> Rolling for bin_num = {bin_num}")
    print(f"Evaluation : {eval}")
    
    return q_da, q_ID_list, S_list, f_P, f_Im, eval


"""
def evaluation_rolling_sddip(scenarios):
        
    da_subp = rolling_da(exp_P_da, exp_P_rt_glob)
    
    da_state = da_subp.get_state_solutions()
                
    q_da = da_state[0]
                        
    f = []
    
    f_DA = [0]*T
        
    q_ID_list = [[0]*T for _ in range(K_eval)]
    S_list = [[0.5*S]+[0]*T for _ in range(K_eval)]
    
    f_P  = [[0]*T for _ in range(K_eval)]
    f_Im = [[0]*T for _ in range(K_eval)]
    
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_eval[n]
                
        rt_init_subp = fw_rt_init(da_state, psi_ID[n][0], P_da)
        rt_init_state = rt_init_subp.get_state_solutions()
        
        f_DA_list = rt_init_subp.get_DA_profit()
        
        for i in range(T):
            f_DA[i] += f_DA_list[i]/K_eval
        
        fcn_value = rt_init_subp.get_settlement_fcn_value()
        
        j = True
            
        for scenario in scenarios_n:
            
            state = rt_init_state
            
            f_scenario = fcn_value
            
            for t in range(T - 1): ## t = 0, ..., T-2
                
                rt_subp = fw_rt(t, state, psi_ID[n][t+1], P_da, scenario[t])
                
                if j:
                    q_ID_list[n][t] = rt_subp.get_ID_solution()
                    S_list[n][t+1] = rt_subp.get_S_solution()
                    f_P[n][t] = rt_subp.get_P_profit()
                    f_Im[n][t] = rt_subp.get_Im_profit()
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T-1])

            if j:
                q_ID_list[n][T-1] = rt_last_subp.get_ID_solution()
                S_list[n][T] = rt_last_subp.get_S_solution()
                f_P[n][T-1] += rt_last_subp.get_P_profit()
                f_Im[n][T-1] += rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()
                        
            f.append(f_scenario)
            
            j = False
        
    mu_hat = np.mean(f)
    
    sigma_hat = np.std(f, ddof=1)  

    z_alpha_half = 1.96  
    
    eval = mu_hat 
    
    print(f"\nRolling Horizon -> SDDiP for price setting = {bin_num}")
    print(f"Evaluation : {eval}")
    
    return q_da, q_ID_list, S_list, f_P, f_Im, eval


def evaluation_SP_sddip(stage_num, scenarios, scenarios_SP):
        
    if stage_num == 2:
        
        exp_P_rt_each = [exp_P_rt_given_P_da(n, Scenario_tree_eval) for n in range(K_eval)]
        
        da_subp = two_stage_da(P_da_eval, exp_P_rt_each)
        
    elif stage_num == 3:
        
        da_subp = three_stage_da(P_da_eval, scenarios_SP)  
          
    da_state = da_subp.get_state_solutions()
                
    q_da = da_state[0]
                        
    f = []
    
    f_DA = [0]*T
        
    q_ID_list = [[0]*T for _ in range(K_eval)]
    S_list = [[0.5*S]+[0]*T for _ in range(K_eval)]
    
    f_P  = [[0]*T for _ in range(K_eval)]
    f_Im = [[0]*T for _ in range(K_eval)]
        
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_eval[n]
                
        rt_init_subp = fw_rt_init(da_state, psi_ID[n][0], P_da)
        rt_init_state = rt_init_subp.get_state_solutions()
        
        f_DA_list = rt_init_subp.get_DA_profit()
        
        for i in range(T):
            f_DA[i] += f_DA_list[i]/K_eval
        
        fcn_value = rt_init_subp.get_settlement_fcn_value()
        
        j = True
            
        for scenario in scenarios_n:
            
            state = rt_init_state
            
            f_scenario = fcn_value
            
            for t in range(T - 1): ## t = 0, ..., T-2
                
                rt_subp = fw_rt(t, state, psi_ID[n][t+1], P_da, scenario[t])
                
                if j:
                    q_ID_list[n][t] = rt_subp.get_ID_solution()
                    S_list[n][t+1] = rt_subp.get_S_solution()
                    f_P[n][t] = rt_subp.get_P_profit()
                    f_Im[n][t] = rt_subp.get_Im_profit()
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T-1])

            if j:
                q_ID_list[n][T-1] = rt_last_subp.get_ID_solution()
                S_list[n][T] = rt_last_subp.get_S_solution()
                f_P[n][T-1] += rt_last_subp.get_P_profit()
                f_Im[n][T-1] += rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()
                        
            f.append(f_scenario)
            
            j = False
        
    mu_hat = np.mean(f)
    
    sigma_hat = np.std(f, ddof=1)  

    z_alpha_half = 1.96  
    
    eval = mu_hat 
    
    print(f"\n{stage_num}-SP -> SDDiP for price setting = {bin_num}")
    print(f"Evaluation : {eval}")
    
    return q_da, q_ID_list, S_list, f_P, f_Im, eval
"""
        
def evaluation_psddip_sddip(K, scenarios, approx_mode):
    
    k_idx = K_list.index(K)
    
    if approx_mode:
        psi_DA_list = psi_DA_approx_list
    else:
        psi_DA_list = psi_DA_exact_list

    da_subp = fw_da(psi_DA_list[k_idx])
    da_state = da_subp.get_state_solutions()
    
    q_da = da_state[0]
    
    f = []
        
    q_ID_list = [[0] * T for _ in range(K_eval)]
    S_list = [[0.5 * S] + [0] * T for _ in range(K_eval)]
    
    f_P  = [[0] * T for _ in range(K_eval)]
    f_Im = [[0] * T for _ in range(K_eval)]
    
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_eval[n]
                
        rt_init_subp = fw_rt_init(da_state, psi_ID[n][0], P_da)
        rt_init_state = rt_init_subp.get_state_solutions()
        
        fcn_value = rt_init_subp.get_settlement_fcn_value()
        
        j = True
            
        for scenario in scenarios_n:
            
            state = rt_init_state
            f_scenario = fcn_value
            
            for t in range(T - 1):
                
                rt_subp = fw_rt(t, state, psi_ID[n][t + 1], P_da, scenario[t])
                
                if j:
                    q_ID_list[n][t] = rt_subp.get_ID_solution()
                    S_list[n][t + 1] = rt_subp.get_S_solution()
                    f_P[n][t] = rt_subp.get_P_profit()
                    f_Im[n][t] = rt_subp.get_Im_profit()
                
                state = rt_subp.get_state_solutions()
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T - 1])

            if j:
                q_ID_list[n][T - 1] = rt_last_subp.get_ID_solution()
                S_list[n][T] = rt_last_subp.get_S_solution()
                f_P[n][T - 1] = rt_last_subp.get_P_profit()
                f_Im[n][T - 1] = rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()
            f.append(f_scenario)
            
            j = False
        
    mu_hat = np.mean(f)
    eval = mu_hat
    
    mode_name = "approx" if approx_mode else "exact"
    print(f"\nPSDDiP ({mode_name}, K={K}) -> SDDiP for bin_num = {bin_num}")
    print(f"Evaluation : {eval}")
    
    return q_da, q_ID_list, S_list, f_P, f_Im, eval


scenarios = scenarios_for_eval

q_da_r, q_ID_r, S_r, f_P_r, f_Im_r, eval_r = evaluation_rolling_rolling(scenarios)
q_da_s2, q_ID_s2, S_s2, f_P_s2, f_Im_s2, eval_s2 = evaluation_2SP_rolling(
    scenarios, scenarios_for_SP
)

q_da_p_approx = []
q_ID_p_approx = []
S_p_approx    = []
f_P_p_approx  = []
f_Im_p_approx = []
eval_p_approx = []

q_da_p_exact = []
q_ID_p_exact = []
S_p_exact    = []
f_P_p_exact  = []
f_Im_p_exact = []
eval_p_exact = []

for K in K_list:
    q_da, q_ID, S_sol, f_P, f_Im, eval = evaluation_psddip_sddip(
        K, scenarios, approx_mode=True
    )
    q_da_p_approx.append(q_da)
    q_ID_p_approx.append(q_ID)
    S_p_approx.append(S_sol)
    f_P_p_approx.append(f_P)
    f_Im_p_approx.append(f_Im)
    eval_p_approx.append(eval)

for K in K_list:
    q_da, q_ID, S_sol, f_P, f_Im, eval = evaluation_psddip_sddip(
        K, scenarios, approx_mode=False
    )
    q_da_p_exact.append(q_da)
    q_ID_p_exact.append(q_ID)
    S_p_exact.append(S_sol)
    f_P_p_exact.append(f_P)
    f_Im_p_exact.append(f_Im)
    eval_p_exact.append(eval)


## 5. Save solutions

def save_solutions_lp(
    bin_num,
    K_list,
    q_da_r, q_ID_r, S_r, f_P_r, f_Im_r, eval_r,
    q_da_s2, q_ID_s2, S_s2, f_P_s2, f_Im_s2, eval_s2,
    q_da_p_approx, q_ID_p_approx, S_p_approx, f_P_p_approx, f_Im_p_approx, eval_p_approx, runtime_p_approx,
    q_da_p_exact, q_ID_p_exact, S_p_exact, f_P_p_exact, f_Im_p_exact, eval_p_exact, runtime_p_exact,
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
        "Rolling -> Rolling": {
            "q_da": q_da_r,
            "q_ID": q_ID_r,
            "S": S_r,
            "f_P": f_P_r,
            "f_Im": f_Im_r,
            "eval": eval_r,
        },
        "2-SP -> Rolling": {
            "q_da": q_da_s2,
            "q_ID": q_ID_s2,
            "S": S_s2,
            "f_P": f_P_s2,
            "f_Im": f_Im_s2,
            "eval": eval_s2,
        },
        "PSDDiP -> SDDiP": {
            "approx": {
                "q_da": q_da_p_approx,
                "q_ID": q_ID_p_approx,
                "S": S_p_approx,
                "f_P": f_P_p_approx,
                "f_Im": f_Im_p_approx,
                "eval": eval_p_approx,
                "runtime": runtime_p_approx,
            },
            "exact": {
                "q_da": q_da_p_exact,
                "q_ID": q_ID_p_exact,
                "S": S_p_exact,
                "f_P": f_P_p_exact,
                "f_Im": f_Im_p_exact,
                "eval": eval_p_exact,
                "runtime": runtime_p_exact,
            },
        },
    }

    save_path = os.path.join(SOL_DIR, filename)
    np.save(save_path, payload, allow_pickle=True)
    print(f"✅ Saved LP solutions to: {save_path}")


# ---- call it once after evaluation is finished ----

runtime_psddip = load_psddip_runtime_lp(bin_num=bin_num, K_list=K_list)
runtime_p_approx = runtime_psddip["approx"]
runtime_p_exact = runtime_psddip["exact"]

save_solutions_lp(
    bin_num=bin_num,
    K_list=K_list,

    q_da_r=q_da_r, q_ID_r=q_ID_r, S_r=S_r, f_P_r=f_P_r, f_Im_r=f_Im_r, eval_r=eval_r,
    q_da_s2=q_da_s2, q_ID_s2=q_ID_s2, S_s2=S_s2, f_P_s2=f_P_s2, f_Im_s2=f_Im_s2, eval_s2=eval_s2,

    q_da_p_approx=q_da_p_approx, q_ID_p_approx=q_ID_p_approx, S_p_approx=S_p_approx,
    f_P_p_approx=f_P_p_approx, f_Im_p_approx=f_Im_p_approx, eval_p_approx=eval_p_approx,
    runtime_p_approx=runtime_p_approx,

    q_da_p_exact=q_da_p_exact, q_ID_p_exact=q_ID_p_exact, S_p_exact=S_p_exact,
    f_P_p_exact=f_P_p_exact, f_Im_p_exact=f_Im_p_exact, eval_p_exact=eval_p_exact,
    runtime_p_exact=runtime_p_exact,

    filename=f"{bin_num}_solutions.npy"
)
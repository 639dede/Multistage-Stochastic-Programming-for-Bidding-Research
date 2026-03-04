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


price_setting = 'cloudy'  # 'cloudy', 'normal', 'sunny'

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

C = 21022.1
S = C
B = C/3

S_min = 0.1*S
S_max = 0.9*S

P_r = 80
P_max = 200

_price_re = re.compile(r'^K(\d+)\.csv$')        # matches K6.csv, K500.csv
_tree_re  = re.compile(r'^scenario_(\d+)\.csv$')# matches scenario_0.csv ...


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


cluster_dir = f'./Stochastic_Approach/Scenarios/Reduced_data/P_da_{price_setting}'
Reduced_P_da, Reduced_Probs = load_clustered_P_da(cluster_dir)


clustered_tree_dir = f'./Stochastic_Approach/Scenarios/Reduced_data/scenario_trees_{price_setting}'
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

evaluation_num = 10

BASE_DIR = Path(__file__).resolve().parent
SCEN_ROOT = BASE_DIR / "scenario_paths" / price_setting

scenarios_for_eval = np.load(
    SCEN_ROOT / "scenarios_eval.npy",
    allow_pickle=True
).tolist()

scenarios_for_SP = np.load(
    SCEN_ROOT / "scenarios_SP.npy",
    allow_pickle=True
).tolist()


## 2. Load ECTG functions for DA stage

PSI_DA_DIR = BASE_DIR / "psi_DA_LP" / price_setting

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


## 3. Load ECTG functions for ID stages

PSI_FULL_DIR = BASE_DIR / "psi_full_LP"
state_path = PSI_FULL_DIR / f"{price_setting}_state.npy"

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
    
    f_DA = [0]*T
    f_P  = [0]*T
    f_E  = [0]*T
    f_Im = [0]*T
    
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_eval[n]  
                    
        exp_P_rt = exp_P_rt_given_P_da(n, Scenario_tree_eval)
                    
        rt_init_subp = rolling_rt_init(da_state, P_da, exp_P_rt)
        rt_init_state = rt_init_subp.get_state_solutions()       
        
        f_DA_list = rt_init_subp.get_DA_profit()
        
        for i in range(T):
            f_DA[i] += f_DA_list[i]/K_eval
        
        fcn_value = rt_init_subp.get_settlement_fcn_value()
        
        for scenario in scenarios_n:
            
            state = rt_init_state
            
            f_scenario = fcn_value
            
            for t in range(T - 1): ## t = 0, ..., T-2
                                    
                rt_subp = rolling_rt(t, state, P_da, exp_P_rt, scenario[t])
                
                f_P[t] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
                f_Im[t] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = rolling_rt_last(state, P_da, scenario[T-1])

            f_P[T-1] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
            f_Im[T-1] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)

            f_scenario += rt_last_subp.get_settlement_fcn_value()
                        
            f.append(f_scenario)
        
    mu_hat = np.mean(f)
    
    sigma_hat = np.std(f, ddof=1)  

    z_alpha_half = 1.96  
    
    eval = mu_hat 

    print(f"\nRolling Horizon -> Rolling Horizon for price setting = {price_setting}")
    print(f"Evaluation : {eval}")
    
    return q_da, eval


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
    
    print(f"\nRolling Horizon -> SDDiP for price setting = {price_setting}")
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
    
    print(f"\n{stage_num}-SP -> SDDiP for price setting = {price_setting}")
    print(f"Evaluation : {eval}")
    
    return q_da, q_ID_list, S_list, f_P, f_Im, eval
    
        
def evaluation_psddip_sddip(K, scenarios, approx_mode):
    
    k_idx = K_list.index(K)
    
    if approx_mode:
        psi_DA_list = psi_DA_approx_list
        DA_probs = Probs_eval
    
    else:
        psi_DA_list = psi_DA_exact_list
        DA_probs = Reduced_Probs[k_idx]

    da_subp = fw_da(DA_probs, psi_DA_list[k_idx])
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
    
    print(f"\nPSDDiP K = {K} -> SDDiP for price setting = {price_setting}")
    print(f"Evaluation : {eval}")
    
    return q_da, q_ID_list, S_list, f_P, f_Im, eval
    

scenarios = scenarios_for_eval


#%% Cut mode index: 'SB' -> 0, 'L-sub' -> 1

evaluation_rolling_rolling(scenarios)
q_da_r, q_ID_r, S_r, f_P_r, f_Im_r, eval_r = evaluation_rolling_sddip(scenarios)
q_da_s2, q_ID_s2, S_s2, f_P_s2, f_Im_s2, eval_s2 = evaluation_SP_sddip(2, scenarios, scenarios_for_SP)  
q_da_s3, q_ID_s3, S_s3, f_P_s3, f_Im_s3, eval_s3 = evaluation_SP_sddip(3, scenarios, scenarios_for_SP)
q_da_p_approx = []
q_ID_p_approx = []
S_p_approx    = []
f_P_p_approx  = []
f_E_p_approx  = []
f_Im_p_approx = []
eval_p_approx = []
q_da_p_exact = []
q_ID_p_exact = []
S_p_exact    = []
f_P_p_exact  = []
f_E_p_exact  = []
f_Im_p_exact = []
eval_p_exact = []


for k_idx, K in enumerate(K_list):

    q_da, q_ID, S_sol, f_P, f_Im, eval = evaluation_psddip_sddip(
        K, scenarios, approx_mode=True
    )

    q_da_p_approx.append(q_da)
    q_ID_p_approx.append(q_ID)
    S_p_approx.append(S_sol)
    f_P_p_approx.append(f_P)
    f_Im_p_approx.append(f_Im)
    eval_p_approx.append(eval)

for k_idx, K in enumerate(K_list):

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
    price_setting,
    K_list,
    q_da_r, q_ID_r, S_r, f_P_r, f_Im_r, eval_r,
    q_da_s2, q_ID_s2, S_s2, f_P_s2, f_Im_s2, eval_s2,
    q_da_s3, q_ID_s3, S_s3, f_P_s3, f_Im_s3, eval_s3,
    q_da_p_approx, q_ID_p_approx, S_p_approx, f_P_p_approx, f_Im_p_approx, eval_p_approx,
    q_da_p_exact, q_ID_p_exact, S_p_exact, f_P_p_exact, f_Im_p_exact, eval_p_exact,
    filename=None
):
    """
    Saves a single npy dict into the same folder as this file's sibling: NestedBenders/Solutions_LP/
    File: <price_setting>_solutions.npy by default.
    """

    # directory: .../NestedBenders/Solutions_LP
    SOL_DIR = os.path.join(os.path.dirname(__file__), "Solutions_LP")
    os.makedirs(SOL_DIR, exist_ok=True)

    if filename is None:
        filename = f"{price_setting}_solutions.npy"

    payload = {
        "meta": {
            "price_setting": price_setting,
            "K_list": list(K_list),
        },
        # Baselines
        "Rolling → SDDiP": {"q_da": q_da_r,  "q_ID": q_ID_r,  "S": S_r,  "f_P": f_P_r,  "f_Im": f_Im_r, "eval": eval_r},
        "2-SP → SDDiP":    {"q_da": q_da_s2, "q_ID": q_ID_s2, "S": S_s2, "f_P": f_P_s2, "f_Im": f_Im_s2, "eval": eval_s2},
        "3-SP → SDDiP":    {"q_da": q_da_s3, "q_ID": q_ID_s3, "S": S_s3, "f_P": f_P_s3, "f_Im": f_Im_s3, "eval": eval_s3},

        # PSDDiP (LP) by K (both approx + exact)
        "PSDDiP": {
            "approx": {
                "q_da": q_da_p_approx,
                "q_ID": q_ID_p_approx,
                "S":    S_p_approx,
                "f_P":  f_P_p_approx,
                "f_Im": f_Im_p_approx,
                "eval": eval_p_approx,
            },
            "exact": {
                "q_da": q_da_p_exact,
                "q_ID": q_ID_p_exact,
                "S":    S_p_exact,
                "f_P":  f_P_p_exact,
                "f_Im": f_Im_p_exact,
                "eval": eval_p_exact,
            },
        },
    }

    save_path = os.path.join(SOL_DIR, filename)
    np.save(save_path, payload, allow_pickle=True)
    print(f"✅ Saved LP solutions to: {save_path}")

# ---- call it once after evaluation is finished ----

save_solutions_lp(
    price_setting=price_setting,
    K_list=K_list,

    q_da_r=q_da_r, q_ID_r=q_ID_r, S_r=S_r, f_P_r=f_P_r, f_Im_r=f_Im_r, eval_r=eval_r,
    q_da_s2=q_da_s2, q_ID_s2=q_ID_s2, S_s2=S_s2, f_P_s2=f_P_s2, f_Im_s2=f_Im_s2, eval_s2=eval_s2,
    q_da_s3=q_da_s3, q_ID_s3=q_ID_s3, S_s3=S_s3, f_P_s3=f_P_s3, f_Im_s3=f_Im_s3, eval_s3=eval_s3,

    q_da_p_approx=q_da_p_approx, q_ID_p_approx=q_ID_p_approx, S_p_approx=S_p_approx,
    f_P_p_approx=f_P_p_approx, f_Im_p_approx=f_Im_p_approx, eval_p_approx=eval_p_approx,

    q_da_p_exact=q_da_p_exact, q_ID_p_exact=q_ID_p_exact, S_p_exact=S_p_exact,
    f_P_p_exact=f_P_p_exact, f_Im_p_exact=f_Im_p_exact, eval_p_exact=eval_p_exact,

    filename=f"{price_setting}_solutions.npy"
)
import os
import signal, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyomo.environ as pyo
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


# Load Energy Forecast list

E_0_path_cloudy = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_cloudy.csv'
np.set_printoptions(suppress=True, precision=4)

E_0_cloudy = np.loadtxt(E_0_path_cloudy, delimiter=',')


E_0_path_normal = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_normal.csv'
np.set_printoptions(suppress=True, precision=4)

E_0_normal = np.loadtxt(E_0_path_normal, delimiter=',')


E_0_path_sunny = './Stochastic_Approach/Scenarios/Energy_forecast/E_0_sunny.csv'
np.set_printoptions(suppress=True, precision=4)

E_0_sunny = np.loadtxt(E_0_path_sunny, delimiter=',')


E_0 = E_0_normal


## 0. Load Price and Scenario csv files

C = 21022.1
S = C
B = C/3

S_min = 0.1*S
S_max = 0.9*S

P_r = 80
P_max = 270

K_list = [1, 3, 6, 10, 15]
cut_list = ['SB', 'L-sub']

price_setting = 'sunny'  # 'cloudy', 'normal', 'sunny'
time_limit = '7200' # '3600', '7200', '10800' 

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

P_da_test = Reduced_P_da[-1]
Probs_test = Reduced_Probs[-1]
Scenario_tree_test = Reduced_scenario_trees[-1]

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

scenarios_for_test = np.load(
    SCEN_ROOT / "scenarios_test.npy",
    allow_pickle=True
).tolist()

scenarios_for_SP = np.load(
    SCEN_ROOT / "scenarios_SP.npy",
    allow_pickle=True
).tolist()


## 2. Load ECTG functions for DA stage

psi_DA_list = []

for cut_mode in cut_list:

    PSI_DA_DIR = BASE_DIR / "psi_DA" / price_setting / cut_mode / time_limit

    psi_DA = []

    for K in K_list:
        psi_DA_path = PSI_DA_DIR / f"psi_DA_{K}.npy"
        if not psi_DA_path.exists():
            raise FileNotFoundError(f"Missing psi_DA file: {psi_DA_path}")
        psi_DA.append(np.load(psi_DA_path, allow_pickle=True).tolist())

    psi_DA_list.append(psi_DA)


## 3. Load ECTG functions for ID stages

PSI_FULL_DIR = BASE_DIR / "psi_full"
state_path = PSI_FULL_DIR / f"{price_setting}_state.npy"

if not state_path.exists():
    raise FileNotFoundError(f"Missing full checkpoint: {state_path}")

state = np.load(state_path, allow_pickle=True).item()

psi_ID = state["psi_ID"]        # this is the list you saved as model.psi


## 4. Evaluate

from NestedBenders.PSDDiP import (
    fw_da,
    fw_rt_init,
    fw_rt,
    fw_rt_last,
    rolling_da,
    rolling_rt_init,
    rolling_rt,
    rolling_rt_last,
    two_stage_da,
    three_stage_da
    )


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
                    
        P_da = P_da_test[n]  
                    
        exp_P_rt = exp_P_rt_given_P_da(n, Scenario_tree_test)
                    
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
                f_E[t] += rt_subp.get_E_profit()/(K_eval*evaluation_num)
                f_Im[t] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = rolling_rt_last(state, P_da, scenario[T-1])

            f_P[T-1] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
            f_E[T-1] += rt_subp.get_E_profit()/(K_eval*evaluation_num)
            f_Im[T-1] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)

            f_scenario += rt_last_subp.get_settlement_fcn_value()
                        
            f.append(f_scenario)
        
    mu_hat = np.mean(f)
    
    sigma_hat = np.std(f, ddof=1)  

    z_alpha_half = 1.96  
    
    eval = mu_hat 

    print(f"\nRolling Horizon -> Rolling Horizon for price setting = {price_setting}")
    print(f"Evaluation : {eval}")
    
    return q_da


def evaluation_rolling_sddip(scenarios):
        
    da_subp = rolling_da(exp_P_da, exp_P_rt_glob)
    
    da_state = da_subp.get_state_solutions()
                
    q_da = da_state[0]
                        
    f = []
    
    f_DA = [0]*T
        
    q_ID_list = [[0]*T for _ in range(K_eval)]
    S_list = [[0.5*S]+[0]*T for _ in range(K_eval)]
    
    f_P  = [[0]*T for _ in range(K_eval)]
    f_E  = [[0]*T for _ in range(K_eval)]
    f_Im = [[0]*T for _ in range(K_eval)]
    
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_test[n]
                
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
                    f_E[n][t] = rt_subp.get_E_profit()
                    f_Im[n][t] = rt_subp.get_Im_profit()
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T-1])

            if j:
                q_ID_list[n][T-1] = rt_last_subp.get_ID_solution()
                S_list[n][T] = rt_last_subp.get_S_solution()
                f_P[n][T-1] += rt_last_subp.get_P_profit()
                f_E[n][T-1] += rt_last_subp.get_E_profit()
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
    
    return q_da, q_ID_list, S_list, f_P, f_E, f_Im


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
    f_E  = [[0]*T for _ in range(K_eval)]
    f_Im = [[0]*T for _ in range(K_eval)]
        
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_test[n]
                
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
                    f_E[n][t] = rt_subp.get_E_profit()
                    f_Im[n][t] = rt_subp.get_Im_profit()
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T-1])

            if j:
                q_ID_list[n][T-1] = rt_last_subp.get_ID_solution()
                S_list[n][T] = rt_last_subp.get_S_solution()
                f_P[n][T-1] += rt_last_subp.get_P_profit()
                f_E[n][T-1] += rt_last_subp.get_E_profit()
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
    
    return q_da, q_ID_list, S_list, f_P, f_E, f_Im
    
        
def evaluation_psddip_sddip(K, cut_mode, scenarios):
    
    k_idx = K_list.index(K)
        
    da_subp = fw_da(Reduced_Probs[k_idx], psi_DA_list[cut_mode][k_idx])
    da_state = da_subp.get_state_solutions()
    
    q_da = da_state[0]
    
    f = []
    
    f_DA = [0]*T
        
    q_ID_list = [[0]*T for _ in range(K_eval)]
    S_list = [[0.5*S]+[0]*T for _ in range(K_eval)]
    
    f_P  = [[0]*T for _ in range(K_eval)]
    f_E  = [[0]*T for _ in range(K_eval)]
    f_Im = [[0]*T for _ in range(K_eval)]
    
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_test[n]
                
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
                    f_E[n][t] = rt_subp.get_E_profit()
                    f_Im[n][t] = rt_subp.get_Im_profit()
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T-1])

            if j:
                q_ID_list[n][T-1] = rt_last_subp.get_ID_solution()
                S_list[n][T] = rt_last_subp.get_S_solution()
                f_P[n][T-1] += rt_last_subp.get_P_profit()
                f_E[n][T-1] += rt_last_subp.get_E_profit()
                f_Im[n][T-1] += rt_last_subp.get_Im_profit()

            f_scenario += rt_last_subp.get_settlement_fcn_value()
                        
            f.append(f_scenario)
            
            j = False
        
    mu_hat = np.mean(f)
    
    sigma_hat = np.std(f, ddof=1)  

    z_alpha_half = 1.96  
    
    eval = mu_hat 
    
    print(f"\nPSDDiP K = {K}, time_lim = {time_limit}, cut = {cut_mode} -> SDDiP for price setting = {price_setting}")
    print(f"Evaluation : {eval}")
    
    return q_da, q_ID_list, S_list, f_P, f_E, f_Im
    

scenarios = scenarios_for_test


#%% Cut mode index: 'SB' -> 0, 'L-sub' -> 1

evaluation_rolling_rolling(scenarios)
q_da_r, q_ID_r, S_r, f_P_r, f_E_r, f_Im_r = evaluation_rolling_sddip(scenarios)
q_da_s2, q_ID_s2, S_s2, f_P_s2, f_E_s2, f_Im_s2 = evaluation_SP_sddip(2, scenarios, scenarios_for_SP)  
q_da_s3, q_ID_s3, S_s3, f_P_s3, f_E_s3, f_Im_s3 = evaluation_SP_sddip(3, scenarios, scenarios_for_SP)
q_da_p = [[] for _ in K_list]
q_ID_p = [[] for _ in K_list]
S_p    = [[] for _ in K_list]
f_P_p  = [[] for _ in K_list]
f_E_p  = [[] for _ in K_list]
f_Im_p = [[] for _ in K_list]


for k_idx, K in enumerate(K_list):
    for c_idx, cut_index in enumerate(range(len(cut_list))):

        q_da, q_ID, S_sol, f_P, f_E, f_Im = evaluation_psddip_sddip(
            K, cut_index, scenarios
        )

        q_da_p[k_idx].append(q_da)
        q_ID_p[k_idx].append(q_ID)
        S_p[k_idx].append(S_sol)
        f_P_p[k_idx].append(f_P)
        f_E_p[k_idx].append(f_E)
        f_Im_p[k_idx].append(f_Im)


## 5. Plotting solutions

### 1) Helpers for DA solutions

def plot_q_da_single(name, q_da, xlabel="Hour", ylabel="Value",
                     ylim=(-1000, 32000), figsize=(7, 4), pause=False):
    y = np.asarray(q_da).reshape(-1)
    if len(y) != 24:
        print(f"Warning: {name} has length {len(y)} (expected 24).")
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

def ensure_solution_dirs(base_dir="solutions", price_settings=("cloudy", "normal", "sunny")):
    
    os.makedirs(base_dir, exist_ok=True)
    for ps in price_settings:
        os.makedirs(os.path.join(base_dir, ps), exist_ok=True)

def save_q_da_solutions(q_da_list, price_setting, base_dir="solutions", filename="q_da_list.npy"):
    
    save_dir = os.path.join(base_dir, price_setting)
    os.makedirs(save_dir, exist_ok=True)

    q_da_dict = {name: np.asarray(q_da).reshape(-1) for name, q_da in q_da_list}
    save_path = os.path.join(save_dir, filename)

    np.save(save_path, q_da_dict, allow_pickle=True)
    print(f"✅ Saved q_da solutions to: {save_path}")

def load_q_da_solutions(price_setting, base_dir="solutions", filename="q_da_list.npy"):

    load_path = os.path.join(base_dir, price_setting, filename)
    q_da_dict = np.load(load_path, allow_pickle=True).item()
    return q_da_dict

q_da_list = [
    ("Rolling → SDDiP", q_da_r),
    ("2-SP → SDDiP",    q_da_s2),
    ("3-SP → SDDiP",    q_da_s3),
]

for k_idx, K in enumerate(K_list):
    for c_idx, cut in enumerate(cut_list):
        name = f"PSDDiP (K={K}, {cut})"
        q_da_list.append((name, q_da_p[k_idx][c_idx]))


ensure_solution_dirs(base_dir="solutions", price_settings=("cloudy", "normal", "sunny"))

save_q_da_solutions(q_da_list, price_setting=price_setting, base_dir="solutions")


### 2) Helpers for ID solutions

#### q_ID

def ensure_solution_dirs_ID(base_dir="solutions_ID", price_settings=("cloudy", "normal", "sunny")):
    os.makedirs(base_dir, exist_ok=True)
    for ps in price_settings:
        os.makedirs(os.path.join(base_dir, ps), exist_ok=True)

def save_q_ID_solutions(q_ID_list, price_setting, base_dir="solutions_ID", filename="q_ID_list.npy"):
    """
    q_ID_list: list of (name, q_ID_matrix) pairs
      - q_ID_matrix is expected shape (K_eval, T) or list-of-lists [[T]*K_eval]
    Saves as dict[name] = np.ndarray of shape (K_eval, T)
    """
    save_dir = os.path.join(base_dir, price_setting)
    os.makedirs(save_dir, exist_ok=True)

    q_ID_dict = {name: np.asarray(q_ID).reshape(np.asarray(q_ID).shape[0], -1) for name, q_ID in q_ID_list}
    save_path = os.path.join(save_dir, filename)

    np.save(save_path, q_ID_dict, allow_pickle=True)
    print(f"✅ Saved q_ID solutions to: {save_path}")

def load_q_ID_solutions(price_setting, base_dir="solutions_ID", filename="q_ID_list.npy"):
    load_path = os.path.join(base_dir, price_setting, filename)
    q_ID_dict = np.load(load_path, allow_pickle=True).item()
    return q_ID_dict

def plot_q_ID_density_single(
    name,
    q_ID_matrix,
    xlabel="Hour",
    ylabel="q_ID",
    ylim=None,
    figsize=(8, 4),
    alpha=0.25,
    pause=False,
    save_path=None,
    show=True,
    overlay_mean=False
):
    """
    Density plot: one thin line per evaluation scenario.
    q_ID_matrix: array-like (K_eval, T)
    """
    q_ID_arr = np.asarray(q_ID_matrix)
    if q_ID_arr.ndim != 2:
        raise ValueError(f"{name}: q_ID_matrix must be 2D (K_eval, T). Got shape {q_ID_arr.shape}")

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

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if pause:
        input("Press Enter for next plot...")

    return fig, ax

q_ID_list = [
    ("Rolling → SDDiP", q_ID_r),
    ("2-SP → SDDiP",    q_ID_s2),
    ("3-SP → SDDiP",    q_ID_s3),
]

for k_idx, K in enumerate(K_list):
    for c_idx, cut in enumerate(cut_list):
        name = f"PSDDiP (K={K}, {cut})"
        q_ID_list.append((name, q_ID_p[k_idx][c_idx]))

#### SoC values

def ensure_solution_dirs_S(base_dir="solutions_S", price_settings=("cloudy", "normal", "sunny")):
    os.makedirs(base_dir, exist_ok=True)
    for ps in price_settings:
        os.makedirs(os.path.join(base_dir, ps), exist_ok=True)

def save_S_solutions(S_list_all, price_setting, base_dir="solutions_S", filename="S_list.npy"):
    """
    S_list_all: list of (name, S_matrix) pairs
      - S_matrix expected shape (K_eval, T+1) or list-of-lists [[T+1]*K_eval]
    Saves as dict[name] = np.ndarray of shape (K_eval, T+1)
    """
    save_dir = os.path.join(base_dir, price_setting)
    os.makedirs(save_dir, exist_ok=True)

    S_dict = {}
    for name, S_mat in S_list_all:
        arr = np.asarray(S_mat)
        if arr.ndim != 2:
            raise ValueError(f"{name}: S must be 2D (K_eval, T+1). Got shape {arr.shape}")
        S_dict[name] = arr

    save_path = os.path.join(save_dir, filename)
    np.save(save_path, S_dict, allow_pickle=True)
    print(f"✅ Saved S solutions to: {save_path}")

def load_S_solutions(price_setting, base_dir="solutions_S", filename="S_list.npy"):
    load_path = os.path.join(base_dir, price_setting, filename)
    S_dict = np.load(load_path, allow_pickle=True).item()
    return S_dict

def plot_S_density_single(
    name,
    S_matrix,
    xlabel="Hour",
    ylabel="S",
    ylim=None,
    figsize=(8, 4),
    alpha=0.25,
    pause=False,
    save_path=None,
    show=True,
    overlay_mean=False
):
    """
    Density plot for S: one thin line per evaluation scenario.
    IMPORTANT: S_matrix must be (K_eval, T+1).
    """
    S_arr = np.asarray(S_matrix)
    if S_arr.ndim != 2:
        raise ValueError(f"{name}: S_matrix must be 2D (K_eval, T+1). Got shape {S_arr.shape}")

    K_eval, Tp1 = S_arr.shape  # Tp1 should be T+1
    hours = np.arange(Tp1)     # 0..T (length T+1)

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
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if pause:
        input("Press Enter for next plot...")

    return fig, ax

S_list_all = [
    ("Rolling → SDDiP", S_r),
    ("2-SP → SDDiP",    S_s2),
    ("3-SP → SDDiP",    S_s3),
]

for k_idx, K in enumerate(K_list):
    for c_idx, cut in enumerate(cut_list):
        name = f"PSDDiP (K={K}, {cut})"
        S_list_all.append((name, S_p[k_idx][c_idx]))


### 3) Save and plot for DA solutions

plot_q_da_single("Rolling → SDDiP", q_da_r, pause=True)
plot_q_da_single("2-SP → SDDiP",    q_da_s2, pause=True)
plot_q_da_single("3-SP → SDDiP",    q_da_s3, pause=True)

for k_idx, K in enumerate(K_list):
    for c_idx, cut in enumerate(cut_list):
        name = f"PSDDiP (K={K}, {cut})"
        plot_q_da_single(name, q_da_p[k_idx][c_idx], pause=True)

# load later and plot anytime

# q_da_dict_loaded = load_q_da_solutions("sunny", base_dir="solutions")
# q_da_list_loaded = list(q_da_dict_loaded.items())
# plot_q_da_separate(q_da_list_loaded, ylim=(-1000, 32000))


### 4) Save and plot for ID solutions

ensure_solution_dirs_ID(base_dir="solutions_ID", price_settings=("cloudy", "normal", "sunny"))
save_q_ID_solutions(q_ID_list, price_setting=price_setting, base_dir="solutions_ID")


#### q_ID

# (Optional) set y-limits if you want consistent visuals across runs
# id_ylim = (-50, 50)
id_ylim = None

# baselines
plot_q_ID_density_single("Rolling → SDDiP", q_ID_r, ylim=id_ylim, pause=True)
plot_q_ID_density_single("2-SP → SDDiP",    q_ID_s2, ylim=id_ylim, pause=True)
plot_q_ID_density_single("3-SP → SDDiP",    q_ID_s3, ylim=id_ylim, pause=True)

# PSDDiP: loop over K and cuts
for k_idx, K in enumerate(K_list):
    for c_idx, cut in enumerate(cut_list):
        name = f"PSDDiP (K={K}, {cut})"
        plot_q_ID_density_single(name, q_ID_p[k_idx][c_idx], ylim=id_ylim, pause=True)

# ----------------------------
# (Optional) Load later & plot
# ----------------------------
# q_ID_dict_loaded = load_q_ID_solutions("sunny", base_dir="solutions_ID")
# q_ID_list_loaded = list(q_ID_dict_loaded.items())
# for name, mat in q_ID_list_loaded:
#     plot_q_ID_density_single(name, mat, ylim=id_ylim, pause=False)


#### Soc values


# --------------
# Save S solutions
# --------------
ensure_solution_dirs_S(base_dir="solutions_S", price_settings=("cloudy", "normal", "sunny"))
save_S_solutions(S_list_all, price_setting=price_setting, base_dir="solutions_S")


# ----------------------------
# Plot S density one-by-one
# ----------------------------

# Optional consistent y-limits (example):
# S_ylim = (0, S_max)  # define if you want
S_ylim = None

# baselines
plot_S_density_single("Rolling → SDDiP", S_r,  ylim=S_ylim, pause=True)
plot_S_density_single("2-SP → SDDiP",    S_s2, ylim=S_ylim, pause=True)
plot_S_density_single("3-SP → SDDiP",    S_s3, ylim=S_ylim, pause=True)

# PSDDiP: loop over K and cuts
for k_idx, K in enumerate(K_list):
    for c_idx, cut in enumerate(cut_list):
        name = f"PSDDiP (K={K}, {cut})"
        plot_S_density_single(name, S_p[k_idx][c_idx], ylim=S_ylim, pause=True)


# ----------------------------
# (Optional) Load later & plot
# ----------------------------
# S_dict_loaded = load_S_solutions("sunny", base_dir="solutions_S")
# S_list_loaded = list(S_dict_loaded.items())
# for name, mat in S_list_loaded:
#     plot_S_density_single(name, mat, ylim=S_ylim, pause=False)


# *Notify done via plot*

def notify_done_via_plot(title="✅ Evaluation finished", subtitle=f"price_setting={price_setting}, cut_mode={cut_mode}, time_limit={time_limit}"):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=20, weight="bold")
    ax.text(0.5, 0.40, subtitle, ha="center", va="center", fontsize=12)
    try:
        fig.canvas.manager.set_window_title("Evaluation — Done")
    except Exception:
        pass
    plt.show()


notify_done_via_plot()
        
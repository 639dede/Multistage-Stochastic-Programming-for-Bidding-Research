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

SOLVER.options['TimeLimit'] = 300
#SOLVER.options['MIPGap'] = 1e-4

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


def evaluation_rolling_sddip(scenarios):
        
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
                
        rt_init_subp = fw_rt_init(da_state, psi_ID[n][0], P_da)
        rt_init_state = rt_init_subp.get_state_solutions()
        
        f_DA_list = rt_init_subp.get_DA_profit()
        
        for i in range(T):
            f_DA[i] += f_DA_list[i]/K_eval
        
        fcn_value = rt_init_subp.get_settlement_fcn_value()
            
        for scenario in scenarios_n:
            
            state = rt_init_state
            
            f_scenario = fcn_value
            
            for t in range(T - 1): ## t = 0, ..., T-2
                
                rt_subp = fw_rt(t, state, psi_ID[n][t+1], P_da, scenario[t])
                
                f_P[t] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
                f_E[t] += rt_subp.get_E_profit()/(K_eval*evaluation_num)
                f_Im[t] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T-1])

            f_P[T-1] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
            f_E[T-1] += rt_subp.get_E_profit()/(K_eval*evaluation_num)
            f_Im[T-1] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)

            f_scenario += rt_last_subp.get_settlement_fcn_value()
                        
            f.append(f_scenario)
        
    mu_hat = np.mean(f)
    
    sigma_hat = np.std(f, ddof=1)  

    z_alpha_half = 1.96  
    
    eval = mu_hat 
    
    print(f"\nRolling Horizon -> SDDiP for price setting = {price_setting}")
    print(f"Evaluation : {eval}")


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
    f_P  = [0]*T
    f_E  = [0]*T
    f_Im = [0]*T
        
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_test[n]
                
        rt_init_subp = fw_rt_init(da_state, psi_ID[n][0], P_da)
        rt_init_state = rt_init_subp.get_state_solutions()
        
        f_DA_list = rt_init_subp.get_DA_profit()
        
        for i in range(T):
            f_DA[i] += f_DA_list[i]/K_eval
        
        fcn_value = rt_init_subp.get_settlement_fcn_value()
            
        for scenario in scenarios_n:
            
            state = rt_init_state
            
            f_scenario = fcn_value
            
            for t in range(T - 1): ## t = 0, ..., T-2
                
                rt_subp = fw_rt(t, state, psi_ID[n][t+1], P_da, scenario[t])
                
                f_P[t] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
                f_E[t] += rt_subp.get_E_profit()/(K_eval*evaluation_num)
                f_Im[t] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T-1])

            f_P[T-1] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
            f_E[T-1] += rt_subp.get_E_profit()/(K_eval*evaluation_num)
            f_Im[T-1] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)

            f_scenario += rt_last_subp.get_settlement_fcn_value()
                        
            f.append(f_scenario)
        
    mu_hat = np.mean(f)
    
    sigma_hat = np.std(f, ddof=1)  

    z_alpha_half = 1.96  
    
    eval = mu_hat 
    
    print(f"\n{stage_num}-SP -> SDDiP for price setting = {price_setting}")
    print(f"Evaluation : {eval}")
    
        
def evaluation_psddip_sddip(K, cut_mode, scenarios):
    
    k_idx = K_list.index(K)
        
    da_subp = fw_da(Reduced_Probs[k_idx], psi_DA_list[cut_mode][k_idx])
    da_state = da_subp.get_state_solutions()
    
    q_da = da_state[0]
    
    f = []
    
    f_DA = [0]*T
    f_P  = [0]*T
    f_E  = [0]*T
    f_Im = [0]*T
    
    for n, scenarios_n in enumerate(scenarios):
                    
        P_da = P_da_test[n]
                
        rt_init_subp = fw_rt_init(da_state, psi_ID[n][0], P_da)
        rt_init_state = rt_init_subp.get_state_solutions()
        
        f_DA_list = rt_init_subp.get_DA_profit()
        
        for i in range(T):
            f_DA[i] += f_DA_list[i]/K_eval
        
        fcn_value = rt_init_subp.get_settlement_fcn_value()
            
        for scenario in scenarios_n:
            
            state = rt_init_state
            
            f_scenario = fcn_value
            
            for t in range(T - 1): ## t = 0, ..., T-2
                
                rt_subp = fw_rt(t, state, psi_ID[n][t+1], P_da, scenario[t])
                
                f_P[t] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
                f_E[t] += rt_subp.get_E_profit()/(K_eval*evaluation_num)
                f_Im[t] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)
                
                state = rt_subp.get_state_solutions()
                
                f_scenario += rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            rt_last_subp = fw_rt_last(state, P_da, scenario[T-1])

            f_P[T-1] += rt_subp.get_P_profit()/(K_eval*evaluation_num)
            f_E[T-1] += rt_subp.get_E_profit()/(K_eval*evaluation_num)
            f_Im[T-1] += rt_subp.get_Im_profit()/(K_eval*evaluation_num)

            f_scenario += rt_last_subp.get_settlement_fcn_value()
                        
            f.append(f_scenario)
        
    mu_hat = np.mean(f)
    
    sigma_hat = np.std(f, ddof=1)  

    z_alpha_half = 1.96  
    
    eval = mu_hat 
    
    print(f"\nPSDDiP K = {K}, time_lim = {time_limit}, cut = {cut_mode} -> SDDiP for price setting = {price_setting}")
    print(f"Evaluation : {eval}")
    

scenarios = scenarios_for_test

#%% Cut mode index: 'SB' -> 0, 'L-sub' -> 1

evaluation_rolling_rolling(scenarios)
evaluation_rolling_sddip(scenarios)  
evaluation_SP_sddip(2, scenarios, scenarios_for_SP)
evaluation_SP_sddip(3, scenarios, scenarios_for_SP)
evaluation_SP_sddip(3, scenarios, scenarios_for_eval)


for K in K_list:
    for cut_index in range(len(cut_list)):
        evaluation_psddip_sddip(K, cut_index, scenarios)


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
        
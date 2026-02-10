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
import copy
import csv

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

inspect = False


price_setting = 'sunny'  # 'cloudy', 'normal', 'sunny'

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


## Load Price and Scenario csv files

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
hours = np.arange(T)

K_list = [len(P_da_list) for P_da_list in Reduced_P_da]



## Plot clustered P_da profiles for each K

for i, P_da_list in enumerate(Reduced_P_da):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for profile in P_da_list:
        ax.plot(hours, profile, color='blue', alpha=0.6)
    
    ax.set_title(f"K = {K_list[i]}: Clustered Day-Ahead Price Profiles", fontsize=20)
    ax.set_xlabel("Hour", fontsize=20)
    ax.set_ylabel("Price (KRW)", fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True)
    ax.set_ylim(-120, 200)
    plt.tight_layout()
    plt.show()


## Plot clustered P_rt profiles for each K

hours = np.arange(24)

for k, scenario_trees in zip(K_list, Reduced_scenario_trees):

    paths = []
    for scenario in scenario_trees:
        N_b = len(scenario[0])
        for b in range(N_b):
            traj = [scenario[t][b][1] for t in range(24)]
            paths.append(traj)

    P = np.asarray(paths)
    if P.ndim != 2 or P.shape[1] != 24:
        raise ValueError(f"Unexpected shape for P (got {P.shape})")

    q25 = np.percentile(P, 25, axis=0)
    q75 = np.percentile(P, 75, axis=0)
    mean = P.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(hours, q25, q75, alpha=0.3, color="green",
                    label='Central 50% (25–75%)')

    n_show = min(15, len(P))
    rng_idx = np.linspace(0, len(P) - 1, n_show, dtype=int)
    for idx in rng_idx:
        ax.plot(hours, P[idx], color='black', alpha=0.2, linewidth=1.0)

    ax.plot(hours, mean, linewidth=1.8, linestyle='--',
            color='orange', label='Mean')

    ax.set_title(f"K = {k}: Real-Time Price Scenarios", fontsize=20)
    ax.set_xlabel("Hour", fontsize=20)
    ax.set_ylabel("P_rt (KRW)", fontsize=20)
    ax.set_ylim(-120, 200)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(['0h', '6h', '12h', '18h', '24h'], fontsize=20)
    ax.grid(True)

    ax.legend(loc='upper right', fontsize=18)
    plt.tight_layout()
    plt.show()


## Plot q_c and E_1 paths for each K

def build_paths_from_trees(scenario_trees, component_idx):
    paths = []
    for scenario in scenario_trees:
        N_b = len(scenario[0])
        for b in range(N_b):
            traj = [scenario[t][b][component_idx] for t in range(24)]
            paths.append(traj)
    return np.asarray(paths, dtype=float)


def build_E1_paths_from_trees(scenario_trees, E_0_vec, deltaE_idx=0):
    E_0_vec = np.asarray(E_0_vec, dtype=float)
    paths = []

    for scenario in scenario_trees:
        N_b = len(scenario[0])
        for b in range(N_b):
            traj = []
            for t in range(24):
                delta_E = scenario[t][b][deltaE_idx]
                if t == 23:
                    delta_E = 1.0
                traj.append(delta_E * E_0_vec[t])
            paths.append(traj)

    return np.asarray(paths, dtype=float)


for k, scenario_trees in zip(K_list, Reduced_scenario_trees):

    # ---------- q_c ----------
    qc_paths = build_paths_from_trees(scenario_trees, component_idx=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    n_show = min(30, len(qc_paths))
    idxs = np.linspace(0, len(qc_paths) - 1, n_show, dtype=int)

    for idx in idxs:
        ax.plot(hours, qc_paths[idx], color="black", alpha=0.25, linewidth=1.0)

    ax.set_title(f"K = {k}: q_c trajectories", fontsize=20)
    ax.set_xlabel("Hour", fontsize=20)
    ax.set_ylabel("q_c", fontsize=20)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(['0h', '6h', '12h', '18h', '24h'], fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- E_1 ----------
    E1_paths = build_E1_paths_from_trees(scenario_trees, E_0_vec=E_0)

    fig, ax = plt.subplots(figsize=(10, 6))
    n_show = min(30, len(E1_paths))
    idxs = np.linspace(0, len(E1_paths) - 1, n_show, dtype=int)

    for idx in idxs:
        ax.plot(hours, E1_paths[idx], color="black", alpha=0.25, linewidth=1.0)

    ax.set_title(f"K = {k}: E_1 trajectories (delta_E × E_0)", fontsize=20)
    ax.set_xlabel("Hour", fontsize=20)
    ax.set_ylabel("E_1", fontsize=20)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(['0h', '6h', '12h', '18h', '24h'], fontsize=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


## Parameters 

T = 24
D = 1

P_r = 80
P_max = 200

E_0 = E_0

C = 21022.1
S = C
B = C/3

S_min = 0.1*S
S_max = 0.9*S
beta = 5*P_max

v = 0.95

gamma_over = 2*P_max
gamma_under = 2*P_max

omega = 0.05*S

E_0_partial = E_0

P_da_evaluate = Reduced_P_da[-1]
Probs_evaluate = Reduced_Probs[-1]

Scenario_tree_evaluate = Reduced_scenario_trees[-1]

K_eval = len(P_da_evaluate)

K_list = [1, 3, 6, 10, 15, K_eval]

dual_tolerance = 1e-3
dual_tolerance_da = 1e-4
tol = 1e-5
Node_num = 1
Lag_iter_UB = 500
Lag_iter_UB_da = 1000
G = 1e7
reg = 1
gamma = 0.5
reg_num = 10

E_0_sum = 0
E_0_partial_max = max(E_0_partial)

for t in range(len(E_0_partial)):
    E_0_sum += E_0_partial[t]


# 2. Subproblems

## 1) Subproblems for PSDDiP

### stage = DA

class fw_da(pyo.ConcreteModel): 
    
    def __init__(self, probs, psi):
        
        super().__init__()

        self.solved = False
        
        self.probs = probs
        
        self.psi = psi
        
        self.K = len(self.psi)
        
        self.T = T

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.KRANGE = pyo.RangeSet(0, self.K-1)
                
        model.PSIRANGE = pyo.Set(model.KRANGE, initialize=lambda _m, k: range(len(self.psi[k])))
        
        # Vars
        
        def _value_index_init(m): 
            
            for k in m.KRANGE:
                for l in m.PSIRANGE[k]:
                    yield (k, l)
                    
        model.VALUE_INDEX = pyo.Set(dimen=2, initialize=_value_index_init)
        
        model.theta = pyo.Var(model.KRANGE, domain = pyo.Reals)
        
        model.q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        # Constraints
        
        def da_bidding_amount_rule(model, t):
            return model.q[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q[t] for t in range(self.T)) <= E_0_sum
        
        def value_fcn_approx_rule(model, k, l):
            
            v, pi_q = self.psi[k][l]
            
            return model.theta[k] <= (
                v
                +sum(pi_q[t]*model.q[t] for t in range(T)) 
                )

        model.da_bidding_amount = pyo.Constraint(model.TIME, rule=da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule=da_overbid_rule)
        model.value_fcn_approx = pyo.Constraint(model.VALUE_INDEX, rule=value_fcn_approx_rule)

        # Obj Fcn
        
        def objective_rule(model):
            return (
                sum(model.theta[k]*self.probs[k] for k in range(self.K))
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense=pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True

    def get_state_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append([pyo.value(self.q[t]) for t in range(self.T)])
        
        return State_var 

    def get_objective_value(self):
       
        if not self.solved:
            self.solve()
            self.solved = True        

        return pyo.value(self.objective)        


### stage = -1

class fw_rt_init(pyo.ConcreteModel): 
    
    def __init__(self, da_prev, psi, P_da):
        
        super().__init__()

        self.solved = False
        
        self.q_prev = da_prev[0]
        
        self.psi = psi
        self.T = T
        self.P_da = P_da
            
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        model.theta = pyo.Var(bounds = (-1e8, 1e8), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.Reals)
                
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.Reals)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to DA stage
        
        def da_q_rule(model, t):
            return model.q_da[t] == self.q_prev[t]
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_amount_rule_1(model):
            return model.q_rt <= model.q_da[0] + omega
        
        def rt_bidding_amount_rule_2(model):
            return model.q_rt >= model.q_da[0] - omega
        
        ## State variable trainsition
        
        def State_SOC_rule(model):
            return model.S == 0.5*S

        def state_Q_da_rule(model, t):
            return model.T_Q[t] == model.q_da[t]
        
        def State_q_rule(model):
            return model.T_q == model.q_rt
        
        def State_E_rule(model):
            return model.T_E == 0
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= (
                self.psi[l][0] 
                + self.psi[l][1]*model.S 
                + sum(self.psi[l][2][t]*model.T_Q[t] for t in range(T)) 
                + self.psi[l][3]*model.T_q 
                + self.psi[l][4]*model.T_E
                )
        
        def settlement_fcn_rule(model):
            return model.f == sum(
                self.P_da[t]*model.q_da[t] for t in range(self.T)
            )
            
        model.da_q_amount = pyo.Constraint(model.TIME, rule = da_q_rule)
        model.rt_bidding_amount_1 = pyo.Constraint(rule = rt_bidding_amount_rule_1)
        model.rt_bidding_amount_2 = pyo.Constraint(rule = rt_bidding_amount_rule_2)
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.state_Q_da = pyo.Constraint(model.TIME, rule = state_Q_da_rule)
        model.state_q = pyo.Constraint(rule = State_q_rule)
        model.state_E = pyo.Constraint(rule = State_E_rule)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)

        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)

        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta + model.f
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense=pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self, tee=inspect)
        self.solved = True

    def get_state_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append(pyo.value(self.S))
        State_var.append([pyo.value(self.T_Q[t]) for t in range(self.T)])
        State_var.append(pyo.value(self.T_q))
        State_var.append(pyo.value(self.T_E))
        
        return State_var 
 
    def get_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True

        solutions = {
            "q_da": [pyo.value(self.q_da[t]) for t in range(self.T)],
        }
        
        return solutions

    def get_settlement_fcn_value(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        return pyo.value(self.f)    

    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        return pyo.value(self.objective)        

    def get_DA_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        DA_profit = [self.P_da[t]*pyo.value(self.q_da[t]) for t in range(self.T)]

        return DA_profit

class fw_rt_init_LP_relax(pyo.ConcreteModel): ## (Backward - Benders' Cut) 
    
    def __init__(self, da_prev, psi, P_da):
        
        super().__init__()

        self.solved = False
        
        self.q_prev = da_prev[0]
        
        self.psi = psi
        self.T = T
        self.P_da = P_da

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_q = pyo.Var(model.TIME, domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Other
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.Reals)
                        
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.Reals)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_q(model, t):
            return model.z_q[t] == self.q_prev[t]
        
        ## Connected to DA stage
        
        def da_q_rule(model, t):
            return model.q_da[t] == model.z_q[t]
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_amount_rule_1(model):
            return model.q_rt <= model.q_da[0] + omega
        
        def rt_bidding_amount_rule_2(model):
            return model.q_rt >= model.q_da[0] - omega
        
        ## State variable trainsition
        
        def State_SOC_rule(model):
            return model.S == 0.5*S

        def state_Q_da_rule(model, t):
            return model.T_Q[t] == model.q_da[t]
        
        def State_q_rule(model):
            return model.T_q == model.q_rt
        
        def State_E_rule(model):
            return model.T_E == 0
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= (
                self.psi[l][0] 
                + self.psi[l][1]*model.S 
                + sum(self.psi[l][2][t]*model.T_Q[t] for t in range(T)) 
                + self.psi[l][3]*model.T_q 
                + self.psi[l][4]*model.T_E
                )
        
        def settlement_fcn_rule(model):
            return model.f == sum(
                self.P_da[t]*model.q_da[t] for t in range(self.T)
            )
        
        model.auxiliary_q = pyo.Constraint(model.TIME, rule = auxiliary_q)
        model.da_q_amount = pyo.Constraint(model.TIME, rule = da_q_rule)
        model.rt_bidding_amount_1 = pyo.Constraint(rule = rt_bidding_amount_rule_1)
        model.rt_bidding_amount_2 = pyo.Constraint(rule = rt_bidding_amount_rule_2)
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.state_Q_da = pyo.Constraint(model.TIME, rule = state_Q_da_rule)
        model.state_q = pyo.Constraint(rule = State_q_rule)
        model.state_E = pyo.Constraint(rule = State_E_rule)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)

        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)

        # Dual(shadow price)
        
        model.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)
        
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta + model.f
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense=pyo.maximize)

    def solve(self):
        self.build_model()
        self.solver_results = SOLVER.solve(self, tee=inspect)
        if self.solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            self.solved = True
        else:
            self.solved = False
        return self.solver_results

    def get_cut_coefficients(self):
        
        if not self.solved:
            results = self.solve()
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                return [
                    3*3600000*(T), 
                    [0 for _ in range(len(self.q_prev))], 
                    ]
        
        psi = []
        psi.append(pyo.value(self.objective))
        
        pi_q = []
        for i in range(self.T):
            pi_q.append(self.dual[self.auxiliary_q[i]])
        psi.append(pi_q)
        
        return psi


### stage = 0, 1, ..., T-1

class fw_rt(pyo.ConcreteModel):

    def __init__(self, stage, T_prev, psi, P_da, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_q_prev = T_prev[2]
        self.T_E_prev = T_prev[3]
        
        self.psi = psi
        
        self.P_da = P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, self.T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.q_rt = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals, initialize = 0.0)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals, initialize = 0.0)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals, initialize = 0.0)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals, initialize = 0.0)
        model.u = pyo.Var(domain = pyo.NonNegativeReals, initialize = 0.0)
        model.Q_c = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        
        model.E_1 = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals, initialize = self.S_prev)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.Reals, initialize = 0.0)
        model.T_q = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.T_E = pyo.Var(domain = pyo.Reals, initialize = 0.0)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(domain = pyo.NonNegativeReals, initialize = 0.0)
        model.phi_under = pyo.Var(domain = pyo.NonNegativeReals, initialize = 0.0)
        
        # Experiment for LP relaxation
                
        ## settlement_fcn_Vars
        
        model.f = pyo.Var(domain = pyo.Reals, initialize = 0.0)
                   
        # Constraints
        
        ## Connected to t-1 stage 
        
        def da_Q_rule(model):
            return model.Q_da == self.T_Q_prev[0]
        
        def rt_q_rule(model):
            return model.q_rt == self.T_q_prev
        
        def rt_E_rule(model):
            return model.E_1 == self.T_E_prev
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == self.S_prev + v*model.c - (1/v)*model.d
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == self.T_Q_prev[t+1]
        
        def State_q_rule(model):
            return model.T_q == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def next_q_rt_rule_1(model):
            return model.q_rt_next <= self.T_Q_prev[1] + omega
        
        def next_q_rt_rule_2(model):
            return model.q_rt_next >= self.T_Q_prev[1] - omega
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.q_rt
        
        ### Imbalance Penalty
        
        def imbalance_over_rule(model):
            return model.u - model.Q_c <= model.phi_over
        
        def imbalance_under_rule(model):
            return model.Q_c - model.u <= model.phi_under
        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= (
                self.psi[l][0] 
                + self.psi[l][1]*model.S 
                + sum(self.psi[l][2][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) 
                + self.psi[l][3]*model.T_q 
                + self.psi[l][4]*model.T_E
                )
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                - gamma_over*model.phi_over 
                - gamma_under*model.phi_under
                )
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_E = pyo.Constraint(rule = State_E_rule)
        
        model.next_q_rt_1 = pyo.Constraint(rule = next_q_rt_rule_1)
        model.next_q_rt = pyo.Constraint(rule = next_q_rt_rule_2)
        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.dispatch = pyo.Constraint(rule = dispatch_rule)
        
        model.imbalance_over = pyo.Constraint(rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(rule = imbalance_under_rule)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)
                
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta + model.f
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self, tee=inspect)
        self.solved = True   

    def get_state_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append(pyo.value(self.S))
        State_var.append([pyo.value(self.T_Q[t]) for t in range(self.T - 1 - self.stage)])
        State_var.append(pyo.value(self.T_q))
        State_var.append(pyo.value(self.T_E))
                
        return State_var 

    def get_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True

        solutions = {
            "q_rt": pyo.value(self.q_rt)
        }

        return solutions

    def get_settlement_fcn_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
        
        return pyo.value(self.f)

    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)
    
    def get_P_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        P_profit = (pyo.value(self.u) - pyo.value(self.Q_da))*self.P_rt

        return P_profit

    def get_Im_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        Im_profit = - gamma_over*pyo.value(self.phi_over) - gamma_under*pyo.value(self.phi_under)

        return Im_profit

    def get_ID_solution(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        ID_solution = pyo.value(self.q_rt)

        return ID_solution

    def get_S_solution(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        S_solution = pyo.value(self.S)

        return S_solution

class fw_rt_LP_relax(pyo.ConcreteModel): ## (Backward - Benders' Cut)

    def __init__(self, stage, T_prev, psi, P_da, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_q_prev = T_prev[2]
        self.T_E_prev = T_prev[3]
        
        self.psi = psi
        
        self.P_da = P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
               
    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
                
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
                
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.Reals)
        
        model.E_1 = pyo.Var(domain = pyo.Reals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.Reals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## settlement_fcn_Vars
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_S(model):
            return model.z_S == self.S_prev
        
        def auxiliary_T_Q(model, t):
            return model.z_T_Q[t] == self.T_Q_prev[t]
        
        def auxiliary_T_q(model):
            return model.z_T_q == self.T_q_prev
        
        def auxiliary_T_E(model):
            return model.z_T_E == self.T_E_prev        
        
        ## Connected to t-1 state 
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d

        def State_Q_rule(model, t):
            return model.T_Q[t] == model.z_T_Q[t+1]
        
        def State_q_rule(model):
            return model.T_q == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def next_q_rt_rule_1(model):
            return model.q_rt_next <= self.z_T_Q[1] + omega
        
        def next_q_rt_rule_2(model):
            return model.q_rt_next >= self.z_T_Q[1] - omega
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.q_rt
        
        
        ### Imbalance Penalty
        
        def imbalance_over_rule(model):
            return model.u - model.Q_c <= model.phi_over
        
        def imbalance_under_rule(model):
            return model.Q_c - model.u <= model.phi_under
        
        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= (
                self.psi[l][0] 
                + self.psi[l][1]*model.S 
                + sum(self.psi[l][2][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) 
                + self.psi[l][3]*model.T_q 
                + self.psi[l][4]*model.T_E)
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                - gamma_over*model.phi_over 
                - gamma_under*model.phi_under
                )
        
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S)
        model.auxiliary_T_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_Q)
        model.auxiliary_T_q = pyo.Constraint(rule = auxiliary_T_q)
        model.auxiliary_T_E = pyo.Constraint(rule = auxiliary_T_E)
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_E = pyo.Constraint(rule = State_E_rule)
        
        model.next_q_rt = pyo.Constraint(rule = next_q_rt_rule_1)
        model.next_q_rt_2 = pyo.Constraint(rule = next_q_rt_rule_2)
        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.dispatch = pyo.Constraint(rule = dispatch_rule)
        
        model.imbalance_over = pyo.Constraint(rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(rule = imbalance_under_rule)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)
                
        # Dual(shadow price)
        
        model.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)
                
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta + model.f
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)
        
    def solve(self):
        self.build_model()
        self.solver_results = SOLVER.solve(self, tee=inspect)
        if self.solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            self.solved = True
        else:
            self.solved = False
        return self.solver_results

    def get_cut_coefficients(self):
 
        if not self.solved:
            results = self.solve()
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                return [
                    3*3600000*(T - self.stage), 
                    0, 
                    [0 for _ in range(len(self.T_Q_prev))], 
                    0, 
                    0
                    ]
        
        psi = []
        psi.append(pyo.value(self.objective))
        psi.append(self.dual[self.auxiliary_S])
        
        pi_T_Q = []
        for i in range(len(self.T_Q_prev)):
            pi_T_Q.append(self.dual[self.auxiliary_T_Q[i]])
        psi.append(pi_T_Q)
        
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi
    

### stage = T

class fw_rt_last(pyo.ConcreteModel): 
    
    def __init__(self, T_prev, P_da, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_q_prev = T_prev[2]
        self.T_E_prev = T_prev[3]

        self.P_da = P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
                
    def build_model(self):
        
        model = self.model()
                
        # Vars
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.S_r = pyo.Var(domain = pyo.Reals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(domain = pyo.NonNegativeReals)     
        
        ## settlement_fcn
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_Q_rule(model):
            return model.Q_da == self.T_Q_prev[0]
        
        def rt_q_rule(model):
            return model.q_rt == self.T_q_prev
        
        def rt_E_rule(model):
            return model.E_1 == self.T_E_prev
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == self.S_prev + v*model.c - (1/v)*model.d
        
        def SOC_recourse_rule_1(model):
            return model.S_r >= beta*(model.S - 0.5*S)
            #return model.S_r >= 0

        def SOC_recourse_rule_2(model):
            return model.S_r >= -beta*(model.S - 0.5*S)
            #return model.S_r >= 0

        ## General Constraints
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.q_rt
        
        ### Imbalance Penalty
        
        def imbalance_over_rule(model):
            return model.u - model.Q_c <= model.phi_over
        
        def imbalance_under_rule(model):
            return model.Q_c - model.u <= model.phi_under
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                - gamma_over*model.phi_over - gamma_under*model.phi_under
                - model.S_r
                )
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.SOC_recourse_1 = pyo.Constraint(rule = SOC_recourse_rule_1)
        model.SOC_recourse_2 = pyo.Constraint(rule = SOC_recourse_rule_2)

        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.dispatch = pyo.Constraint(rule = dispatch_rule)
        
        model.imbalance_over = pyo.Constraint(rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(rule = imbalance_under_rule)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
                         
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.f
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense=pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self, tee=inspect)
        self.solved = True

    def get_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True

        solutions = {
            "Q_rt": pyo.value(self.q_rt)
        }

        return solutions

    def get_settlement_fcn_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
        
        return pyo.value(self.f)

    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)

    def get_last_SOC_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.S)

    def get_P_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        P_profit = (pyo.value(self.u) - pyo.value(self.Q_da))*self.P_rt

        return P_profit

    def get_Im_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        Im_profit = - gamma_over*pyo.value(self.phi_over) - gamma_under*pyo.value(self.phi_under)

        return Im_profit

    def get_ID_solution(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        ID_solution = pyo.value(self.q_rt)

        return ID_solution

    def get_S_solution(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        SOC_solution = pyo.value(self.S)

        return SOC_solution

class fw_rt_last_LP_relax(pyo.ConcreteModel): ## (Backward)
           
    def __init__(self, T_prev, P_da, delta):
        
        super().__init__()

        self.solved = False
                
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_q_prev = T_prev[2]
        self.T_E_prev = T_prev[3]

        self.P_da = P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
       
    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
                
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
                
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.S_r = pyo.Var(domain = pyo.Reals)
        
        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(domain = pyo.NonNegativeReals)        
        
        ## settlement_fcn
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_S(model):
            return model.z_S == self.S_prev
        
        def auxiliary_T_Q(model, t):
            return model.z_T_Q[t] == self.T_Q_prev[t]
        
        def auxiliary_T_q(model):
            return model.z_T_q == self.T_q_prev
        
        def auxiliary_T_E(model):
            return model.z_T_E == self.T_E_prev
        
        ## Connected to t-1 state 
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d
        
        def SOC_recourse_rule_1(model):
            return model.S_r >= beta*(model.S - 0.5*S)
            #return model.S_r >= 0
        
        def SOC_recourse_rule_2(model):
            return model.S_r >= -beta*(model.S - 0.5*S)
            #return model.S_r >= 0
        
        ## General Constraints
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.q_rt
    
    
        ### Imbalance Penalty
        
        def imbalance_over_rule(model):
            return model.u - model.Q_c <= model.phi_over
        
        def imbalance_under_rule(model):
            return model.Q_c - model.u <= model.phi_under
  
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                - gamma_over*model.phi_over - gamma_under*model.phi_under
                - model.S_r
                )
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S)
        model.auxiliary_T_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_Q)
        model.auxiliary_T_q = pyo.Constraint(rule = auxiliary_T_q)
        model.auxiliary_T_E = pyo.Constraint(rule = auxiliary_T_E)
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.SOC_recourse_1 = pyo.Constraint(rule = SOC_recourse_rule_1)
        model.SOC_recourse_2 = pyo.Constraint(rule = SOC_recourse_rule_2)

        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.dispatch = pyo.Constraint(rule = dispatch_rule)
        
        model.imbalance_over = pyo.Constraint(rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(rule = imbalance_under_rule) 
                      
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        
        # Dual(Shadow price)
          
        model.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)
          
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.f 
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense=pyo.maximize)

    def solve(self):
        self.build_model()
        self.solver_results = SOLVER.solve(self, tee=inspect)
        if self.solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            self.solved = True
        else:
            self.solved = False
        return self.solver_results

    def get_cut_coefficients(self):
        if not self.solved:
            results = self.solve()
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                return [
                    3*3600000, 
                    0, 
                    [0 for _ in range(len(self.T_Q_prev))], 
                    0, 
                    0
                    ]
        
        psi = []
        psi.append(pyo.value(self.objective))
        psi.append(self.dual[self.auxiliary_S])

        pi_T_Q = []
        for i in range(len(self.T_Q_prev)):
            pi_T_Q.append(self.dual[self.auxiliary_T_Q[i]])
        psi.append(pi_T_Q)
        
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi  


## 2) Subproblems for Rolling Horizon

class rolling_da(pyo.ConcreteModel):
    
    def __init__(self, exp_P_da, exp_P_rt):
        
        super().__init__()

        self.solved = False
        
        self.exp_P_da = exp_P_da
        self.exp_P_rt = exp_P_rt
        
        self.T = T

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)

        # Vars
        
        # Day Ahead variables
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        # Intraday bidding variables
        
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals
        )
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        ## settlement fcn Vars
        
        model.f_DA = pyo.Var(model.TIME, domain = pyo.Reals)
        model.f_RT = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## DA bidding & market rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= E_0_sum
        
        ## RT bidding & market rules
        
        def rt_bidding_amount_rule_1(model, t):
            return model.q_rt[t] <= model.q_da[t] + omega
        
        def rt_bidding_amount_rule_2(model, t):
            return model.q_rt[t] >= model.q_da[t] - omega
    
        ## Operations rules
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.q_rt[t]
        
        def SOC_init_rule(model):
            return model.S[-1] == 0.5*S
        
        def SOC_last_rule(model):
            return model.S[self.T-1] == 0.5*S
        
        def SOC_balance_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def generation_rule(model, t):
            return model.g[t] <= E_0_partial[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]
        
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, t):
            return model.u[t] - model.Q_c[t] <= model.phi_over[t]
        
        def imbalance_under_rule(model, t):
            return model.Q_c[t] - model.u[t] <= model.phi_under[t]
        
        ## settlement fcn
        
        def da_settlement_fcn_rule(model, t):
            return model.f_DA[t] == self.exp_P_da[t]*model.q_da[t] 
        
        def rt_settlement_fcn_rule(model, t):
            return model.f_RT[t] == (
                (model.u[t] - model.q_da[t])*self.exp_P_rt[t] 
                - gamma_over*model.phi_over[t] 
                - gamma_under*model.phi_under[t]
                )
        
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
        model.rt_bidding_amount_1 = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule_1)
        model.rt_bidding_amount_2 = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule_2)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.SOC_init = pyo.Constraint(rule = SOC_init_rule)
        model.SOC_last = pyo.Constraint(rule = SOC_last_rule)
        model.SOC_balance = pyo.Constraint(model.TIME, rule = SOC_balance_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.imbalance_over = pyo.Constraint(model.TIME, rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(model.TIME, rule = imbalance_under_rule)
        model.da_settlement_fcn = pyo.Constraint(model.TIME, rule = da_settlement_fcn_rule)
        model.rt_settlement_fcn = pyo.Constraint(model.TIME, rule = rt_settlement_fcn_rule)
        
        # Objective
        
        def objective_rule(model):
            return sum(
                model.f_DA[t] + model.f_RT[t] 
                for t in model.TIME
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)
        
    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True   
    
    def get_state_solutions(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append([pyo.value(self.q_da[t]) for t in range(self.T)])
        
        return State_var    

class rolling_rt_init(pyo.ConcreteModel):
       
    def __init__(self, da_state, P_da, exp_P_rt):
        
        super().__init__()

        self.solved = False
        
        self.q_da_prev = da_state[0]
        
        self.P_da = P_da
        self.exp_P_rt = exp_P_rt
                
        self.T = T

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)

        # Vars

        # Intraday bidding variables

        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals
        )
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        ## settlement fcn Vars
        
        model.f_DA = pyo.Var(model.TIME, domain = pyo.Reals)
        model.f_RT = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## RT bidding & market rules
        
        def rt_bidding_amount_rule_1(model, t):
            return model.q_rt[t] <= self.q_da_prev[t] + omega
        
        def rt_bidding_amount_rule_2(model, t):
            return model.q_rt[t] >= self.q_da_prev[t] - omega
        
        ## Operations rules
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.q_rt[t]
        
        def SOC_init_rule(model):
            return model.S[-1] == 0.5*S
        
        def SOC_last_rule(model):
            return model.S[self.T-1] == 0.5*S
        
        def SOC_balance_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def generation_rule(model, t):
            return model.g[t] <= E_0_partial[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        ## Imbalance Penalty
        
        def imbalance_over_rule(model, t):
            return model.u[t] - model.Q_c[t] <= model.phi_over[t]
        
        def imbalance_under_rule(model, t):
            return model.Q_c[t] - model.u[t] <= model.phi_under[t]
        
        ## settlement fcn
        
        def da_settlement_fcn_rule(model, t):
            return model.f_DA[t] == self.P_da[t]*self.q_da_prev[t] 
        
        def rt_settlement_fcn_rule(model, t):
            return model.f_RT[t] == (
                (model.u[t] - self.q_da_prev[t])*self.exp_P_rt[t] 
                - gamma_over*model.phi_over[t] 
                - gamma_under*model.phi_under[t]
                )
        
        model.rt_bidding_amount_1 = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule_1)
        model.rt_bidding_amount_2 = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule_2)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.SOC_init = pyo.Constraint(rule = SOC_init_rule)
        model.SOC_last = pyo.Constraint(rule = SOC_last_rule)
        model.SOC_balance = pyo.Constraint(model.TIME, rule = SOC_balance_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.imbalance_over = pyo.Constraint(model.TIME, rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(model.TIME, rule = imbalance_under_rule)
        model.da_settlement_fcn = pyo.Constraint(model.TIME, rule = da_settlement_fcn_rule)
        model.rt_settlement_fcn = pyo.Constraint(model.TIME, rule = rt_settlement_fcn_rule)
        
        # Objective
        
        def objective_rule(model):
            return sum(
                model.f_DA[t] + model.f_RT[t] 
                for t in model.TIME
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True   
    
    def get_state_solutions(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
            
        State_var = []
        
        State_var.append(pyo.value(self.S[-1]))
        State_var.append([pyo.value(self.q_da_prev[t]) for t in range(self.T)])
        State_var.append(pyo.value(self.q_rt[0]))
        State_var.append(0)
        
        return State_var    

    def get_settlement_fcn_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True

        fcn_value = sum(
            pyo.value(self.f_DA[t]) 
            for t in range(self.T)
            )
        
        return fcn_value

    def get_DA_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        DA_profit = [self.P_da[t]*pyo.value(self.q_da_prev[t]) for t in range(self.T)]

        return DA_profit

class rolling_rt(pyo.ConcreteModel):
       
    def __init__(self, stage, state, P_da, exp_P_rt, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = state[0]
        self.Q_prev = state[1]
        self.q_rt_prev = state[2]
        self.E_prev = state[3]
        
        self.P_da = P_da
        self.exp_P_rt = exp_P_rt
                
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(self.stage, T-1)
        model.BIDTIME = pyo.RangeSet(self.stage+1, T-1)
        model.BIDTIME_NEXT = pyo.RangeSet(self.stage+2, T-1)
        model.TIME_ESS = pyo.RangeSet(self.stage-1, T-1)

        # Vars

        # Intraday bidding variables

        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        # Intraday operation variables
        
        model.S = pyo.Var(
            model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals, initialize = 0.0
        )
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        ## settlement fcn Vars
        
        model.f_RT = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## DA awarded amounts
        
        def da_awarded_amount_rule(model, t):
            return model.Q_da[t] == self.Q_prev[t]
        
        ## RT bidding & market rules
        
        def rt_bidding_amount_next_rule_1(model):
            return model.q_rt[self.stage+1] <= model.Q_da[self.stage+1] + omega
        
        def rt_bidding_amount_next_rule_2(model):
            return model.q_rt[self.stage+1] >= model.Q_da[self.stage+1] - omega
        
        def rt_bidding_amount_rule_1(model, t):
            return model.q_rt[t] <= model.Q_da[t] + omega
        
        def rt_bidding_amount_rule_2(model, t):
            return model.q_rt[t] >= model.Q_da[t] - omega
        
        ## Operations rules
        
        def dispatch_stage_rule(model):
            return model.Q_c[self.stage] == (1 + self.delta_c)*self.q_rt_prev
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.q_rt[t]
        
        def SOC_init_rule(model):
            return model.S[self.stage-1] == self.S_prev
        
        def SOC_last_rule(model):
            return model.S[self.T-1] == 0.5*S
        
        def SOC_balance_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def generation_stage_rule(model):
            return model.g[self.stage] <= self.E_prev
        
        def generation_next_rule(model):
            return model.g[self.stage+1] <= self.delta_E_0*E_0_partial[self.stage+1]
        
        def generation_rule(model, t):
            return model.g[t] <= E_0_partial[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]
        
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, t):
            return model.u[t] - model.Q_c[t] <= model.phi_over[t]
        
        def imbalance_under_rule(model, t):
            return model.Q_c[t] - model.u[t] <= model.phi_under[t]
        
        ## settlement fcn
        
        def rt_settlement_fcn_stage_rule(model):
            return model.f_RT[self.stage] == (
                (model.u[self.stage] - model.Q_da[self.stage])*self.P_rt 
                - gamma_over*model.phi_over[self.stage] 
                - gamma_under*model.phi_under[self.stage]
                )
        
        def rt_settlement_fcn_rule(model, t):
            return model.f_RT[t] == (
                (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] 
                - gamma_over*model.phi_over[t] 
                - gamma_under*model.phi_under[t]
                )
        
        model.da_awarded_amount = pyo.Constraint(model.TIME, rule = da_awarded_amount_rule)
        model.rt_bidding_amount_next = pyo.Constraint(rule = rt_bidding_amount_next_rule_1)
        model.rt_bidding_amount_next_2 = pyo.Constraint(rule = rt_bidding_amount_next_rule_2)
        model.rt_bidding_amount_1 = pyo.Constraint(model.BIDTIME_NEXT, rule = rt_bidding_amount_rule_1)
        model.rt_bidding_amount_2 = pyo.Constraint(model.BIDTIME_NEXT, rule = rt_bidding_amount_rule_2)
        model.dispatch_stage = pyo.Constraint(rule = dispatch_stage_rule)
        model.dispatch = pyo.Constraint(model.BIDTIME, rule = dispatch_rule)
        model.SOC_init = pyo.Constraint(rule = SOC_init_rule)
        model.SOC_last = pyo.Constraint(rule = SOC_last_rule)
        model.SOC_balance = pyo.Constraint(model.TIME, rule = SOC_balance_rule)
        model.generation_stage = pyo.Constraint(rule = generation_stage_rule)
        model.generation_next = pyo.Constraint(rule = generation_next_rule)
        model.generation = pyo.Constraint(model.BIDTIME_NEXT, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.imbalance_over = pyo.Constraint(model.TIME, rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(model.TIME, rule = imbalance_under_rule)
        model.rt_settlement_stage_fcn = pyo.Constraint(rule = rt_settlement_fcn_stage_rule)
        model.rt_settlement_fcn = pyo.Constraint(model.BIDTIME, rule = rt_settlement_fcn_rule)
        
        # Objective
        
        def objective_rule(model):
            return sum(
                model.f_RT[t] 
                for t in model.TIME
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True   
    
    def get_state_solutions(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
            
        State_var = []
        
        State_var.append(pyo.value(self.S[self.stage]))
        State_var.append(self.Q_prev)
        State_var.append(pyo.value(self.q_rt[self.stage+1]))
        State_var.append(self.delta_E_0*E_0_partial[self.stage+1])
        
        return State_var    

    def get_settlement_fcn_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True

        fcn_value = pyo.value(self.f_RT[self.stage])
        
        return fcn_value

    def get_P_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        P_profit = (pyo.value(self.u[self.stage]) - pyo.value(self.Q_da[self.stage]))*self.P_rt

        return P_profit

    def get_Im_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        Im_profit = - gamma_over*pyo.value(self.phi_over[self.stage]) - gamma_under*pyo.value(self.phi_under[self.stage])

        return Im_profit

class rolling_rt_last(pyo.ConcreteModel):
       
    def __init__(self, state, P_da, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T-1
        
        self.S_prev = state[0]
        self.Q_prev = state[1]
        self.q_rt_prev = state[2]
        self.E_prev = state[3]
        
        self.P_da = P_da
        
        self.P_abs = [0 for _ in range(T)]
        
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(self.stage, T-1)
        model.TIME_ESS = pyo.RangeSet(self.stage-1, T-1)

        # Vars
        
        # Day Ahead variables

        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals, initialize = 0.0
        )
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        ## settlement fcn Vars
        
        model.f_RT = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## DA awarded amounts
        
        def da_awarded_amount_rule(model, t):
            return model.Q_da[t] == self.Q_prev[t]
        
        ## Operations rules
        
        def dispatch_stage_rule(model):
            return model.Q_c[self.stage] == (1 + self.delta_c)*self.q_rt_prev
        
        def SOC_init_rule(model):
            return model.S[self.stage-1] == self.S_prev
        
        def SOC_last_rule(model):
            return model.S[self.stage] == 0.5*S
        
        def SOC_balance_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def generation_stage_rule(model):
            return model.g[self.stage] <= self.E_prev
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]
        
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, t):
            return model.u[t] - model.Q_c[t] <= model.phi_over[t]
        
        def imbalance_under_rule(model, t):
            return model.Q_c[t] - model.u[t] <= model.phi_under[t]
        
        ## settlement fcn
        
        def rt_settlement_fcn_stage_rule(model):
            return model.f_RT[self.stage] == (
                (model.u[self.stage] - model.Q_da[self.stage])*self.P_rt 
                - gamma_over*model.phi_over[self.stage] 
                - gamma_under*model.phi_under[self.stage]
                )

        model.da_awarded_amount = pyo.Constraint(model.TIME, rule = da_awarded_amount_rule)
        model.dispatch_stage = pyo.Constraint(rule = dispatch_stage_rule)
        model.SOC_init = pyo.Constraint(rule = SOC_init_rule)
        model.SOC_last = pyo.Constraint(rule = SOC_last_rule)
        model.SOC_balance = pyo.Constraint(model.TIME, rule = SOC_balance_rule)
        model.generation_stage = pyo.Constraint(rule = generation_stage_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.imbalance_over = pyo.Constraint(model.TIME, rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(model.TIME, rule = imbalance_under_rule)
        model.rt_settlement_stage_fcn = pyo.Constraint(rule = rt_settlement_fcn_stage_rule)
        
        # Objective
        
        def objective_rule(model):
            return sum(
                model.f_RT[t] 
                for t in model.TIME
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True   

    def get_settlement_fcn_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True

        fcn_value = pyo.value(self.f_RT[self.stage])
        
        return fcn_value


## 3) 2-stage stochastic programming model

class two_stage_da(pyo.ConcreteModel):
    
    def __init__(self, P_da_params, P_rt_exp_list):
        
        super().__init__()

        self.solved = False
        
        self.P_da_params = P_da_params
        self.P_rt_exp_list = P_rt_exp_list
        
        self.K = len(P_da_params)
                
        self.T = T

    def build_model(self):
        
        model = self.model()
        
        model.DANODE = pyo.RangeSet(0, self.K-1)
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)

        # Vars
        
        # Day Ahead variables
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        # Intraday bidding variables
        
        model.q_rt = pyo.Var(model.DANODE, model.TIME, domain = pyo.NonNegativeReals)
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.DANODE, model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals
        )
        
        model.g = pyo.Var(model.DANODE, model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.DANODE, model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.DANODE, model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.DANODE, model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.DANODE, model.TIME, domain = pyo.NonNegativeReals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(model.DANODE, model.TIME, domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(model.DANODE, model.TIME, domain = pyo.NonNegativeReals)
        
        ## settlement fcn Vars
        
        model.f_DA = pyo.Var(model.DANODE, model.TIME, domain = pyo.Reals)
        model.f_RT = pyo.Var(model.DANODE, model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## DA bidding & market rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= E_0_sum
        
        ## RT bidding & market rules
        
        def rt_bidding_amount_rule_1(model,k ,t):
            return model.q_rt[k, t] <= model.q_da[t] + omega
        
        def rt_bidding_amount_rule_2(model, k, t):
            return model.q_rt[k, t] >= model.q_da[t] - omega
        
        ## Operations rules
        
        def dispatch_rule(model, k, t):
            return model.Q_c[k, t] == model.q_rt[k, t]
        
        def SOC_init_rule(model, k):
            return model.S[k, -1] == 0.5*S
        
        def SOC_last_rule(model, k):
            return model.S[k, self.T-1] == 0.5*S
        
        def SOC_balance_rule(model, k, t):
            return model.S[k, t] == model.S[k, t-1] + v*model.c[k, t] - (1/v)*model.d[k, t]
        
        def generation_rule(model, k, t):
            return model.g[k, t] <= E_0_partial[t]
        
        def charge_rule(model, k, t):
            return model.c[k, t] <= model.g[k, t]
        
        def electricity_supply_rule(model, k, t):
            return model.u[k, t] == model.g[k, t] + model.d[k, t] - model.c[k, t]
    
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, k, t):
            return model.u[k, t] - model.Q_c[k, t] <= model.phi_over[k, t]
        
        def imbalance_under_rule(model, k, t):
            return model.Q_c[k, t] - model.u[k, t] <= model.phi_under[k, t]
        
        ## settlement fcn
        
        def da_settlement_fcn_rule(model, k, t):
            return model.f_DA[k, t] == self.P_da_params[k][t]*model.q_da[t] 
        
        def rt_settlement_fcn_rule(model, k, t):
            return model.f_RT[k, t] == (
                (model.u[k, t] - model.q_da[t])*self.P_rt_exp_list[k][t] 
                - gamma_over*model.phi_over[k, t] 
                - gamma_under*model.phi_under[k, t]
                )
        
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
        model.rt_bidding_amount_1 = pyo.Constraint(model.DANODE, model.TIME, rule = rt_bidding_amount_rule_1)
        model.rt_bidding_amount_2 = pyo.Constraint(model.DANODE, model.TIME, rule = rt_bidding_amount_rule_2)

        model.dispatch = pyo.Constraint(model.DANODE, model.TIME, rule = dispatch_rule)
        model.SOC_init = pyo.Constraint(model.DANODE, rule = SOC_init_rule)
        model.SOC_last = pyo.Constraint(model.DANODE, rule = SOC_last_rule)
        model.SOC_balance = pyo.Constraint(model.DANODE, model.TIME, rule = SOC_balance_rule)
        model.generation = pyo.Constraint(model.DANODE, model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.DANODE, model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.DANODE, model.TIME, rule = electricity_supply_rule)
        model.imbalance_over = pyo.Constraint(model.DANODE, model.TIME, rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(model.DANODE, model.TIME, rule = imbalance_under_rule)
        model.da_settlement_fcn = pyo.Constraint(model.DANODE, model.TIME, rule = da_settlement_fcn_rule)
        model.rt_settlement_fcn = pyo.Constraint(model.DANODE, model.TIME, rule = rt_settlement_fcn_rule)
        
        # Objective
        
        def objective_rule(model):
            return sum(
                sum(
                    model.f_DA[k, t] + model.f_RT[k, t] for k in model.DANODE
                    )*(1/len(self.P_da_params)) 
                for t in model.TIME
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)
        
    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True   
    
    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)

    def get_state_solutions(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append([pyo.value(self.q_da[t]) for t in range(self.T)])
        
        return State_var    

## 4) 3-stage stochastic programming model

class three_stage_da(pyo.ConcreteModel):
    
    def __init__(self, P_da_params, scenario_paths):
        
        super().__init__()

        self.solved = False
        
        self.P_da_params = P_da_params
        self.scenario_paths = scenario_paths
        
        self.K = len(scenario_paths)
        self.M = len(scenario_paths[0])
        
        self.delta_E_0_paths = [[[0 for _ in range(T)] for _ in range(self.M)] for _ in range(self.K)]
        self.P_rt_paths = [[[0 for _ in range(T)] for _ in range(self.M)] for _ in range(self.K)]
        self.delta_c_paths = [[[0 for _ in range(T)] for _ in range(self.M)] for _ in range(self.K)]
                
        self.T = T

        self._Param_setting()

    def _Param_setting(self):

        for k in range(self.K):
            
            scenario_path_list = self.scenario_paths[k]
            
            for m in range(self.M):
                
                scenario_path = scenario_path_list[m]
                
                for t in range(self.T):
                    
                    self.delta_E_0_paths[k][m][t] = scenario_path[t][0]
                    self.P_rt_paths[k][m][t] = scenario_path[t][1]
                    self.delta_c_paths[k][m][t] = scenario_path[t][2]
                    
    def build_model(self):
        
        model = self.model()
        
        model.DANODE = pyo.RangeSet(0, self.K-1)
        model.IDNODE = pyo.RangeSet(0, self.M-1)
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)

        # Vars
        
        # Day Ahead variables
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        # Intraday bidding variables
        
        model.q_rt = pyo.Var(
            model.DANODE, model.IDNODE, model.TIME, domain = pyo.NonNegativeReals
            )
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.DANODE, model.IDNODE, model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals
        )
        
        model.g = pyo.Var(model.DANODE, model.IDNODE, model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.DANODE, model.IDNODE, model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.DANODE, model.IDNODE, model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.DANODE, model.IDNODE, model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.DANODE, model.IDNODE, model.TIME, domain = pyo.NonNegativeReals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(model.DANODE, model.IDNODE, model.TIME, domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(model.DANODE, model.IDNODE, model.TIME, domain = pyo.NonNegativeReals)
        
        ## settlement fcn Vars
        
        model.f_DA = pyo.Var(model.DANODE, model.TIME, domain = pyo.Reals)
        model.f_RT = pyo.Var(model.DANODE, model.IDNODE, model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## DA bidding & market rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= E_0_sum
        
        ## RT bidding & market rules

        def rt_bidding_amount_init_rule(model, k, m):
            return model.q_rt[k, 0, 0] == model.q_rt[k, m, 0]

        def rt_bidding_amount_rule_1(model, k, m, t):
            return model.q_rt[k, m, t] <= model.q_da[t] + omega
        
        def rt_bidding_amount_rule_2(model, k, m, t):
            return model.q_rt[k, m, t] >= model.q_da[t] - omega
        
        ## Operations rules

        def dispatch_rule(model, k, m, t):
            return model.Q_c[k, m, t] == (1 + self.delta_c_paths[k][m][t])*model.q_rt[k, m, t]

        def SOC_init_rule(model, k, m):
            return model.S[k, m, -1] == 0.5*S
        
        def SOC_last_rule(model, k, m):
            return model.S[k, m, self.T-1] == 0.5*S
        
        def SOC_balance_rule(model, k, m, t):
            return model.S[k, m, t] == model.S[k, m, t-1] + v*model.c[k, m, t] - (1/v)*model.d[k, m, t]
        
        def generation_rule(model, k, m, t):
            return model.g[k, m, t] <= self.delta_E_0_paths[k][m][t]*E_0_partial[t]
        
        def charge_rule(model, k, m, t):
            return model.c[k, m, t] <= model.g[k, m, t]
        
        def electricity_supply_rule(model, k, m, t):
            return model.u[k, m, t] == model.g[k, m, t] + model.d[k, m, t] - model.c[k, m, t]
        
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, k, m, t):
            return model.u[k, m, t] - model.Q_c[k, m, t] <= model.phi_over[k, m, t]
        
        def imbalance_under_rule(model, k, m, t):
            return model.Q_c[k, m, t] - model.u[k, m, t] <= model.phi_under[k, m, t]
        
        ## settlement fcn

        def da_settlement_fcn_rule(model, k, t):
            return model.f_DA[k, t] == self.P_da_params[k][t]*model.q_da[t] 

        def rt_settlement_fcn_rule(model, k, m, t):
            return model.f_RT[k, m, t] == (
                (model.u[k, m, t] - model.q_da[t])*self.P_rt_paths[k][m][t] 
                - gamma_over*model.phi_over[k, m, t] 
                - gamma_under*model.phi_under[k, m, t]
                )
        
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
        model.rt_bidding_amount_init = pyo.Constraint(model.DANODE, model.IDNODE, rule = rt_bidding_amount_init_rule)
        model.rt_bidding_amount_1 = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = rt_bidding_amount_rule_1)
        model.rt_bidding_amount_2 = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = rt_bidding_amount_rule_2)
        model.dispatch = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = dispatch_rule)
        model.SOC_init = pyo.Constraint(model.DANODE, model.IDNODE, rule = SOC_init_rule)
        model.SOC_last = pyo.Constraint(model.DANODE, model.IDNODE, rule = SOC_last_rule)
        model.SOC_balance = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = SOC_balance_rule)
        model.generation = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = electricity_supply_rule)
        model.imbalance_over = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = imbalance_under_rule)
        model.da_settlement_fcn = pyo.Constraint(model.DANODE, model.TIME, rule = da_settlement_fcn_rule)
        model.rt_settlement_fcn = pyo.Constraint(model.DANODE, model.IDNODE, model.TIME, rule = rt_settlement_fcn_rule)
        
        # Objective
        
        def objective_rule(model):
            return sum(
                sum(
                    model.f_DA[k, t] + sum(model.f_RT[k, m, t] for m in model.IDNODE)*(1/self.M) 
                    for k in model.DANODE
                    )*(1/self.K) 
                for t in model.TIME
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)
        
    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True   
    
    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)

    def get_state_solutions(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append([pyo.value(self.q_da[t]) for t in range(self.T)])
        
        return State_var  


# 3. Algorithms instances


## PSDDiP Algorithm

### Sub processes for parallel computing in backward pass

def inner_product(t, pi, sol):
    
    return sum(pi[i]*sol[i] for i in [0, 2, 3]) + sum(pi[1][j]*sol[1][j] for j in range(T - t))


def basic_cut_rt(t, STAGE=T):
    # Matches _initialize_psi() for RT stage t
    v0 = 3 * 3600000 * (STAGE - t)
    psi_sub = [
        v0,                         # intercept
        0,                          # coeff for sol[0]
        [0 for _ in range(STAGE-t)],# coeffs for sol[1][*]
        0, 0                     # coeffs for sol[2], sol[3], sol[4]
    ]
    return psi_sub, v0

def basic_pi_rt(t, STAGE=T):
    # For process_lag_* which returns (pi, L)
    L0 = 3 * 3600000 * (STAGE - t)
    pi0 = [0, [0 for _ in range(STAGE-t)], 0, 0]
    return pi0, L0


#### Sub processes for SB cut generation

def process_single_subproblem_last_stage(j, prev_solution, P_da, delta, cut_mode):
    
    t_last = T - 1
    
    try:
        fw = fw_rt_last_LP_relax(prev_solution, P_da, delta)
        psi_sub = fw.get_cut_coefficients()

        if cut_mode in ['B']:
            v = psi_sub[0]

        return psi_sub, v
    
    except Exception:
        # Return the SAME “basic” cut as _initialize_psi
        return basic_cut_rt(t_last, STAGE=T)

def process_single_subproblem_inner_stage(j, t, prev_solution, psi_next, P_da, delta, cut_mode):
    
    try:
        fw = fw_rt_LP_relax(t, prev_solution, psi_next, P_da, delta)
        psi_sub = fw.get_cut_coefficients()

        if cut_mode in ['B']:
            v = psi_sub[0]

        return psi_sub, v
    
    except Exception:
        
        return basic_cut_rt(t, STAGE=T)


class PSDDiPModel:
        
    def __init__(
        self, 
        STAGE = T, 
        DA_params_reduced = Reduced_P_da[0], 
        DA_prob = Reduced_Probs[0],
        ID_params_reduced = Reduced_scenario_trees[0],
        DA_params_full = P_da_evaluate,
        DA_prob_full = Probs_evaluate,
        ID_params_full = Scenario_tree_evaluate,
        sample_num = 1000,
        alpha = 0.95, 
        tol = 0.005,
        breakstage_selection = 4,
        stopping_counter_limit = 5,
        approx_mode = False,
        ):

        ## STAGE = -1, 0, ..., STAGE - 1
        ## forward_scenarios(last iteration) = M
        ## baackward_scenarios = N_t

        self.STAGE = STAGE
        
        self.DA_params = DA_params_reduced
        self.DA_probs = DA_prob
        self.RT_params = ID_params_reduced
        
        self.DA_params_full = DA_params_full
        self.DA_probs_full = DA_prob_full
        self.ID_params_full = ID_params_full
        
        self.M = sample_num
                         
        self.alpha = alpha
        self.tol = tol
                
        self.breakstage_selection = breakstage_selection
                
        self.iteration = 0
        
        self.K = len(self.DA_params)
        self.K_eval = len(self.DA_params_full)   
            
        self.N_t = len(self.RT_params[0][0])
        
        self.approx_mode = approx_mode
                   
        self.start_time = time.time()
        self.running_time = 0
        
        self.gap = 1
        self.conv_rate = 2e3
        
        self.stopping_counter = 0
        self.stopping_counter_limit = stopping_counter_limit
                
        self.LB = [-np.inf]
        self.UB = [np.inf]
        
        self.eval = 0

        self.forward_solutions_da = [] ## Day Ahead Solution (Day Ahead Bidding)
        
        self.forward_solutions = [  ## x(-1), ..., x(T - 2)
                [
                    [] for _ in range(self.STAGE)
                ] for _ in range(self.K)
            ]
        
        self.q_da = [0 for _ in range(self.STAGE)]
        self.f_DA = [0 for _ in range(self.STAGE)]
        
        self.f_P = [0 for _ in range(self.STAGE)]
        self.f_E = [0 for _ in range(self.STAGE)]
        self.f_Im = [0 for _ in range(self.STAGE)]
        
        self.S_last_list = []
        
        self.psi_da = [     ## t = -1 -> Day-Ahead Stage (Multi-cut)
                [] for _ in range(self.K)
            ] 
        
        self.psi_da_1 = [
                [] for _ in range(self.K_eval)
            ]
        
        self.psi = [  ## t = {0 -> -1}, ..., {T - 1 -> T - 2}
                [
                    [] for _ in range(self.STAGE)
                ] for _ in range(self.K)
            ] 
                
        self._initialize_psi()
    
    # Initialize approximation of ECTGs        
    def _initialize_psi(self):
        
        self.psi_da = [
                    [                   
                        [
                            3*3600000*T,
                            [0 for _ in range(self.STAGE)]
                        ]
                    ] for _ in range(self.K)
            ]
        
        for k in range(self.K):
            for t in range(self.STAGE): ## psi(-1), ..., psi(T - 2)
                self.psi[k][t].append([
                    3*3600000*(self.STAGE - t), 
                    0, 
                    [0 for _ in range(self.STAGE - t)], 
                    0, 0
                    ])

    # Cut selection
    def cut_selection(self):
        
        ## cut selection for DA stage
        
        for k in range(self.K):
        
            min_index_da_list = []
        
            for n in range(self.iteration):
                
                sol = self.forward_solutions_da[n]
                
                V_da = []
                
                for coeff_da in self.psi_da[k]:
                        
                    V_da.append(
                        coeff_da[0] 
                        + sum(coeff_da[1][t]*sol[0][t] for t in range(self.STAGE))
                    )
                
                min_index_da = min(enumerate(V_da), key=lambda x: x[1])[0]

                min_index_da_list.append(min_index_da)

            min_index_da_list = list(dict.fromkeys(min_index_da_list))
            
            psi_da = [self.psi_da[k][i] for i in min_index_da_list] 
            
            self.psi_da[k] = psi_da
        
        ## cut selection for ID stage
        
        for k in range(self.K):
            
            for t in range(self.STAGE):
                
                min_index_rt_list = []
                
                for n in range(self.iteration):
                    
                    sol = self.forward_solutions[k][t][n]
                    
                    V_rt = []
                    
                    for coeff_rt in self.psi[k][t]:
                        
                        V_rt.append(
                            coeff_rt[0] 
                            + coeff_rt[1]*sol[0]
                            + sum(coeff_rt[2][t]*sol[1][t] for t in range(self.STAGE - t))
                            + coeff_rt[3]*sol[2]
                            + coeff_rt[4]*sol[3]
                        )
                    
                    min_index_rt = min(enumerate(V_rt), key=lambda x: x[1])[0]

                    min_index_rt_list.append(min_index_rt)
                
                min_index_rt_list = list(dict.fromkeys(min_index_rt_list))
                
                psi_rt = [self.psi[k][t][i] for i in min_index_rt_list]
                
                self.psi[k][t] = psi_rt
    
    
    # Forward Pass      
    def sample_scenarios(self):
        
        scenarios = []
        
        for k in range(self.K): 
            scenario = []
            
            scenario_params = self.RT_params[k]
            
            for stage_params in scenario_params:
                param = random.choice(stage_params)  
                scenario.append(param)
                
            scenarios.append(scenario)
                     
        return scenarios
    
    def sample_scenarios_for_stopping(self):
        
        scenarios = []
        
        for k in range(self.K):
            scenarios_k = []
            
            for _ in range(self.M):
                scenario = [random.choice(stage_params)
                            for stage_params in self.RT_params[k]]
                
                scenarios_k.append(scenario)
                
            scenarios.append(scenarios_k)
            
        return scenarios
            
    def sample_scenarios_1(self):
        
        scenarios = []
        
        for n in range(self.K_eval): 
            
            k = self.find_cluster_index_for_evaluation(n)
            
            scenario = []
            
            scenario_params = self.RT_params[k]
            
            for stage_params in scenario_params:
                param = random.choice(stage_params)  
                scenario.append(param)
                
            scenarios.append(scenario)
                     
        return scenarios
    
    def sample_scenarios_for_stopping_1(self):
        
        scenarios = []
        
        for n in range(self.K_eval):
            
            k = self.find_cluster_index_for_evaluation(n)
            
            scenarios_k = []
            
            for _ in range(self.M):
                scenario = [random.choice(stage_params)
                            for stage_params in self.RT_params[k]]
                
                scenarios_k.append(scenario)
                
            scenarios.append(scenarios_k)
            
        return scenarios
   
   
    def find_cluster_index_for_evaluation(self, n):
    
        actual_P_da = np.array(self.DA_params_full[n])
        centers = np.array(self.DA_params)  
        dists = np.linalg.norm(centers - actual_P_da, axis=1)
        
        return int(np.argmin(dists))
   
                  
    def forward_pass(self, scenarios):
        
        fw_da_subp = fw_da(self.DA_probs, self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
        
        self.forward_solutions_da.append(fw_da_state)
        
        mu_hat = 0.0
        
        S_last = []
                
        for k, scenario in enumerate(scenarios):
            
            P_da = self.DA_params[k]
                        
            fw_rt_init_subp = fw_rt_init(fw_da_state, self.psi[k][0], P_da)
            fw_rt_init_state = fw_rt_init_subp.get_state_solutions()
            
            self.forward_solutions[k][0].append(fw_rt_init_state) ## x(-1)
            
            state = fw_rt_init_state
            
            f_scenario = fw_rt_init_subp.get_settlement_fcn_value()
            
            for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                
                fw_rt_subp = fw_rt(t, state, self.psi[k][t+1], P_da, scenario[t])
                
                state = fw_rt_subp.get_state_solutions()
                
                self.forward_solutions[k][t+1].append(state)
                f_scenario += fw_rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            fw_rt_last_subp = fw_rt_last(state, P_da, scenario[self.STAGE-1])

            f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
            last_SOC_value = fw_rt_last_subp.get_last_SOC_value()
            S_last.append(last_SOC_value)
                                    
            mu_hat += f_scenario*self.DA_probs[k]
                
        self.S_last_list.append(S_last)

        z_alpha_half = 1.96  
        
        #self.LB.append(mu_hat - z_alpha_half * (sigma_hat / np.sqrt(self.M))) 
        self.LB.append(mu_hat) 
        
        return mu_hat
      
    def forward_pass_for_stopping(self, scenarios):
        
        fw_da_subp = fw_da(self.DA_probs, self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
        
        self.forward_solutions_da.append(fw_da_state)
        
        mu_hat = 0.0
        
        for k, scenarios_k in enumerate(scenarios):
            
            f = []
            
            P_da = self.DA_params[k]
            
            fw_rt_init_subp = fw_rt_init(fw_da_state, self.psi[k][0], P_da)
            fw_rt_init_state = fw_rt_init_subp.get_state_solutions()
            
            self.forward_solutions[k][0].append(fw_rt_init_state) ## x(-1)
            
            f_init = fw_rt_init_subp.get_settlement_fcn_value()
            
            for scenario in scenarios_k:
                
                state = fw_rt_init_state
                
                f_scenario = f_init
                
                for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                    
                    fw_rt_subp = fw_rt(t, state, self.psi[k][t+1], P_da, scenario[t])
                    
                    state = fw_rt_subp.get_state_solutions()
                    
                    self.forward_solutions[k][t+1].append(state)
                    f_scenario += fw_rt_subp.get_settlement_fcn_value()
                
                ## t = T-1
                
                fw_rt_last_subp = fw_rt_last(state, P_da, scenario[self.STAGE-1])

                f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
                            
                f.append(f_scenario)
            
            mu_hat += np.mean(f)*self.DA_probs[k]
            
        #sigma_hat = np.std(f, ddof=1)  

        #z_alpha_half = 1.96  
        
        #self.LB.append(mu_hat - z_alpha_half * (sigma_hat / np.sqrt(self.M))) 
        self.LB.append(mu_hat) 
        
        return mu_hat

    def forward_pass_1(self, scenarios):
        
        fw_da_subp = fw_da(self.DA_probs, self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
        
        self.forward_solutions_da.append(fw_da_state)
        
        f = []
        
        S_last = []
        
        for n, scenario in enumerate(scenarios):
            
            P_da = self.DA_params_full[n]
            
            k = self.find_cluster_index_for_evaluation(n)
            
            psi_ID_init = self.psi[k][0]
            
            fw_rt_init_subp = fw_rt_init(fw_da_state, psi_ID_init, P_da)
            fw_rt_init_state = fw_rt_init_subp.get_state_solutions()
            
            self.forward_solutions[k][0].append(fw_rt_init_state) ## x(-1)
            
            state = fw_rt_init_state
            
            f_scenario = fw_rt_init_subp.get_settlement_fcn_value()
            
            for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                
                psi_ID = self.psi[k][t+1]
                
                fw_rt_subp = fw_rt(t, state, psi_ID, P_da, scenario[t])
                
                state = fw_rt_subp.get_state_solutions()
                
                self.forward_solutions[k][t+1].append(state)
                f_scenario += fw_rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            fw_rt_last_subp = fw_rt_last(state, P_da, scenario[self.STAGE-1])

            f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
            last_SOC_value = fw_rt_last_subp.get_last_SOC_value()
            S_last.append(last_SOC_value)
                                    
            f.append(f_scenario)
                
        self.S_last_list.append(S_last)
        
        mu_hat = np.mean(f)
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        #self.LB.append(mu_hat - z_alpha_half * (sigma_hat / np.sqrt(self.M))) 
        self.LB.append(mu_hat) 
        
        return mu_hat
      
    def forward_pass_for_stopping_1(self, scenarios):
        
        fw_da_subp = fw_da(self.DA_probs, self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
        
        self.forward_solutions_da.append(fw_da_state)
        
        mu_hat = 0.0
        
        for n, scenarios_n in enumerate(scenarios):
            
            f = []
            
            P_da = self.DA_params_full[n]
            
            k = self.find_cluster_index_for_evaluation(n)
            
            psi_ID_init = self.psi[k][0]
            
            fw_rt_init_subp = fw_rt_init(fw_da_state, psi_ID_init, P_da)
            fw_rt_init_state = fw_rt_init_subp.get_state_solutions()
            
            self.forward_solutions[k][0].append(fw_rt_init_state) ## x(-1)
            
            f_init = fw_rt_init_subp.get_settlement_fcn_value()
            
            for scenario in scenarios_n:
                
                state = fw_rt_init_state
                
                f_scenario = f_init
                
                for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                    
                    psi_ID = self.psi[k][t+1]
                    fw_rt_subp = fw_rt(t, state, psi_ID, P_da, scenario[t])
                    
                    state = fw_rt_subp.get_state_solutions()
                    
                    self.forward_solutions[k][t+1].append(state)
                    f_scenario += fw_rt_subp.get_settlement_fcn_value()
                
                ## t = T-1
                
                fw_rt_last_subp = fw_rt_last(state, P_da, scenario[self.STAGE-1])

                f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
                            
                f.append(f_scenario)
            
            mu_hat += np.mean(f)*self.DA_probs_full[n]
            
        #sigma_hat = np.std(f, ddof=1)  

        #z_alpha_half = 1.96  
        
        #self.LB.append(mu_hat - z_alpha_half * (sigma_hat / np.sqrt(self.M))) 
        self.LB.append(mu_hat) 
        
        return mu_hat

       
    # Backward Pass    
    def inner_product(self, t, pi, sol):
        
        return sum(pi[i]*sol[i] for i in [0, 2, 3]) + sum(pi[1][j]*sol[1][j] for j in range(self.STAGE - t))

    def inner_product_da(self, pi, sol):
        
        return sum(sum(pi[i][t]*sol[i][t] for t in range(self.STAGE)) for i in [0]) 


    def backward_pass(self):

        for k, P_da in enumerate(self.DA_params):
            
            stage_params = self.RT_params[k]

            with mp.Pool() as pool:

                t_last        = self.STAGE - 1
                prev_solution = self.forward_solutions[k][t_last][-1]
                deltas_last   = stage_params[t_last]

                last_args = [
                    (j, prev_solution, P_da, deltas_last[j], 'B')
                    for j in range(self.N_t)
                ]
                
                last_results = pool.starmap(
                    process_single_subproblem_last_stage,
                    last_args
                )

                v_sum   = 0.0
                pi_mean = [0, [0], 0, 0]
                
                for psi_sub, v in last_results:
                    
                    v_sum += v
                    pi_mean[0]    += psi_sub[1]    / self.N_t
                    pi_mean[1][0] += psi_sub[2][0] / self.N_t
                    pi_mean[2]    += psi_sub[3]    / self.N_t
                    pi_mean[3]    += psi_sub[4]    / self.N_t

                v = v_sum/self.N_t - self.inner_product(t_last, pi_mean, prev_solution)

                cut_coeff = [v] + pi_mean
                self.psi[k][t_last].append(cut_coeff)

                for t in range(self.STAGE - 2, -1, -1):
                    prev_solution = self.forward_solutions[k][t][-1]
                    psi_next      = self.psi[k][t+1]
                    deltas        = stage_params[t]

                    inner_args = [
                        (j, t, prev_solution, psi_next, P_da, deltas[j], 'B')
                        for j in range(self.N_t)
                    ]
                    inner_results = pool.starmap(
                        process_single_subproblem_inner_stage,
                        inner_args
                    )

                    v_sum   = 0.0
                    pi_mean = [0, [0]*(self.STAGE - t), 0, 0]
                    
                    for psi_sub, v in inner_results:
                        v_sum += v
                        pi_mean[0] += psi_sub[1] / self.N_t
                        
                        for i in range(self.STAGE - t):
                            pi_mean[1][i] += psi_sub[2][i] / self.N_t
                        pi_mean[2] += psi_sub[3] / self.N_t
                        pi_mean[3] += psi_sub[4] / self.N_t

                    v = v_sum/self.N_t - self.inner_product(t, pi_mean, prev_solution)

                    cut_coeff = [v] + pi_mean
                    self.psi[k][t].append(cut_coeff)

        prev_solution = self.forward_solutions_da[-1]

        for k, P_da in enumerate(self.DA_params):
                        
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(
                prev_solution, self.psi[k][0], P_da
                )
            
            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            pi = [psi_sub[1]] 

            v = psi_sub[0] - self.inner_product_da(pi, prev_solution)
            
            cut_coeff = [v] + [pi[0]] 
            self.psi_da[k].append(cut_coeff)
        
        fw_da_for_UB = fw_da(self.DA_probs, self.psi_da)
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))
  
    def backward_pass_1(self):

        for k, P_da in enumerate(self.DA_params):
            
            stage_params = self.RT_params[k]

            with mp.Pool() as pool:

                t_last        = self.STAGE - 1
                prev_solution = self.forward_solutions[k][t_last][-1]
                deltas_last   = stage_params[t_last]

                last_args = [
                    (j, prev_solution, P_da, deltas_last[j], 'B')
                    for j in range(self.N_t)
                ]
                
                last_results = pool.starmap(
                    process_single_subproblem_last_stage,
                    last_args
                )

                v_sum   = 0.0
                pi_mean = [0, [0], 0, 0]
                
                for psi_sub, v in last_results:
                    
                    v_sum += v
                    pi_mean[0]    += psi_sub[1]    / self.N_t
                    pi_mean[1][0] += psi_sub[2][0] / self.N_t
                    pi_mean[2]    += psi_sub[3]    / self.N_t
                    pi_mean[3]    += psi_sub[4]    / self.N_t

                v = v_sum/self.N_t - self.inner_product(t_last, pi_mean, prev_solution)

                cut_coeff = [v] + pi_mean
                self.psi[k][t_last].append(cut_coeff)

                for t in range(self.STAGE - 2, -1, -1):
                    prev_solution = self.forward_solutions[k][t][-1]
                    psi_next      = self.psi[k][t+1]
                    deltas        = stage_params[t]

                    inner_args = [
                        (j, t, prev_solution, psi_next, P_da, deltas[j], 'B')
                        for j in range(self.N_t)
                    ]
                    inner_results = pool.starmap(
                        process_single_subproblem_inner_stage,
                        inner_args
                    )

                    v_sum   = 0.0
                    pi_mean = [0, [0]*(self.STAGE - t), 0, 0]
                    
                    for psi_sub, v in inner_results:
                        v_sum += v
                        pi_mean[0] += psi_sub[1] / self.N_t
                        
                        for i in range(self.STAGE - t):
                            pi_mean[1][i] += psi_sub[2][i] / self.N_t
                        pi_mean[2] += psi_sub[3] / self.N_t
                        pi_mean[3] += psi_sub[4] / self.N_t

                    v = v_sum/self.N_t - self.inner_product(t, pi_mean, prev_solution)

                    cut_coeff = [v] + pi_mean
                    self.psi[k][t].append(cut_coeff)

        prev_solution = self.forward_solutions_da[-1]

        for n, P_da in enumerate(self.DA_params_full):
            
            k_nearest = self.find_cluster_index_for_evaluation(n)
            
            psi_ID = self.psi[k_nearest][0]
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(
                prev_solution, psi_ID, P_da
                )
            
            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            pi = [psi_sub[1]] 

            v = psi_sub[0] - self.inner_product_da(pi, prev_solution)
            
            cut_coeff = [v] + [pi[0]] 
            self.psi_da_1[n].append(cut_coeff)
        
        fw_da_for_UB = fw_da(self.DA_probs_full, self.psi_da_1)
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))

    # Main Loop
    def stopping_criterion(self):
        
        self.gap = (self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]
        
        if self.iteration >= 5:
            self.conv_rate = abs(self.UB[self.iteration] - self.UB[self.iteration -3])
        
        self.running_time = time.time() - self.start_time
        
        if (
            self.gap <= self.tol 
        ):
            self.stopping_counter += 1
          
        if (
            self.stopping_counter >= self.stopping_counter_limit
        ):
            return True

        return False

    def run_sddip(self):
        
        final_pass = False

        self.start_time = time.time()
        self.running_time = 0
    
        while True:
            
            if self.approx_mode:
                
                if not final_pass and self.stopping_criterion():
                    final_pass = True
                    
                    print("\n>>> Stopping criterion met. Performing final pass with M scenarios...")
                    
                elif final_pass:
                                    
                    break
                    
                self.iteration += 1

                if final_pass:
                    scenarios = self.sample_scenarios_for_stopping_1()
                        
                    self.forward_pass_for_stopping_1(scenarios)
                        
                else:
                    scenarios = self.sample_scenarios_1()

                    self.forward_pass_1(scenarios)

                #print(f"  LB for iter = {self.iteration} updated to: {self.LB[self.iteration]:.4f}")

                self.backward_pass_1()
                
                self.gap = (self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]
                
                if self.iteration >= self.breakstage_selection + 1:
                    
                    self.cut_selection()
                
                #print(f"cut_num = {(len(self.psi[0][2]))}")
                
                #print(f"  UB for iter = {self.iteration} updated to: {self.UB[self.iteration]:.4f}")

            else:
                
                if not final_pass and self.stopping_criterion():
                    final_pass = True
                    
                    print("\n>>> Stopping criterion met. Performing final pass with M scenarios...")
                    
                elif final_pass:
                                    
                    break
                    
                self.iteration += 1

                if final_pass:
                    scenarios = self.sample_scenarios_for_stopping()
                        
                    self.forward_pass_for_stopping(scenarios)
                        
                else:
                    scenarios = self.sample_scenarios()

                    self.forward_pass(scenarios)

                #print(f"  LB for iter = {self.iteration} updated to: {self.LB[self.iteration]:.4f}")

                self.backward_pass()
                
                self.gap = (self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]
                
                if self.iteration >= self.breakstage_selection + 1:
                    
                    self.cut_selection()
                
                #print(f"cut_num = {(len(self.psi[0][2]))}")
                
                #print(f"  UB for iter = {self.iteration} updated to: {self.UB[self.iteration]:.4f}")

        print(f"\nPSDDiPModel for price setting = {price_setting}")
        print(f"SDDiP complete. for T = {self.STAGE}, k = {self.K}")
        print(f"Final LB = {self.LB[self.iteration]:.4f}, UB = {self.UB[self.iteration]:.4f}, gap = {self.gap:.4f}")
        print(f"iteration : {self.iteration}, total_time : {self.running_time:.2f} seconds\n")


# 4. Main execution

if __name__ == "__main__":

    ## 1. Sample scenario paths and save as .npy
    
    random.seed(42)
    np.random.seed(42)

    evaluation_num = int(300/K_eval)
    
    approx_mode = False  

    def sample_scenario_paths(Scenario_tree, N):
        scenarios = []
        for k in range(K_eval):
            scenarios_k = []
            for _ in range(N):
                scenario = [
                    random.choice(stage_params)
                    for stage_params in Scenario_tree[k]
                ]
                scenarios_k.append(scenario)
            scenarios.append(scenarios_k)
        return scenarios
    
    # Sample
    scenarios_for_eval = sample_scenario_paths(Scenario_tree_evaluate, evaluation_num)
    scenarios_for_test = sample_scenario_paths(Scenario_tree_evaluate, evaluation_num)
    scenarios_for_SP = sample_scenario_paths(Scenario_tree_evaluate, evaluation_num)

    # Base directory
    BASE_OUTDIR = Path(f"./Stochastic_Approach/NestedBenders/scenario_paths/{price_setting}")
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

    # Save as npy (single file each)
    np.save(BASE_OUTDIR / "scenarios_eval.npy", np.asarray(scenarios_for_eval, dtype=object))
    np.save(BASE_OUTDIR / "scenarios_test.npy", np.asarray(scenarios_for_test, dtype=object))
    np.save(BASE_OUTDIR / "scenarios_SP.npy", np.asarray(scenarios_for_SP, dtype=object))

    print("✅ Scenario paths saved as .npy:")
    print(BASE_OUTDIR)
    
    
    ## 2. Run full-PSDDiP to get psi_ID and save as npy
    
    """
    def save_psddip_state(model, base_dir, price_setting):
        
        base_dir = Path(base_dir)
        base_dir.mkdir(exist_ok=True)

        state = {
            "psi_ID": model.psi,
            "psi_DA": model.psi_da,
            "iteration": model.iteration,
            "LB": model.LB,
            "UB": model.UB,
            "forward_solutions": model.forward_solutions,
            "forward_solutions_da": model.forward_solutions_da,
        }

        np.save(
            base_dir / f"{price_setting}_state.npy",
            state,
            allow_pickle=True
        )

        print("✅ PSDDiP checkpoint saved")

    def load_psddip_state(model, base_dir, price_setting):
        base_dir = Path(base_dir)
        state_path = base_dir / f"{price_setting}_state.npy"

        if not state_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {state_path}")

        state = np.load(state_path, allow_pickle=True).item()

        model.psi = state["psi_ID"]
        model.psi_da = state["psi_DA"]
        model.iteration = state["iteration"]
        model.LB = state["LB"]
        model.UB = state["UB"]
        model.forward_solutions = state["forward_solutions"]
        model.forward_solutions_da = state["forward_solutions_da"]

        print(f"✅ PSDDiP checkpoint loaded (iteration={model.iteration})")
    
    BASE_DIR = Path(__file__).resolve().parent
    PSI_DIR  = BASE_DIR / "psi_full_LP"
    PSI_DIR.mkdir(exist_ok=True)
    
    psddip_multi_full = PSDDiPModel(
        STAGE=T,
        DA_params_reduced=P_da_evaluate,
        DA_prob=Probs_evaluate,
        ID_params_reduced=Scenario_tree_evaluate,
        DA_params_full=P_da_evaluate,
        DA_prob_full=Probs_evaluate,
        sample_num=evaluation_num,
        alpha=0.95,
        tol=1e-4,
        breakstage_selection=10,
        stopping_counter_limit=9,
        approx_mode=approx_mode,
    )

    checkpoint_path = PSI_DIR / f"{price_setting}_state.npy"
    
    if checkpoint_path.exists():
        load_psddip_state(
            psddip_multi_full,
            PSI_DIR,
            price_setting
        )
    else:
        print("ℹ️ No checkpoint found — starting fresh")
    
    
    psddip_multi_full.run_sddip()
    
    save_psddip_state(
        psddip_multi_full,
        PSI_DIR,
        price_setting
    )"""
 
 
    """### Convergence of Final SOC Across Iterations (scenario-wise + statistics)

    import numpy as np
    import matplotlib.pyplot as plt

    # S_last_list_convergence: shape (N_iter, N_scenarios)
    S_last_list_convergence = psddip_multi_full.S_last_list
    S_arr = np.asarray(S_last_list_convergence)

    N_iter, N_scenarios = S_arr.shape
    iterations = np.arange(1, N_iter + 1)

    # statistics across scenarios (per iteration)
    mean_S = S_arr.mean(axis=1)
    q10 = np.quantile(S_arr, 0.10, axis=1)
    q90 = np.quantile(S_arr, 0.90, axis=1)

    plt.figure(figsize=(10, 6))

    # 1) scenario-wise trajectories (thin, transparent)
    for s in range(N_scenarios):
        plt.plot(iterations, S_arr[:, s], color="gray", alpha=0.25)

    # 2) quantile band
    plt.fill_between(iterations, q10, q90, color="tab:blue", alpha=0.25, label="10–90% quantile")

    # 3) mean trajectory
    plt.plot(iterations, mean_S, color="black", linewidth=2.5, label="Mean")

    plt.xlabel("Iteration")
    plt.ylabel("Final State of Charge (SOC)")
    plt.title("Convergence of Final SOC Across Iterations")

    plt.ylim(0, S)        # physical SOC bounds
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()"""
    
 
 
    ## 3. Run each PSDDiP for K in K_list to get psi_DA and save as npy
    """    
    for k, K in enumerate([K for K in K_list if K <= 30]):
        
        psddip_multi_da = PSDDiPModel(
            STAGE = T,
            DA_params_reduced=Reduced_P_da[k],
            DA_prob=Reduced_Probs[k],
            ID_params_reduced=Reduced_scenario_trees[k],
            DA_params_full=P_da_evaluate,
            DA_prob_full=Probs_evaluate,
            sample_num=int(500/K),
            alpha = 0.95,
            tol=1e-4,
            breakstage_selection=10,
            stopping_counter_limit=4,
            approx_mode=approx_mode,
        )
        
        psddip_multi_da.run_sddip()
        
        BASE_DIR = Path(__file__).resolve().parent           # NestedBenders
                
        PSI_DIR  = BASE_DIR / "psi_DA_LP" / f"{price_setting}"
        PSI_DIR.mkdir(parents=True, exist_ok=True)

        psi_path = PSI_DIR / f"psi_DA_{K}.npy"

        np.save(
            psi_path,
            np.array(psddip_multi_da.psi_da, dtype=object),
            allow_pickle=True
        )

    print("✅ psd_DA saved as .npy:")
    """
 
    ## 4. Notify done via plot
    
    def notify_done_via_plot(title="✅ PSDDiP finished", subtitle="hyb-40 & hyb-80 complete"):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=20, weight="bold")
        ax.text(0.5, 0.40, subtitle, ha="center", va="center", fontsize=12)
        try:
            fig.canvas.manager.set_window_title("PSDDiP — Done")
        except Exception:
            pass
        plt.show()
    

    notify_done_via_plot()
        

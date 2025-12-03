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



# Load Price and Scenario csv files

K_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 500, 500]

price_setting = 'sunny'  # 'cloudy', 'normal', 'sunny'

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
    # collect K folders like 'K1','K5',... (numbers only)
    k_dirs = [d for d in os.listdir(base_dir)
              if d.startswith('K') and d[1:].isdigit()
              and os.path.isdir(os.path.join(base_dir, d))]
    k_dirs.sort(key=lambda d: int(d[1:]))

    Reduced_scenario_trees = []

    for kdir in k_dirs:
        k_path = os.path.join(base_dir, kdir)
        # only scenario_*.csv; ignore probs.csv or other files
        scen_files = [n for n in os.listdir(k_path)
                      if _tree_re.match(n) and n.endswith('.csv')]
        scen_files.sort(key=lambda n: int(_tree_re.match(n).group(1)))

        trees = []
        for fname in scen_files:
            fpath = os.path.join(k_path, fname)
            data = np.loadtxt(fpath, delimiter=',')

            if data.ndim == 1:      # ensure 2-D
                data = data.reshape(1, -1)

            # reconstruct: 24 time levels; rows = [t, b, ...branch...]
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


# Plotting

"""
T = 24
hours = np.arange(T)

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
outdir = './Stochastic_Approach/Scenarios/P_rt_fan_byK'
os.makedirs(outdir, exist_ok=True)

for k, scenario_trees in zip(K_list, Reduced_scenario_trees):
    # --- collect all trajectories for this K into an array [n_paths, 24] ---
    paths = []
    for scenario in scenario_trees:
        N_b = len(scenario[0])  # branches at hour 0
        for b in range(N_b):
            traj = [scenario[t][b][1] for t in range(24)]
            paths.append(traj)
    P = np.asarray(paths)  # shape: (n_paths, 24)
    if P.ndim != 2 or P.shape[1] != 24:
        raise ValueError(f"Unexpected shape for P (got {P.shape})")

    # --- compute quantiles hour-by-hour ---
    q25 = np.percentile(P, 25, axis=0)
    q75 = np.percentile(P, 75, axis=0)
    mean = P.mean(axis=0)

    # --- plot fan chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Central 50% band (green)
    band_50 = ax.fill_between(hours, q25, q75, alpha=0.3, color="green", label='Central 50% (25–75%)')

    # Sample paths in **black**
    n_show = min(15, len(P))  # show up to 15 paths
    rng_idx = np.linspace(0, len(P) - 1, n_show, dtype=int)
    for idx in rng_idx:
        ax.plot(hours, P[idx], color='black', alpha=0.2, linewidth=1.0)

    # Mean line (orange dashed)
    mean_line, = ax.plot(hours, mean, linewidth=1.8, linestyle='--', color='orange', label='Mean')

    # Axes, ticks, limits
    ax.set_title(f"K = {k}: Real-Time Price Scenarios", fontsize=20)
    ax.set_xlabel("Hour", fontsize=20)
    ax.set_ylabel("P_rt (KRW)", fontsize=20)
    ax.set_ylim(-120, 200)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(['0h', '6h', '12h', '18h', '24h'], fontsize=20)
    ax.grid(True)

    # Legend (only Central 50% + Mean)
    ax.legend(
        loc='upper right',
        frameon=True,
        fontsize=18,        # font size for labels like "Central 50%"
        title_fontsize=18   # font size for "Legend"
    )

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f'P_rt_fan_K{k}.png'), dpi=300)
    plt.show()
    plt.close(fig)
"""

# SDDiP Model Parameters

T = 24
D = 1

E_0 = E_0

#print(f'E_0 = {E_0}')

C = 21022.1
S = C*3
B = C

S_min = 0.1*S
S_max = 0.9*S

P_r = 80
P_max = 270

v = 0.95

gamma_over = P_max
gamma_under = P_max

if T == 24:
    Total_time = 3600*3
    
    E_0_partial = E_0
    
    Reduced_P_da = Reduced_P_da
    Reduced_scenario_trees = Reduced_scenario_trees
    
elif T in [7, 10]:
    Total_time = 10
    start = 9
    E_0_partial = E_0[start:start+T]
    Reduced_P_da = [np.array(cluster)[:, start:start+T] for cluster in Reduced_P_da]
    Reduced_scenario_trees = [
        [scenario_tree[start:start+T] for scenario_tree in trees_by_k]
        for trees_by_k in Reduced_scenario_trees
    ]

elif T in [2, 4]:
    Total_time = 20
    start = 9
    E_0_partial = E_0[start:start+T]
    Reduced_P_da = [np.array(cluster)[:, start:start+T] for cluster in Reduced_P_da]
    Reduced_scenario_trees = [
        [scenario_tree[start:start+T] for scenario_tree in trees_by_k]
        for trees_by_k in Reduced_scenario_trees
    ]

P_da_evaluate = Reduced_P_da[-2]
Scenario_tree_evaluate = Reduced_scenario_trees[-2]

P_da_test = Reduced_P_da[-1]
Scenario_tree_test = Reduced_scenario_trees[-1]


K_eval = len(P_da_evaluate)


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
    
K = [1.23*E_0_partial[t] + 1.02*B for t in range(T)]

M_gen = [[1.04*K[t], 2*K[t]] for t in range(T)]


# Subproblems for SDDiP

## stage = DA

class fw_da(pyo.ConcreteModel): 
    
    def __init__(self, psi):
        
        super().__init__()

        self.solved = False
        
        self.psi = psi
        
        self.T = T

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
                
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        model.b = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals, initialize = 0.0)
        model.q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        # Constraints
        
        def da_bidding_amount_rule(model, t):
            return model.q[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q[t] for t in range(self.T)) <= E_0_sum
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= (
                self.psi[l][0]
                +sum(self.psi[l][1][t]*model.b[t] for t in range(T)) 
                +sum(self.psi[l][2][t]*model.q[t] for t in range(T)) 
                )

        model.da_bidding_amount = pyo.Constraint(model.TIME, rule=da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule=da_overbid_rule)
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)

        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta
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
        
        State_var.append([pyo.value(self.b[t]) for t in range(self.T)])
        State_var.append([pyo.value(self.q[t]) for t in range(self.T)])
        
        return State_var 

    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        return pyo.value(self.objective)        

## stage = DA (Multi-cut)

class fw_da_multicut(pyo.ConcreteModel): 
    
    def __init__(self, psi):
        
        super().__init__()

        self.solved = False
        
        self.psi = psi
        
        self.K = len(self.psi)
        
        self.T = T

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.KRANGE = pyo.RangeSet(0, self.K-1)
                
        model.PSIRANGE = pyo.Set(model.KRANGE, initialize=lambda _m, k: range(len(self.psi[k])))
        
        # Vars
        
        def _value_index_init(m): ## processing different dimension of psi[k]
            for k in m.KRANGE:
                for l in m.PSIRANGE[k]:
                    yield (k, l)
                    
        model.VALUE_INDEX = pyo.Set(dimen=2, initialize=_value_index_init)
        
        model.theta = pyo.Var(model.KRANGE, domain = pyo.Reals)
        
        model.b = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals, initialize = -P_r)
        model.q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        # Constraints
        
        def da_bidding_amount_rule(model, t):
            return model.q[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q[t] for t in range(self.T)) <= E_0_sum
        
        def value_fcn_approx_rule(model, k, l):
            
            v, pi_b, pi_q = self.psi[k][l]
            
            return model.theta[k] <= (
                v
                +sum(pi_b[t]*model.b[t] for t in range(T)) 
                +sum(pi_q[t]*model.q[t] for t in range(T)) 
                )

        model.da_bidding_amount = pyo.Constraint(model.TIME, rule=da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule=da_overbid_rule)
        model.value_fcn_approx = pyo.Constraint(model.VALUE_INDEX, rule=value_fcn_approx_rule)

        # Obj Fcn
        
        def objective_rule(model):
            return (
                sum(model.theta[k] for k in range(self.K))/self.K
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
        
        State_var.append([pyo.value(self.b[t]) for t in range(self.T)])
        State_var.append([pyo.value(self.q[t]) for t in range(self.T)])
        
        return State_var 

    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        return pyo.value(self.objective)        


## stage = -1

class fw_rt_init(pyo.ConcreteModel): 
    
    def __init__(self, da_prev, psi, P_da):
        
        super().__init__()

        self.solved = False
        
        self.b_prev = da_prev[0]
        self.q_prev = da_prev[1]
        
        self.psi = psi
        self.T = T
        self.P_da = P_da
        
        self.M_price = [[0, 0] for _ in range(self.T)]
        
        self._BigM_setting()
    
    def _BigM_setting(self):
        
        for t in range(self.T):
            
            if self.P_da[t] >= 0:
                self.M_price[t][0] = 10
                self.M_price[t][1] = self.P_da[t] + P_r

            else:
                self.M_price[t][0] = -self.P_da[t] + 10
                self.M_price[t][1] = self.P_da[t] + P_r
            
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_o = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to DA stage
        
        def da_b_rule(model, t):
            return model.b_da[t] == self.b_prev[t]
        
        def da_q_rule(model, t):
            return model.q_da[t] == self.q_prev[t]
        
        ## Day-Ahead Market Rules
        
        def market_clearing_1_rule(model, t):
            return model.b_da[t] - self.P_da[t] <= self.M_price[t][0]*(1 - model.n_da[t])
        
        def market_clearing_2_rule(model, t):
            return self.P_da[t] - model.b_da[t] <= self.M_price[t][1]*model.n_da[t]
        
        def market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def market_clearing_4_rule(model, t):
            return model.Q_da[t] <= M_gen[t][0]*model.n_da[t]
        
        def market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - M_gen[t][0]*(1 - model.n_da[t])
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_amount_rule(model):
            return model.q_rt <= E_0_partial[0] + B
        
        def rt_overbid_rule(model):
            return model.T_o <= E_0_sum
        
        ## State variable trainsition
        
        def State_SOC_rule(model):
            return model.S == 0.5*S

        def state_Q_da_rule(model, t):
            return model.T_Q[t] == model.Q_da[t]
        
        def State_o_rule(model):
            return model.T_o == model.q_rt
        
        def State_b_rule(model):
            return model.T_b == model.b_rt
        
        def State_q_rule(model):
            return model.T_q == model.q_rt
        
        def State_E_rule(model):
            return model.T_E == 0
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= (
                self.psi[l][0] 
                + self.psi[l][1]*model.S 
                + sum(self.psi[l][2][t]*model.T_Q[t] for t in range(T)) 
                + self.psi[l][3]*model.T_o 
                + self.psi[l][4]*model.T_b 
                + self.psi[l][5]*model.T_q 
                + self.psi[l][6]*model.T_E
                )
        
        def settlement_fcn_rule(model):
            return model.f == sum(
                self.P_da[t]*model.Q_da[t] for t in range(self.T)
            )
            
        model.da_b_amount = pyo.Constraint(model.TIME, rule = da_b_rule)
        model.da_q_amount = pyo.Constraint(model.TIME, rule = da_q_rule)
        model.market_clearing_1 = pyo.Constraint(model.TIME, rule = market_clearing_1_rule)
        model.market_clearing_2 = pyo.Constraint(model.TIME, rule = market_clearing_2_rule)
        model.market_clearing_3 = pyo.Constraint(model.TIME, rule = market_clearing_3_rule)
        model.market_clearing_4 = pyo.Constraint(model.TIME, rule = market_clearing_4_rule)
        model.market_clearing_5 = pyo.Constraint(model.TIME, rule = market_clearing_5_rule)
        model.rt_bidding_amount = pyo.Constraint(rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)    
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.state_Q_da = pyo.Constraint(model.TIME, rule = state_Q_da_rule)
        model.state_o = pyo.Constraint(rule = State_o_rule)
        model.state_b = pyo.Constraint(rule = State_b_rule)
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
        SOLVER.solve(self)
        self.solved = True

    def get_state_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append(pyo.value(self.S))
        State_var.append([pyo.value(self.T_Q[t]) for t in range(self.T)])
        State_var.append(pyo.value(self.T_o))
        State_var.append(pyo.value(self.T_b))
        State_var.append(pyo.value(self.T_q))
        State_var.append(pyo.value(self.T_E))
        
        return State_var 
 
    def get_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True

        solutions = {
            "b_da": [pyo.value(self.b_da[t]) for t in range(self.T)],
            "b_rt": pyo.value(self.b_rt),
            "Q_da": [pyo.value(self.Q_da[t]) for t in range(self.T)],
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

        DA_profit = [self.P_da[t]*pyo.value(self.Q_da[t]) for t in range(self.T)]

        return DA_profit

class fw_rt_init_LP_relax(pyo.ConcreteModel): ## (Backward - Benders' Cut) 
    
    def __init__(self, da_prev, psi, P_da):
        
        super().__init__()

        self.solved = False
        
        self.b_prev = da_prev[0]
        self.q_prev = da_prev[1]
        
        self.psi = psi
        self.T = T
        self.P_da = P_da
        
        self.M_price = [[0, 0] for t in range(self.T)]
        
        self._BigM_setting()
    
    def _BigM_setting(self):
        
        for t in range(self.T):
            
            if self.P_da[t] >= 0:
                self.M_price[t][0] = 10
                self.M_price[t][1] = self.P_da[t] + P_r

            else:
                self.M_price[t][0] = -self.P_da[t] + 10
                self.M_price[t][1] = self.P_da[t] + P_r
   
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.z_q = pyo.Var(model.TIME, domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Other
        
        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, bounds = (0, 1), domain = pyo.Reals)
        
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_o = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_b(model, t):
            return model.z_b[t] == self.b_prev[t]
        
        def auxiliary_q(model, t):
            return model.z_q[t] == self.q_prev[t]
        
        ## Connected to DA stage
        
        def da_b_rule(model, t):
            return model.b_da[t] == model.z_b[t]
        
        def da_q_rule(model, t):
            return model.q_da[t] == model.z_q[t]
        
        ## Day-Ahead Market Rules
        
        def market_clearing_1_rule(model, t):
            return model.b_da[t] - self.P_da[t] <= self.M_price[t][0]*(1 - model.n_da[t])
        
        def market_clearing_2_rule(model, t):
            return self.P_da[t] - model.b_da[t] <= self.M_price[t][1]*model.n_da[t]
        
        def market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def market_clearing_4_rule(model, t):
            return model.Q_da[t] <= M_gen[t][0]*model.n_da[t]
        
        def market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - M_gen[t][0]*(1 - model.n_da[t])
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_amount_rule(model):
            return model.q_rt <= E_0_partial[0] + B
        
        def rt_overbid_rule(model):
            return model.T_o <= E_0_sum
        
        ## State variable trainsition
        
        def State_SOC_rule(model):
            return model.S == 0.5*S

        def state_Q_da_rule(model, t):
            return model.T_Q[t] == model.Q_da[t]
        
        def State_o_rule(model):
            return model.T_o == model.q_rt
        
        def State_b_rule(model):
            return model.T_b == model.b_rt
        
        def State_q_rule(model):
            return model.T_q == model.q_rt
        
        def State_E_rule(model):
            return model.T_E == 0
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= (
                self.psi[l][0] 
                + self.psi[l][1]*model.S 
                + sum(self.psi[l][2][t]*model.T_Q[t] for t in range(T)) 
                + self.psi[l][3]*model.T_o 
                + self.psi[l][4]*model.T_b 
                + self.psi[l][5]*model.T_q 
                + self.psi[l][6]*model.T_E
                )
        
        def settlement_fcn_rule(model):
            return model.f == sum(
                self.P_da[t]*model.Q_da[t] for t in range(self.T)
            )
        
        model.auxiliary_b = pyo.Constraint(model.TIME, rule = auxiliary_b)
        model.auxiliary_q = pyo.Constraint(model.TIME, rule = auxiliary_q)
        model.da_b_amount = pyo.Constraint(model.TIME, rule = da_b_rule)
        model.da_q_amount = pyo.Constraint(model.TIME, rule = da_q_rule)
        model.market_clearing_1 = pyo.Constraint(model.TIME, rule = market_clearing_1_rule)
        model.market_clearing_2 = pyo.Constraint(model.TIME, rule = market_clearing_2_rule)
        model.market_clearing_3 = pyo.Constraint(model.TIME, rule = market_clearing_3_rule)
        model.market_clearing_4 = pyo.Constraint(model.TIME, rule = market_clearing_4_rule)
        model.market_clearing_5 = pyo.Constraint(model.TIME, rule = market_clearing_5_rule)
        model.rt_bidding_amount = pyo.Constraint(rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)    
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.state_Q_da = pyo.Constraint(model.TIME, rule = state_Q_da_rule)
        model.state_o = pyo.Constraint(rule = State_o_rule)
        model.state_b = pyo.Constraint(rule = State_b_rule)
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
        self.solver_results = SOLVER.solve(self, tee=False)
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
                    [0 for _ in range(len(self.b_prev))], 
                    [0 for _ in range(len(self.q_prev))], 
                    ]
        
        psi = []
        psi.append(pyo.value(self.objective))
        
        pi_b = []
        for i in range(self.T):
            pi_b.append(self.dual[self.auxiliary_b[i]])
        psi.append(pi_b)
        
        pi_q = []
        for i in range(self.T):
            pi_q.append(self.dual[self.auxiliary_q[i]])
        psi.append(pi_q)
        
        return psi

class fw_rt_init_Lagrangian(pyo.ConcreteModel): ## (Backward - Benders' Cut) 
    
    def __init__(self, pi, psi, P_da):
        
        super().__init__()

        self.solved = False
        
        self.pi = pi
        
        self.psi = psi
        self.T = T
        self.P_da = P_da
        
        self.M_price = [[0, 0] for t in range(self.T)]
        
        self._BigM_setting()
    
    def _BigM_setting(self):
        
        for t in range(self.T):
            
            if self.P_da[t] >= 0:
                self.M_price[t][0] = 10
                self.M_price[t][1] = self.P_da[t] + P_r

            else:
                self.M_price[t][0] = -self.P_da[t] + 10
                self.M_price[t][1] = self.P_da[t] + P_r
         
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.z_q = pyo.Var(model.TIME, domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Other
        
        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, bounds = (0, 1), domain = pyo.Reals)
        
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_o = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to DA stage
        
        def da_b_rule(model, t):
            return model.b_da[t] == model.z_b[t]
        
        def da_q_rule(model, t):
            return model.q_da[t] == model.z_q[t]
        
        ## Day-Ahead Market Rules
        
        def market_clearing_1_rule(model, t):
            return model.b_da[t] - self.P_da[t] <= self.M_price[t][0]*(1 - model.n_da[t])
        
        def market_clearing_2_rule(model, t):
            return self.P_da[t] - model.b_da[t] <= self.M_price[t][1]*model.n_da[t]
        
        def market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def market_clearing_4_rule(model, t):
            return model.Q_da[t] <= M_gen[t][0]*model.n_da[t]
        
        def market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - M_gen[t][0]*(1 - model.n_da[t])
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_amount_rule(model):
            return model.q_rt <= E_0_partial[0] + B
        
        def rt_overbid_rule(model):
            return model.T_o <= E_0_sum
        
        ## State variable trainsition
        
        def State_SOC_rule(model):
            return model.S == 0.5*S

        def state_Q_da_rule(model, t):
            return model.T_Q[t] == model.Q_da[t]
        
        def State_o_rule(model):
            return model.T_o == model.q_rt
        
        def State_b_rule(model):
            return model.T_b == model.b_rt
        
        def State_q_rule(model):
            return model.T_q == model.q_rt
        
        def State_E_rule(model):
            return model.T_E == 0
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= (
                self.psi[l][0] 
                + self.psi[l][1]*model.S 
                + sum(self.psi[l][2][t]*model.T_Q[t] for t in range(T)) 
                + self.psi[l][3]*model.T_o 
                + self.psi[l][4]*model.T_b 
                + self.psi[l][5]*model.T_q 
                + self.psi[l][6]*model.T_E
                )
        
        def settlement_fcn_rule(model):
            return model.f == sum(
                self.P_da[t]*model.Q_da[t] for t in range(self.T)
            )
        
        model.da_b_amount = pyo.Constraint(model.TIME, rule = da_b_rule)
        model.da_q_amount = pyo.Constraint(model.TIME, rule = da_q_rule)
        model.market_clearing_1 = pyo.Constraint(model.TIME, rule = market_clearing_1_rule)
        model.market_clearing_2 = pyo.Constraint(model.TIME, rule = market_clearing_2_rule)
        model.market_clearing_3 = pyo.Constraint(model.TIME, rule = market_clearing_3_rule)
        model.market_clearing_4 = pyo.Constraint(model.TIME, rule = market_clearing_4_rule)
        model.market_clearing_5 = pyo.Constraint(model.TIME, rule = market_clearing_5_rule)
        model.rt_bidding_amount = pyo.Constraint(rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)    
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.state_Q_da = pyo.Constraint(model.TIME, rule = state_Q_da_rule)
        model.state_o = pyo.Constraint(rule = State_o_rule)
        model.state_b = pyo.Constraint(rule = State_b_rule)
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
                - (
                    sum(self.pi[0][i]*model.z_b[i] for i in range(T))
                    + sum(self.pi[1][j]*model.z_q[j] for j in range(T))
                )
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense=pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)
    
    def get_auxiliary_value(self):
        if not self.solved:
            self.solve()
            self.solved = True

        z = [
            [pyo.value(self.z_b[t]) for t in range(T)],
            [pyo.value(self.z_q[t]) for t in range(T)],
        ]
        
        return z



## stage = 0, 1, ..., T-1

class fw_rt(pyo.ConcreteModel):

    def __init__(self, stage, T_prev, psi, P_da, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_o_prev = T_prev[2]
        self.T_b_prev = T_prev[3]
        self.T_q_prev = T_prev[4]
        self.T_E_prev = T_prev[5]
        
        self.psi = psi
        
        self.P_da = P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        
        self.M_price = [0, 0]
        
        self.P_abs = 0
        
        self._Param_setting()
        self._BigM_setting()

    def _Param_setting(self):
        
        self.P_abs = max(self.P_rt - self.P_da[self.stage], 0)

    def _BigM_setting(self):

        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        else:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90       
        
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, self.T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_o = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(domain = pyo.NonNegativeReals)
        
        # Experiment for LP relaxation
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        model.n_1 = pyo.Var(domain = pyo.Binary)
        
        ## settlement_fcn_Vars
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints
        
        ## Connected to t-1 stage 
        
        def da_Q_rule(model):
            return model.Q_da == self.T_Q_prev[0]
        
        def rt_b_rule(model):
            return model.b_rt == self.T_b_prev
        
        def rt_q_rule(model):
            return model.q_rt == self.T_q_prev
        
        def rt_E_rule(model):
            return model.E_1 == self.T_E_prev
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == self.S_prev + v*model.c - (1/v)*model.d
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == self.T_Q_prev[t+1]
        
        def State_o_rule(model):
            return model.T_o == self.T_o_prev + model.q_rt_next
        
        def State_b_rule(model):
            return model.T_b == model.b_rt_next
        
        def State_q_rule(model):
            return model.T_q == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_o <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0_partial[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= self.M_price[0]*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= self.M_price[1]*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage][0]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage][0]*(1 - model.n_rt)
        
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.Q_rt
       
        ## f MIP reformulation   
        
        ### Linearize f_E
        
        def minmax_rule_1_1(model):
            return model.m_1 >= model.Q_da - model.u
        
        def minmax_rule_1_2(model):
            return model.m_1 >= 0

        def minmax_rule_1_3(model):
            return model.m_1 <= model.Q_da - model.u + M_gen[self.stage][0]*(1 - model.n_1)
        
        def minmax_rule_1_4(model):
            return model.m_1 <= M_gen[self.stage][0]*model.n_1
        
        def minmax_rule_2_1(model):
            return model.m_2 == model.m_1*self.P_abs
        
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
                + self.psi[l][3]*model.T_o
                + self.psi[l][4]*model.T_b 
                + self.psi[l][5]*model.T_q 
                + self.psi[l][6]*model.T_E)
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                + self.m_2 
                - gamma_over*model.phi_over 
                - gamma_under*model.phi_under
                + model.u*P_r
                )
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_o = pyo.Constraint(rule = State_o_rule)
        model.State_b = pyo.Constraint(rule = State_b_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_E = pyo.Constraint(rule = State_E_rule)
        
        model.overbid = pyo.Constraint(rule = overbid_rule)
        model.next_q_rt = pyo.Constraint(rule = next_q_rt_rule)
        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.market_clearing_1 = pyo.Constraint(rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(rule = market_clearing_rule_5)
        model.dispatch = pyo.Constraint(rule = dispatch_rule)

        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        
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
        SOLVER.solve(self)
        self.solved = True   

    def get_state_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append(pyo.value(self.S))
        State_var.append([pyo.value(self.T_Q[t]) for t in range(self.T - 1 - self.stage)])
        State_var.append(pyo.value(self.T_o))
        State_var.append(pyo.value(self.T_b))
        State_var.append(pyo.value(self.T_q))
        State_var.append(pyo.value(self.T_E))
        
        return State_var 

    def get_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True

        solutions = {
            "b_rt": pyo.value(self.b_rt),
            "Q_rt": pyo.value(self.Q_rt)
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

    def get_E_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        E_profit = pyo.value(self.m_2) 

        return E_profit

    def get_Im_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        Im_profit = - gamma_over*pyo.value(self.phi_over) - gamma_under*pyo.value(self.phi_under)

        return Im_profit

class fw_rt_LP_relax(pyo.ConcreteModel): ## (Backward - Benders' Cut)

    def __init__(self, stage, T_prev, psi, P_da, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_o_prev = T_prev[2]
        self.T_b_prev = T_prev[3]
        self.T_q_prev = T_prev[4]
        self.T_E_prev = T_prev[5]
        
        self.psi = psi
        
        self.P_da = P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        
        self.M_price = [0, 0]
                
        self.P_abs = 0
        
        self._Param_setting()
        self._BigM_setting()
        
    def _Param_setting(self):
        
        self.P_abs = max(self.P_rt - self.P_da[self.stage], 0)

    def _BigM_setting(self):

        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        else:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90       
       
    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
                
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_o = pyo.Var(domain = pyo.Reals)
        model.z_T_b = pyo.Var(domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_o = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
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
        
        def auxiliary_T_o(model):
            return model.z_T_o == self.T_o_prev
        
        def auxiliary_T_b(model):
            return model.z_T_b == self.T_b_prev
        
        def auxiliary_T_q(model):
            return model.z_T_q == self.T_q_prev
        
        def auxiliary_T_E(model):
            return model.z_T_E == self.T_E_prev        
        
        ## Connected to t-1 state 
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d

        def State_Q_rule(model, t):
            return model.T_Q[t] == model.z_T_Q[t+1]
        
        def State_o_rule(model):
            return model.T_o == model.z_T_o + model.q_rt_next
        
        def State_b_rule(model):
            return model.T_b == model.b_rt_next
        
        def State_q_rule(model):
            return model.T_q == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_o <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0_partial[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= self.M_price[0]*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= self.M_price[1]*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage][0]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage][0]*(1 - model.n_rt)
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.Q_rt
         
        ## f MIP reformulation   
        
        ### Linearize f_E
        
        def minmax_rule_1_1(model):
            return model.m_1 >= model.Q_da - model.u
        
        def minmax_rule_1_2(model):
            return model.m_1 >= 0

        def minmax_rule_1_3(model):
            return model.m_1 <= model.Q_da - model.u + M_gen[self.stage][0]*(1 - model.n_1)
        
        def minmax_rule_1_4(model):
            return model.m_1 <= M_gen[self.stage][0]*model.n_1
        
        def minmax_rule_2_1(model):
            return model.m_2 == model.m_1*self.P_abs
        
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
                + self.psi[l][3]*model.T_o
                + self.psi[l][4]*model.T_b 
                + self.psi[l][5]*model.T_q 
                + self.psi[l][6]*model.T_E)
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                + self.m_2 
                - gamma_over*model.phi_over - gamma_under*model.phi_under
                + model.u*P_r
                )
        
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S)
        model.auxiliary_T_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_Q)
        model.auxiliary_T_o = pyo.Constraint(rule = auxiliary_T_o)
        model.auxiliary_T_b = pyo.Constraint(rule = auxiliary_T_b)
        model.auxiliary_T_q = pyo.Constraint(rule = auxiliary_T_q)
        model.auxiliary_T_E = pyo.Constraint(rule = auxiliary_T_E)
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_o = pyo.Constraint(rule = State_o_rule)
        model.State_b = pyo.Constraint(rule = State_b_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_E = pyo.Constraint(rule = State_E_rule)
        
        model.overbid = pyo.Constraint(rule = overbid_rule)
        model.next_q_rt = pyo.Constraint(rule = next_q_rt_rule)
        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.market_clearing_1 = pyo.Constraint(rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(rule = market_clearing_rule_5)
        
        model.dispatch = pyo.Constraint(rule = dispatch_rule)
        
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        
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
        self.solver_results = SOLVER.solve(self, tee=False)
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
                    0, 
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
        
        psi.append(self.dual[self.auxiliary_T_o])
        psi.append(self.dual[self.auxiliary_T_b])
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi

class fw_rt_Lagrangian(pyo.ConcreteModel): ## (Backward - Strengthened Benders' Cut)

    def __init__(self, stage, pi, psi, P_da, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        self.pi = pi
        self.psi = psi
        
        self.P_da = P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T

        self.M_price = [0, 0]
        
        self.P_abs = 0
        
        self._Param_setting()
        self._BigM_setting()

    def _Param_setting(self):
        
        self.P_abs = max(self.P_rt - self.P_da[self.stage], 0)

    def _BigM_setting(self):

        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        else:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90       
            
    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
                
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(bounds = (S_min, S_max), domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, bounds = (0, 2*B), domain = pyo.Reals)
        model.z_T_o = pyo.Var(bounds = (0, E_0_sum), domain = pyo.Reals, initialize = 0.0)
        model.z_T_b = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.z_T_q = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage] + B), domain = pyo.Reals)
        model.z_T_E = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage]), domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_o = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(domain = pyo.Reals)
        model.T_q = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(domain = pyo.NonNegativeReals)
         
        # Experiment for LP relaxation
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        model.n_1 = pyo.Var(domain = pyo.Binary)          
        
        #model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn_Vars
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints   
        
        ## Connected to t-1 state 
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == model.z_T_Q[t+1]
        
        def State_o_rule(model):
            return model.T_o == model.z_T_o + model.q_rt_next
        
        def State_b_rule(model):
            return model.T_b == model.b_rt_next
        
        def State_q_rule(model):
            return model.T_q == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_o <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0_partial[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= self.M_price[0]*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= self.M_price[1]*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage][0]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage][0]*(1 - model.n_rt)
        
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.Q_rt
         
        ## f MIP reformulation   
        
        ### Linearize f_E
        
        def minmax_rule_1_1(model):
            return model.m_1 >= model.Q_da - model.u
        
        def minmax_rule_1_2(model):
            return model.m_1 >= 0

        def minmax_rule_1_3(model):
            return model.m_1 <= model.Q_da - model.u + M_gen[self.stage][0]*(1 - model.n_1)
        
        def minmax_rule_1_4(model):
            return model.m_1 <= M_gen[self.stage][0]*model.n_1
        
        def minmax_rule_2_1(model):
            return model.m_2 == model.m_1*self.P_abs
        
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
                + self.psi[l][3]*model.T_o
                + self.psi[l][4]*model.T_b
                + self.psi[l][5]*model.T_q
                + self.psi[l][6]*model.T_E
                )
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                + self.m_2 
                - gamma_over*model.phi_over - gamma_under*model.phi_under
                + model.u*P_r
                )

        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_o = pyo.Constraint(rule = State_o_rule)
        model.State_b = pyo.Constraint(rule = State_b_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_E = pyo.Constraint(rule = State_E_rule)
        
        model.overbid = pyo.Constraint(rule = overbid_rule)
        model.next_q_rt = pyo.Constraint(rule = next_q_rt_rule)
        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.market_clearing_1 = pyo.Constraint(rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(rule = market_clearing_rule_5)
        model.dispatch = pyo.Constraint(rule = dispatch_rule)

        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        
        model.imbalance_over = pyo.Constraint(rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(rule = imbalance_under_rule)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)
                
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta 
                + model.f 
                - (
                    self.pi[0]*model.z_S 
                    + sum(self.pi[1][j]*model.z_T_Q[j] for j in range(T - self.stage)) 
                    + self.pi[2]*model.z_T_o
                    + self.pi[3]*model.z_T_b 
                    + self.pi[4]*model.z_T_q 
                    + self.pi[5]*model.z_T_E
                    )
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
    
    def get_auxiliary_value(self):
        if not self.solved:
            self.solve()
            self.solved = True

        z = [
            pyo.value(self.z_S),
            [pyo.value(self.z_T_Q[t]) for t in range(self.T - self.stage)],
            pyo.value(self.z_T_o),
            pyo.value(self.z_T_b),
            pyo.value(self.z_T_q),
            pyo.value(self.z_T_E)
        ]
        
        return z


## stage = T

class fw_rt_last(pyo.ConcreteModel): 
    
    def __init__(self, T_prev, P_da, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_o_prev = T_prev[2]
        self.T_b_prev = T_prev[3]
        self.T_q_prev = T_prev[4]
        self.T_E_prev = T_prev[5]

        self.P_da = P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T

        self.M_price = [0, 0]
                
        self.P_abs = 0
        
        self._Param_setting()
        self._BigM_setting()

    def _Param_setting(self):
        
        self.P_abs = max(self.P_rt - self.P_da[self.stage], 0)

    def _BigM_setting(self):

        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        else:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90       
       
    def build_model(self):
        
        model = self.model()
                
        # Vars
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(domain = pyo.NonNegativeReals)
        
        # Experiment for LP relaxation
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        model.n_1 = pyo.Var(domain = pyo.Binary)        
        
        #model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_Q_rule(model):
            return model.Q_da == self.T_Q_prev[0]
        
        def rt_b_rule(model):
            return model.b_rt == self.T_b_prev
        
        def rt_q_rule(model):
            return model.q_rt == self.T_q_prev
        
        def rt_E_rule(model):
            return model.E_1 == self.T_E_prev
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == self.S_prev + v*model.c - (1/v)*model.d
        
        def State_SOC_rule_last(model):
            return model.S == 0.5*S
        
        ## General Constraints
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= self.M_price[0]*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= self.M_price[1]*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage][0]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage][0]*(1 - model.n_rt)
        
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.Q_rt
         
        ## f MIP reformulation   
        
        ### Linearize f_E
        
        def minmax_rule_1_1(model):
            return model.m_1 >= model.Q_da - model.u
        
        def minmax_rule_1_2(model):
            return model.m_1 >= 0

        def minmax_rule_1_3(model):
            return model.m_1 <= model.Q_da - model.u + M_gen[self.stage][0]*(1 - model.n_1)
        
        def minmax_rule_1_4(model):
            return model.m_1 <= M_gen[self.stage][0]*model.n_1
        
        def minmax_rule_2_1(model):
            return model.m_2 == model.m_1*self.P_abs
        
        ### Imbalance Penalty
        
        def imbalance_over_rule(model):
            return model.u - model.Q_c <= model.phi_over
        
        def imbalance_under_rule(model):
            return model.Q_c - model.u <= model.phi_under
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                + self.m_2 
                - gamma_over*model.phi_over - gamma_under*model.phi_under
                + model.u*P_r                
                )
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        ## model.State_SOC_last = pyo.Constraint(rule = State_SOC_rule_last)

        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.market_clearing_1 = pyo.Constraint(rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(rule = market_clearing_rule_5)
        model.dispatch = pyo.Constraint(rule = dispatch_rule)
        
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        
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
        SOLVER.solve(self)
        self.solved = True

    def get_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True

        solutions = {
            "b_rt": pyo.value(self.b_rt),
            "Q_rt": pyo.value(self.Q_rt)
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

class fw_rt_last_LP_relax(pyo.ConcreteModel): ## (Backward)
           
    def __init__(self, T_prev, P_da, delta):
        
        super().__init__()

        self.solved = False
                
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_o_prev = T_prev[2]
        self.T_b_prev = T_prev[3]
        self.T_q_prev = T_prev[4]
        self.T_E_prev = T_prev[5]

        self.P_da = P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T

        self.M_price = [0, 0]
                
        self.P_abs = 0
        
        self._Param_setting()
        self._BigM_setting()
        
    def _Param_setting(self):
        
        self.P_abs = max(self.P_rt - self.P_da[self.stage], 0)

    def _BigM_setting(self):

        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        else:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90       
       
    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
                
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_o = pyo.Var(domain = pyo.Reals)
        model.z_T_b = pyo.Var(domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
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
        
        def auxiliary_T_o(model):
            return model.z_T_o == self.T_o_prev
        
        def auxiliary_T_b(model):
            return model.z_T_b == self.T_b_prev
        
        def auxiliary_T_q(model):
            return model.z_T_q == self.T_q_prev
        
        def auxiliary_T_E(model):
            return model.z_T_E == self.T_E_prev
        
        ## Connected to t-1 state 
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d
        
        def State_SOC_rule_last(model):
            return model.S == 0.5*S
        
        ## General Constraints
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= self.M_price[0]*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= self.M_price[1]*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage][0]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage][0]*(1 - model.n_rt)
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.Q_rt
         
        ## f MIP reformulation   
        
        ### Linearize f_E
        
        def minmax_rule_1_1(model):
            return model.m_1 >= model.Q_da - model.u
        
        def minmax_rule_1_2(model):
            return model.m_1 >= 0

        def minmax_rule_1_3(model):
            return model.m_1 <= model.Q_da - model.u + M_gen[self.stage][0]*(1 - model.n_1)
        
        def minmax_rule_1_4(model):
            return model.m_1 <= M_gen[self.stage][0]*model.n_1
        
        def minmax_rule_2_1(model):
            return model.m_2 == model.m_1*self.P_abs
        
        ### Imbalance Penalty
        
        def imbalance_over_rule(model):
            return model.u - model.Q_c <= model.phi_over
        
        def imbalance_under_rule(model):
            return model.Q_c - model.u <= model.phi_under
  
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                + self.m_2 
                - gamma_over*model.phi_over - gamma_under*model.phi_under
                + model.u*P_r
                )
        
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S)
        model.auxiliary_T_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_Q)
        model.auxiliary_T_o = pyo.Constraint(rule = auxiliary_T_o)
        model.auxiliary_T_b = pyo.Constraint(rule = auxiliary_T_b)
        model.auxiliary_T_q = pyo.Constraint(rule = auxiliary_T_q)
        model.auxiliary_T_E = pyo.Constraint(rule = auxiliary_T_E)
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        ## model.State_SOC_last = pyo.Constraint(rule = State_SOC_rule_last)

        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.market_clearing_1 = pyo.Constraint(rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(rule = market_clearing_rule_5)
        
        model.dispatch = pyo.Constraint(rule = dispatch_rule)
        
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        
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
        self.solver_results = SOLVER.solve(self, tee=False)
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
                    0, 
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
        
        psi.append(self.dual[self.auxiliary_T_o])
        psi.append(self.dual[self.auxiliary_T_b])
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi  
     
class fw_rt_last_Lagrangian(pyo.ConcreteModel): ## (Backward - Strengthened Benders' Cut)
           
    def __init__(self, pi, P_da, delta):
        
        super().__init__()

        self.solved = False

        self.pi = pi
        
        self.stage = T - 1

        self.P_da = P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        
        self.P_abs = 0
        
        self._Param_setting()
        self._BigM_setting()

    def _Param_setting(self):
        
        self.P_abs = max(self.P_rt - self.P_da[self.stage], 0)

    def _BigM_setting(self):

        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 10
            self.M_price[1] = self.P_rt + 90

        else:
            
            self.M_price[0] = -self.P_rt + 10
            self.M_price[1] = self.P_rt + 90       
       
    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
                
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(bounds = (S_min, S_max), domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, bounds = (0, 2*B), domain = pyo.Reals)
        model.z_T_o = pyo.Var(bounds = (0, E_0_sum), domain = pyo.Reals, initialize = 0.0)
        model.z_T_b = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.z_T_q = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage] + B), domain = pyo.Reals)
        model.z_T_E = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage]), domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)

        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(domain = pyo.NonNegativeReals)
        
        # Experiment for LP relaxation
        
        model.n_rt = pyo.Var(domain = pyo.Binary)        
        model.n_1 = pyo.Var(domain = pyo.Binary)        
        
        #model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d
        
        def State_SOC_rule_last(model):
            return model.S == 0.5*S
        
        ## General Constraints
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= self.M_price[0]*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= self.M_price[1]*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage][0]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage][0]*(1 - model.n_rt)
        
        def dispatch_rule(model):
            return model.Q_c == (1 + self.delta_c)*model.Q_rt
         
        ## f MIP reformulation   
        
        ### Linearize f_E
        
        def minmax_rule_1_1(model):
            return model.m_1 >= model.Q_da - model.u
        
        def minmax_rule_1_2(model):
            return model.m_1 >= 0

        def minmax_rule_1_3(model):
            return model.m_1 <= model.Q_da - model.u + M_gen[self.stage][0]*(1 - model.n_1)
        
        def minmax_rule_1_4(model):
            return model.m_1 <= M_gen[self.stage][0]*model.n_1
        
        def minmax_rule_2_1(model):
            return model.m_2 == model.m_1*self.P_abs
        
        ### Imbalance Penalty
        
        def imbalance_over_rule(model):
            return model.u - model.Q_c <= model.phi_over
        
        def imbalance_under_rule(model):
            return model.Q_c - model.u <= model.phi_under
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f == (
                (model.u - model.Q_da)*self.P_rt 
                + self.m_2 
                - gamma_over*model.phi_over - gamma_under*model.phi_under
                + model.u*P_r
                )
        
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        ## model.State_SOC_last = pyo.Constraint(rule = State_SOC_rule_last)

        model.generation = pyo.Constraint(rule = generation_rule)
        model.charge = pyo.Constraint(rule = charge_rule)
        model.electricity_supply = pyo.Constraint(rule = electricity_supply_rule)
        model.market_clearing_1 = pyo.Constraint(rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(rule = market_clearing_rule_5)
        model.dispatch = pyo.Constraint(rule = dispatch_rule)

        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)

        model.imbalance_over = pyo.Constraint(rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(rule = imbalance_under_rule)

        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
          
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.f - (
                    self.pi[0]*model.z_S 
                    + sum(self.pi[1][j]*model.z_T_Q[j] for j in range(T - self.stage)) 
                    + self.pi[2]*model.z_T_o
                    + self.pi[3]*model.z_T_b 
                    + self.pi[4]*model.z_T_q 
                    + self.pi[5]*model.z_T_E
                    )
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense=pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True

    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True

        return pyo.value(self.objective)         

    def get_auxiliary_value(self):
        if not self.solved:
            self.solve()
            self.solved = True

        z = [
            pyo.value(self.z_S),
            [pyo.value(self.z_T_Q[t]) for t in range(self.T - self.stage)],
            pyo.value(self.z_T_o),
            pyo.value(self.z_T_b),
            pyo.value(self.z_T_q),
            pyo.value(self.z_T_E)
        ]
        
        return z


## Solve Convex Lagrangian Dual Problem

class dual_approx_sub(pyo.ConcreteModel): ## Subgradient method
    
    def __init__(self, stage, reg, pi):
        
        super().__init__()
        
        self.solved = False
        
        self.stage = stage
        
        self.reg = reg
        self.pi = pi
        
        self.T = T
    
        self._build_model()
    
    def _build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, self.T - self.stage - 1)
        
        # Vars
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        model.pi_S = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.pi_Q = pyo.Var(model.TIME, domain = pyo.Reals, initialize = 0.0)
        model.pi_o = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.pi_b = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.pi_q = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.pi_E = pyo.Var(domain = pyo.Reals, initialize = 0.0)
                
        # Constraints
        
        def initialize_theta_rule(model):
            return model.theta >= -10000000
        
        model.initialize_theta = pyo.Constraint(rule = initialize_theta_rule)
        
        model.dual_fcn_approx = pyo.ConstraintList() 
        
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta
                + self.reg*(
                    + (model.pi_S)**2
                    + sum((model.pi_Q[t])**2 for t in range(self.T - self.stage))
                    + (model.pi_o)**2 
                    + (model.pi_b)**2 
                    + (model.pi_q)**2 
                    + (model.pi_E)**2 
                )
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.minimize)

    def add_plane(self, coeff):
        
        lamb = coeff[0]
        k = coeff[1]
                
        model = self.model()
        
        model.dual_fcn_approx.add(model.theta >= (
            lamb 
            + k[0]*model.pi_S 
            + sum(k[1][t]*model.pi_Q[t] for t in range(self.T - self.stage)) 
            + k[2]*model.pi_o 
            + k[3]*model.pi_b 
            + k[4]*model.pi_q 
            + k[5]*model.pi_E)
            )
    
    def solve(self):
        
        SOLVER.solve(self)
        self.solved = True
        
    def get_solution_value(self):
        
        self.solve()
        self.solved = True

        pi = [
            pyo.value(self.pi_S),
            [pyo.value(self.pi_Q[t]) for t in range(self.T - self.stage)],
            pyo.value(self.pi_o),
            pyo.value(self.pi_b),
            pyo.value(self.pi_q),
            pyo.value(self.pi_E)
        ]
        
        return pi
    
    def get_objective_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True

        return pyo.value(self.theta) 

class dual_approx_sub_da(pyo.ConcreteModel): ## Subgradient method
    
    def __init__(self, reg, pi):
        
        super().__init__()
        
        self.solved = False
        
        self.reg = reg
        self.pi = pi
        
        self.T = T
    
        self._build_model()
    
    def _build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, self.T - 1)
        
        # Vars
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        model.pi_b = pyo.Var(model.TIME, domain = pyo.Reals, initialize = 0.0)
        model.pi_q = pyo.Var(model.TIME, domain = pyo.Reals, initialize = 0.0)
                
        # Constraints
        
        def initialize_theta_rule(model):
            return model.theta >= -10000000
        
        model.initialize_theta = pyo.Constraint(rule = initialize_theta_rule)
        
        model.dual_fcn_approx = pyo.ConstraintList() 
        
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta
                + self.reg*(
                    + sum((model.pi_b[t])**2 for t in range(self.T))
                    + sum((model.pi_q[t])**2 for t in range(self.T))
                )
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.minimize)

    def add_plane(self, coeff):
        
        lamb = coeff[0]
        k = coeff[1]
                
        model = self.model()
        
        model.dual_fcn_approx.add(model.theta >= (
                lamb 
                + sum(k[0][t]*model.pi_b[t] for t in range(self.T)) 
                + sum(k[1][t]*model.pi_q[t] for t in range(self.T)) 
                )
            )
    
    def solve(self):
        
        SOLVER.solve(self)
        self.solved = True
        
    def get_solution_value(self):
        
        self.solve()
        self.solved = True

        pi = [
            [pyo.value(self.pi_b[t]) for t in range(self.T)],
            [pyo.value(self.pi_q[t]) for t in range(self.T)],
        ]
        
        return pi
    
    def get_objective_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True

        return pyo.value(self.theta) 


# SubProblems for Rolling Horizon

class rolling_da(pyo.ConcreteModel):
    
    def __init__(self, exp_P_da, exp_P_rt):
        
        super().__init__()

        self.solved = False
        
        self.exp_P_da = exp_P_da
        self.exp_P_rt = exp_P_rt
        
        self.exp_P_abs = [0 for _ in range(T)]
        
        self.T = T
        
        self.M_price_DA = [[0, 0] for _ in range(self.T)]
        self.M_price_RT = [[0, 0] for _ in range(self.T)]
        
        self._Param_setting()
        self._BigM_setting()

    def _Param_setting(self):

        for t in range(self.T):
            
            self.exp_P_abs[t] = max(self.exp_P_rt[t] - self.exp_P_da[t], 0)

    def _BigM_setting(self):
        
        for t in range(self.T):
            
            if self.exp_P_da[t] >= 0:
                self.M_price_DA[t][0] = 10
                self.M_price_DA[t][1] = self.exp_P_da[t] + 90

            else:
                self.M_price_DA[t][0] = -self.exp_P_da[t] + 10
                self.M_price_DA[t][1] = self.exp_P_da[t] + 90
                
            if self.exp_P_rt[t] >= 0:
                self.M_price_RT[t][0] = 10
                self.M_price_RT[t][1] = self.exp_P_rt[t] + 90

            else:
                self.M_price_RT[t][0] = -self.exp_P_rt[t] + 10
                self.M_price_RT[t][1] = self.exp_P_rt[t] + 90

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)

        # Vars
        
        # Day Ahead variables
        
        model.b_da = pyo.Var(
            model.TIME, bounds = (-P_r, 0), domain = pyo.Reals, initialize = 0.0
            )
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)

        # Intraday bidding variables

        model.b_rt = pyo.Var(
            model.TIME, bounds = (-P_r, 0), domain = pyo.Reals, initialize = 0.0
        )
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals
        )
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
                
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)

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
        
        def da_market_clearing_1_rule(model, t):
            return model.b_da[t] - self.exp_P_da[t] <= self.M_price_DA[t][0]*(1 - model.n_da[t])
        
        def da_market_clearing_2_rule(model, t):
            return self.exp_P_da[t] - model.b_da[t] <= self.M_price_DA[t][1]*model.n_da[t]
        
        def da_market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def da_market_clearing_4_rule(model, t):
            return model.Q_da[t] <= M_gen[t][0]*model.n_da[t]
        
        def da_market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - M_gen[t][0]*(1 - model.n_da[t])
        
        ## RT bidding & market rules
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= E_0_partial[t] + B
        
        def rt_overbid_rule(model):
            return sum(model.q_rt[t] for t in range(self.T)) <= E_0_sum
        
        def market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= self.M_price_RT[t][0]*(1 - model.n_rt[t])
        
        def market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= self.M_price_RT[t][1]*model.n_rt[t] 
        
        def market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= M_gen[t][0]*model.n_rt[t]
        
        def market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - M_gen[t][0]*(1 - model.n_rt[t])
        
        ## Operations rules
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def SOC_init_rule(model):
            return model.S[-1] == 0.5*S
        
        def SOC_balance_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def generation_rule(model, t):
            return model.g[t] <= E_0_partial[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]
        
        ## min, max reformulation constraints
        
        def minmax_rule_1_1(model, t):
            return model.m_1[t] >= model.Q_da[t] - model.u[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] >= 0

        def minmax_rule_1_3(model, t):
            return model.m_1[t] <= model.Q_da[t] - model.u[t] + M_gen[t][0]*(1 - model.n_1[t])
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] <= M_gen[t][0]*model.n_1[t]
        
        def minmax_rule_2_1(model, t):
            return model.m_2[t] == model.m_1[t]*self.exp_P_abs[t]
        
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, t):
            return model.u[t] - model.Q_c[t] <= model.phi_over[t]
        
        def imbalance_under_rule(model, t):
            return model.Q_c[t] - model.u[t] <= model.phi_under[t]
        
        ## settlement fcn
        
        def da_settlement_fcn_rule(model, t):
            return model.f_DA[t] == self.exp_P_da[t]*model.Q_da[t] 
        
        def rt_settlement_fcn_rule(model, t):
            return model.f_RT[t] == (
                (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] 
                + self.m_2[t] 
                - gamma_over*model.phi_over[t] 
                - gamma_under*model.phi_under[t]
                + model.u[t]*P_r
                )
        
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.TIME, rule = da_market_clearing_5_rule)
        model.rt_bidding_amount = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)
        model.market_clearing_1 = pyo.Constraint(model.TIME, rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(model.TIME, rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(model.TIME, rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(model.TIME, rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(model.TIME, rule = market_clearing_rule_5)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.SOC_init = pyo.Constraint(rule = SOC_init_rule)
        model.SOC_balance = pyo.Constraint(model.TIME, rule = SOC_balance_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_1)
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
        
        State_var.append([pyo.value(self.b_da[t]) for t in range(self.T)])
        State_var.append([pyo.value(self.q_da[t]) for t in range(self.T)])
        
        return State_var    

class rolling_rt_init(pyo.ConcreteModel):
       
    def __init__(self, da_state, P_da, exp_P_rt):
        
        super().__init__()

        self.solved = False
        
        self.b_da_prev = da_state[0]
        self.q_da_prev = da_state[1]
        
        self.P_da = P_da
        self.exp_P_rt = exp_P_rt
        
        self.exp_P_abs = [0 for _ in range(T)]
        
        self.T = T
        
        self.M_price_DA = [[0, 0] for _ in range(self.T)]
        self.M_price_RT = [[0, 0] for _ in range(self.T)]
        
        self._Param_setting()
        self._BigM_setting()

    def _Param_setting(self):

        for t in range(self.T):
            
            self.exp_P_abs[t] = max(self.exp_P_rt[t] - self.P_da[t], 0)

    def _BigM_setting(self):
        
        for t in range(self.T):
            
            if self.P_da[t] >= 0:
                self.M_price_DA[t][0] = 10
                self.M_price_DA[t][1] = self.P_da[t] + 90

            else:
                self.M_price_DA[t][0] = -self.P_da[t] + 10
                self.M_price_DA[t][1] = self.P_da[t] + 90
                
            if self.exp_P_rt[t] >= 0:
                self.M_price_RT[t][0] = 10
                self.M_price_RT[t][1] = self.exp_P_rt[t] + 90

            else:
                self.M_price_RT[t][0] = -self.exp_P_rt[t] + 10
                self.M_price_RT[t][1] = self.exp_P_rt[t] + 90

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)

        # Vars
        
        # Day Ahead variables

        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)

        # Intraday bidding variables

        model.b_rt = pyo.Var(
            model.TIME, bounds = (-P_r, 0), domain = pyo.Reals, initialize = 0.0
        )
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals
        )
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
                
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)

        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        ## settlement fcn Vars
        
        model.f_DA = pyo.Var(model.TIME, domain = pyo.Reals)
        model.f_RT = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## DA bidding & market rules
        
        def da_market_clearing_1_rule(model, t):
            return self.b_da_prev[t] - self.P_da[t] <= self.M_price_DA[t][0]*(1 - model.n_da[t])
        
        def da_market_clearing_2_rule(model, t):
            return self.P_da[t] - self.b_da_prev[t] <= self.M_price_DA[t][1]*model.n_da[t]
        
        def da_market_clearing_3_rule(model, t):
            return model.Q_da[t] <= self.q_da_prev[t]
        
        def da_market_clearing_4_rule(model, t):
            return model.Q_da[t] <= M_gen[t][0]*model.n_da[t]
        
        def da_market_clearing_5_rule(model, t):
            return model.Q_da[t] >= self.q_da_prev[t] - M_gen[t][0]*(1 - model.n_da[t])
        
        ## RT bidding & market rules
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= E_0_partial[t] + B
        
        def rt_overbid_rule(model):
            return sum(model.q_rt[t] for t in range(self.T)) <= E_0_sum
        
        def market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= self.M_price_RT[t][0]*(1 - model.n_rt[t])
        
        def market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= self.M_price_RT[t][1]*model.n_rt[t] 
        
        def market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= M_gen[t][0]*model.n_rt[t]
        
        def market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - M_gen[t][0]*(1 - model.n_rt[t])
        
        ## Operations rules
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def SOC_init_rule(model):
            return model.S[-1] == 0.5*S
        
        def SOC_balance_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def generation_rule(model, t):
            return model.g[t] <= E_0_partial[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]
        
        ## min, max reformulation constraints
        
        def minmax_rule_1_1(model, t):
            return model.m_1[t] >= model.Q_da[t] - model.u[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] >= 0

        def minmax_rule_1_3(model, t):
            return model.m_1[t] <= model.Q_da[t] - model.u[t] + M_gen[t][0]*(1 - model.n_1[t])
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] <= M_gen[t][0]*model.n_1[t]
        
        def minmax_rule_2_1(model, t):
            return model.m_2[t] == model.m_1[t]*self.exp_P_abs[t]
        
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, t):
            return model.u[t] - model.Q_c[t] <= model.phi_over[t]
        
        def imbalance_under_rule(model, t):
            return model.Q_c[t] - model.u[t] <= model.phi_under[t]
        
        ## settlement fcn
        
        def da_settlement_fcn_rule(model, t):
            return model.f_DA[t] == self.P_da[t]*model.Q_da[t] 
        
        def rt_settlement_fcn_rule(model, t):
            return model.f_RT[t] == (
                (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] 
                + self.m_2[t] 
                - gamma_over*model.phi_over[t] 
                - gamma_under*model.phi_under[t]
                + model.u[t]*P_r
                )
        
        model.da_market_clearing_1 = pyo.Constraint(model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.TIME, rule = da_market_clearing_5_rule)
        model.rt_bidding_amount = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)
        model.market_clearing_1 = pyo.Constraint(model.TIME, rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(model.TIME, rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(model.TIME, rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(model.TIME, rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(model.TIME, rule = market_clearing_rule_5)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.SOC_init = pyo.Constraint(rule = SOC_init_rule)
        model.SOC_balance = pyo.Constraint(model.TIME, rule = SOC_balance_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_1)
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
        State_var.append([pyo.value(self.Q_da[t]) for t in range(self.T)])
        State_var.append(pyo.value(self.q_rt[0]))
        State_var.append(pyo.value(self.b_rt[0]))
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

        DA_profit = [self.P_da[t]*pyo.value(self.Q_da[t]) for t in range(self.T)]

        return DA_profit

class rolling_rt(pyo.ConcreteModel):
       
    def __init__(self, stage, state, P_da, exp_P_rt, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = state[0]
        self.Q_prev = state[1]
        self.o_prev = state[2]
        self.b_rt_prev = state[3]
        self.q_rt_prev = state[4]
        self.E_prev = state[5]
        
        self.P_da = P_da
        self.exp_P_rt = exp_P_rt
        
        self.P_abs = [0 for _ in range(T)]
        
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        
        self.M_price_DA = [[0, 0] for _ in range(self.T)]
        self.M_price_RT = [[0, 0] for _ in range(self.T)]
        
        self._Param_setting()
        self._BigM_setting()

    def _Param_setting(self):

        self.P_abs[self.stage] = max(self.P_rt - self.P_da[self.stage], 0)
            
        for t in range(self.stage+1, self.T):
            
            self.P_abs[t] = max(self.exp_P_rt[t] - self.P_da[t], 0)

    def _BigM_setting(self):
            
        if self.P_rt >= 0:
            self.M_price_RT[self.stage][0] = 10
            self.M_price_RT[self.stage][1] = self.P_rt + 90

        else:
            self.M_price_RT[self.stage][0] = -self.P_rt + 10
            self.M_price_RT[self.stage][1] = self.P_rt + 90
        
        for t in range(self.stage+1, self.T):
                
            if self.exp_P_rt[t] >= 0:
                self.M_price_RT[t][0] = 10
                self.M_price_RT[t][1] = self.exp_P_rt[t] + 90

            else:
                self.M_price_RT[t][0] = -self.exp_P_rt[t] + 10
                self.M_price_RT[t][1] = self.exp_P_rt[t] + 90

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(self.stage, T-1)
        model.BIDTIME = pyo.RangeSet(self.stage+1, T-1)
        model.BIDTIME_NEXT = pyo.RangeSet(self.stage+2, T-1)
        model.TIME_ESS = pyo.RangeSet(self.stage-1, T-1)

        # Vars
        
        # Day Ahead variables

        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        # Intraday bidding variables

        model.b_rt = pyo.Var(
            model.BIDTIME, bounds = (-P_r, 0), domain = pyo.Reals, initialize = 0.0
        )
        model.q_rt = pyo.Var(model.BIDTIME, domain = pyo.NonNegativeReals)

        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals, initialize = 0.0
        )
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
                
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)

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
        
        def rt_bidding_amount_next_rule(model):
            return model.q_rt[self.stage+1] <= self.delta_E_0*E_0_partial[self.stage+1] + B
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= E_0_partial[t] + B
        
        def rt_overbid_rule(model):
            return self.o_prev + sum(model.q_rt[t] for t in range(self.stage+1, self.T)) <= E_0_sum
        
        def market_clearing_stage_rule_1(model):
            return self.b_rt_prev - self.P_rt <= self.M_price_RT[self.stage][0]*(1 - model.n_rt[self.stage])
        
        def market_clearing_stage_rule_2(model):
            return self.P_rt - self.b_rt_prev <= self.M_price_RT[self.stage][1]*model.n_rt[self.stage] 
        
        def market_clearing_stage_rule_3(model):
            return model.Q_rt[self.stage] <= self.q_rt_prev
        
        def market_clearing_stage_rule_4(model):
            return model.Q_rt[self.stage] <= M_gen[self.stage][0]*model.n_rt[self.stage]    
        
        def market_clearing_stage_rule_5(model):
            return model.Q_rt[self.stage] >= self.q_rt_prev - M_gen[self.stage][0]*(1 - model.n_rt[self.stage])    
        
        def market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= self.M_price_RT[t][0]*(1 - model.n_rt[t])
        
        def market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= self.M_price_RT[t][1]*model.n_rt[t] 
        
        def market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= M_gen[t][0]*model.n_rt[t]
        
        def market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - M_gen[t][0]*(1 - model.n_rt[t])
        
        ## Operations rules
        
        def dispatch_stage_rule(model):
            return model.Q_c[self.stage] == (1 + self.delta_c)*model.Q_rt[self.stage]
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def SOC_init_rule(model):
            return model.S[self.stage-1] == self.S_prev
        
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
        
        ## min, max reformulation constraints
        
        def minmax_rule_1_1(model, t):
            return model.m_1[t] >= model.Q_da[t] - model.u[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] >= 0

        def minmax_rule_1_3(model, t):
            return model.m_1[t] <= model.Q_da[t] - model.u[t] + M_gen[t][0]*(1 - model.n_1[t])
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] <= M_gen[t][0]*model.n_1[t]
        
        def minmax_rule_2_1(model, t):
            return model.m_2[t] == model.m_1[t]*self.P_abs[t]
        
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, t):
            return model.u[t] - model.Q_c[t] <= model.phi_over[t]
        
        def imbalance_under_rule(model, t):
            return model.Q_c[t] - model.u[t] <= model.phi_under[t]
        
        ## settlement fcn
        
        def rt_settlement_fcn_stage_rule(model):
            return model.f_RT[self.stage] == (
                (model.u[self.stage] - model.Q_da[self.stage])*self.P_rt 
                + self.m_2[self.stage] 
                - gamma_over*model.phi_over[self.stage] 
                - gamma_under*model.phi_under[self.stage]
                + model.u[self.stage]*P_r
                )
        
        def rt_settlement_fcn_rule(model, t):
            return model.f_RT[t] == (
                (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] 
                + self.m_2[t] 
                - gamma_over*model.phi_over[t] 
                - gamma_under*model.phi_under[t]
                + model.u[t]*P_r
                )
        
        model.da_awarded_amount = pyo.Constraint(model.TIME, rule = da_awarded_amount_rule)
        model.rt_bidding_amount_next = pyo.Constraint(rule = rt_bidding_amount_next_rule)
        model.rt_bidding_amount = pyo.Constraint(model.BIDTIME_NEXT, rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)
        model.market_clearing_stage_1 = pyo.Constraint(rule = market_clearing_stage_rule_1)
        model.market_clearing_stage_2 = pyo.Constraint(rule = market_clearing_stage_rule_2)
        model.market_clearing_stage_3 = pyo.Constraint(rule = market_clearing_stage_rule_3)
        model.market_clearing_stage_4 = pyo.Constraint(rule = market_clearing_stage_rule_4)
        model.market_clearing_stage_5 = pyo.Constraint(rule = market_clearing_stage_rule_5)
        model.market_clearing_1 = pyo.Constraint(model.BIDTIME, rule = market_clearing_rule_1)
        model.market_clearing_2 = pyo.Constraint(model.BIDTIME, rule = market_clearing_rule_2)
        model.market_clearing_3 = pyo.Constraint(model.BIDTIME, rule = market_clearing_rule_3)
        model.market_clearing_4 = pyo.Constraint(model.BIDTIME, rule = market_clearing_rule_4)
        model.market_clearing_5 = pyo.Constraint(model.BIDTIME, rule = market_clearing_rule_5)
        model.dispatch_stage = pyo.Constraint(rule = dispatch_stage_rule)
        model.dispatch = pyo.Constraint(model.BIDTIME, rule = dispatch_rule)
        model.SOC_init = pyo.Constraint(rule = SOC_init_rule)
        model.SOC_balance = pyo.Constraint(model.TIME, rule = SOC_balance_rule)
        model.generation_stage = pyo.Constraint(rule = generation_stage_rule)
        model.generation_next = pyo.Constraint(rule = generation_next_rule)
        model.generation = pyo.Constraint(model.BIDTIME_NEXT, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_1)
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
        State_var.append(self.o_prev + pyo.value(self.q_rt[self.stage+1]))
        State_var.append(pyo.value(self.b_rt[self.stage+1]))
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

    def get_E_profit(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        E_profit = pyo.value(self.m_2[self.stage]) 

        return E_profit

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
        self.o_prev = state[2]
        self.b_rt_prev = state[3]
        self.q_rt_prev = state[4]
        self.E_prev = state[5]
        
        self.P_da = P_da
        
        self.P_abs = [0 for _ in range(T)]
        
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        
        self.M_price_RT = [[0, 0] for _ in range(self.T)]
        
        self._Param_setting()
        self._BigM_setting()

    def _Param_setting(self):

        self.P_abs[self.stage] = max(self.P_rt - self.P_da[self.stage], 0)

    def _BigM_setting(self):
            
        if self.P_rt >= 0:
            self.M_price_RT[self.stage][0] = 10
            self.M_price_RT[self.stage][1] = self.P_rt + 90

        else:
            self.M_price_RT[self.stage][0] = -self.P_rt + 10
            self.M_price_RT[self.stage][1] = self.P_rt + 90

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(self.stage, T-1)
        model.TIME_ESS = pyo.RangeSet(self.stage-1, T-1)

        # Vars
        
        # Day Ahead variables

        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)

        # Intraday bidding variables

        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        # Intraday operation variables
        
        model.S = pyo.Var(
            model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals, initialize = 0.0
        )
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
                
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)

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

        def market_clearing_stage_rule_1(model):
            return self.b_rt_prev - self.P_rt <= self.M_price_RT[self.stage][0]*(1 - model.n_rt[self.stage])
        
        def market_clearing_stage_rule_2(model):
            return self.P_rt - self.b_rt_prev <= self.M_price_RT[self.stage][1]*model.n_rt[self.stage] 
        
        def market_clearing_stage_rule_3(model):
            return model.Q_rt[self.stage] <= self.q_rt_prev
        
        def market_clearing_stage_rule_4(model):
            return model.Q_rt[self.stage] <= M_gen[self.stage][0]*model.n_rt[self.stage]    
        
        def market_clearing_stage_rule_5(model):
            return model.Q_rt[self.stage] >= self.q_rt_prev - M_gen[self.stage][0]*(1 - model.n_rt[self.stage])    
        
        ## Operations rules
        
        def dispatch_stage_rule(model):
            return model.Q_c[self.stage] == (1 + self.delta_c)*model.Q_rt[self.stage]
        
        def SOC_init_rule(model):
            return model.S[self.stage-1] == self.S_prev
        
        def SOC_balance_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def generation_stage_rule(model):
            return model.g[self.stage] <= self.E_prev
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]
        
        ## min, max reformulation constraints
        
        def minmax_rule_1_1(model, t):
            return model.m_1[t] >= model.Q_da[t] - model.u[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] >= 0

        def minmax_rule_1_3(model, t):
            return model.m_1[t] <= model.Q_da[t] - model.u[t] + M_gen[t][0]*(1 - model.n_1[t])
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] <= M_gen[t][0]*model.n_1[t]
        
        def minmax_rule_2_1(model, t):
            return model.m_2[t] == model.m_1[t]*self.P_abs[t]
        
        ## Imbalance Penalty
        
        def imbalance_over_rule(model, t):
            return model.u[t] - model.Q_c[t] <= model.phi_over[t]
        
        def imbalance_under_rule(model, t):
            return model.Q_c[t] - model.u[t] <= model.phi_under[t]
        
        ## settlement fcn
        
        def rt_settlement_fcn_stage_rule(model):
            return model.f_RT[self.stage] == (
                (model.u[self.stage] - model.Q_da[self.stage])*self.P_rt 
                + self.m_2[self.stage] 
                - gamma_over*model.phi_over[self.stage] 
                - gamma_under*model.phi_under[self.stage]
                + model.u[self.stage]*P_r
                )

        model.da_awarded_amount = pyo.Constraint(model.TIME, rule = da_awarded_amount_rule)
        model.market_clearing_stage_1 = pyo.Constraint(rule = market_clearing_stage_rule_1)
        model.market_clearing_stage_2 = pyo.Constraint(rule = market_clearing_stage_rule_2)
        model.market_clearing_stage_3 = pyo.Constraint(rule = market_clearing_stage_rule_3)
        model.market_clearing_stage_4 = pyo.Constraint(rule = market_clearing_stage_rule_4)
        model.market_clearing_stage_5 = pyo.Constraint(rule = market_clearing_stage_rule_5)
        model.dispatch_stage = pyo.Constraint(rule = dispatch_stage_rule)
        model.SOC_init = pyo.Constraint(rule = SOC_init_rule)
        model.SOC_balance = pyo.Constraint(model.TIME, rule = SOC_balance_rule)
        model.generation_stage = pyo.Constraint(rule = generation_stage_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_1)
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



# PSDDiP Algorithm

def inner_product(t, pi, sol):
    
    return sum(pi[i]*sol[i] for i in [0, 2, 3, 4, 5]) + sum(pi[1][j]*sol[1][j] for j in range(T - t))


def process_single_subproblem_last_stage(j, prev_solution, P_da, delta, cut_mode):
    
    fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, P_da, delta)
    psi_sub = fw_rt_last_LP_relax_subp.get_cut_coefficients()
    
    if cut_mode in ['B']:
        v = psi_sub[0]
        
    else:
        lag = fw_rt_last_Lagrangian([psi_sub[i] for i in range(1, 7)], P_da, delta)
        v = lag.get_objective_value()
        
    return psi_sub, v            

def process_single_subproblem_inner_stage(j, t, prev_solution, psi_next, P_da, delta, cut_mode):
    
    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, psi_next, P_da, delta)
    psi_sub = fw_rt_LP_relax_subp.get_cut_coefficients()
    
    if cut_mode in ['B']:
        v = psi_sub[0]
        
    else:
        lag = fw_rt_Lagrangian(t, [psi_sub[i] for i in range(1, 7)], psi_next, P_da, delta)
        v = lag.get_objective_value()
        
    return psi_sub, v


def process_lag_last_stage(j, prev_solution, P_da, delta):

    fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, P_da, delta)
    
    pi = fw_rt_last_LP_relax_subp.get_cut_coefficients()[1:]
    
    gap = 1       
    lamb = 0
    k_lag = [0, [0], 0, 0, 0, 0]
    l = 10000000
    
    dual_subp_sub_last = dual_approx_sub(T-1, reg, pi)
    
    fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
    
    L = fw_rt_last_Lag_subp.get_objective_value()
    z = fw_rt_last_Lag_subp.get_auxiliary_value()
    
    Lag_iter = 1
                
    pi_minobj = 10000000

    while gap >= dual_tolerance:            
        
        if Lag_iter >= reg_num:
            
            dual_subp_sub_last.reg = 0
        
        lamb = L + inner_product(T - 1, pi, z)
        
        for l in [0, 2, 3, 4, 5]:
            
            k_lag[l] = prev_solution[l] - z[l]
        
        for l in [1]:
            
            k_lag[l][0] = prev_solution[l][0] - z[l][0]
        
        dual_coeff = [lamb, k_lag]
                                    
        dual_subp_sub_last.add_plane(dual_coeff)
        pi = dual_subp_sub_last.get_solution_value()
        obj = dual_subp_sub_last.get_objective_value()
        
        fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
                
        L = fw_rt_last_Lag_subp.get_objective_value()
        z = fw_rt_last_Lag_subp.get_auxiliary_value()
                                                            
        pi_obj = L + inner_product(T - 1, pi, prev_solution)
                        
        if pi_obj < pi_minobj:
            
            pi_minobj = pi_obj
            pi_min = pi
        
        gap = (pi_obj - obj)/(abs(pi_obj) + 1)
                                                                    
        Lag_iter += 1
        
        if Lag_iter == Lag_iter_UB:
            
            break

    return pi, L

def process_lag_inner_stage(j, t, prev_solution, psi_next, P_da, delta):

    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, psi_next, P_da, delta)
    
    pi = fw_rt_LP_relax_subp.get_cut_coefficients()[1:]
        
    gap = 1       
    lamb = 0
    k_lag = [0, [0 for _ in range(T - t)], 0, 0, 0, 0]
    l = 10000000*(T - t)
    
    dual_subp_sub = dual_approx_sub(t, reg, pi)
    fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, psi_next, P_da, delta)
    
    L = fw_rt_Lag_subp.get_objective_value()
    z = fw_rt_Lag_subp.get_auxiliary_value()
    
    Lag_iter = 1
                    
    pi_minobj = 10000000*(T - t)
                    
    while gap >= dual_tolerance:
        
        if Lag_iter >= reg_num:
            
            dual_subp_sub.reg = 0
        
        lamb = L + inner_product(t, pi, z)
        
        for l in [0, 2, 3, 4, 5]:
            
            k_lag[l] = prev_solution[l] - z[l]
            
        for l in [1]:
            
            for i in range(T - t):
                
                k_lag[l][i] = prev_solution[l][i] - z[l][i]
                            
        dual_coeff = [lamb, k_lag]
                        
        dual_subp_sub.add_plane(dual_coeff)
        pi = dual_subp_sub.get_solution_value()
        obj = dual_subp_sub.get_objective_value()
                            
        fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, psi_next, P_da, delta)
                
        L = fw_rt_Lag_subp.get_objective_value()
        z = fw_rt_Lag_subp.get_auxiliary_value()
                                
        pi_obj = L + inner_product(t, pi, prev_solution)
        
        if pi_obj < pi_minobj:
        
            pi_minobj = pi_obj
            pi_min = pi
        
        gap = (pi_obj - obj)/(abs(pi_obj) + 1)
                                                                
        Lag_iter += 1    
        
        if Lag_iter == Lag_iter_UB:
        
            break
        
    return pi, L


def process_hyb_last_stage(j, prev_solution, P_da, delta, threshold):

    fw_rt_last_LP_relax_subp    = fw_rt_last_LP_relax(prev_solution, P_da, delta)
    
    psi_sub   = fw_rt_last_LP_relax_subp.get_cut_coefficients()
    
    pi = psi_sub[1:]
    
    P_rt = delta[1]

    if P_rt <= P_da[T-1] + threshold:
        
        fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
        v = fw_rt_last_Lagrangian_subp.get_objective_value()
        
    else:

        gap = 1       
        lamb = 0
        k_lag = [0, [0], 0, 0, 0, 0]
        l = 10000000

        dual_subp_sub_last = dual_approx_sub(T - 1, reg, pi)
        
        fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
        
        L = fw_rt_last_Lag_subp.get_objective_value()
        z = fw_rt_last_Lag_subp.get_auxiliary_value()
        
        Lag_iter = 1
                    
        pi_minobj = 10000000
        
        while gap >= dual_tolerance:            
            
            if Lag_iter >= reg_num:
                
                dual_subp_sub_last.reg = 0
            
            lamb = L + inner_product(T - 1, pi, z)
            
            for l in [0, 2, 3, 4, 5]:
                
                k_lag[l] = prev_solution[l] - z[l]
            
            for l in [1]:
                
                k_lag[l][0] = prev_solution[l][0] - z[l][0]
            
            dual_coeff = [lamb, k_lag]            
            
            dual_subp_sub_last.add_plane(dual_coeff)
            pi = dual_subp_sub_last.get_solution_value()
            obj = dual_subp_sub_last.get_objective_value()
            
            fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
                        
            L = fw_rt_last_Lag_subp.get_objective_value()
            z = fw_rt_last_Lag_subp.get_auxiliary_value()
                                                                
            pi_obj = L + inner_product(T - 1, pi, prev_solution)
                            
            if pi_obj < pi_minobj:
                
                pi_minobj = pi_obj
                pi_min = pi
            
            gap = (pi_obj - obj)/(abs(pi_obj) + 1)
                                                                            
            Lag_iter += 1
            
            if Lag_iter == Lag_iter_UB:
                
                break
            
        v = L

    return pi, v

def process_hyb_inner_stage(j, t, prev_solution, psi_next, P_da, delta, threshold):

    fw_rt_LP_relax_subp    = fw_rt_LP_relax(t, prev_solution, psi_next, P_da, delta)
    
    psi_sub   = fw_rt_LP_relax_subp.get_cut_coefficients()
    
    pi = psi_sub[1:]
    
    P_rt = delta[1]

    if P_rt <= P_da[t] + threshold:
        
        fw_rt_Lagrangian_subp = fw_rt_Lagrangian(t, pi, psi_next, P_da, delta)
        v = fw_rt_Lagrangian_subp.get_objective_value()
        
    else:
        
        gap = 1
        lamb = 0
        k_lag = [0, [0 for _ in range(T - t)], 0, 0, 0, 0]
        l = 10000000*(T - t)
                        
        dual_subp_sub = dual_approx_sub(t, reg, pi)
        
        fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, psi_next, P_da, delta)
        
        L = fw_rt_Lag_subp.get_objective_value()
        z = fw_rt_Lag_subp.get_auxiliary_value()    
                        
        Lag_iter = 1
                        
        pi_minobj = 10000000*(T - t)
        
        while gap >= dual_tolerance:
            
            if Lag_iter >= reg_num:
                
                dual_subp_sub.reg = 0
            
            lamb = L + inner_product(t, pi, z)
            
            for l in [0, 2, 3, 4, 5]:
                
                k_lag[l] = prev_solution[l] - z[l]
                
            for l in [1]:
                
                for i in range(T - t):
                    
                    k_lag[l][i] = prev_solution[l][i] - z[l][i]
                                
            dual_coeff = [lamb, k_lag]
                                
            dual_subp_sub.add_plane(dual_coeff)
            pi = dual_subp_sub.get_solution_value()
            obj = dual_subp_sub.get_objective_value()
                                
            fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, psi_next, P_da, delta)
                        
            L = fw_rt_Lag_subp.get_objective_value()
            z = fw_rt_Lag_subp.get_auxiliary_value()
                                    
            pi_obj = L + inner_product(t, pi, prev_solution)
            
            if pi_obj < pi_minobj:
            
                pi_minobj = pi_obj
                pi_min = pi
            
            gap = (pi_obj - obj)/(abs(pi_obj) + 1)
                                                                    
            Lag_iter += 1    
            
            if Lag_iter == Lag_iter_UB:
            
                break
        v = L

    return pi, v


class PSDDiPModel:
        
    def __init__(
        self, 
        STAGE = T, 
        DA_params = [],
        ID_params = [],
        scenarios_for_eval = [],
        DA_params_reduced = Reduced_P_da[0], 
        DA_params_prob = Reduced_Probs[0],
        RT_params_reduced = Reduced_scenario_trees[0],
        scenario_exp = Reduced_scenario_trees[0],
        sample_num = 1000,
        evaluation_num = 10,
        alpha = 0.95, 
        cut_mode = 'B',
        tol = 0.001,
        parallel_mode = 0,
        ):

        ## STAGE = -1, 0, ..., STAGE - 1
        ## forward_scenarios(last iteration) = M
        ## baackward_scenarios = N_t

        self.STAGE = STAGE
        
        self.DA_params_evaluation = DA_params
        self.ID_params_evaluation = ID_params
        
        self.DA_params = DA_params_reduced
        self.Prob = DA_params_prob
        self.RT_params = RT_params_reduced
        
        self.M = sample_num
        self.N = evaluation_num
        
        self.scenarios_for_eval = scenarios_for_eval
        
        self.scenario_exp = scenario_exp
        
        self.alpha = alpha
        self.cut_mode = cut_mode
        self.tol = tol
        
        self.parallel_mode = parallel_mode
                
        self.iteration = 0
        
        self.K = len(self.DA_params)
        self.N_t = len(self.RT_params[0][0])
            
        self.K_eval = len(self.DA_params_evaluation)
                
        self.start_time = time.time()
        self.running_time = 0
        
        self.gap = 1
                
        self.LB = [-np.inf]
        self.UB = [np.inf]
        
        self.eval = 0

        self.forward_solutions_da = [] ## Day Ahead Solution (Day Ahead Bidding)
        
        self.forward_solutions = [  ## x(-1), ..., x(T - 2)
                [
                    [] for _ in range(self.STAGE)
                ] for _ in range(self.K)
            ]
        
        self.b_da = [0 for _ in range(self.STAGE)]
        self.q_da = [0 for _ in range(self.STAGE)]
        self.f_DA = [0 for _ in range(self.STAGE)]
        
        self.f_P = [0 for _ in range(self.STAGE)]
        self.f_E = [0 for _ in range(self.STAGE)]
        self.f_Im = [0 for _ in range(self.STAGE)]
        
        self.psi_da = [     ## t = -1 -> Day-Ahead Stage (Multi-cut)
                [] for _ in range(self.K)
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
                            [0 for _ in range(self.STAGE)],
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
                    0, 0, 0, 0
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
                        + sum(coeff_da[2][t]*sol[1][t] for t in range(self.STAGE))
                    )
                
                min_index_da = min(enumerate(V_da), key=lambda x: x[1])[0]

                min_index_da_list.append(min_index_da)

            min_index_da_list = list(dict.fromkeys(min_index_da_list))
            
            psi_da = [self.psi_da[k][i] for i in min_index_da_list] 
            
            self.psi_da[k] = psi_da
        
        ## cut selection for RT stage
        
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
                            + coeff_rt[5]*sol[4]
                            + coeff_rt[6]*sol[5]
                        )
                    
                    min_index_rt = min(enumerate(V_rt), key=lambda x: x[1])[0]

                    min_index_rt_list.append(min_index_rt)
                
                min_index_rt_list = list(dict.fromkeys(min_index_rt_list))
                
                psi_rt = [self.psi[k][t][i] for i in min_index_rt_list]
                
                self.psi[k][t] = psi_rt
             
    # For rolling horizon evaluation                 
        exp_P_rt_list = []
        
        scenario_tree_rt = self.scenario_exp[0]
        
        for t in range(self.STAGE):
            
            branches_t = scenario_tree_rt[t]
            
            exp_P_rt = 0
            
            for b in branches_t:   
                
                exp_P_rt += b[1]/len(branches_t)
            
            exp_P_rt_list.append(exp_P_rt)
                            
        return exp_P_rt_list 
    
    def exp_P_rt_given_P_da(self, n):
    
        exp_P_rt_list = []
        
        scenario_tree_rt = self.ID_params_evaluation[n]
        
        for t in range(self.STAGE):
            
            branches_t = scenario_tree_rt[t]
            
            exp_P_rt = 0
            
            for b in branches_t:   
                
                exp_P_rt += b[1]/len(branches_t)
            
            exp_P_rt_list.append(exp_P_rt)
                            
        return exp_P_rt_list

                
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

    # Find nearest reduced P_da 
    def find_cluster_index_for_evaluation(self, n):
        
        actual_P_da = np.array(self.DA_params_evaluation[n])
        centers = np.array(self.DA_params)  
        dists = np.linalg.norm(centers - actual_P_da, axis=1)
        
        return int(np.argmin(dists))
            
            
    def forward_pass(self, scenarios):
        
        fw_da_subp = fw_da_multicut(self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
        
        self.forward_solutions_da.append(fw_da_state)
        
        f = []
        
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
                        
            f.append(f_scenario)
        
        mu_hat = np.mean(f)
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        #self.LB.append(mu_hat - z_alpha_half * (sigma_hat / np.sqrt(self.M))) 
        self.LB.append(mu_hat) 
        
        return mu_hat
      
    def forward_pass_for_stopping(self, scenarios):
        
        fw_da_subp = fw_da_multicut(self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
        
        self.forward_solutions_da.append(fw_da_state)
        
        f = []
        
        for k, scenarios_k in enumerate(scenarios):
            
            P_da = self.DA_params[k]
            
            fw_rt_init_subp = fw_rt_init(fw_da_state, self.psi[k][0], P_da)
            fw_rt_init_state = fw_rt_init_subp.get_state_solutions()
            
            self.forward_solutions[k][0].append(fw_rt_init_state) ## x(-1)
            
            for scenario in scenarios_k:
                
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
                            
                f.append(f_scenario)
            
        mu_hat = np.mean(f)
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        #self.LB.append(mu_hat - z_alpha_half * (sigma_hat / np.sqrt(self.M))) 
        self.LB.append(mu_hat) 
        
        return mu_hat

    def forward_pass_for_eval(self, scenarios):
        
        fw_da_subp = fw_da_multicut(self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
        
        self.b_da = fw_da_state[0]
        self.q_da = fw_da_state[1]
        
        f = []
        
        for n, scenarios_n in enumerate(scenarios):
            
            k = self.find_cluster_index_for_evaluation(n)
            
            P_da = self.DA_params_evaluation[n]  
            
            fw_rt_init_subp = fw_rt_init(fw_da_state, self.psi[k][0], P_da)
            fw_rt_init_state = fw_rt_init_subp.get_state_solutions()
            
            f_DA_list = fw_rt_init_subp.get_DA_profit()
            
            for i in range(self.STAGE):
                self.f_DA[i] += f_DA_list[i]/self.K_eval
                        
            for scenario in scenarios_n:
                
                state = fw_rt_init_state
                
                f_scenario = fw_rt_init_subp.get_settlement_fcn_value()
                
                for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                    
                    fw_rt_subp = fw_rt(t, state, self.psi[k][t+1], P_da, scenario[t])
                    
                    self.f_P[t] += fw_rt_subp.get_P_profit()/(self.K_eval*self.N)
                    self.f_E[t] += fw_rt_subp.get_E_profit()/(self.K_eval*self.N)
                    self.f_Im[t] += fw_rt_subp.get_Im_profit()/(self.K_eval*self.N)
                    
                    state = fw_rt_subp.get_state_solutions()
                    
                    f_scenario += fw_rt_subp.get_settlement_fcn_value()
                
                ## t = T-1
                
                fw_rt_last_subp = fw_rt_last(state, P_da, scenario[self.STAGE-1])

                f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
                            
                f.append(f_scenario)
            
        mu_hat = np.mean(f)
        
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        self.eval = mu_hat 
   
    def forward_pass_for_eval_roll(self, scenarios):
        
        fw_da_subp = fw_da_multicut(self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
        
        self.b_da = fw_da_state[0]
        self.q_da = fw_da_state[1]
        
        f = []
        
        for n, scenarios_n in enumerate(scenarios):
                        
            P_da = self.DA_params_evaluation[n]  
            
            exp_P_rt = self.exp_P_rt_given_P_da(n)
            
            rt_init_subp = rolling_rt_init(fw_da_state, P_da, exp_P_rt)
            rt_init_state = rt_init_subp.get_state_solutions()
            
            f_DA_list = rt_init_subp.get_DA_profit()
            
            for i in range(self.STAGE):
                self.f_DA[i] += f_DA_list[i]/self.K_eval
            
            fcn_value = rt_init_subp.get_settlement_fcn_value()
               
            for scenario in scenarios_n:
                
                state = rt_init_state
                
                f_scenario = fcn_value
                
                for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                    
                    rt_subp = rolling_rt(t, state, P_da, exp_P_rt, scenario[t])
                    
                    self.f_P[t] += rt_subp.get_P_profit()/(self.K_eval*self.N)
                    self.f_E[t] += rt_subp.get_E_profit()/(self.K_eval*self.N)
                    self.f_Im[t] += rt_subp.get_Im_profit()/(self.K_eval*self.N)
                    
                    state = rt_subp.get_state_solutions()
                    
                    f_scenario += rt_subp.get_settlement_fcn_value()
                
                ## t = T-1
                
                rt_last_subp = rolling_rt_last(state, P_da, scenario[self.STAGE-1])

                self.f_P[self.STAGE-1] += rt_subp.get_P_profit()/(self.K_eval*self.N)
                self.f_E[self.STAGE-1] += rt_subp.get_E_profit()/(self.K_eval*self.N)
                self.f_Im[self.STAGE-1] += rt_subp.get_Im_profit()/(self.K_eval*self.N)

                f_scenario += rt_last_subp.get_settlement_fcn_value()
                            
                f.append(f_scenario)
            
        mu_hat = np.mean(f)
        
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        self.eval = mu_hat 
       
        
    def inner_product(self, t, pi, sol):
        
        return sum(pi[i]*sol[i] for i in [0, 2, 3, 4, 5]) + sum(pi[1][j]*sol[1][j] for j in range(self.STAGE - t))

    def inner_product_da(self, pi, sol):
        
        return sum(sum(pi[i][t]*sol[i][t] for t in range(self.STAGE)) for i in [0, 1]) 


    def backward_pass(self):
        
        BL = ['B']
        SBL = ['SB', 'L-sub', 'L-lev']
        
        for k, P_da in enumerate(self.DA_params):
            
            stage_params = self.RT_params[k]
            
            ## t = {T-1 -> T-2}
            
            v_sum = 0 
            pi_mean = [0, [0], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[k][self.STAGE - 1][-1]
            
            for j in range(self.N_t): 
                
                delta = stage_params[self.STAGE - 1][j]  
                    
                fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, P_da, delta)
                
                psi_sub = fw_rt_last_LP_relax_subp.get_cut_coefficients()
                
                pi_mean[0] += psi_sub[1]/self.N_t
                pi_mean[1][0] += psi_sub[2][0]/self.N_t
                pi_mean[2] += psi_sub[3]/self.N_t
                pi_mean[3] += psi_sub[4]/self.N_t
                pi_mean[4] += psi_sub[5]/self.N_t
                pi_mean[5] += psi_sub[6]/self.N_t
                
                if self.cut_mode in BL:
                    
                    v_sum += psi_sub[0]
                
                elif self.cut_mode in SBL or self.cut_mode.startswith('hyb'):
                    
                    fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian([psi_sub[i] for i in range(1, 7)], P_da, delta)

                    v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
                
            if self.cut_mode in BL:   
                    
                v = v_sum/self.N_t - self.inner_product(self.STAGE - 1, pi_mean, prev_solution)
            
            elif self.cut_mode in SBL or self.cut_mode.startswith('hyb'):
                
                v = v_sum/self.N_t
                
            cut_coeff = []
            
            cut_coeff.append(v)
            
            for i in range(6):
                cut_coeff.append(pi_mean[i])
            
            self.psi[k][T-1].append(cut_coeff)
            
            #print(f"last_stage_cut_coeff = {self.psi[T-1]}")
            
            ## t = {T-2 -> T-3}, ..., {0 -> -1}
            for t in range(self.STAGE - 2, -1, -1): 
                    
                v_sum = 0 
                pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
                
                prev_solution = self.forward_solutions[k][t][-1]
                
                for j in range(self.N_t):
                    
                    delta = stage_params[t][j]
                    
                    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[k][t+1], P_da, delta)
    
                    psi_sub = fw_rt_LP_relax_subp.get_cut_coefficients()
                    
                    pi_mean[0] += psi_sub[1]/self.N_t
                    
                    for i in range(self.STAGE - t):
                        pi_mean[1][i] += psi_sub[2][i]/self.N_t
                        
                    pi_mean[2] += psi_sub[3]/self.N_t
                    pi_mean[3] += psi_sub[4]/self.N_t
                    pi_mean[4] += psi_sub[5]/self.N_t
                    pi_mean[5] += psi_sub[6]/self.N_t
                    
                    if self.cut_mode in BL:
                        
                        v_sum += psi_sub[0]
                        
                    elif self.cut_mode in SBL or self.cut_mode.startswith('hyb'):
                        
                        fw_rt_Lagrangian_subp = fw_rt_Lagrangian(t, [psi_sub[i] for i in range(1, 7)], self.psi[k][t+1], P_da, delta)

                        v_sum += fw_rt_Lagrangian_subp.get_objective_value()
                
                if self.cut_mode in BL:
                    
                    v = v_sum/self.N_t - self.inner_product(t, pi_mean, prev_solution)
            
                if self.cut_mode in SBL or self.cut_mode.startswith('hyb'):
                    
                    v = v_sum/self.N_t
            
                cut_coeff = []
                
                cut_coeff.append(v)
                
                for i in range(6):
                    cut_coeff.append(pi_mean[i])
            
                self.psi[k][t].append(cut_coeff)
                #print(f"stage {t - 1} _cut_coeff = {self.psi[t]}")
                     
        v_sum = 0
        pi_mean = [[0 for _ in range(self.STAGE)], [0 for _ in range(self.STAGE)]]
        
        prev_solution = self.forward_solutions_da[-1] 
        
        for k, P_da in enumerate(self.DA_params):
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(
                prev_solution, self.psi[k][0], P_da
                )
            
            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            pi = [psi_sub[1]] + [psi_sub[2]]

            if self.cut_mode in BL or self.cut_mode.startswith('hyb'):
                v = psi_sub[0]
                
            else:
                lag = fw_rt_init_Lagrangian([psi_sub[i] for i in [1,2]], self.psi[k][0], P_da)
                v = lag.get_objective_value()


            cut_coeff = [v] + [pi[0]] + [pi[1]]
            self.psi_da[k].append(cut_coeff)
        
        fw_da_for_UB = fw_da_multicut(self.psi_da)
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))

    def backward_pass_Lagrangian(self):
        
        for k, P_da in enumerate(self.DA_params):
                
            stage_params = self.RT_params[k]    
                
            ## t = {T-1 -> T-2}
            
            v_sum = 0 
            pi_mean = [0, [0], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[k][self.STAGE - 1][-1]
                    
            for j in range(self.N_t): 
                
                delta = stage_params[self.STAGE - 1][j]      
                
                fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, P_da, delta)
                
                pi_LP = fw_rt_last_LP_relax_subp.get_cut_coefficients()[1:]
                
                pi = pi_LP
                pi_min = pi_LP
                
                gap = 1       
                lamb = 0
                k_lag = [0, [0], 0, 0, 0, 0]
                l = 10000000
                            
                dual_subp_sub_last = dual_approx_sub(self.STAGE - 1, reg, pi_LP)
                
                fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
                
                L = fw_rt_last_Lag_subp.get_objective_value()
                z = fw_rt_last_Lag_subp.get_auxiliary_value()
                
                Lag_iter = 1
                            
                pi_minobj = 10000000
                
                while gap >= dual_tolerance:            
                    
                    if Lag_iter >= reg_num:
                        
                        dual_subp_sub_last.reg = 0
                    
                    lamb = L + self.inner_product(self.STAGE - 1, pi, z)
                    
                    for l in [0, 2, 3, 4, 5]:
                        
                        k_lag[l] = prev_solution[l] - z[l]
                    
                    for l in [1]:
                        
                        k_lag[l][0] = prev_solution[l][0] - z[l][0]
                    
                    dual_coeff = [lamb, k_lag]
                                    
                    if self.cut_mode == 'L-sub':
                        
                        dual_subp_sub_last.add_plane(dual_coeff)
                        pi = dual_subp_sub_last.get_solution_value()
                        obj = dual_subp_sub_last.get_objective_value()
                    
                    fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
                    
                    start_time = time.time()
                    
                    L = fw_rt_last_Lag_subp.get_objective_value()
                    z = fw_rt_last_Lag_subp.get_auxiliary_value()
                                                                        
                    pi_obj = L + self.inner_product(self.STAGE - 1, pi, prev_solution)
                                    
                    if pi_obj < pi_minobj:
                        
                        pi_minobj = pi_obj
                        pi_min = pi
                    
                    if pi_obj == -G:
                        Lag_iter += 1
                        continue
                    
                    gap = (pi_obj - obj)/(pi_obj+G)
                                            
                    #print(f"k = {k}, \npi = {pi} \n, \ngap = {gap}, \npi_obj = {pi_obj}, \nobj = {obj}")
                                                
                    Lag_iter += 1
                    
                    if Lag_iter == Lag_iter_UB:
                        
                        break
                                
                pi_mean[0] += pi[0]/self.N_t
                pi_mean[1][0] += pi[1][0]/self.N_t
                pi_mean[2] += pi[2]/self.N_t
                pi_mean[3] += pi[3]/self.N_t
                pi_mean[4] += pi[4]/self.N_t
                pi_mean[5] += pi[5]/self.N_t
                
                v_sum += L
                    
            v = v_sum/self.N_t
                
            cut_coeff = []
            
            cut_coeff.append(v)
            
            for i in range(6):
                cut_coeff.append(pi_mean[i])
            
            self.psi[k][self.STAGE - 1].append(cut_coeff)
                    
            #print(f"last_stage_cut_coeff = {self.psi[T-1]}")
            
            ## t = {T-2 -> T-3}, ..., {0 -> -1}
            
            for t in range(self.STAGE - 2, -1, -1): 
                
                v_sum = 0 
                pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
                
                prev_solution = self.forward_solutions[k][t][-1]
                            
                for j in range(self.N_t):
                                    
                    delta = stage_params[t][j]
                    
                    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[k][t+1], P_da, delta)

                    pi_LP = fw_rt_LP_relax_subp.get_cut_coefficients()[1:]
                    
                    pi = pi_LP
                    pi_min = pi_LP
                                    
                    gap = 1
                    lamb = 0
                    k_lag = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
                    l = 10000000*(self.STAGE - t)
                        
                    dual_subp_sub = dual_approx_sub(t, reg, pi_LP)
                    
                    fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[k][t+1], P_da, delta)
                    
                    L = fw_rt_Lag_subp.get_objective_value()
                    z = fw_rt_Lag_subp.get_auxiliary_value()    
                                    
                    Lag_iter = 1
                                    
                    pi_minobj = 10000000*(self.STAGE - t)
                    
                    while gap >= dual_tolerance:
                        
                        if Lag_iter >= reg_num:
                            
                            dual_subp_sub.reg = 0
                        
                        lamb = L + self.inner_product(t, pi, z)
                        
                        for l in [0, 2, 3, 4, 5]:
                            
                            k_lag[l] = prev_solution[l] - z[l]
                            
                        for l in [1]:
                            
                            for i in range(self.STAGE - t):
                                
                                k_lag[l][i] = prev_solution[l][i] - z[l][i]
                                            
                        dual_coeff = [lamb, k_lag]
                                            
                        if self.cut_mode == 'L-sub':
                            dual_subp_sub.add_plane(dual_coeff)
                            pi = dual_subp_sub.get_solution_value()
                            obj = dual_subp_sub.get_objective_value()
                                            
                        fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[k][t+1], P_da, delta)
                        
                        start_time = time.time()
                        
                        L = fw_rt_Lag_subp.get_objective_value()
                        z = fw_rt_Lag_subp.get_auxiliary_value()
                                                
                        pi_obj = L + self.inner_product(t, pi, prev_solution)
                        
                        if pi_obj < pi_minobj:
                        
                            pi_minobj = pi_obj
                            pi_min = pi
                        
                        if pi_obj == -G:
                            Lag_iter += 1
                            continue
                        
                        gap = (pi_obj - obj)/(pi_obj+G)
                                                                                
                        Lag_iter += 1    
                        
                        if Lag_iter == Lag_iter_UB:
                        
                            break
                                                                                
                    pi_mean[0] += pi[0]/self.N_t
                    
                    for i in range(self.STAGE - t):
                        pi_mean[1][i] += pi[1][i]/self.N_t
                        
                    pi_mean[2] += pi[2]/self.N_t
                    pi_mean[3] += pi[3]/self.N_t
                    pi_mean[4] += pi[4]/self.N_t
                    pi_mean[5] += pi[5]/self.N_t
                                    
                    v_sum += L
                                
                v = v_sum/self.N_t
            
                cut_coeff = []
                
                cut_coeff.append(v)
                
                for i in range(6):
                    cut_coeff.append(pi_mean[i])
            
                self.psi[k][t].append(cut_coeff)
                #print(f"stage {t - 1} _cut_coeff = {self.psi[t]}")
        
        prev_solution = self.forward_solutions_da[-1] 
        
        for k, P_da in enumerate(self.DA_params):
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(
                prev_solution, self.psi[k][0], P_da
                )
            
            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            pi = [psi_sub[1]] + [psi_sub[2]]
                
            lag = fw_rt_init_Lagrangian([psi_sub[i] for i in [1,2]], self.psi[k][0], P_da)
            v = lag.get_objective_value()

            cut_coeff = [v] + [pi[0]] + [pi[1]]
            self.psi_da[k].append(cut_coeff)
        
        fw_da_for_UB = fw_da_multicut(self.psi_da)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   

    def backward_pass_hybrid(self):
        
        for k, P_da in enumerate(self.DA_params):
                
            stage_params = self.RT_params[k]    
                
            ## t = {T-1 -> T-2}
                        
            gap = 1  
            
            v_sum = 0 
            pi_mean = [0, [0], 0, 0, 0, 0]
            
            threshold = int(self.cut_mode.split('-')[1])
                        
            prev_solution = self.forward_solutions[k][self.STAGE - 1][-1]
                    
            for j in range(self.N_t): 
                
                delta = stage_params[self.STAGE - 1][j]      
                
                P_rt_branch = delta[1]
                P_da_branch = P_da[self.STAGE - 1]
                
                fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, P_da, delta)
                
                psi_sub = fw_rt_last_LP_relax_subp.get_cut_coefficients()
                
                if P_rt_branch <= P_da_branch + threshold:
                        
                    pi_mean[0] += psi_sub[1]/self.N_t
                    pi_mean[1][0] += psi_sub[2][0]/self.N_t
                    pi_mean[2] += psi_sub[3]/self.N_t
                    pi_mean[3] += psi_sub[4]/self.N_t
                    pi_mean[4] += psi_sub[5]/self.N_t
                    pi_mean[5] += psi_sub[6]/self.N_t    
                    
                    fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian([psi_sub[i] for i in range(1, 7)], P_da, delta)

                    v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
                    
                else:
                            
                    pi_LP = psi_sub[1:]
                    
                    pi = pi_LP
                    pi_min = pi_LP
                    
                    gap = 1       
                    lamb = 0
                    k_lag = [0, [0], 0, 0, 0, 0]
                    l = 10000000
                                
                    dual_subp_sub_last = dual_approx_sub(self.STAGE - 1, reg, pi_LP)
                    
                    fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
                    
                    L = fw_rt_last_Lag_subp.get_objective_value()
                    z = fw_rt_last_Lag_subp.get_auxiliary_value()
                    
                    Lag_iter = 1
                                
                    pi_minobj = 10000000
                    
                    while gap >= dual_tolerance:            
                        
                        if Lag_iter >= reg_num:
                            
                            dual_subp_sub_last.reg = 0
                        
                        lamb = L + self.inner_product(self.STAGE - 1, pi, z)
                        
                        for l in [0, 2, 3, 4, 5]:
                            
                            k_lag[l] = prev_solution[l] - z[l]
                        
                        for l in [1]:
                            
                            k_lag[l][0] = prev_solution[l][0] - z[l][0]
                        
                        dual_coeff = [lamb, k_lag]            
                        
                        dual_subp_sub_last.add_plane(dual_coeff)
                        pi = dual_subp_sub_last.get_solution_value()
                        obj = dual_subp_sub_last.get_objective_value()
                        
                        fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
                        
                        start_time = time.time()
                        
                        L = fw_rt_last_Lag_subp.get_objective_value()
                        z = fw_rt_last_Lag_subp.get_auxiliary_value()
                                                                            
                        pi_obj = L + self.inner_product(self.STAGE - 1, pi, prev_solution)
                                        
                        if pi_obj < pi_minobj:
                            
                            pi_minobj = pi_obj
                            pi_min = pi
                        
                        if pi_obj == -G:
                            Lag_iter += 1
                            continue
                        
                        gap = (pi_obj - obj)/(pi_obj+G)
                                                
                        #print(f"k = {k}, \npi = {pi} \n, \ngap = {gap}, \npi_obj = {pi_obj}, \nobj = {obj}")
                                                    
                        Lag_iter += 1
                        
                        if Lag_iter == Lag_iter_UB:
                            
                            break
                                    
                    pi_mean[0] += pi[0]/self.N_t
                    pi_mean[1][0] += pi[1][0]/self.N_t
                    pi_mean[2] += pi[2]/self.N_t
                    pi_mean[3] += pi[3]/self.N_t
                    pi_mean[4] += pi[4]/self.N_t
                    pi_mean[5] += pi[5]/self.N_t
                    
                    v_sum += L
                    
            v = v_sum/self.N_t
                
            cut_coeff = []
            
            cut_coeff.append(v)
            
            for i in range(6):
                cut_coeff.append(pi_mean[i])
            
            self.psi[k][self.STAGE - 1].append(cut_coeff)
                    
            #print(f"last_stage_cut_coeff = {self.psi[T-1]}")
            
            ## t = {T-2 -> T-3}, ..., {0 -> -1}
            
            for t in range(self.STAGE - 2, -1, -1): 
                
                v_sum = 0 
                pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
                
                prev_solution = self.forward_solutions[k][t][-1]
                            
                for j in range(self.N_t):
                                    
                    delta = stage_params[t][j]
                    
                    P_rt_branch = delta[1]
                    P_da_branch = P_da[t]
                    
                    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[k][t+1], P_da, delta)

                    psi_sub = fw_rt_LP_relax_subp.get_cut_coefficients()
                    
                    if P_rt_branch <= P_da_branch + threshold:
                        
                        pi_mean[0] += psi_sub[1]/self.N_t
                        
                        for i in range(self.STAGE - t):
                            pi_mean[1][i] += psi_sub[2][i]/self.N_t
                            
                        pi_mean[2] += psi_sub[3]/self.N_t
                        pi_mean[3] += psi_sub[4]/self.N_t
                        pi_mean[4] += psi_sub[5]/self.N_t
                        pi_mean[5] += psi_sub[6]/self.N_t
                        
                        fw_rt_Lagrangian_subp = fw_rt_Lagrangian(t, [psi_sub[i] for i in range(1, 7)], self.psi[k][t+1], P_da, delta)

                        v_sum += fw_rt_Lagrangian_subp.get_objective_value()

                    else: 
                        
                        pi_LP = psi_sub[1:]
                        
                        pi = pi_LP
                        pi_min = pi_LP
                                        
                        gap = 1
                        lamb = 0
                        k_lag = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
                        l = 10000000*(self.STAGE - t)
                            
                        dual_subp_sub = dual_approx_sub(t, reg, pi_LP)
                        
                        fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[k][t+1], P_da, delta)
                        
                        L = fw_rt_Lag_subp.get_objective_value()
                        z = fw_rt_Lag_subp.get_auxiliary_value()    
                                        
                        Lag_iter = 1
                                        
                        pi_minobj = 10000000*(self.STAGE - t)
                        
                        while gap >= dual_tolerance:
                            
                            if Lag_iter >= reg_num:
                                
                                dual_subp_sub.reg = 0
                            
                            lamb = L + self.inner_product(t, pi, z)
                            
                            for l in [0, 2, 3, 4, 5]:
                                
                                k_lag[l] = prev_solution[l] - z[l]
                                
                            for l in [1]:
                                
                                for i in range(self.STAGE - t):
                                    
                                    k_lag[l][i] = prev_solution[l][i] - z[l][i]
                                                
                            dual_coeff = [lamb, k_lag]
                                                
                            dual_subp_sub.add_plane(dual_coeff)
                            pi = dual_subp_sub.get_solution_value()
                            obj = dual_subp_sub.get_objective_value()
                                                
                            fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[k][t+1], P_da, delta)
                            
                            start_time = time.time()
                            
                            L = fw_rt_Lag_subp.get_objective_value()
                            z = fw_rt_Lag_subp.get_auxiliary_value()
                                                    
                            pi_obj = L + self.inner_product(t, pi, prev_solution)
                            
                            if pi_obj < pi_minobj:
                            
                                pi_minobj = pi_obj
                                pi_min = pi
                            
                            if pi_obj == -G:
                                Lag_iter += 1
                                continue
                            
                            gap = (pi_obj - obj)/(pi_obj+G)
                                                                                    
                            Lag_iter += 1    
                            
                            if Lag_iter == Lag_iter_UB:
                            
                                break
                                                                                    
                        pi_mean[0] += pi[0]/self.N_t
                        
                        for i in range(self.STAGE - t):
                            pi_mean[1][i] += pi[1][i]/self.N_t
                            
                        pi_mean[2] += pi[2]/self.N_t
                        pi_mean[3] += pi[3]/self.N_t
                        pi_mean[4] += pi[4]/self.N_t
                        pi_mean[5] += pi[5]/self.N_t
                                        
                        v_sum += L
                                
                v = v_sum/self.N_t
            
                cut_coeff = []
                
                cut_coeff.append(v)
                
                for i in range(6):
                    cut_coeff.append(pi_mean[i])
            
                self.psi[k][t].append(cut_coeff)
                #print(f"stage {t - 1} _cut_coeff = {self.psi[t]}")
        
        prev_solution = self.forward_solutions_da[-1] 
        
        for k, P_da in enumerate(self.DA_params):
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(
                prev_solution, self.psi[k][0], P_da
                )
            
            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            pi = [psi_sub[1]] + [psi_sub[2]]

            lag = fw_rt_init_Lagrangian([psi_sub[i] for i in [1,2]], self.psi[k][0], P_da)
            v = lag.get_objective_value()

            cut_coeff = [v] + [pi[0]] + [pi[1]]
            self.psi_da[k].append(cut_coeff)
        
        fw_da_for_UB = fw_da_multicut(self.psi_da)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   


    def backward_pass_1(self):
     
        BL  = ['B']
        SBL = ['SB', 'L-sub', 'L-lev']

        for k, P_da in enumerate(self.DA_params):
            
            stage_params = self.RT_params[k]

            with mp.Pool() as pool:

                t_last        = self.STAGE - 1
                prev_solution = self.forward_solutions[k][t_last][-1]
                deltas_last   = stage_params[t_last]

                last_args = [
                    (j, prev_solution, P_da, deltas_last[j], self.cut_mode)
                    for j in range(self.N_t)
                ]
                
                last_results = pool.starmap(
                    process_single_subproblem_last_stage,
                    last_args
                )

                v_sum   = 0.0
                pi_mean = [0, [0], 0, 0, 0, 0]
                
                for psi_sub, v in last_results:
                    
                    v_sum += v
                    pi_mean[0]    += psi_sub[1]    / self.N_t
                    pi_mean[1][0] += psi_sub[2][0] / self.N_t
                    pi_mean[2]    += psi_sub[3]    / self.N_t
                    pi_mean[3]    += psi_sub[4]    / self.N_t
                    pi_mean[4]    += psi_sub[5]    / self.N_t
                    pi_mean[5]    += psi_sub[6]    / self.N_t

                if self.cut_mode in BL:
                    v = v_sum/self.N_t - self.inner_product(t_last, pi_mean, prev_solution)
                
                else:
                    v = v_sum/self.N_t

                cut_coeff = [v] + pi_mean
                self.psi[k][t_last].append(cut_coeff)

                for t in range(self.STAGE - 2, -1, -1):
                    prev_solution = self.forward_solutions[k][t][-1]
                    psi_next      = self.psi[k][t+1]
                    deltas        = stage_params[t]

                    inner_args = [
                        (j, t, prev_solution, psi_next, P_da, deltas[j], self.cut_mode)
                        for j in range(self.N_t)
                    ]
                    inner_results = pool.starmap(
                        process_single_subproblem_inner_stage,
                        inner_args
                    )

                    v_sum   = 0.0
                    pi_mean = [0, [0]*(self.STAGE - t), 0, 0, 0, 0]
                    
                    for psi_sub, v in inner_results:
                        v_sum += v
                        pi_mean[0] += psi_sub[1] / self.N_t
                        
                        for i in range(self.STAGE - t):
                            pi_mean[1][i] += psi_sub[2][i] / self.N_t
                        pi_mean[2] += psi_sub[3] / self.N_t
                        pi_mean[3] += psi_sub[4] / self.N_t
                        pi_mean[4] += psi_sub[5] / self.N_t
                        pi_mean[5] += psi_sub[6] / self.N_t

                    if self.cut_mode in BL:
                        v = v_sum/self.N_t - self.inner_product(t, pi_mean, prev_solution)
                    
                    else:
                        v = v_sum/self.N_t

                    cut_coeff = [v] + pi_mean
                    self.psi[k][t].append(cut_coeff)

        prev_solution = self.forward_solutions_da[-1]

        for k, P_da in enumerate(self.DA_params):
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(
                prev_solution, self.psi[k][0], P_da
                )
            
            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            pi = [psi_sub[1]] + [psi_sub[2]]

            if self.cut_mode in BL or self.cut_mode.startswith('hyb'):
                v = psi_sub[0]
                
            else:
                lag = fw_rt_init_Lagrangian([psi_sub[i] for i in [1,2]], self.psi[k][0], P_da)
                v = lag.get_objective_value()


            cut_coeff = [v] + [pi[0]] + [pi[1]]
            self.psi_da[k].append(cut_coeff)
        
        fw_da_for_UB = fw_da_multicut(self.psi_da)
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))

    def backward_pass_Lagrangian_1(self):
        
        for k, P_da in enumerate(self.DA_params):
            
            stage_params = self.RT_params[k]

            with mp.Pool() as pool:

                t_last        = self.STAGE - 1
                prev_solution = self.forward_solutions[k][t_last][-1]
                deltas_last   = stage_params[t_last]
                
                last_args = [
                    (j, prev_solution, P_da, deltas_last[j])
                    for j in range(self.N_t)
                ]
                
                last_results = pool.starmap(process_lag_last_stage, last_args)

                v_sum = 0.0
                pi_mean = [0, [0], 0, 0, 0, 0]
                
                for pi, L in last_results:
                    
                    v_sum += L
                    pi_mean[0]    += pi[0] / self.N_t
                    pi_mean[1][0] += pi[1][0] / self.N_t
                    pi_mean[2]    += pi[2] / self.N_t
                    pi_mean[3]    += pi[3] / self.N_t
                    pi_mean[4]    += pi[4] / self.N_t
                    pi_mean[5]    += pi[5] / self.N_t

                v = v_sum/self.N_t
                
                cut_coeff = [v] + pi_mean
                
                self.psi[k][t_last].append(cut_coeff)

                for t in range(self.STAGE-2, -1, -1):
                    
                    prev = self.forward_solutions[k][t][-1]
                    psi_next = self.psi[k][t+1]
                    
                    args_inner = [
                        (j, t, prev, psi_next, P_da, stage_params[t][j])
                        for j in range(self.N_t)
                    ]
                    
                    results_inner = pool.starmap(process_lag_inner_stage,
                                                 args_inner)

                    v_sum = 0.0
                    pi_mean = [0, [0]*(self.STAGE-t), 0, 0, 0, 0]
                    
                    for pi, L in results_inner:
                        
                        v_sum += L
                        pi_mean[0] += pi[0] / self.N_t
                        
                        for i in range(self.STAGE-t):
                            pi_mean[1][i] += pi[1][i] / self.N_t
                            
                        pi_mean[2] += pi[2] / self.N_t
                        pi_mean[3] += pi[3] / self.N_t
                        pi_mean[4] += pi[4] / self.N_t
                        pi_mean[5] += pi[5] / self.N_t

                    v = v_sum/self.N_t
                    cut_coeff = [v] + pi_mean
                    self.psi[k][t].append(cut_coeff)
                    
        
        prev_da = self.forward_solutions_da[-1]

        for k, P_da in enumerate(self.DA_params):
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(
                prev_da, self.psi[k][0], P_da
                )
            
            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            pi = [psi_sub[1]] + [psi_sub[2]]
                
            lag = fw_rt_init_Lagrangian([psi_sub[i] for i in [1,2]], self.psi[k][0], P_da)
            v = lag.get_objective_value()

            cut_coeff = [v] + [pi[0]] + [pi[1]]
            self.psi_da[k].append(cut_coeff)

        fw_da_for_UB = fw_da_multicut(self.psi_da)
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))

    def backward_pass_hybrid_1(self):
        
        for k, P_da in enumerate(self.DA_params):
            
            stage_params = self.RT_params[k]

            threshold = int(self.cut_mode.split('-')[1])

            with mp.Pool() as pool:

                t_last        = self.STAGE - 1
                prev_solution = self.forward_solutions[k][t_last][-1]
                deltas_last   = stage_params[t_last]
                
                last_args = [
                    (j, prev_solution, P_da, deltas_last[j], threshold)
                    for j in range(self.N_t)
                ]
                
                last_results = pool.starmap(process_hyb_last_stage, last_args)

                v_sum = 0.0
                pi_mean = [0, [0], 0, 0, 0, 0]
                
                for pi, L in last_results:
                    
                    v_sum += L
                    pi_mean[0]    += pi[0] / self.N_t
                    pi_mean[1][0] += pi[1][0] / self.N_t
                    pi_mean[2]    += pi[2] / self.N_t
                    pi_mean[3]    += pi[3] / self.N_t
                    pi_mean[4]    += pi[4] / self.N_t
                    pi_mean[5]    += pi[5] / self.N_t

                v = v_sum/self.N_t
                
                cut_coeff = [v] + pi_mean
                
                self.psi[k][t_last].append(cut_coeff)

                for t in range(self.STAGE-2, -1, -1):
                    
                    prev = self.forward_solutions[k][t][-1]
                    psi_next = self.psi[k][t+1]
                    
                    args_inner = [
                        (j, t, prev, psi_next, P_da, stage_params[t][j], threshold)
                        for j in range(self.N_t)
                    ]
                    
                    results_inner = pool.starmap(process_hyb_inner_stage, args_inner)

                    v_sum = 0.0
                    pi_mean = [0, [0]*(self.STAGE-t), 0, 0, 0, 0]
                    
                    for pi, L in results_inner:
                        
                        v_sum += L
                        pi_mean[0] += pi[0] / self.N_t
                        
                        for i in range(self.STAGE-t):
                            pi_mean[1][i] += pi[1][i] / self.N_t
                            
                        pi_mean[2] += pi[2] / self.N_t
                        pi_mean[3] += pi[3] / self.N_t
                        pi_mean[4] += pi[4] / self.N_t
                        pi_mean[5] += pi[5] / self.N_t

                    v = v_sum/self.N_t
                    cut_coeff = [v] + pi_mean
                    self.psi[k][t].append(cut_coeff)
                    
        prev_da = self.forward_solutions_da[-1]

        for k, P_da in enumerate(self.DA_params):
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(
                prev_da, self.psi[k][0], P_da
                )
            
            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            pi = [psi_sub[1]] + [psi_sub[2]]
                
            lag = fw_rt_init_Lagrangian([psi_sub[i] for i in [1,2]], self.psi[k][0], P_da)
            v = lag.get_objective_value()

            cut_coeff = [v] + [pi[0]] + [pi[1]]
            self.psi_da[k].append(cut_coeff)

        fw_da_for_UB = fw_da_multicut(self.psi_da)
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))


    def stopping_criterion(self, tol = 1e-5):
        
        self.gap = (self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]
        
        self.running_time = time.time() - self.start_time
        
        #print(f"run_time = {self.running_time}, criteria = {Total_time}")
        
        if (
            self.running_time > Total_time
        ):
            return True

        return False

    def run_sddip(self):
        
        Timeline = 0
        
        while Timeline <= 0:
        
            final_pass = False

            self.start_time = time.time()
            self.running_time = 0
        
            while True:
                
                if not final_pass and self.stopping_criterion():
                    final_pass = True
                    
                    print("\n>>> Stopping criterion met. Performing final pass with M scenarios...")
                
                elif final_pass:
                    
                    Timeline += 1
                    
                    break
                    
                self.iteration += 1
                #print(f"\n=== Iteration {self.iteration} ===")

                if final_pass:
                    scenarios = self.sample_scenarios_for_stopping()
                    self.scenarios_for_eval = self.scenarios_for_eval
                        
                    if self.cut_mode in ['B', 'SB', 'L-sub']:
                        self.forward_pass_for_stopping(scenarios)
                        self.forward_pass_for_eval_roll(self.scenarios_for_eval)
                        
                    else:
                        print("Not a proposed cut")
                        break
                else:
                    scenarios = self.sample_scenarios()

                    if self.cut_mode in ['B', 'SB', 'L-sub']:
                        self.forward_pass(scenarios)
                        
                    else:
                        print("Not a proposed cut")
                        break

                #print(f"  LB for iter = {self.iteration} updated to: {self.LB[self.iteration]:.4f}")
                
                if self.parallel_mode == 0:
                
                    if self.cut_mode in ['B', 'SB']:
                        self.backward_pass()
                        
                    elif self.cut_mode in ['L-sub']:
                        
                        if self.iteration <= 4:
                            self.backward_pass()
                            
                        else:
                            
                            self.backward_pass_Lagrangian()

                    else:
                        print("Not a proposed cut")
                        break

                elif self.parallel_mode == 1:
                
                    if self.cut_mode in ['B', 'SB']:
                        self.backward_pass_1()
                        
                    elif self.cut_mode in ['L-sub']:
                        
                        if self.iteration <= 4:
                            self.backward_pass_1()
                            
                        else:

                            self.backward_pass_Lagrangian_1()
                            
                    else:
                        print("Not a proposed cut")
                        break
                
                if self.iteration >= 5:
                    
                    self.cut_selection()
                
                #print(f"cut_num = {(len(self.psi[0][2]))}")
                
                #print(f"  UB for iter = {self.iteration} updated to: {self.UB[self.iteration]:.4f}")

            print(f"\nPSDDiPModel for price setting = {price_setting}, running_time = {Total_time*Timeline}")
            print(f"used cuts = {(len(self.psi[0][2]))}")
            print(f"SDDiP complete. for T = {self.STAGE}, k = {self.K}, cut mode = {self.cut_mode}")
            print(f"Final LB = {self.LB[self.iteration]:.4f}, UB = {self.UB[self.iteration]:.4f}, gap = {(self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]:.4f}")
            print(f"Evaluation : {self.eval}, iteration : {self.iteration}")


# Rolling Horizon Model

class RollingHorizonModel:
    
    def __init__(
        self, 
        STAGE = T, 
        DA_params_eval = P_da_test, 
        Scenario_tree_eval = Scenario_tree_test,
        scenario_eval = [], 
        DA_praram_exp = Reduced_P_da[0],
        scenario_exp = Reduced_scenario_trees[0],
        evaluation_num = 10,
        ):
        
        self.STAGE = STAGE
        
        self.DA_params_eval = DA_params_eval
        self.ID_praram_eval = Scenario_tree_eval
        
        self.scenario_eval = scenario_eval
        
        self.exp_P_da = DA_praram_exp[0]
        self.scenario_exp = scenario_exp
        self.exp_P_rt = self._expectation_P_rt()
        
        self.K_eval = len(self.DA_params_eval)
        self.N = evaluation_num
        
        self.M = self.K_eval*evaluation_num
        
        self.eval = 0

        self.b_da = [0 for _ in range(self.STAGE)]
        self.q_da = [0 for _ in range(self.STAGE)]
        self.f_DA = [0 for _ in range(self.STAGE)]
        
        self.f_P = [0 for _ in range(self.STAGE)]
        self.f_E = [0 for _ in range(self.STAGE)]
        self.f_Im = [0 for _ in range(self.STAGE)]

    def _expectation_P_rt(self):
        
        exp_P_rt_list = []
        
        scenario_tree_rt = self.scenario_exp[0]
        
        for t in range(self.STAGE):
            
            branches_t = scenario_tree_rt[t]
            
            exp_P_rt = 0
            
            for b in branches_t:   
                
                exp_P_rt += b[1]/len(branches_t)
            
            exp_P_rt_list.append(exp_P_rt)
                            
        return exp_P_rt_list 
    
    def exp_P_rt_given_P_da(self, n):
    
        exp_P_rt_list = []
        
        scenario_tree_rt = self.ID_praram_eval[n]
        
        for t in range(self.STAGE):
            
            branches_t = scenario_tree_rt[t]
            
            exp_P_rt = 0
            
            for b in branches_t:   
                
                exp_P_rt += b[1]/len(branches_t)
            
            exp_P_rt_list.append(exp_P_rt)
                            
        return exp_P_rt_list

    def rolling_horizon(self):
        
        da_subp = rolling_da(self.exp_P_da, self.exp_P_rt)
        
        da_state = da_subp.get_state_solutions()
                 
        self.b_da = da_state[0]
        self.q_da = da_state[1]    
                            
        f = []
        
        for n, scenarios_n in enumerate(self.scenario_eval):
                        
            P_da = self.DA_params_eval[n]  
                        
            exp_P_rt = self.exp_P_rt_given_P_da(n)
                        
            rt_init_subp = rolling_rt_init(da_state, P_da, exp_P_rt)
            rt_init_state = rt_init_subp.get_state_solutions()       
            
            f_DA_list = rt_init_subp.get_DA_profit()
            
            for i in range(self.STAGE):
                self.f_DA[i] += f_DA_list[i]/self.K_eval
            
            fcn_value = rt_init_subp.get_settlement_fcn_value()
            
            for scenario in scenarios_n:
                
                state = rt_init_state
                
                f_scenario = fcn_value
                
                for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                                        
                    rt_subp = rolling_rt(t, state, P_da, exp_P_rt, scenario[t])
                    
                    self.f_P[t] += rt_subp.get_P_profit()/(self.K_eval*self.N)
                    self.f_E[t] += rt_subp.get_E_profit()/(self.K_eval*self.N)
                    self.f_Im[t] += rt_subp.get_Im_profit()/(self.K_eval*self.N)
                    
                    state = rt_subp.get_state_solutions()
                    
                    f_scenario += rt_subp.get_settlement_fcn_value()
                
                ## t = T-1
                
                rt_last_subp = rolling_rt_last(state, P_da, scenario[self.STAGE-1])

                self.f_P[self.STAGE-1] += rt_subp.get_P_profit()/(self.K_eval*self.N)
                self.f_E[self.STAGE-1] += rt_subp.get_E_profit()/(self.K_eval*self.N)
                self.f_Im[self.STAGE-1] += rt_subp.get_Im_profit()/(self.K_eval*self.N)

                f_scenario += rt_last_subp.get_settlement_fcn_value()
                            
                f.append(f_scenario)
            
        mu_hat = np.mean(f)
        
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        self.eval = mu_hat 

        print(f"\nRolling Horizon for price setting = {price_setting}")
        print(f"Rolling Horizon complete. for T = {self.STAGE}")
        print(f"Evaluation : {self.eval}, scenario_num : {len(self.scenario_eval)}")



if __name__ == "__main__":

    # Full length of T = 24
    
    l_1 = 0
    l_2 = 0
    l_3 = 2
    parallel = 1
    
    ## (l, k, sample_num) : (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 10), (9, 15), (10, 500) 
    
    sample_num_1 = int(1000/K_list[l_1])
    sample_num_2 = int(1000/K_list[l_2])
    sample_num_3 = int(1000/K_list[l_3])    
    
    evaluation_num = 10

    test_mode = True

    if test_mode:
    
        P_da = P_da_test
        Scenario_tree = Scenario_tree_test
        
    else:
        
        P_da = P_da_evaluate
        Scenario_tree = Scenario_tree_evaluate
            
    def sample_scenarios_for_evaluation(N):
        
        scenarios = []
        
        for k in range(K_eval):
            scenarios_k = []
            
            for _ in range(N):
                scenario = [random.choice(stage_params)
                            for stage_params in Scenario_tree[k]]
                
                scenarios_k.append(scenario)
                
            scenarios.append(scenarios_k)
            
        return scenarios

    scenarios_for_eval = sample_scenarios_for_evaluation(evaluation_num)

    psddip_multi_1 = PSDDiPModel(
            STAGE = T,
            DA_params=P_da,
            ID_params=Scenario_tree,
            scenarios_for_eval=scenarios_for_eval,
            DA_params_reduced=Reduced_P_da[l_1],
            DA_params_prob=Reduced_Probs[l_1],
            RT_params_reduced=Reduced_scenario_trees[l_1],
            scenario_exp = Reduced_scenario_trees[0],
            sample_num=sample_num_1,
            evaluation_num=evaluation_num,
            alpha = 0.95,
            cut_mode='SB',
            tol=0.00000001,
            parallel_mode=parallel,
        )
    
    psddip_multi_2 = PSDDiPModel(
            STAGE = T,
            DA_params=P_da,
            ID_params=Scenario_tree,
            scenarios_for_eval=scenarios_for_eval,
            DA_params_reduced=Reduced_P_da[l_2],
            DA_params_prob=Reduced_Probs[l_2],
            RT_params_reduced=Reduced_scenario_trees[l_2],
            scenario_exp = Reduced_scenario_trees[0],
            sample_num=sample_num_1,
            evaluation_num=evaluation_num,
            alpha = 0.95,
            cut_mode='L-sub',
            tol=0.00000001,
            parallel_mode=parallel,
        )

    psddip_multi_3 = PSDDiPModel(
            STAGE = T,
            DA_params=P_da,
            ID_params=Scenario_tree,
            scenarios_for_eval=scenarios_for_eval,
            DA_params_reduced=Reduced_P_da[l_3],
            DA_params_prob=Reduced_Probs[l_3],
            RT_params_reduced=Reduced_scenario_trees[l_3],
            scenario_exp = Reduced_scenario_trees[0],
            sample_num=sample_num_1,
            evaluation_num=evaluation_num,
            alpha = 0.95,
            cut_mode='L-sub',
            tol=0.00000001,
            parallel_mode=parallel,
        )

    rolling_1 = RollingHorizonModel(
        STAGE = T,
        DA_params_eval=P_da,
        Scenario_tree_eval=Scenario_tree,
        scenario_eval=scenarios_for_eval,
        DA_praram_exp=Reduced_P_da[0],
        scenario_exp=Reduced_scenario_trees[0],
        evaluation_num=evaluation_num,
    )


    psddip_multi_1.run_sddip()
    psddip_multi_2.run_sddip()
    psddip_multi_3.run_sddip()
    
    rolling_1.rolling_horizon()
    
    
    b_da_1 = psddip_multi_1.b_da
    b_da_2 = psddip_multi_2.b_da
    b_da_3 = psddip_multi_3.b_da
    b_da_4 = rolling_1.b_da

    q_da_1 = psddip_multi_1.q_da
    q_da_2 = psddip_multi_2.q_da
    q_da_3 = psddip_multi_3.q_da
    q_da_4 = rolling_1.q_da

    f_DA_1 = psddip_multi_1.f_DA
    f_DA_2 = psddip_multi_2.f_DA
    f_DA_3 = psddip_multi_3.f_DA
    f_DA_4 = rolling_1.f_DA
    
    f_P_1 = psddip_multi_1.f_P
    f_P_2 = psddip_multi_2.f_P
    f_P_3 = psddip_multi_3.f_P
    f_P_4 = rolling_1.f_P
    
    f_E_1 = psddip_multi_1.f_E
    f_E_2 = psddip_multi_2.f_E
    f_E_3 = psddip_multi_3.f_E
    f_E_4 = rolling_1.f_E
   
    f_Im_1 = psddip_multi_1.f_Im
    f_Im_2 = psddip_multi_2.f_Im
    f_Im_3 = psddip_multi_3.f_Im
    f_Im_4 = rolling_1.f_Im

    def save_arrays_DA():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sol_dir = os.path.join(current_dir, f"DA_solutions_{price_setting}")

        os.makedirs(sol_dir, exist_ok=True)

        data_dict = {
            "b_da_1": b_da_1,
            "q_da_1": q_da_1,
            "f_P_1": f_DA_1,
            "b_da_2": b_da_2,
            "q_da_2": q_da_2,
            "f_P_2": f_DA_2,
            "b_da_3": b_da_3,
            "q_da_3": q_da_3,
            "f_P_3": f_DA_3,
            "b_da_4": b_da_4,
            "q_da_4": q_da_4,
            "f_P_4": f_DA_4,
        }

        for name, arr in data_dict.items():
            df = pd.DataFrame(arr)
            df.to_csv(os.path.join(sol_dir, f"{name}.csv"), index=False)

        print("✓ All DA-solution CSV files saved inside DA_solutions/ folder.")

    def save_arrays_ID():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sol_dir = os.path.join(current_dir, f"ID_solutions_{price_setting}")

        os.makedirs(sol_dir, exist_ok=True)

        data_dict = {
            "f_P_1": f_P_1,
            "f_E_1": f_E_1,
            "f_Im_1": f_Im_1,
            "f_P_2": f_P_2,
            "f_E_2": f_E_2,
            "f_Im_2": f_Im_2,
            "f_P_3": f_P_3,
            "f_E_3": f_E_3,
            "f_Im_3": f_Im_3,
            "f_P_4": f_P_4,
            "f_E_4": f_E_4,
            "f_Im_4": f_Im_4,
        }

        for name, arr in data_dict.items():
            df = pd.DataFrame(arr)  # 3 x 24
            df.to_csv(os.path.join(sol_dir, f"{name}.csv"), index=False)

        print("✓ All ID-solution CSV files saved inside ID_solutions/ folder.")
    
    
    
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
    

    # Run everything
    save_arrays_DA()
    save_arrays_ID()
    notify_done_via_plot()
        

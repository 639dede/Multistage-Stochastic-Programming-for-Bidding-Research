import os
import sys
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

from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.opt import TerminationCondition, SolverStatus

warnings.filterwarnings("ignore")
logging.getLogger("pyomo.core").setLevel(logging.ERROR)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

SOLVER.options['TimeLimit'] = 1000
#SOLVER.options['MIPGap'] = 1e-4

assert SOLVER.available(), f"Solver {solver} is available."


# Load Energy Forecast list

E_0_path = './Stochastic_Approach/Scenarios/Energy_forecast/E_0.csv'
np.set_printoptions(suppress=True, precision=4)
E_0 = np.loadtxt(E_0_path, delimiter=',')


# Load Price and Scenario csv files

def load_clustered_P_da(directory_path):
    Reduced_P_da = []
    files = sorted(os.listdir(directory_path), key=lambda x: int(x.split('K')[-1].split('.csv')[0]))

    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(directory_path, file)
            data = np.loadtxt(file_path, delimiter=',')
            if data.ndim == 1:
                data = data.reshape(1, -1)  # Fix for K = 1
            Reduced_P_da.append(data.tolist())  # Convert to list

    return Reduced_P_da

cluster_dir = './Stochastic_Approach/Scenarios/Clustered_P_da'
Reduced_P_da = load_clustered_P_da(cluster_dir)

K_list = [1, 5, 10, 25, 60, 100, 1000]
    
T = 24
hours = np.arange(T)

"""
for i, P_da_list in enumerate(Reduced_P_da):
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile in P_da_list:
        ax.plot(hours, profile, color='blue', alpha=0.6)
    ax.set_title(f"K = {K_list[i]}: Clustered Day-Ahead Price Profiles")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Price")
    ax.grid(True)
    plt.tight_layout()
    plt.show()
"""

def load_scenario_trees(base_dir):
    Reduced_scenario_trees = []
    k_folders = sorted(os.listdir(base_dir), key=lambda x: int(x[1:]))  # Assumes folders named K1, K5, ...

    for k_folder in k_folders:
        k_path = os.path.join(base_dir, k_folder)
        scenario_files = sorted(os.listdir(k_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

        scenario_trees = []
        for file in scenario_files:
            file_path = os.path.join(k_path, file)
            data = np.loadtxt(file_path, delimiter=',')

            tree = [[] for _ in range(24)]
            for row in data:
                t = int(row[0])
                branch = row[2:].tolist()  # Skip t and b
                tree[t].append(branch)
            scenario_trees.append(tree)

        Reduced_scenario_trees.append(scenario_trees)

    return Reduced_scenario_trees

clustered_tree_dir = './Stochastic_Approach/Scenarios/Clustered_scenario_trees'
Reduced_scenario_trees = load_scenario_trees(clustered_tree_dir)

"""
fig, axes = plt.subplots(len(K_list), 1, figsize=(12, 2.5 * len(K_list)), sharex=True, constrained_layout=True)
hours = np.arange(24)

for i, (k, (scenario_trees, P_da_list)) in enumerate(zip(K_list, zip(Reduced_scenario_trees, Reduced_P_da))):
    ax = axes[i] if len(K_list) > 1 else axes

    for P_da, scenario in zip(P_da_list, scenario_trees):
        ax.plot(hours, P_da, color='blue', linewidth=2.0, alpha=0.8)  # Day-ahead

        N_t = len(scenario[0])
        for b in range(N_t):
            P_rt_values = [scenario[t][b][1] for t in range(24)]
            ax.plot(hours, P_rt_values, color='black', linewidth=0.6, alpha=0.3)  # Real-time branches

    ax.set_title(f"K = {k}: P_da (blue) & P_rt branches (black)", fontsize=10)
    ax.set_ylabel("Price")
    ax.set_ylim(-120, 200)
    ax.grid(True)

axes[-1].set_xlabel("Hour")
"""

plt.show()


# SDDiP Model Parameters

T = 4

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
    Total_time = 18000
    
    E_0_partial = E_0
    
    Reduced_P_da = Reduced_P_da
    Reduced_scenario_trees = Reduced_scenario_trees
    
elif T in [7, 10]:
    Total_time = 1500
    start = 9
    E_0_partial = E_0[start:start+T]
    Reduced_P_da = [np.array(cluster)[:, start:start+T] for cluster in Reduced_P_da]
    Reduced_scenario_trees = [
        [scenario_tree[start:start+T] for scenario_tree in trees_by_k]
        for trees_by_k in Reduced_scenario_trees
    ]

elif T in [2, 4]:
    Total_time = 30
    start = 9
    E_0_partial = E_0[start:start+T]
    Reduced_P_da = [np.array(cluster)[:, start:start+T] for cluster in Reduced_P_da]
    Reduced_scenario_trees = [
        [scenario_tree[start:start+T] for scenario_tree in trees_by_k]
        for trees_by_k in Reduced_scenario_trees
    ]

P_da_evaluate = Reduced_P_da[-1]
Sceanrio_tree_evaluate = Reduced_scenario_trees[-1]

"""
fig, axes = plt.subplots(len(K_list), 1, figsize=(12, 2.5 * len(K_list)), sharex=True, constrained_layout=True)
hours = np.arange(T)

for i, (k, (scenario_trees, P_da_list)) in enumerate(zip(K_list, zip(Reduced_scenario_trees, Reduced_P_da))):
    ax = axes[i] if len(K_list) > 1 else axes

    for P_da, scenario in zip(P_da_list, scenario_trees):
        ax.plot(hours, P_da, color='blue', linewidth=2.0, alpha=0.8)  # Day-ahead

        N_t = len(scenario[0])
        for b in range(N_t):
            P_rt_values = [scenario[t][b][1] for t in range(T)]
            ax.plot(hours, P_rt_values, color='black', linewidth=0.6, alpha=0.3)  # Real-time branches

    ax.set_title(f"K = {k}: P_da (blue) & P_rt branches (black)", fontsize=10)
    ax.set_ylabel("Price")
    ax.set_ylim(-120, 200)
    ax.grid(True)

axes[-1].set_xlabel("Hour")

plt.show()

"""

dual_tolerance = 1e-7
tol = 1e-5
Node_num = 1
Lag_iter_UB = 500

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
        
        #model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
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


# PSDDiP Algorithm
                                 
class PSDDiPModel:
        
    def __init__(
        self, 
        STAGE = T, 
        DA_params = [],
        RT_params = [],
        DA_params_reduced = Reduced_P_da[0], 
        RT_params_reduced = Reduced_scenario_trees[0],
        sample_num = 1000,
        evaluation_num = 10,
        alpha = 0.95, 
        cut_mode = 'B',
        tol = 0.001,
        ):

        ## STAGE = -1, 0, ..., STAGE - 1
        ## forward_scenarios(last iteration) = M
        ## baackward_scenarios = N_t

        self.STAGE = STAGE
        
        self.DA_params_evaluation = DA_params
        self.RT_params_evaluation = RT_params
        
        self.DA_params = DA_params_reduced
        self.RT_params = RT_params_reduced
        
        self.M = sample_num
        self.N = evaluation_num
        
        self.alpha = alpha
        self.cut_mode = cut_mode
        self.tol = tol
                
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
        
        self.psi_da = [] ## t = -1 -> Day-Ahead Stage
        
        self.psi = [  ## t = {0 -> -1}, ..., {T - 1 -> T - 2}
                [
                    [] for _ in range(self.STAGE)
                ] for _ in range(self.K)
            ] 
        
        self._initialize_psi()
        
    def _initialize_psi(self):
        
        self.psi_da = [
                    [3*3600000*T,
                     [0 for _ in range(self.STAGE)],
                     [0 for _ in range(self.STAGE)]
                    ]
            ]
        
        for k in range(self.K):
            for t in range(self.STAGE): ## psi(-1), ..., psi(T - 2)
                self.psi[k][t].append([
                    3*3600000*(self.STAGE - t), 
                    0, 
                    [0 for _ in range(self.STAGE - t)], 
                    0, 0, 0, 0
                    ])

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
    
    def sample_scenarios_for_evaluation(self):
        
        scenarios = []
        
        for k in range(self.K_eval):
            scenarios_k = []
            
            for _ in range(self.N):
                scenario = [random.choice(stage_params)
                            for stage_params in self.RT_params_evaluation[k]]
                
                scenarios_k.append(scenario)
                
            scenarios.append(scenarios_k)
            
        return scenarios

    def find_cluster_index_for_evaluation(self, n):
        
        P_da = 10
        k = 10
        
        return k
            
    def forward_pass(self, scenarios):
        
        fw_da_subp = fw_da(self.psi_da)
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
        
        fw_da_subp = fw_da(self.psi_da)
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
        
        fw_da_subp = fw_da(self.psi_da)
        fw_da_state = fw_da_subp.get_state_solutions()
                
        f = []
        
        for n, scenarios_n in enumerate(scenarios):
            
            k = self.find_cluster_index_for_evaluation(n)
            
            P_da = self.DA_params[k]  ## <--- bitch
            
            fw_rt_init_subp = fw_rt_init(fw_da_state, self.psi[k][0], P_da)
            fw_rt_init_state = fw_rt_init_subp.get_state_solutions()
                        
            for scenario in scenarios_n:
                
                state = fw_rt_init_state
                
                f_scenario = fw_rt_init_subp.get_settlement_fcn_value()
                
                for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                    
                    fw_rt_subp = fw_rt(t, state, self.psi[k][t+1], P_da, scenario[t])
                    
                    state = fw_rt_subp.get_state_solutions()
                    
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


    def inner_product(self, t, pi, sol):
        
        return sum(pi[i]*sol[i] for i in [0, 2, 3, 4, 5]) + sum(pi[1][j]*sol[1][j] for j in range(self.STAGE - t))

    def backward_pass(self):
        
        BL = ['B']
        SBL = ['SB', 'L-sub', 'L-lev', 'hyb']
        
        for k, P_da in enumerate(self.DA_params):
            
            stage_params = self.RT_params[k]
            
            ## t = {T-1 -> T-2}
            
            v_sum = 0 
            pi_mean = [0, [0], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[k][self.STAGE - 1][0]
            
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
                
                elif self.cut_mode in SBL:
                    
                    fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian([psi_sub[i] for i in range(1, 7)], P_da, delta)

                    v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
                
            if self.cut_mode in BL:   
                    
                v = v_sum/self.N_t - self.inner_product(self.STAGE - 1, pi_mean, prev_solution)
            
            elif self.cut_mode in SBL:
                
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
                
                prev_solution = self.forward_solutions[k][t][0]
                
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
                        
                    elif self.cut_mode in SBL:
                        
                        fw_rt_Lagrangian_subp = fw_rt_Lagrangian(t, [psi_sub[i] for i in range(1, 7)], self.psi[k][t+1], P_da, delta)

                        v_sum += fw_rt_Lagrangian_subp.get_objective_value()
                
                if self.cut_mode in BL:
                    
                    v = v_sum/self.N_t - self.inner_product(t, pi_mean, prev_solution)
            
                if self.cut_mode in SBL:
                    
                    v = v_sum/self.N_t
            
                cut_coeff = []
                
                cut_coeff.append(v)
                
                for i in range(6):
                    cut_coeff.append(pi_mean[i])
            
                self.psi[k][t].append(cut_coeff)
                #print(f"stage {t - 1} _cut_coeff = {self.psi[t]}")
                     
        v_sum = 0
        pi_mean = [[0 for _ in range(self.STAGE)], [0 for _ in range(self.STAGE)]]
        
        prev_solution = self.forward_solutions_da[0] 
        
        for k, P_da in enumerate(self.DA_params):
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(prev_solution, self.psi[k][0], P_da)

            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            for i in range(self.STAGE):
                pi_mean[0][i] += psi_sub[1][i]/self.K
                pi_mean[1][i] += psi_sub[2][i]/self.K
            
            if self.cut_mode in BL:
                
                v_sum += psi_sub[0]
                
            elif self.cut_mode in SBL:
                
                fw_rt_init_Lagrangian_subp = fw_rt_init_Lagrangian(
                    [psi_sub[i] for i in [1, 2]], 
                    self.psi[k][0], 
                    P_da
                    )

                v_sum += fw_rt_init_Lagrangian_subp.get_objective_value()
            
        if self.cut_mode in BL:
            
            v = v_sum/self.K - self.inner_product_da(pi_mean, prev_solution)
            
        elif self.cut_mode in SBL:
            
            v = v_sum/self.K
        
        cut_coeff = []
        
        cut_coeff.append(v)
        
        for i in [0, 1]:
            cut_coeff.append(pi_mean[i])
        
        self.psi_da.append(cut_coeff)
        
        self.forward_solutions_da = []
        
        self.forward_solutions = [  
                [
                    [] for _ in range(self.STAGE)
                ] for _ in range(self.K)
            ]
        
        fw_da_for_UB = fw_da(self.psi_da)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value())) 

    def backward_pass_Lagrangian(self):
        
        for k, P_da in enumerate(self.DA_params):
                
            stage_params = self.RT_params[k]    
                
            ## t = {T-1 -> T-2}
            
            v_sum = 0 
            pi_mean = [0, [0], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[k][self.STAGE - 1][0]
                    
            for j in range(self.N_t): 
                
                delta = stage_params[self.STAGE - 1][j]      
                
                fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, P_da, delta)
                
                pi_LP = fw_rt_last_LP_relax_subp.get_cut_coefficients()[1:]
                
                pi = pi_LP
                pi_min = pi_LP
                
                reg = 0.00001
                G = 10000000
                lev = 0.9
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
                    
                    if Lag_iter >= 3:
                        
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
                
                prev_solution = self.forward_solutions[k][t][0]
                            
                for j in range(self.N_t):
                                    
                    delta = stage_params[t][j]
                    
                    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[k][t+1], P_da, delta)

                    pi_LP = fw_rt_LP_relax_subp.get_cut_coefficients()[1:]
                    
                    pi = pi_LP
                    pi_min = pi_LP
                                    
                    lev = 0.9
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
                        
                        if Lag_iter >= 3:
                            
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

        v_sum = 0
        pi_mean = [[0 for _ in range(self.STAGE)], [0 for _ in range(self.STAGE)]]
        
        prev_solution = self.forward_solutions_da[0] 
        
        for k, P_da in enumerate(self.DA_params):
            
            fw_rt_init_LP_relax_subp = fw_rt_init_LP_relax(prev_solution, self.psi[k][0], P_da)

            psi_sub = fw_rt_init_LP_relax_subp.get_cut_coefficients()
            
            for i in range(self.STAGE):
                pi_mean[0][i] += psi_sub[1][i]/self.K
                pi_mean[1][i] += psi_sub[2][i]/self.K
                
            fw_rt_init_Lagrangian_subp = fw_rt_init_Lagrangian(
                [psi_sub[i] for i in [1, 2]], 
                self.psi[k][0], 
                P_da
                )

            v_sum += fw_rt_init_Lagrangian_subp.get_objective_value()
                        
        v = v_sum/self.K
        
        cut_coeff = []
        
        cut_coeff.append(v)
        
        for i in [0, 1]:
            cut_coeff.append(pi_mean[i])
        
        self.psi_da.append(cut_coeff)

        self.forward_solutions_da = []
        
        self.forward_solutions = [  
                [
                    [] for _ in range(self.STAGE)
                ] for _ in range(self.K)
            ]
        
        fw_da_for_UB = fw_da(self.psi_da)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   

    def backward_pass_hybrid(self):
                
        ## t = {T-1 -> T-2}
        
        v_sum = 0 
        pi_mean = [0, [0], 0, 0, 0, 0]
        
        Lag_elapsed_time = 0
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
                
        for j in range(self.N_t): 
            
            delta = stage_params[T - 1][j]      
            
            P_rt = delta[1]
            P_da = P_da_partial[T - 1]
            
            fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, delta, 0)
            
            psi_sub = fw_rt_last_LP_relax_subp.get_cut_coefficients()
            
            if P_rt <= P_da:
                
                pi_mean[0] += psi_sub[1]/self.N_t
                pi_mean[1][0] += psi_sub[2][0]/self.N_t
                pi_mean[2] += psi_sub[3]/self.N_t
                pi_mean[3] += psi_sub[4]/self.N_t
                pi_mean[4] += psi_sub[5]/self.N_t
                pi_mean[5] += psi_sub[6]/self.N_t
                                    
                fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian([psi_sub[i] for i in range(1, 7)], delta)

                v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
                
            else:
                
                pi_LP = psi_sub[1:]
                
                pi = pi_LP
                pi_min = pi_LP
                
                reg = 0.00001
                G = 10000000
                lev = 0.9
                gap = 1       
                lamb = 0
                k = [0, [0], 0, 0, 0, 0]
                l = 10000000
                            
                dual_subp_sub_last = dual_approx_sub(self.STAGE - 1, reg, pi_LP)
                #dual_subp_lev_last = dual_approx_lev(self.STAGE - 1, reg, pi_LP, l)
                
                fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, delta)
                
                L = fw_rt_last_Lag_subp.get_objective_value()
                z = fw_rt_last_Lag_subp.get_auxiliary_value()
                
                Lag_iter = 1
                            
                pi_minobj = 10000000
                
                while gap >= dual_tolerance:            
                    
                    if Lag_iter >= 3:
                    
                        dual_subp_sub_last.reg = 0
                    
                    lamb = L + self.inner_product(self.STAGE - 1, pi, z)
                    
                    for l in [0, 2, 3, 4, 5]:
                        
                        k[l] = prev_solution[l] - z[l]
                    
                    for l in [1]:
                        
                        k[l][0] = prev_solution[l][0] - z[l][0]
                    
                    dual_coeff = [lamb, k]
                                    
                    if self.cut_mode == 'hyb':
                        
                        dual_subp_sub_last.add_plane(dual_coeff)
                        pi = dual_subp_sub_last.get_solution_value()
                        obj = dual_subp_sub_last.get_objective_value()
                        
                    elif self.cut_mode == 'L-lev':
                                                                
                        dual_subp_sub_last.add_plane(dual_coeff)
                        #dual_subp_lev_last.add_plane(dual_coeff)
                                            
                        f_lb = dual_subp_sub_last.get_objective_value()
                        f_ub = pi_minobj
                                            
                        l = f_lb + lev*(f_ub - f_lb)
                                            
                        #dual_subp_lev_last.level = l
                        #dual_subp_lev_last.pi = pi_min
                                            
                        #pi = dual_subp_lev_last.get_solution_value()
                        
                        obj = f_lb
                    
                    fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, delta)
                    
                    start_time = time.time()
                    
                    L = fw_rt_last_Lag_subp.get_objective_value()
                    z = fw_rt_last_Lag_subp.get_auxiliary_value()
                              
                    Lag_elapsed_time += time.time() - start_time
                                    
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
        
        self.psi[self.STAGE - 1].append(cut_coeff)
                
        #print(f"last_stage_cut_coeff = {self.psi[T-1]}")
        
        ## t = {T-2 -> T-3}, ..., {0 -> -1}
        
        for t in range(self.STAGE - 2, -1, -1): 
            
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
                        
            for j in range(self.N_t):
                                
                delta = stage_params[t][j]
                
                P_rt = delta[1]
                P_da = P_da_partial[t]
                
                fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[t+1], delta, 0)

                psi_sub = fw_rt_LP_relax_subp.get_cut_coefficients()
                
                if P_rt <= P_da:
                    
                    pi_mean[0] += psi_sub[1]/self.N_t
                
                    for i in range(self.STAGE - t):
                        pi_mean[1][i] += psi_sub[2][i]/self.N_t
                        
                    pi_mean[2] += psi_sub[3]/self.N_t
                    pi_mean[3] += psi_sub[4]/self.N_t
                    pi_mean[4] += psi_sub[5]/self.N_t
                    pi_mean[5] += psi_sub[6]/self.N_t
                        
                    fw_rt_Lagrangian_subp = fw_rt_Lagrangian(t, [psi_sub[i] for i in range(1, 7)], self.psi[t+1], delta)

                    v_sum += fw_rt_Lagrangian_subp.get_objective_value()
                
                else:
                
                    pi_LP = psi_sub[1:]
                    
                    pi = pi_LP
                    pi_min = pi_LP
                                    
                    lev = 0.9
                    G = 10000000
                    reg = 0.00001
                    gap = 1
                    lamb = 0
                    k = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
                    l = 10000000*(self.STAGE - t)
                        
                    dual_subp_sub = dual_approx_sub(t, reg, pi_LP)
                    dual_subp_lev = dual_approx_lev(t, reg, pi_LP, l)
                    
                    fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[t+1], delta)
                    
                    L = fw_rt_Lag_subp.get_objective_value()
                    z = fw_rt_Lag_subp.get_auxiliary_value()    
                                    
                    Lag_iter = 1
                                    
                    pi_minobj = 10000000*(self.STAGE - t)
                    
                    while gap >= dual_tolerance:
                        
                        if Lag_iter >= 3:
                            
                            dual_subp_sub.reg = 0
                        
                        lamb = L + self.inner_product(t, pi, z)
                        
                        for l in [0, 2, 3, 4, 5]:
                            
                            k[l] = prev_solution[l] - z[l]
                            
                        for l in [1]:
                            
                            for i in range(self.STAGE - t):
                                
                                k[l][i] = prev_solution[l][i] - z[l][i]
                                            
                        dual_coeff = [lamb, k]
                                            
                        if self.cut_mode == 'hyb':
                            dual_subp_sub.add_plane(dual_coeff)
                            pi = dual_subp_sub.get_solution_value()
                            obj = dual_subp_sub.get_objective_value()
                        
                        elif self.cut_mode == 'L-lev':
                            
                            dual_subp_sub.add_plane(dual_coeff)
                            dual_subp_lev.add_plane(dual_coeff)
                            
                            f_lb = dual_subp_sub.get_objective_value()
                            f_ub = pi_minobj
                            
                            l = f_lb + lev*(f_ub - f_lb)
                            
                            dual_subp_lev.level = l
                            dual_subp_lev.pi = pi_min

                            pi = dual_subp_lev.get_solution_value()
                            
                            obj = f_lb
                                            
                        fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[t+1], delta)
                        
                        start_time = time.time()
                        
                        L = fw_rt_Lag_subp.get_objective_value()
                        z = fw_rt_Lag_subp.get_auxiliary_value()
                        
                        Lag_elapsed_time += time.time() - start_time
                        
                        pi_obj = L + self.inner_product(t, pi, prev_solution)
                        
                        if pi_obj < pi_minobj:
                        
                            pi_minobj = pi_obj
                            pi_min = pi
                        
                        if pi_obj == -G:
                            Lag_iter += 1
                            continue
                        
                        gap = (pi_obj - obj)/(pi_obj+G)
                                                                                
                        Lag_iter += 1      
                        #print(Lag_iter)                  
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
        
            self.psi[t].append(cut_coeff)
            #print(f"stage {t - 1} _cut_coeff = {self.psi[t]}")

        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0])
        
        self.Lag_elapsed_time_list.append(Lag_elapsed_time)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   

    def stopping_criterion(self, tol = 1e-5):
        
        self.gap = (self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]
        
        self.running_time = time.time() - self.start_time
        
        print(f"run_time = {self.running_time}, criteria = {Total_time}")
        
        if self.iteration <= 3:
            baseline = 1e9
            
        elif self.iteration >= 4:
            baseline = self.UB[self.iteration-3]
        
        if (
            (baseline - self.UB[self.iteration])/self.UB[self.iteration] < self.tol 
            or self.running_time > Total_time
        ):
            return True

        return False

    def run_sddip(self):
        
        final_pass = False

        self.start_time = time.time()
        
        while True:
            
            if not final_pass and self.stopping_criterion():
                final_pass = True
                
                print("\n>>> Stopping criterion met. Performing final pass with M scenarios...")
            
            elif final_pass:
                break

            self.iteration += 1
            print(f"\n=== Iteration {self.iteration} ===")

            if final_pass:
                scenarios = self.sample_scenarios_for_stopping()
                    
                if self.cut_mode in ['B', 'SB', 'L-sub', 'L-lev', 'hyb']:
                    self.forward_pass_for_stopping(scenarios)
                    
                else:
                    print("Not a proposed cut")
                    break
            else:
                scenarios = self.sample_scenarios()

                if self.cut_mode in ['B', 'SB', 'L-sub', 'L-lev', 'hyb']:
                    self.forward_pass(scenarios)
                    
                else:
                    print("Not a proposed cut")
                    break

            print(f"  LB for iter = {self.iteration} updated to: {self.LB[self.iteration]:.4f}")

            if self.cut_mode in ['B', 'SB']:
                self.backward_pass()
                
            elif self.cut_mode in ['L-sub', 'L-lev']:
                
                if self.iteration <= 4:
                    self.backward_pass()
                    
                else:
                    self.backward_pass_Lagrangian()
                    
            elif self.cut_mode == 'hyb':
                
                if self.iteration <= 4:
                    self.backward_pass()
                    
                else:
                    self.backward_pass_hybrid()
            else:
                print("Not a proposed cut")
                break

            print(f"  UB for iter = {self.iteration} updated to: {self.UB[self.iteration]:.4f}")

        print("\nSDDiP complete.")
        print(f"Final LB = {self.LB[self.iteration]:.4f}, UB = {self.UB[self.iteration]:.4f}, gap = {(self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]:.4f}")



if __name__ == "__main__":
    
    #print(len(Reduced_scenario_trees))
    
    psddip_1 = PSDDiPModel(
            STAGE = T,
            DA_params=P_da_evaluate,
            RT_params=Sceanrio_tree_evaluate,
            DA_params_reduced=Reduced_P_da[1],
            RT_params_reduced=Reduced_scenario_trees[1],
            sample_num=50,
            evaluation_num=10,
            alpha = 0.95,
            cut_mode='L-sub',
            tol=0.00000001
        )
    
    psddip_2 = PSDDiPModel(
            STAGE = T,
            DA_params=P_da_evaluate,
            RT_params=Sceanrio_tree_evaluate,
            DA_params_reduced=Reduced_P_da[1],
            RT_params_reduced=Reduced_scenario_trees[1],
            sample_num=50,
            evaluation_num=10,
            alpha = 0.95,
            cut_mode='SB',
            tol=0.00000001
        )
    
    psddip_1.run_sddip()
    psddip_2.run_sddip()
        

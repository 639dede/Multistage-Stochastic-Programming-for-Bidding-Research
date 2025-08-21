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

SOLVER.options['TimeLimit'] = 1000
#SOLVER.options['MIPGap'] = 1e-4

assert SOLVER.available(), f"Solver {solver} is available."


# Data

Total_time = 18000

C = 21022.1
S = C*3
B = C

S_min = 0.1*S
S_max = 0.9*S

P_r = 80
P_max = 200

v = 0.95

gamma_over = P_max
gamma_under = P_max

T = 24


# === 1. Load E_0 ===
E_0_path = './Stochastic_Approach/Scenarios/Energy_forecast/E_0.csv'
np.set_printoptions(suppress=True, precision=4)
E_0 = np.loadtxt(E_0_path, delimiter=',')


# === 2. Load P_da_list ===
P_da_dir = './Stochastic_Approach/Scenarios/P_da_settings'
P_da_list = []
for i in range(3):  # or however many instances you have
    path = os.path.join(P_da_dir, f'P_da_{i}.csv')
    P_da = pd.read_csv(path, header=None).squeeze().tolist()
    P_da_list.append(P_da)


# === 3. Load scenario_trees ===
tree_dir = './Stochastic_Approach/Scenarios/Trees_settings'
scenario_trees = []

for i in range(3):
    tree_path = os.path.join(tree_dir, f'Tree_{i}.csv')
    df = pd.read_csv(tree_path)

    # Group by branch ID and convert each to list of [x0, P_rt, x2]
    grouped = df.groupby('branch')
    tree = [grouped.get_group(idx)[['x0', 'P_rt', 'x2']].values.tolist() for idx in grouped.groups]
    scenario_trees.append(tree)
    

if T <= 10:

    E_0_partial = [E_0[t] for t in range(9, 19)]
    
elif T == 24:
    
    E_0_partial = E_0

E_0_sum = 0
E_0_partial_max = max(E_0_partial)

for t in range(len(E_0_partial)):
    E_0_sum += E_0_partial[t]

    
K = [1.28*E_0_partial[t] + 1.04*B for t in range(T)]

M_gen = [[1.04*K[t], 2*K[t]] for t in range(T)]

dual_tolerance = 1e-5
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

## stage = -1

class fw_da(pyo.ConcreteModel): 
    
    def __init__(self, psi, P_da):
        
        super().__init__()

        self.solved = False
        self.psi = psi
        self.T = T
        
        self.P_da = P_da
        
        self.M_price = [[0, 0] for t in range(self.T)]
        
        self._BigM_setting()
    
    def _BigM_setting(self):
        
        for t in range(self.T):

            self.M_price[t][0] = 400
            self.M_price[t][1] = 400

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
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= E_0_sum
        
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
                + self.psi[l][6]*model.T_E)
        
        def settlement_fcn_rule(model):
            return model.f == sum(
                self.P_da[t]*model.Q_da[t] for t in range(self.T)
            )
            
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
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
        """
        self.M_price[0] = 400
        self.M_price[1] = 400
        """
        
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

class fw_rt_Lagrangian(pyo.ConcreteModel): ## stage = 0, 1, ..., T-1 (Backward - Strengthened Benders' Cut)

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
        """
        self.M_price[0] = 400
        self.M_price[1] = 400
        """
        
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

class fw_rt_last_LP_relax(pyo.ConcreteModel): ## stage = T (Backward)
           
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
     
class fw_rt_last_Lagrangian(pyo.ConcreteModel): ## stage = T (Backward - Strengthened Benders' Cut)
           
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


## Dual Problem

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


# Scenario Tree generation

class ScenarioNode:
    
    def __init__(self, name, stage, param, prob=1.0):
        
        self.name = name
        self.stage = stage
        self.param = param
        self.prob = prob
        self.children = []  

    def add_child(self, child_node):
        
        self.children.append(child_node)

class RecombiningScenarioTree:
    
    def __init__(self, stage_num ,branch, params):
        
        self.stage_num = stage_num
        self.branch = branch
        self.params = params
        
        self.root = ScenarioNode(name = "root", param = None, stage = -1, prob = 1.0)
        self.nodes = [[] for t in range(self.stage_num)]
        
        self._build_recombining_tree()
        
    def _build_recombining_tree(self):
        
        for i in range(self.branch):
            
            param = self.params[0][i]
            child_node = ScenarioNode(
                name = f"Stage0_Node{i}",
                stage = 0,
                param = param,
                prob = 1/self.branch
            )
            self.root.add_child(child_node)
            self.nodes[0].append(child_node)
         
        for t in range(self.stage_num - 1):  
              
            stage_nodes = self.nodes[t]
            
            for i in range(len(stage_nodes)):
                                
                for j in range(self.branch):
                    
                    param = self.params[t + 1][j]
                    child_node = ScenarioNode(
                        name = f"Stage{t+1}_Node{j}",
                        stage = t + 1,
                        param = param,
                        prob = 1/(self.branch)**(t + 1)
                    )
                    stage_nodes[i].add_child(child_node)
                    self.nodes[t+1].append(child_node)
                    
    def print_tree(self):

        def _print_node(node, indent = 0):
            
            print("  " * indent + f"{node.name} (stage={node.stage}, prob={node.prob})" + f"param = {node.param}")
            for child in node.children:
                _print_node(child, indent + 1)

        _print_node(self.root)
                                        
    def scenarios(self):
        scenarios = []
        
        def _dfs(node, path):
            path.append(node)
            
            if node.stage == self.stage_num - 1:
                scenarios.append(path.copy())
            else:
                for child in node.children:
                    _dfs(child, path)
            
            path.pop()

        for child in self.root.children:
            _dfs(child, [])
        
        return scenarios
                    
    def node_scenarios(self):
        
        node_scenarios = {}
            
        for t in range(self.stage_num):
            for node_id, node_obj in enumerate(self.nodes[t]):
                node_scenarios[t, node_id] = set()

        for s_idx, path in enumerate(self.scenarios()):
            for t in range(self.stage_num):
                node_obj = path[t]

                node_id = self.nodes[t].index(node_obj)
                node_scenarios[t, node_id].add(s_idx)
                
        return node_scenarios



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
    
    reg = 0.00001
    G = 10000000
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
        
        if Lag_iter >= 3:
            
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
        
        if pi_obj == -G:
            Lag_iter += 1
            continue
        
        gap = (pi_obj - obj)/(pi_obj+G)
                                                                    
        Lag_iter += 1
        
        if Lag_iter == Lag_iter_UB:
            
            break

    return pi, L

def process_lag_inner_stage(j, t, prev_solution, psi_next, P_da, delta):

    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, psi_next, P_da, delta)
    
    pi = fw_rt_LP_relax_subp.get_cut_coefficients()[1:]
        
    reg = 0.00001
    G = 10000000
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
        
        if Lag_iter >= 3:
            
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
        
        if pi_obj == -G:
            Lag_iter += 1
            continue
        
        gap = (pi_obj - obj)/(pi_obj+G)
                                                                
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

        G = 10000000
        gap = 1       
        lamb = 0
        k_lag = [0, [0], 0, 0, 0, 0]
        l = 10000000
        reg = 0.00001

        dual_subp_sub_last = dual_approx_sub(T - 1, reg, pi)
        
        fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, P_da, delta)
        
        L = fw_rt_last_Lag_subp.get_objective_value()
        z = fw_rt_last_Lag_subp.get_auxiliary_value()
        
        Lag_iter = 1
                    
        pi_minobj = 10000000
        
        while gap >= dual_tolerance:            
            
            if Lag_iter >= 3:
                
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
            
            if pi_obj == -G:
                Lag_iter += 1
                continue
            
            gap = (pi_obj - obj)/(pi_obj+G)
                                                                            
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
        
        G = 10000000
        gap = 1
        lamb = 0
        k_lag = [0, [0 for _ in range(T - t)], 0, 0, 0, 0]
        l = 10000000*(T - t)
        reg = 0.00001
                        
        dual_subp_sub = dual_approx_sub(t, reg, pi)
        
        fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, psi_next, P_da, delta)
        
        L = fw_rt_Lag_subp.get_objective_value()
        z = fw_rt_Lag_subp.get_auxiliary_value()    
                        
        Lag_iter = 1
                        
        pi_minobj = 10000000*(T - t)
        
        while gap >= dual_tolerance:
            
            if Lag_iter >= 3:
                
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
            
            if pi_obj == -G:
                Lag_iter += 1
                continue
            
            gap = (pi_obj - obj)/(pi_obj+G)
                                                                    
            Lag_iter += 1    
            
            if Lag_iter == Lag_iter_UB:
            
                break
        v = L

    return pi, v



# SDDiP Algorithm
                                 
class SDDiPModel:
        
    def __init__(
        self, 
        STAGE = T, 
        P_da = P_da_list[0],
        stage_params = None, 
        forward_scenario_num = 100, 
        cut_mode = 'B',
        alpha = 0.95, 
        time_lim = 3600,
        parallel_mode = False
        ):

        ## STAGE = -1, 0, ..., STAGE - 1
        ## forward_scenarios(last iteration) = M
        ## baackward_scenarios = N_t

        self.STAGE = STAGE
        self.P_da = P_da
        self.stage_params = stage_params
        self.M = forward_scenario_num
        self.cut_mode = cut_mode
        self.alpha = alpha
        self.time_lim = time_lim
        self.parallel_mode = parallel_mode
        
        self.N_t = len(self.stage_params[0])
        
        self.iteration = 0
        
        self.final_pass = False
        
        self.start_time = time.time()
        self.running_time = 0
        
        self.Lag_elapsed_time_list = []
        self.Total_Lag_iter_list = []
        
        self.gap = 1
                
        self.LB = [-np.inf]
        self.UB = [np.inf]

        self.forward_solutions = [  ## T(-1), ..., T(T - 2)
            [] for _ in range(self.STAGE)
        ]
        
        self.psi = [[] for _ in range(self.STAGE)] ## t = {0 -> -1}, ..., {T - 1 -> T - 2}
        
        self.b_da_final = [] 
        self.Q_da_final = []
        self.b_rt_final = []
        self.Q_rt_final = []
        
        self._initialize_psi()
        
    def _initialize_psi(self):
        
        for t in range(self.STAGE): ## psi(-1), ..., psi(T - 2)
            self.psi[t].append([
                3*3600000*(T - t), 
                0, 
                [0 for _ in range(self.STAGE - t)], 
                0, 0, 0, 0
                ])

    def sample_scenarios(self, M):
        
        scenarios = []
        
        for _ in range(M): 
            scenario = []
            for stage in self.stage_params:
                param = random.choice(stage)  
                scenario.append(param)
            scenarios.append(scenario)

        return scenarios
       
    def forward_pass(self, scenarios):
        
        f = []
        
        local_b_da = []
        local_Q_da = []
        local_b_rt = []
        local_Q_rt = []
                
        for k, scenario in enumerate(scenarios):
            
            fw_da_subp = fw_da(self.psi[0], self.P_da)
            fw_da_state = fw_da_subp.get_state_solutions()
            
            b_da = [pyo.value(fw_da_subp.b_da[t]) for t in range(self.STAGE)]
            Q_da = [pyo.value(fw_da_subp.Q_da[t]) for t in range(self.STAGE)]
            local_b_da.append(b_da)
            local_Q_da.append(Q_da)
            
            self.forward_solutions[0].append(fw_da_state)
            
            state = fw_da_state
            
            f_scenario = fw_da_subp.get_settlement_fcn_value()
            
            for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                
                fw_rt_subp = fw_rt(t, state, self.psi[t+1], self.P_da, scenario[t])
                
                state = fw_rt_subp.get_state_solutions()
                
                local_b_rt.append((k,t,pyo.value(fw_rt_subp.b_rt)))
                local_Q_rt.append((k,t,pyo.value(fw_rt_subp.Q_rt)))
                
                self.forward_solutions[t+1].append(state)
                f_scenario += fw_rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            fw_rt_last_subp = fw_rt_last(state, self.P_da, scenario[self.STAGE-1])

            f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
            
            local_b_rt.append((k,self.STAGE-1,pyo.value(fw_rt_last_subp.b_rt)))
            local_Q_rt.append((k,self.STAGE-1,pyo.value(fw_rt_last_subp.Q_rt)))
            
            f.append(f_scenario)
        
        if self.final_pass:

            self.b_da_final = np.array(local_b_da)
            self.Q_da_final = np.array(local_Q_da)

            b_rt_arr = np.zeros((len(scenarios), self.STAGE))
            Q_rt_arr = np.zeros_like(b_rt_arr)
            
            for k,t,val in local_b_rt:
                b_rt_arr[k,t] = val
                
            for k,t,val in local_Q_rt:
                Q_rt_arr[k,t] = val
                
            self.b_rt_final = b_rt_arr
            self.Q_rt_final = Q_rt_arr
        
        mu_hat = np.mean(f)
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        #self.LB.append(mu_hat - z_alpha_half * (sigma_hat / np.sqrt(self.M))) 
        self.LB.append(mu_hat) 
        
        return mu_hat

    def inner_product(self, t, pi, sol):
        
        return sum(pi[i]*sol[i] for i in [0, 2, 3, 4, 5]) + sum(pi[1][j]*sol[1][j] for j in range(self.STAGE - t))


    def backward_pass(self):
                
        ## t = {T-1 -> T-2}
        
        BL = ['B']
        SBL = ['SB', 'L-sub', 'L-lev']
        
        v_sum = 0 
        pi_mean = [0, [0], 0, 0, 0, 0]
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
        
        for j in range(self.N_t): 
            
            delta = self.stage_params[T - 1][j]      
            
            fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(
                prev_solution, 
                self.P_da, 
                delta
                )
            
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
                
                fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian(
                    [psi_sub[i] for i in range(1, 7)], 
                    self.P_da, 
                    delta
                    )

                v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
            
        if self.cut_mode in BL:   
                 
            v = v_sum/self.N_t - self.inner_product(self.STAGE - 1, pi_mean, prev_solution)
        
        elif self.cut_mode in SBL or self.cut_mode.startswith('hyb'):
            
            v = v_sum/self.N_t
            
        cut_coeff = []
        
        cut_coeff.append(v)
        
        for i in range(6):
            cut_coeff.append(pi_mean[i])
        
        self.psi[T-1].append(cut_coeff)
                
        ## t = {T-2 -> T-3}, ..., {0 -> -1}
        for t in range(self.STAGE - 2, -1, -1): 
                
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
            
            for j in range(self.N_t):
                
                delta = self.stage_params[t][j]
                
                fw_rt_LP_relax_subp = fw_rt_LP_relax(
                    t, 
                    prev_solution, 
                    self.psi[t+1], 
                    self.P_da, 
                    delta
                    )
   
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
                    
                    fw_rt_Lagrangian_subp = fw_rt_Lagrangian(
                        t, 
                        [psi_sub[i] for i in range(1, 7)], 
                        self.psi[t+1], 
                        self.P_da,
                        delta
                        )

                    v_sum += fw_rt_Lagrangian_subp.get_objective_value()
            
            if self.cut_mode in BL:
                
                v = v_sum/self.N_t - self.inner_product(t, pi_mean, prev_solution)
        
            if self.cut_mode in SBL or self.cut_mode.startswith('hyb'):
                
                v = v_sum/self.N_t
        
            cut_coeff = []
            
            cut_coeff.append(v)
            
            for i in range(6):
                cut_coeff.append(pi_mean[i])
        
            self.psi[t].append(cut_coeff)

        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0], self.P_da)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value())) 

    def backward_pass_Lagrangian(self):
                
        ## t = {T-1 -> T-2}
        
        v_sum = 0 
        pi_mean = [0, [0], 0, 0, 0, 0]
        
        Lag_elapsed_time = 0
        Total_Lag_iter = 0
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
                
        for j in range(self.N_t): 
            
            delta = self.stage_params[T - 1][j]      
            
            fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(
                prev_solution, 
                self.P_da, 
                delta
                )
            
            pi_LP = fw_rt_last_LP_relax_subp.get_cut_coefficients()[1:]
            
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
            
            fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, self.P_da, delta)
            
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
                                                    
                dual_subp_sub_last.add_plane(dual_coeff)
                pi = dual_subp_sub_last.get_solution_value()
                obj = dual_subp_sub_last.get_objective_value()
                
                fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, self.P_da, delta)
                
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
                                                                                      
                Lag_iter += 1
                
                if Lag_iter == Lag_iter_UB:
                    
                    break
            
            Total_Lag_iter += Lag_iter
            
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
                        
        ## t = {T-2 -> T-3}, ..., {0 -> -1}
        
        for t in range(self.STAGE - 2, -1, -1): 
            
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
                        
            for j in range(self.N_t):
                                
                delta = self.stage_params[t][j]
                
                fw_rt_LP_relax_subp = fw_rt_LP_relax(
                    t, 
                    prev_solution, 
                    self.psi[t+1], 
                    self.P_da,
                    delta
                    )

                pi_LP = fw_rt_LP_relax_subp.get_cut_coefficients()[1:]
                
                pi = pi_LP
                pi_min = pi_LP
                                
                lev = 0.9
                gap = 1
                lamb = 0
                k = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
                l = 10000000*(self.STAGE - t)
                      
                dual_subp_sub = dual_approx_sub(t, reg, pi_LP)
                
                fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[t+1], self.P_da, delta)
                
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
                                        
                    dual_subp_sub.add_plane(dual_coeff)
                    pi = dual_subp_sub.get_solution_value()
                    obj = dual_subp_sub.get_objective_value()
                                        
                    fw_rt_Lag_subp = fw_rt_Lagrangian(
                        t, 
                        pi, 
                        self.psi[t+1], 
                        self.P_da,
                        delta
                        )
                    
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
                    
                    if Lag_iter == Lag_iter_UB:
                    
                        break
                
                Total_Lag_iter += Lag_iter
                                                        
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

        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0], self.P_da)
        
        self.Lag_elapsed_time_list.append(Lag_elapsed_time)
        self.Total_Lag_iter_list.append(Total_Lag_iter)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   

    def backward_pass_hybrid(self):
                
        ## t = {T-1 -> T-2}
        
        reg = 0.00001
        G = 10000000
        gap = 1 
        
        v_sum = 0 
        pi_mean = [0, [0], 0, 0, 0, 0]
        
        threshold = int(self.cut_mode.split('-')[1])
                
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
                
        for j in range(self.N_t): 
            
            delta = self.stage_params[T - 1][j]      
            
            P_rt = delta[1]
            P_da = self.P_da[T - 1]
            
            fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(
                prev_solution, 
                self.P_da,
                delta
                )
            
            psi_sub = fw_rt_last_LP_relax_subp.get_cut_coefficients()
            
            if P_rt <= P_da + threshold:
                
                pi_mean[0] += psi_sub[1]/self.N_t
                pi_mean[1][0] += psi_sub[2][0]/self.N_t
                pi_mean[2] += psi_sub[3]/self.N_t
                pi_mean[3] += psi_sub[4]/self.N_t
                pi_mean[4] += psi_sub[5]/self.N_t
                pi_mean[5] += psi_sub[6]/self.N_t
                                    
                fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian(
                    [psi_sub[i] for i in range(1, 7)], 
                    self.P_da,
                    delta
                    )

                v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
                
            else:
                
                pi_LP = psi_sub[1:]
                
                pi = pi_LP
                pi_min = pi_LP
                
                reg = 0.00001
                G = 10000000
                gap = 1       
                lamb = 0
                k = [0, [0], 0, 0, 0, 0]
                l = 10000000
                            
                dual_subp_sub_last = dual_approx_sub(self.STAGE - 1, reg, pi_LP)
                
                fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, self.P_da, delta)
                
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

                    dual_subp_sub_last.add_plane(dual_coeff)
                    pi = dual_subp_sub_last.get_solution_value()
                    obj = dual_subp_sub_last.get_objective_value()
                    
                    fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, self.P_da, delta)
                    
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
                        
        ## t = {T-2 -> T-3}, ..., {0 -> -1}
        
        for t in range(self.STAGE - 2, -1, -1): 
            
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
                        
            for j in range(self.N_t):
                                
                delta = self.stage_params[t][j]
                
                P_rt = delta[1]
                P_da = self.P_da[t]
                
                fw_rt_LP_relax_subp = fw_rt_LP_relax(
                    t, 
                    prev_solution, 
                    self.psi[t+1],
                    self.P_da, 
                    delta, 
                    )

                psi_sub = fw_rt_LP_relax_subp.get_cut_coefficients()
                
                if P_rt <= P_da + threshold:
                    
                    pi_mean[0] += psi_sub[1]/self.N_t
                
                    for i in range(self.STAGE - t):
                        pi_mean[1][i] += psi_sub[2][i]/self.N_t
                        
                    pi_mean[2] += psi_sub[3]/self.N_t
                    pi_mean[3] += psi_sub[4]/self.N_t
                    pi_mean[4] += psi_sub[5]/self.N_t
                    pi_mean[5] += psi_sub[6]/self.N_t
                        
                    fw_rt_Lagrangian_subp = fw_rt_Lagrangian(
                        t, 
                        [psi_sub[i] for i in range(1, 7)], 
                        self.psi[t+1], 
                        self.P_da,
                        delta
                        )

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
                    
                    fw_rt_Lag_subp = fw_rt_Lagrangian(
                        t, 
                        pi, 
                        self.psi[t+1], 
                        self.P_da,
                        delta
                        )
                    
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
                                            
                        dual_subp_sub.add_plane(dual_coeff)
                        pi = dual_subp_sub.get_solution_value()
                        obj = dual_subp_sub.get_objective_value()
                                            
                        fw_rt_Lag_subp = fw_rt_Lagrangian(
                            t, 
                            pi, 
                            self.psi[t+1], 
                            self.P_da,
                            delta
                            )
                        
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
        
            self.psi[t].append(cut_coeff)

        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0], self.P_da)
                
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   


    def backward_pass_1(self):
                
        ## t = {T-1 -> T-2}
        
        BL = ['B']
        SBL = ['SB', 'L-sub', 'L-lev']
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
        
        with mp.Pool() as pool:

            t_last        = self.STAGE - 1
            deltas_last   = self.stage_params[t_last]

            last_args = [
                (j, prev_solution, self.P_da, deltas_last[j], self.cut_mode)
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
            self.psi[t_last].append(cut_coeff)

            for t in range(self.STAGE - 2, -1, -1):
                
                prev_solution = self.forward_solutions[t][0]
                psi_next      = self.psi[t+1]
                deltas        = self.stage_params[t]

                inner_args = [
                    (j, t, prev_solution, psi_next, self.P_da, deltas[j], self.cut_mode)
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
                self.psi[t].append(cut_coeff)

        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0], self.P_da)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value())) 

    def backward_pass_Lagrangian_1(self):
                
        ## t = {T-1 -> T-2}
        
        Lag_elapsed_time = 0
        Total_Lag_iter = 0
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
                
        with mp.Pool() as pool:

            t_last        = self.STAGE - 1
            deltas_last   = self.stage_params[t_last]
            
            last_args = [
                (j, prev_solution, self.P_da, deltas_last[j])
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
            
            self.psi[t_last].append(cut_coeff)

            for t in range(self.STAGE-2, -1, -1):
                
                prev = self.forward_solutions[t][0]
                psi_next = self.psi[t+1]
                
                args_inner = [
                    (j, t, prev, psi_next, self.P_da, self.stage_params[t][j])
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
                self.psi[t].append(cut_coeff)
                    
        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0], self.P_da)
        
        self.Lag_elapsed_time_list.append(Lag_elapsed_time)
        self.Total_Lag_iter_list.append(Total_Lag_iter)
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   

    def backward_pass_hybrid_1(self):
                
        ## t = {T-1 -> T-2}
        
        reg = 0.00001
        G = 10000000
        gap = 1 
        
        v_sum = 0 
        pi_mean = [0, [0], 0, 0, 0, 0]
        
        threshold = int(self.cut_mode.split('-')[1])
                
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
                
        with mp.Pool() as pool:

            t_last        = self.STAGE - 1
            deltas_last   = self.stage_params[t_last]
            
            last_args = [
                (j, prev_solution, self.P_da, deltas_last[j], threshold)
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
            
            self.psi[t_last].append(cut_coeff)

            for t in range(self.STAGE-2, -1, -1):
                
                prev = self.forward_solutions[t][0]
                psi_next = self.psi[t+1]
                
                args_inner = [
                    (j, t, prev, psi_next, self.P_da, self.stage_params[t][j], threshold)
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
                self.psi[t].append(cut_coeff)
                    
        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0], self.P_da)
                
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   


    def stopping_criterion(self, tol = 1e-3):
        
        self.gap = (self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]
        
        self.running_time = time.time() - self.start_time
                
        if self.running_time > self.time_lim:
            return True

        return False

    def run_sddip(self):
        
        self.start_time = time.time()
        
        while True:
            
            if not self.final_pass and self.stopping_criterion():
                self.final_pass = True
                
                print("\n>>> Stopping criterion met. Performing final pass with M scenarios...")
            
            elif self.final_pass:
                break

            self.iteration += 1
            #print(f"\n=== Iteration {self.iteration} ===")

            if self.final_pass:
                scenarios = self.sample_scenarios(self.M)
                
            else:
                scenarios = self.sample_scenarios(1)

            if self.cut_mode in ['B', 'SB', 'L-sub', 'L-lev'] or self.cut_mode.startswith('hyb'):
                self.forward_pass(scenarios)
                
            else:
                print("Not a proposed cut")
                break

            #print(f"  LB for iter = {self.iteration} updated to: {self.LB[self.iteration]:.4f}")

            if not self.parallel_mode:
            
                if self.cut_mode in ['B', 'SB']:
                    self.backward_pass()
                    
                elif self.cut_mode in ['L-sub', 'L-lev']:
                    
                    if self.iteration <= 4:
                        self.backward_pass()
                        
                    else:
                        self.backward_pass_Lagrangian()
                        
                elif self.cut_mode.startswith('hyb'):
                    
                    if self.iteration <= 4:
                        self.backward_pass()
                        
                    else:
                        self.backward_pass_hybrid()
                else:
                    print("Not a proposed cut")
                    break
            
            else:
            
                if self.cut_mode in ['B', 'SB']:
                    self.backward_pass_1()
                    
                elif self.cut_mode in ['L-sub', 'L-lev']:
                    
                    if self.iteration <= 4:
                        self.backward_pass_1()
                        
                    else:
                        self.backward_pass_Lagrangian_1()
                        
                elif self.cut_mode.startswith('hyb'):
                    
                    if self.iteration <= 4:
                        self.backward_pass_1()
                        
                    else:
                        self.backward_pass_hybrid_1()
                else:
                    print("Not a proposed cut")
                    break
            
            #print(f"  UB for iter = {self.iteration} updated to: {self.UB[self.iteration]:.4f}")

        print("\nSDDiP complete.")
        print(f"Cut = {self.cut_mode}, Final LB = {self.LB[self.iteration]:.4f}, UB = {self.UB[self.iteration]:.4f}, gap = {(self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]:.4f}")



if __name__ == "__main__":
    
    plot_mode = False
    
    fw_scenario_num = 200  ## M
    time_lim = 7200
    
    l = 0
    
    P_da = P_da_list[l]
    stage_params = scenario_trees[l]
      
    P_rt_per_hour = [ [branch[1] for branch in stage] for stage in stage_params ]     
        
    def convergence_Comparison(
        plot = True, 
        P_da = P_da,
        stage_params = stage_params
        ):
            
        sddip_1 = SDDiPModel(
                P_da = P_da,
                stage_params=stage_params,
                forward_scenario_num=fw_scenario_num,
                cut_mode='SB',
                time_lim=time_lim,
                parallel_mode=False
            )
        
        sddip_2 = SDDiPModel(
                P_da = P_da,
                stage_params=stage_params,
                forward_scenario_num=fw_scenario_num,
                cut_mode='L-sub',
                time_lim=time_lim,
                parallel_mode=False
            )
        
        sddip_3 = SDDiPModel(
                P_da = P_da,
                stage_params=stage_params,
                forward_scenario_num=fw_scenario_num,
                cut_mode='hyb-0',
                time_lim=time_lim,
                parallel_mode=False
            )
        
        sddip_4 = SDDiPModel(
                P_da = P_da,
                stage_params=stage_params,
                forward_scenario_num=fw_scenario_num,
                cut_mode='hyb-40',
                time_lim=time_lim,
                parallel_mode=False
            )
        
        sddip_5 = SDDiPModel(
                P_da = P_da,
                stage_params=stage_params,
                forward_scenario_num=fw_scenario_num,
                cut_mode='hyb-80',
                time_lim=time_lim,
                parallel_mode=False
            )
        
        """
        DEF_start_time = time.time()
        DEF_obj = DEF_1.get_objective_value()
        DEF_end_time = time.time()
        time_DEF = DEF_end_time - DEF_start_time
        """

        sddip_1.run_sddip()
        
        LB_1_list = sddip_1.LB 
        UB_1_list = sddip_1.UB
        
        gap_SDDiP_1 = (sddip_1.UB[sddip_1.iteration] - sddip_1.LB[sddip_1.iteration])/sddip_1.UB[sddip_1.iteration]
        
        iteration_1 = sddip_1.iteration
        
        
        sddip_2.run_sddip()
        
        LB_2_list = sddip_2.LB
        UB_2_list = sddip_2.UB
        
        gap_SDDiP_2 = (sddip_2.UB[sddip_2.iteration] - sddip_2.LB[sddip_2.iteration])/sddip_2.UB[sddip_2.iteration]
        
        iteration_2 = sddip_2.iteration
        
        
        sddip_3.run_sddip()

        LB_3_list = sddip_3.LB
        UB_3_list = sddip_3.UB
        
        gap_SDDiP_3 = (sddip_3.UB[sddip_3.iteration] - sddip_3.LB[sddip_3.iteration])/sddip_3.UB[sddip_3.iteration]        
        
        iteration_3 = sddip_3.iteration
        
        
        sddip_4.run_sddip()

        LB_4_list = sddip_4.LB
        UB_4_list = sddip_4.UB
        
        gap_SDDiP_4 = (sddip_4.UB[sddip_4.iteration] - sddip_4.LB[sddip_4.iteration])/sddip_4.UB[sddip_4.iteration]        
        
        iteration_4 = sddip_4.iteration
        
        
        sddip_5.run_sddip()

        LB_5_list = sddip_5.LB
        UB_5_list = sddip_5.UB
        
        gap_SDDiP_5 = (sddip_5.UB[sddip_5.iteration] - sddip_5.LB[sddip_5.iteration])/sddip_5.UB[sddip_5.iteration]        
        
        iteration_5 = sddip_5.iteration
        
        """       
        ## Plot SDDiP results
        
        if plot:    
            plt.figure(figsize=(7,5))
            
            iterations = range(num_iter+2)
            
            plt.plot(iterations, LB_1_list, label=f"LB ({sddip_1.cut_mode})", color='tab:blue')
            plt.plot(iterations, UB_1_list, label=f"UB ({sddip_1.cut_mode})", color='tab:blue', linestyle='--')
            plt.fill_between(iterations, LB_1_list, UB_1_list, alpha=0.1, color='tab:blue')
            
            plt.plot(iterations, LB_2_list, label=f"LB ({sddip_2.cut_mode})", color='tab:orange')
            plt.plot(iterations, UB_2_list, label=f"UB ({sddip_2.cut_mode})", color='tab:orange', linestyle='--')
            plt.fill_between(iterations, LB_2_list, UB_2_list, alpha=0.1, color='tab:orange')
        
            plt.plot(iterations, LB_3_list, label=f"LB ({sddip_3.cut_mode})", color='tab:green')
            plt.plot(iterations, UB_3_list, label=f"UB ({sddip_3.cut_mode})", color='tab:green', linestyle='--')
            plt.fill_between(iterations, LB_3_list, UB_3_list, alpha=0.1, color='tab:green')
                        
            plt.xlabel('Iteration')
            plt.ylabel('Bound')
            plt.legend()
            
            plt.ylim(0, 7000000*T)
            
            plt.show()
        
        else:    
            
            plt.figure(figsize=(7,5))
            
            iterations = range(num_iter+2)
            
            List_ya = [0 for i in iterations]
            
            plt.plot(iterations, List_ya, label=f"LB ({sddip_1.cut_mode})", marker='o', color='tab:blue')
            plt.plot(iterations, List_ya, label=f"UB ({sddip_1.cut_mode})", marker='^', color='tab:blue', linestyle='--')
            plt.fill_between(iterations, List_ya, List_ya, alpha=0.1, color='tab:blue')

            plt.plot(iterations, List_ya, label=f"LB ({sddip_2.cut_mode})", marker='o', color='tab:orange')
            plt.plot(iterations, List_ya, label=f"UB ({sddip_2.cut_mode})", marker='^', color='tab:orange', linestyle='--')
            plt.fill_between(iterations, List_ya, List_ya, alpha=0.1, color='tab:orange')

            plt.plot(iterations, List_ya, label=f"LB ({sddip_3.cut_mode})", marker='o', color='tab:green')
            plt.plot(iterations, List_ya, label=f"UB ({sddip_3.cut_mode})", marker='^', color='tab:green', linestyle='--')
            plt.fill_between(iterations, List_ya, List_ya, alpha=0.1, color='tab:green')
            
            plt.xlabel('Iteration')
            plt.ylabel('Bound')
            plt.legend()
            
            plt.ylim(0, 7000000*T)
            
            plt.show()
        """   
          
        print(f"SDDiP1 optimality gap = {gap_SDDiP_1:.4f}")
        print(f"SDDiP2 optimality gap = {gap_SDDiP_2:.4f}")
        print(f"SDDiP3 optimality gap = {gap_SDDiP_3:.4f}")      
        print(f"SDDiP4 optimality gap = {gap_SDDiP_4:.4f}")
        print(f"SDDiP5 optimality gap = {gap_SDDiP_5:.4f}")   
                
        print(f"SDDiP1 final LB = {LB_1_list[-1]}, UB = {UB_1_list[-1]}")
        print(f"SDDiP2 final LB = {LB_2_list[-1]}, UB = {UB_2_list[-1]}")
        print(f"SDDiP3 final LB = {LB_3_list[-1]}, UB = {UB_3_list[-1]}")
        print(f"SDDiP4 final LB = {LB_4_list[-1]}, UB = {UB_4_list[-1]}")
        print(f"SDDiP5 final LB = {LB_5_list[-1]}, UB = {UB_5_list[-1]}")

        print(f"SDDiP1 iteration = {iteration_1}")
        print(f"SDDiP2 iteration = {iteration_2}")
        print(f"SDDiP3 iteration = {iteration_3}")
        print(f"SDDiP4 iteration = {iteration_4}")
        print(f"SDDiP5 iteration = {iteration_5}")
        
        ### Plot solutions
        
        b_da_final_1 = sddip_1.b_da_final        
        Q_da_final_1 = sddip_1.Q_da_final        
        b_rt_final_1 = sddip_1.b_rt_final        
        Q_rt_final_1 = sddip_1.Q_rt_final 
        
        b_da_final_2 = sddip_2.b_da_final           
        Q_da_final_2 = sddip_2.Q_da_final
        b_rt_final_2 = sddip_2.b_rt_final
        Q_rt_final_2 = sddip_2.Q_rt_final
        
        b_da_final_3 = sddip_3.b_da_final           
        Q_da_final_3 = sddip_3.Q_da_final
        b_rt_final_3 = sddip_3.b_rt_final
        Q_rt_final_3 = sddip_3.Q_rt_final

        b_da_final_4 = sddip_4.b_da_final           
        Q_da_final_4 = sddip_4.Q_da_final
        b_rt_final_4 = sddip_4.b_rt_final
        Q_rt_final_4 = sddip_4.Q_rt_final
        
        b_da_final_5 = sddip_5.b_da_final           
        Q_da_final_5 = sddip_5.Q_da_final
        b_rt_final_5 = sddip_5.b_rt_final
        Q_rt_final_5 = sddip_5.Q_rt_final
        
        hours = np.arange(T)
                
        def density_plot(all_curves, ylim, title, ax):
            for curve in all_curves:
                ax.plot(hours, curve, color='black', alpha=0.05, linewidth=1)
            ax.set_ylim(ylim)
            ax.set_xlim(0, T-1)
            ax.set_title(title)
            ax.set_xlabel("Hour")
            ax.grid(True)        
        
        def solution_plot(b_da, Q_da, b_rt, Q_rt, title):    
            
            fig, axes = plt.subplots(2,2, figsize=(12,8))
            density_plot(
                b_da,     
                [-P_r-20, 10],               
                "b_da density",   
                axes[0,0]
                )
            density_plot(
                Q_da,     
                [0, E_0_partial_max+30000],       
                "Q_da density",   
                axes[0,1]
                )
            density_plot(
                b_rt,     
                [-P_r-20, 10],              
                "b_rt density",   
                axes[1,0]
                )
            density_plot(
                Q_rt,     
                [0, E_0_partial_max+30000],       
                "Q_rt density",   
                axes[1,1]
                )
            
            fig.suptitle(title, fontsize=16, y=1.03)
            plt.tight_layout()
            plt.show()
        
        if plot:
                
            solution_plot(
                b_da_final_1, 
                Q_da_final_1, 
                b_rt_final_1, 
                Q_rt_final_1, 
                title="SDDiP 1 solution plot"
                )
            
            solution_plot(
                b_da_final_2, 
                Q_da_final_2, 
                b_rt_final_2, 
                Q_rt_final_2, 
                title="SDDiP 2 solution plot"
                )        
            
            solution_plot(
                b_da_final_3, 
                Q_da_final_3, 
                b_rt_final_3, 
                Q_rt_final_3, 
                title="SDDiP 3 solution plot"
                ) 
            
            solution_plot(
                b_da_final_4, 
                Q_da_final_4, 
                b_rt_final_4, 
                Q_rt_final_4, 
                title="SDDiP 4 solution plot"
                )  
                        
            solution_plot(
                b_da_final_5, 
                Q_da_final_5, 
                b_rt_final_5, 
                Q_rt_final_5, 
                title="SDDiP 5 solution plot"
                )        
            
    convergence_Comparison(plot_mode)
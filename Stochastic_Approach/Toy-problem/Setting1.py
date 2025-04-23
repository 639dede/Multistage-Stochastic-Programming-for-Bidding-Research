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

#warnings.filterwarnings("ignore")
#logging.getLogger("pyomo.core").setLevel(logging.ERROR)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

from Scenarios import Scenario

solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

SOLVER.options['TimeLimit'] = 1200

assert SOLVER.available(), f"Solver {solver} is available."

# Generate Scenario

# SDDiP Model

C = 21022.1
S = C*3
B = C
S_min = 0.1*S
S_max = 0.9*S
P_r = 80
v = 0.95
T = 2

E_0 = Scenario.E_0

scenario_generator = Scenario.Setting1_scenario(E_0)

E_0_partial = [E_0[t] for t in range(9, 9 + T)]

E_0_sum = 0
E_0_partial_max = max(E_0_partial)

for t in range(len(E_0_partial)):
    E_0_sum += E_0_partial[t]

P_da = scenario_generator.P_da

P_da_partial = [141.04237333851853, 139.02077847557203, 140.6656414585262, -60.223343222316, 141.1443379704008, 139.23105760780754, 141.57109079814273, 143.93082813230762, 138.80926230668373, 124.39462311589915]

K = [1.22*E_0_partial[t] + 1.01*B for t in range(T)]

M_gen = [[K[t], 2*K[t]] for t in range(T)]

M_set_fcn = 10800000

epsilon = 0.0000000000000000001

# Subproblems for SDDiP


## stage = -1

class fw_da(pyo.ConcreteModel): 
    
    def __init__(self, psi):
        
        super().__init__()

        self.solved = False
        self.psi = psi
        self.T = T
        
        self.P_da = P_da_partial
        
        self.M_price = [[0, 0] for t in range(self.T)]
        
        self._BigM_setting()
    
    def _BigM_setting(self):
        
        for t in range(self.T):
    
            if self.P_da[t] >= 0:

                self.M_price[t][0] = 400
                self.M_price[t][1] = 400
                
            else:

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
        model.T_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_q = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b_rt = pyo.Var(domain = pyo.Reals)
        model.T_q_rt = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        # Constraints
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= E_0_sum
        
        def market_clearing_1_rule(model, t):
            return model.b_da[t] - P_da[t] <= self.M_price[t][0]*(1 - model.n_da[t])
        
        def market_clearing_2_rule(model, t):
            return P_da[t] - model.b_da[t] <= self.M_price[t][1]*model.n_da[t]
        
        def market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def market_clearing_4_rule(model, t):
            return model.Q_da[t] <= M_gen[t][0]*model.n_da[t]
        
        def market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - M_gen[t][0]*(1 - model.n_da[t])
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_price_rule(model):
            return model.b_rt <= model.b_da[0]
        
        def rt_bidding_amount_rule(model):
            return model.q_rt <= B
        
        def rt_overbid_rule(model):
            return model.T_q <= E_0_sum
        
        ## State variable trainsition
        
        def State_SOC_rule(model):
            return model.S == 0.5*S
        
        def state_b_da_rule(model, t):
            return model.T_b[t] == model.b_da[t]
        
        def state_Q_da_rule(model, t):
            return model.T_Q[t] == model.Q_da[t]
        
        def State_q_rule(model):
            return model.T_q == model.q_rt
        
        def State_b_rt_rule(model):
            return model.T_b_rt == model.b_rt
        
        def State_q_rt_rule(model):
            return model.T_q_rt == model.q_rt
        
        def State_E_rule(model):
            return model.T_E == 0
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
            
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
        model.market_clearing_1 = pyo.Constraint(model.TIME, rule = market_clearing_1_rule)
        model.market_clearing_2 = pyo.Constraint(model.TIME, rule = market_clearing_2_rule)
        model.market_clearing_3 = pyo.Constraint(model.TIME, rule = market_clearing_3_rule)
        model.market_clearing_4 = pyo.Constraint(model.TIME, rule = market_clearing_4_rule)
        model.market_clearing_5 = pyo.Constraint(model.TIME, rule = market_clearing_5_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_rule) 
        model.rt_bidding_amount = pyo.Constraint(rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)    
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.state_b_da = pyo.Constraint(model.TIME, rule = state_b_da_rule)
        model.state_Q_da = pyo.Constraint(model.TIME, rule = state_Q_da_rule)
        model.state_q = pyo.Constraint(rule = State_q_rule)
        model.state_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.state_q_rt = pyo.Constraint(rule = State_q_rt_rule)
        model.state_E = pyo.Constraint(rule = State_E_rule)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)

        # Obj Fcn
        
        model.objective = pyo.Objective(expr=model.theta, sense=pyo.maximize)

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
        State_var.append([pyo.value(self.T_b[t]) for t in range(self.T)])
        State_var.append([pyo.value(self.T_Q[t]) for t in range(self.T)])
        State_var.append(pyo.value(self.T_q))
        State_var.append(pyo.value(self.T_b_rt))
        State_var.append(pyo.value(self.T_q_rt))
        State_var.append(pyo.value(self.T_E))
        
        return State_var 

    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        return pyo.value(self.objective)        

### State Var Binarized version

class fw_da_bin(pyo.ConcreteModel): 
    
    def __init__(self, psi):
        
        super().__init__()

        self.solved = False
        self.psi = psi
        self.T = T
        
        self.P_da = P_da_partial
        
        self.M_price = [[0, 0] for t in range(self.T)]
        
        self.Round = []
        
        self._BigM_setting()
        
        self._Bin_setting()
    
    def _BigM_setting(self):
        
        for t in range(self.T):
    
            if self.P_da[t] >= 0:

                self.M_price[t][0] = 400
                self.M_price[t][1] = 400
                
            else:

                self.M_price[t][0] = 400
                self.M_price[t][1] = 400
    
    def _Bin_setting(self):
        
        # S
        self.Round[0] = math.ceil(math.log2(S_max)) 
        # T_b
        self.Round[1] = math.ceil(math.log2(P_r)) 
        # T_Q
        self.Round[2] = math.ceil(math.log2(E_0_partial_max)) 
        # T_q
        self.Round[3] = math.ceil(math.log2(E_0_sum))
        # T_b_rt
        self.Round[4] = math.ceil(math.log2(P_r))
        # T_q_rt
        self.Round[5] = math.ceil(math.log2(B))
        # T_E
        self.Round[6] = math.ceil(math.log2(1.2*E_0_partial_max))
                        
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.S_ROUND = pyo.RangeSet(0, self.Round[0]) 
        model.b_ROUND = pyo.RangeSet(0, self.Round[1])
        model.Q_ROUND = pyo.RangeSet(0, self.Round[2])
        model.q_ROUND = pyo.RangeSet(0, self.Round[3])
        model.b_rt_ROUND = pyo.RangeSet(0, self.Round[4])
        model.q_rt_ROUND = pyo.RangeSet(0, self.Round[5])
        model.E_ROUND = pyo.RangeSet(0, self.Round[6])
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        # Vars
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## Binary State Vars
        
        model.b_S = pyo.Var(model.S_ROUND, domain = pyo.Binary)
        model.b_b = pyo.Var(model.TIME, model.b_ROUND, domain = pyo.Binary)
        model.b_Q = pyo.Var(model.TIME, model.Q_ROUND, domain = pyo.Binary)
        model.b_q = pyo.Var(model.q_ROUND, domain = pyo.Binary)
        model.b_b_rt = pyo.Var(model.b_rt_ROUND, domain = pyo.Binary)
        model.b_q_rt = pyo.Var(model.q_rt_ROUND, domain = pyo.Binary)
        model.b_E = pyo.Var(model.E_ROUND, domain = pyo.Binary)
        
        # Constraints
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= E_0_partial[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= E_0_sum
        
        def market_clearing_1_rule(model, t):
            return model.b_da[t] - P_da[t] <= self.M_price[t][0]*(1 - model.n_da[t])
        
        def market_clearing_2_rule(model, t):
            return P_da[t] - model.b_da[t] <= self.M_price[t][1]*model.n_da[t]
        
        def market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def market_clearing_4_rule(model, t):
            return model.Q_da[t] <= M_gen[t][0]*model.n_da[t]
        
        def market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - M_gen[t][0]*(1 - model.n_da[t])
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_price_rule(model):
            return model.b_rt <= model.b_da[0]
        
        def rt_bidding_amount_rule(model):
            return model.q_rt <= B
        
        def rt_overbid_rule(model):
            return sum(model.b_q[k]*(2**k) for k in range(self.Round[3] + 1)) <= E_0_sum
        
        ## State variable trainsition
        
        def State_SOC_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) == math.ceil(0.5*S)
        
        def state_b_da_rule(model, t):
            return -sum(model.b_b[t, k]*(2**k) for k in range(self.Round[1] + 1)) == model.b_da[t]
        
        def state_Q_da_rule(model, t):
            return sum(model.b_Q[t, k]*(2**k) for k in range(self.Round[2] + 1)) == model.Q_da[t]
        
        def State_q_rule(model):
            return sum(model.b_q[k]*(2**k) for k in range(self.Round[3] + 1)) == model.q_rt
        
        def State_b_rt_rule(model):
            return -sum(model.b_b_rt[k]*(2**k) for k in range(self.Round[4] + 1)) == model.b_rt
        
        def State_q_rt_rule(model):
            return sum(model.b_q_rt[k]*(2**k) for k in range(self.Round[5] + 1)) == model.q_rt
        
        def State_E_rule(model):
            return sum(model.b_E[k]*(2**k) for k in range(self.Round[6] + 1)) == 0
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + sum(self.psi[l][1][k]*model.b_S[k] for k in range(self.Round[0] + 1)) + sum(sum(self.psi[l][2][t][k]*model.b_b[t, k] for k in range(self.Round[1] + 1)) for t in range(T)) + sum(sum(self.psi[l][3][t][k]*model.b_Q[t, k] for k in range(self.Round[2] + 1)) for t in range(T)) + sum(self.psi[l][4][k]*model.b_q[k] for k in range(self.Round[3] + 1)) + sum(self.psi[l][5][k]*model.b_b_rt[k] for k in range(self.Round[4] + 1)) + sum(self.psi[l][6][k]*model.b_q_rt[k] for k in range(self.Round[5] + 1)) + sum(self.psi[l][7][k]*model.b_E[k] for k in range(self.Round[6] + 1))
            
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
        model.market_clearing_1 = pyo.Constraint(model.TIME, rule = market_clearing_1_rule)
        model.market_clearing_2 = pyo.Constraint(model.TIME, rule = market_clearing_2_rule)
        model.market_clearing_3 = pyo.Constraint(model.TIME, rule = market_clearing_3_rule)
        model.market_clearing_4 = pyo.Constraint(model.TIME, rule = market_clearing_4_rule)
        model.market_clearing_5 = pyo.Constraint(model.TIME, rule = market_clearing_5_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_rule) 
        model.rt_bidding_amount = pyo.Constraint(rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)    
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.state_b_da = pyo.Constraint(model.TIME, rule = state_b_da_rule)
        model.state_Q_da = pyo.Constraint(model.TIME, rule = state_Q_da_rule)
        model.state_q = pyo.Constraint(rule = State_q_rule)
        model.state_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.state_q_rt = pyo.Constraint(rule = State_q_rt_rule)
        model.state_E = pyo.Constraint(rule = State_E_rule)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)

        # Obj Fcn
        
        model.objective = pyo.Objective(expr=model.theta, sense=pyo.maximize)

    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True

    def get_state_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        State_var = []
        
        State_var.append([pyo.value(self.b_S[k]) for k in range(self.Round[0] + 1)])
        State_var.append([[pyo.value(self.b_b[t, k]) for k in range(self.Round[1][t] + 1)] for t in range(self.T)])
        State_var.append([[pyo.value(self.b_Q[t, k]) for k in range(self.Round[2][t] + 1)] for t in range(self.T)])
        State_var.append([pyo.value(self.b_q[k]) for k in range(self.Round[3] + 1)])
        State_var.append([pyo.value(self.b_b_rt[k]) for k in range(self.Round[4] + 1)])
        State_var.append([pyo.value(self.b_q_rt[k]) for k in range(self.Round[5] + 1)])
        State_var.append([pyo.value(self.b_E[k]) for k in range(self.Round[6] + 1)])
        
        return State_var 

    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True        

        return pyo.value(self.objective)        
       
  
## stage = 0, 1, ..., T-1

class fw_rt(pyo.ConcreteModel):

    def __init__(self, stage, T_prev, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_b_prev = T_prev[1]
        self.T_Q_prev = T_prev[2]
        self.T_q_prev = T_prev[3]
        self.T_b_rt_prev = T_prev[4]
        self.T_q_rt_prev = T_prev[5]
        self.T_E_prev = T_prev[6]
        
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                      

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, self.T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_q = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b_rt = pyo.Var(domain = pyo.Reals)
        model.T_q_rt = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        
        # Experiment for LP relaxation
        """
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        """
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == self.T_b_prev[0]
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= self.T_b_prev[1]
        
        def da_Q_rule(model):
            return model.Q_da == self.T_Q_prev[0]
        
        def rt_b_rule(model):
            return model.b_rt == self.T_b_rt_prev
        
        def rt_q_rule(model):
            return model.q_rt == self.T_q_rt_prev
        
        def rt_E_rule(model):
            return model.E_1 == self.T_E_prev
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == self.S_prev + v*model.c - (1/v)*model.d
    
        def State_b_rule(model, t):
            return model.T_b[t] == self.T_b_prev[t+1]
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == self.T_Q_prev[t+1]
        
        def State_q_rule(model):
            return model.T_q == self.T_q_prev + model.q_rt_next
        
        def State_b_rt_rule(model):
            return model.T_b_rt == model.b_rt_next
        
        def State_q_rt_rule(model):
            return model.T_q_rt == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
             
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4

        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        
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
        State_var.append([pyo.value(self.T_b[t]) for t in range(self.T - 1 - self.stage)])
        State_var.append([pyo.value(self.T_Q[t]) for t in range(self.T - 1 - self.stage)])
        State_var.append(pyo.value(self.T_q))
        State_var.append(pyo.value(self.T_b_rt))
        State_var.append(pyo.value(self.T_q_rt))
        State_var.append(pyo.value(self.T_E))
        
        return State_var 

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

class fw_rt_Alt(pyo.ConcreteModel): 

    def __init__(self, stage, T_prev, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_b_prev = T_prev[1]
        self.T_Q_prev = T_prev[2]
        self.T_q_prev = T_prev[3]
        self.T_b_rt_prev = T_prev[4]
        self.T_q_rt_prev = T_prev[5]
        self.T_E_prev = T_prev[6]
        
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                         

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-2-self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_q = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b_rt = pyo.Var(domain = pyo.Reals)
        model.T_q_rt = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        
        model.m_4_1_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_1_minus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_minus = pyo.Var(domain = pyo.NonNegativeReals)  
        model.l_4_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.l_4_2 = pyo.Var(domain = pyo.NonNegativeReals)
   
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_1 = pyo.Var(domain = pyo.Binary)
        model.n_4_2 = pyo.Var(domain = pyo.Binary)
        
        #model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == self.T_b_prev[0]
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= self.T_b_prev[1]
        
        def da_Q_rule(model):
            return model.Q_da == self.T_Q_prev[0]
        
        def rt_b_rule(model):
            return model.b_rt == self.T_b_rt_prev
        
        def rt_q_rule(model):
            return model.q_rt == self.T_q_rt_prev
        
        def rt_E_rule(model):
            return model.E_1 == self.T_E_prev
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == self.S_prev + v*model.c - (1/v)*model.d
    
        def State_b_rule(model, t):
            return model.T_b[t] == self.T_b_prev[t+1]
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == self.T_Q_prev[t+1]
        
        def State_q_rule(model):
            return model.T_q == self.T_q_prev + model.q_rt_next
        
        def State_b_rt_rule(model):
            return model.T_b_rt == model.b_rt_next
        
        def State_q_rt_rule(model):
            return model.T_q_rt == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3

        def minmax_rule_4_1(model):
            return model.m_4_1 == model.m_4_1_plus - model.m_4_1_minus
        
        def minmax_rule_4_2(model):
            return model.m_4_1_plus <= self.M_set_decomp[0][0]*(1 - model.n_4_1)
        
        def minmax_rule_4_3(model):
            return model.m_4_1_minus <= self.M_set_decomp[0][1]*model.n_4_1
        
        def minmax_rule_4_4(model):
            return model.m_4_2 == model.m_4_2_plus - model.m_4_2_minus
        
        def minmax_rule_4_5(model):
            return model.m_4_2_plus <= self.M_set_decomp[1][0]*(1 - model.n_4_2)
        
        def minmax_rule_4_6(model):
            return model.m_4_2_minus <= self.M_set_decomp[1][1]*model.n_4_2
        
        def minmax_rule_4_7(model):
            return model.m_4_1_plus - model.m_4_2_plus <= self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4_2_plus - model.m_4_1_plus <= self.M_set_decomp[1][0]*model.n_4

        def minmax_rule_4_9(model):
            return model.m_4 == model.m_4_1_plus + model.l_4_2 - model.l_4_1
        
        def minmax_rule_4_10(model):
            return model.l_4_1 <= model.m_4_1_plus       
        
        def minmax_rule_4_11(model):
            return model.l_4_1 <= self.M_set_decomp[0][0]*model.n_4
        
        def minmax_rule_4_12(model):
            return model.l_4_1 >= model.m_4_1_plus - self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_13(model):
            return model.l_4_2 <= model.m_4_2_plus       
        
        def minmax_rule_4_14(model):
            return model.l_4_2 <= self.M_set_decomp[1][0]*model.n_4
        
        def minmax_rule_4_15(model):
            return model.l_4_2 >= model.m_4_2_plus - self.M_set_decomp[1][0]*(1 - model.n_4)
            
        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        model.minmax_4_9 = pyo.Constraint(rule = minmax_rule_4_9)
        model.minmax_4_10 = pyo.Constraint(rule = minmax_rule_4_10)
        model.minmax_4_11 = pyo.Constraint(rule = minmax_rule_4_11)
        model.minmax_4_12 = pyo.Constraint(rule = minmax_rule_4_12)
        model.minmax_4_13 = pyo.Constraint(rule = minmax_rule_4_13)
        model.minmax_4_14 = pyo.Constraint(rule = minmax_rule_4_14)
        model.minmax_4_15 = pyo.Constraint(rule = minmax_rule_4_15)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        
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
        State_var.append([pyo.value(self.T_b[t]) for t in range(self.T - 1 - self.stage)])
        State_var.append([pyo.value(self.T_Q[t]) for t in range(self.T - 1 - self.stage)])
        State_var.append(pyo.value(self.T_q))
        State_var.append(pyo.value(self.T_b_rt))
        State_var.append(pyo.value(self.T_q_rt))
        State_var.append(pyo.value(self.T_E))
        
        return State_var 

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

    def __init__(self, stage, T_prev, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_b_prev = T_prev[1]
        self.T_Q_prev = T_prev[2]
        self.T_q_prev = T_prev[3]
        self.T_b_rt_prev = T_prev[4]
        self.T_q_rt_prev = T_prev[5]
        self.T_E_prev = T_prev[6]
        
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                              

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_b_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_q = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b_rt = pyo.Var(domain = pyo.Reals)
        model.T_q_rt = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_S(model):
            return model.z_S == self.S_prev
        
        def auxiliary_T_b(model, t):
            return model.z_T_b[t] == self.T_b_prev[t]
        
        def auxiliary_T_Q(model, t):
            return model.z_T_Q[t] == self.T_Q_prev[t]
        
        def auxiliary_T_q(model):
            return model.z_T_q == self.T_q_prev
        
        def auxiliary_T_b_rt(model):
            return model.z_T_b_rt == self.T_b_rt_prev
        
        def auxiliary_T_q_rt(model):
            return model.z_T_q_rt == self.T_q_rt_prev
        
        def auxiliary_T_E(model):
            return model.z_T_E == self.T_E_prev        
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= model.z_T_b[1]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d
    
        def State_b_rule(model, t):
            return model.T_b[t] == model.z_T_b[t+1]
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == model.z_T_Q[t+1]
        
        def State_q_rule(model):
            return model.T_q == model.z_T_q + model.q_rt_next
        
        def State_b_rt_rule(model):
            return model.T_b_rt == model.b_rt_next
        
        def State_q_rt_rule(model):
            return model.T_q_rt == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])           
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
             
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4
        
                
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S)
        model.auxiliary_T_b = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_b)
        model.auxiliary_T_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_Q)
        model.auxiliary_T_q = pyo.Constraint(rule = auxiliary_T_q)
        model.auxiliary_T_b_rt = pyo.Constraint(rule = auxiliary_T_b_rt)
        model.auxiliary_T_q_rt = pyo.Constraint(rule = auxiliary_T_q_rt)
        model.auxiliary_T_E = pyo.Constraint(rule = auxiliary_T_E)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        

        
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
                return [3*3600000*(T - self.stage), 0, [0 for _ in range(len(self.T_b_prev))],
                        [0 for _ in range(len(self.T_Q_prev))], 0, 0, 0, 0]
        
        psi = []
        psi.append(pyo.value(self.objective))
        psi.append(self.dual[self.auxiliary_S])
        
        pi_T_b = []
        for i in range(len(self.T_b_prev)):
            pi_T_b.append(self.dual[self.auxiliary_T_b[i]])
        psi.append(pi_T_b)
        
        pi_T_Q = []
        for i in range(len(self.T_Q_prev)):
            pi_T_Q.append(self.dual[self.auxiliary_T_Q[i]])
        psi.append(pi_T_Q)
        
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_b_rt])
        psi.append(self.dual[self.auxiliary_T_q_rt])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi

class fw_rt_LP_relax_Alt(pyo.ConcreteModel): 

    def __init__(self, stage, T_prev, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_b_prev = T_prev[1]
        self.T_Q_prev = T_prev[2]
        self.T_q_prev = T_prev[3]
        self.T_b_rt_prev = T_prev[4]
        self.T_q_rt_prev = T_prev[5]
        self.T_E_prev = T_prev[6]
        
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                               

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_b_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_q = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b_rt = pyo.Var(domain = pyo.Reals)
        model.T_q_rt = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        
        model.m_4_1_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_1_minus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_minus = pyo.Var(domain = pyo.NonNegativeReals)  
        model.l_4_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.l_4_2 = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_S(model):
            return model.z_S == self.S_prev
        
        def auxiliary_T_b(model, t):
            return model.z_T_b[t] == self.T_b_prev[t]
        
        def auxiliary_T_Q(model, t):
            return model.z_T_Q[t] == self.T_Q_prev[t]
        
        def auxiliary_T_q(model):
            return model.z_T_q == self.T_q_prev
        
        def auxiliary_T_b_rt(model):
            return model.z_T_b_rt == self.T_b_rt_prev
        
        def auxiliary_T_q_rt(model):
            return model.z_T_q_rt == self.T_q_rt_prev
        
        def auxiliary_T_E(model):
            return model.z_T_E == self.T_E_prev        
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= model.z_T_b[1]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d
    
        def State_b_rule(model, t):
            return model.T_b[t] == model.z_T_b[t+1]
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == model.z_T_Q[t+1]
        
        def State_q_rule(model):
            return model.T_q == model.z_T_q + model.q_rt_next
        
        def State_b_rt_rule(model):
            return model.T_b_rt == model.b_rt_next
        
        def State_q_rt_rule(model):
            return model.T_q_rt == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
             
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4
        
        def minmax_rule_4_1(model):
            return model.m_4_1 == model.m_4_1_plus - model.m_4_1_minus
        
        def minmax_rule_4_2(model):
            return model.m_4_1_plus <= self.M_set_decomp[0][0]*(1 - model.n_4_1)
        
        def minmax_rule_4_3(model):
            return model.m_4_1_minus <= self.M_set_decomp[0][1]*model.n_4_1
        
        def minmax_rule_4_4(model):
            return model.m_4_2 == model.m_4_2_plus - model.m_4_2_minus
        
        def minmax_rule_4_5(model):
            return model.m_4_2_plus <= self.M_set_decomp[1][0]*(1 - model.n_4_2)
        
        def minmax_rule_4_6(model):
            return model.m_4_2_minus <= self.M_set_decomp[1][1]*model.n_4_2
        
        def minmax_rule_4_7(model):
            return model.m_4_1_plus - model.m_4_2_plus <= self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4_2_plus - model.m_4_1_plus <= self.M_set_decomp[1][0]*model.n_4

        def minmax_rule_4_9(model):
            return model.m_4 == model.m_4_1_plus + model.l_4_2 - model.l_4_1
        
        def minmax_rule_4_10(model):
            return model.l_4_1 <= model.m_4_1_plus       
        
        def minmax_rule_4_11(model):
            return model.l_4_1 <= self.M_set_decomp[0][0]*model.n_4
        
        def minmax_rule_4_12(model):
            return model.l_4_1 >= model.m_4_1_plus - self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_13(model):
            return model.l_4_2 <= model.m_4_2_plus       
        
        def minmax_rule_4_14(model):
            return model.l_4_2 <= self.M_set_decomp[1][0]*model.n_4
        
        def minmax_rule_4_15(model):
            return model.l_4_2 >= model.m_4_2_plus - self.M_set_decomp[1][0]*(1 - model.n_4)
        

        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S)
        model.auxiliary_T_b = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_b)
        model.auxiliary_T_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_Q)
        model.auxiliary_T_q = pyo.Constraint(rule = auxiliary_T_q)
        model.auxiliary_T_b_rt = pyo.Constraint(rule = auxiliary_T_b_rt)
        model.auxiliary_T_q_rt = pyo.Constraint(rule = auxiliary_T_q_rt)
        model.auxiliary_T_E = pyo.Constraint(rule = auxiliary_T_E)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        model.minmax_4_9 = pyo.Constraint(rule = minmax_rule_4_9)
        model.minmax_4_10 = pyo.Constraint(rule = minmax_rule_4_10)
        model.minmax_4_11 = pyo.Constraint(rule = minmax_rule_4_11)
        model.minmax_4_12 = pyo.Constraint(rule = minmax_rule_4_12)
        model.minmax_4_13 = pyo.Constraint(rule = minmax_rule_4_13)
        model.minmax_4_14 = pyo.Constraint(rule = minmax_rule_4_14)
        model.minmax_4_15 = pyo.Constraint(rule = minmax_rule_4_15)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        

        
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
                return [3*3600000*(T - self.stage), 0, [0 for _ in range(len(self.T_b_prev))],
                        [0 for _ in range(len(self.T_Q_prev))], 0, 0, 0, 0]
        
        psi = []
        psi.append(pyo.value(self.objective))
        psi.append(self.dual[self.auxiliary_S])
        
        pi_T_b = []
        for i in range(len(self.T_b_prev)):
            pi_T_b.append(self.dual[self.auxiliary_T_b[i]])
        psi.append(pi_T_b)
        
        pi_T_Q = []
        for i in range(len(self.T_Q_prev)):
            pi_T_Q.append(self.dual[self.auxiliary_T_Q[i]])
        psi.append(pi_T_Q)
        
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_b_rt])
        psi.append(self.dual[self.auxiliary_T_q_rt])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi


class fw_rt_Lagrangian(pyo.ConcreteModel): ## stage = 0, 1, ..., T-1 (Backward - Strengthened Benders' Cut)

    def __init__(self, stage, pi, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        self.pi = pi
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                               

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(bounds = (S_min, S_max), domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, bounds = (0, 2*B), domain = pyo.Reals)
        model.z_T_q = pyo.Var(bounds = (0, E_0_sum), domain = pyo.Reals, initialize = 0.0)
        model.z_T_b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage] + B), domain = pyo.Reals)
        model.z_T_E = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage]), domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_q = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b_rt = pyo.Var(domain = pyo.Reals)
        model.T_q_rt = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        
        # Experiment for LP relaxation
        """
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        """
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints   
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= model.z_T_b[1]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d
    
        def State_b_rule(model, t):
            return model.T_b[t] == model.z_T_b[t+1]
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == model.z_T_Q[t+1]
        
        def State_q_rule(model):
            return model.T_q == model.z_T_q + model.q_rt_next
        
        def State_b_rt_rule(model):
            return model.T_b_rt == model.b_rt_next
        
        def State_q_rt_rule(model):
            return model.T_q_rt == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3

        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4
        
        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)   
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
    
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)
                
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta + model.f - (self.pi[0]*model.z_S + sum(self.pi[1][j]*model.z_T_b[j] for j in range(T - self.stage)) + sum(self.pi[2][j]*model.z_T_Q[j] for j in range(T - self.stage)) + self.pi[3]*model.z_T_q + self.pi[4]*model.z_T_b_rt + self.pi[5]*model.z_T_q_rt + self.pi[6]*model.z_T_E)
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
            [pyo.value(self.z_T_b[t]) for t in range(self.T - self.stage)],
            [pyo.value(self.z_T_Q[t]) for t in range(self.T - self.stage)],
            pyo.value(self.z_T_q),
            pyo.value(self.z_T_b_rt),
            pyo.value(self.z_T_q_rt),
            pyo.value(self.z_T_E)
        ]
        
        return z

class fw_rt_Lagrangian_Alt(pyo.ConcreteModel): 

    def __init__(self, stage, pi, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        self.pi = pi
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                                  

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_b_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.S = pyo.Var(bounds = (S_min, S_max), domain = pyo.NonNegativeReals)
        model.T_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.T_Q = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.T_q = pyo.Var(domain = pyo.NonNegativeReals)
        model.T_b_rt = pyo.Var(domain = pyo.Reals)
        model.T_q_rt = pyo.Var(domain = pyo.Reals)
        model.T_E = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        
        model.m_4_1_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_1_minus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_minus = pyo.Var(domain = pyo.NonNegativeReals)  
        model.l_4_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.l_4_2 = pyo.Var(domain = pyo.NonNegativeReals)
   
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_1 = pyo.Var(domain = pyo.Binary)
        model.n_4_2 = pyo.Var(domain = pyo.Binary)
        
        #model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints   
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= model.z_T_b[1]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
        def rt_E_rule(model):
            return model.E_1 == model.z_T_E
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == model.z_S + v*model.c - (1/v)*model.d
    
        def State_b_rule(model, t):
            return model.T_b[t] == model.z_T_b[t+1]
        
        def State_Q_rule(model, t):
            return model.T_Q[t] == model.z_T_Q[t+1]
        
        def State_q_rule(model):
            return model.T_q == model.z_T_q + model.q_rt_next
        
        def State_b_rt_rule(model):
            return model.T_b_rt == model.b_rt_next
        
        def State_q_rt_rule(model):
            return model.T_q_rt == model.q_rt_next
        
        def State_E_rule(model):
            return model.T_E == self.delta_E_0*E_0_partial[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
          
        def minmax_rule_4_1(model):
            return model.m_4_1 == model.m_4_1_plus - model.m_4_1_minus
        
        def minmax_rule_4_2(model):
            return model.m_4_1_plus <= self.M_set_decomp[0][0]*(1 - model.n_4_1)
        
        def minmax_rule_4_3(model):
            return model.m_4_1_minus <= self.M_set_decomp[0][1]*model.n_4_1
        
        def minmax_rule_4_4(model):
            return model.m_4_2 == model.m_4_2_plus - model.m_4_2_minus
        
        def minmax_rule_4_5(model):
            return model.m_4_2_plus <= self.M_set_decomp[1][0]*(1 - model.n_4_2)
        
        def minmax_rule_4_6(model):
            return model.m_4_2_minus <= self.M_set_decomp[1][1]*model.n_4_2
        
        def minmax_rule_4_7(model):
            return model.m_4_1_plus - model.m_4_2_plus <= self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4_2_plus - model.m_4_1_plus <= self.M_set_decomp[1][0]*model.n_4

        def minmax_rule_4_9(model):
            return model.m_4 == model.m_4_1_plus + model.l_4_2 - model.l_4_1
        
        def minmax_rule_4_10(model):
            return model.l_4_1 <= model.m_4_1_plus       
        
        def minmax_rule_4_11(model):
            return model.l_4_1 <= self.M_set_decomp[0][0]*model.n_4
        
        def minmax_rule_4_12(model):
            return model.l_4_1 >= model.m_4_1_plus - self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_13(model):
            return model.l_4_2 <= model.m_4_2_plus       
        
        def minmax_rule_4_14(model):
            return model.l_4_2 <= self.M_set_decomp[1][0]*model.n_4
        
        def minmax_rule_4_15(model):
            return model.l_4_2 >= model.m_4_2_plus - self.M_set_decomp[1][0]*(1 - model.n_4)
            
        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)  
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        model.minmax_4_9 = pyo.Constraint(rule = minmax_rule_4_9)
        model.minmax_4_10 = pyo.Constraint(rule = minmax_rule_4_10)
        model.minmax_4_11 = pyo.Constraint(rule = minmax_rule_4_11)
        model.minmax_4_12 = pyo.Constraint(rule = minmax_rule_4_12)
        model.minmax_4_13 = pyo.Constraint(rule = minmax_rule_4_13)
        model.minmax_4_14 = pyo.Constraint(rule = minmax_rule_4_14)
        model.minmax_4_15 = pyo.Constraint(rule = minmax_rule_4_15)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)
                
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta + model.f - (self.pi[0]*model.z_S + sum(self.pi[1][j]*model.z_T_b[j] for j in range(T - self.stage)) + sum(self.pi[2][j]*model.z_T_Q[j] for j in range(T - self.stage)) + self.pi[3]*model.z_T_q + self.pi[4]*model.z_T_b_rt + self.pi[5]*model.z_T_q_rt + self.pi[6]*model.z_T_E)
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

### State Var Binarized version    

class fw_rt_bin(pyo.ConcreteModel):

    def __init__(self, stage, b_prev, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.b_S_prev = b_prev[0]
        self.b_b_prev = b_prev[1]
        self.b_Q_prev = b_prev[2]
        self.b_q_prev = b_prev[3]
        self.b_b_rt_prev = b_prev[4]
        self.b_q_rt_prev = b_prev[5]
        self.b_E_prev = b_prev[6]
        
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self.Round = []
        
        self._BigM_setting()

        self._Bin_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                      

    def _Bin_setting(self):
        
        # S
        self.Round[0] = math.ceil(math.log2(S_max)) 
        # T_b
        self.Round[1] = math.ceil(math.log2(P_r))
        # T_Q
        self.Round[2] = math.ceil(math.log2(E_0_partial_max)) 
        # T_q
        self.Round[3] = math.ceil(math.log2(E_0_sum))
        # T_b_rt
        self.Round[4] = math.ceil(math.log2(P_r))
        # T_q_rt
        self.Round[5] = math.ceil(math.log2(1.2*E_0_partial_max + B))
        # T_E
        self.Round[6] = math.ceil(math.log2(1.2*E_0_partial_max))  
        
        self.bin_num =  math.ceil(math.log2(P_r) + 1)
        
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-2-self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi)-1)
        
        model.S_ROUND = pyo.RangeSet(0, self.Round[0]) 
        model.b_ROUND = pyo.RangeSet(0, self.Round[1])
        model.Q_ROUND = pyo.RangeSet(0, self.Round[2])
        model.q_ROUND = pyo.RangeSet(0, self.Round[3])
        model.b_rt_ROUND = pyo.RangeSet(0, self.Round[4])
        model.q_rt_ROUND = pyo.RangeSet(0, self.Round[5])
        model.E_ROUND = pyo.RangeSet(0, self.Round[6])
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num)
        
        # Vars
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## Binary State Vars
        
        model.b_S = pyo.Var(model.S_ROUND, domain = pyo.Binary)
        model.b_b = pyo.Var(model.TIME, model.b_ROUND, domain = pyo.Binary)
        model.b_Q = pyo.Var(model.TIME, model.Q_ROUND, domain = pyo.Binary)
        model.b_q = pyo.Var(model.q_ROUND, domain = pyo.Binary)
        model.b_b_rt = pyo.Var(model.b_rt_ROUND, domain = pyo.Binary)
        model.b_q_rt = pyo.Var(model.q_rt_ROUND, domain = pyo.Binary)
        model.b_E = pyo.Var(model.E_ROUND, domain = pyo.Binary)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        
        # Experiment for LP relaxation
        """
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        """
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == -sum(self.b_b_prev[0][k]*(2**k) for k in range(self.Round[1] + 1))
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= -sum(self.b_b_prev[1][k]*(2**k) for k in range(self.Round[1] + 1))
        
        def da_Q_rule(model):
            return model.Q_da == sum(self.b_Q_prev[0][k]*(2**k) for k in range(self.Round[2] + 1))
        
        def rt_b_rule(model):
            return model.b_rt == -sum(self.b_b_rt_prev[k]*(2**k) for k in range(self.Round[4] + 1))
        
        def rt_q_rule(model):
            return model.q_rt == sum(self.b_q_rt_prev[k]*(2**k) for k in range(self.Round[5] + 1))
        
        def rt_E_rule(model):
            return model.E_1 == sum(self.b_E_prev[k]*(2**k) for k in range(self.Round[6] + 1))
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) == sum(self.b_S_prev[k]*(2**k) for k in range(self.Round[0] + 1)) + v*model.c - (1/v)*model.d

        def State_SOC_UB_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) <= S_max
        
        def State_SOC_LB_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) >= S_min
                
        def State_b_rule(model, t, k):
            return model.b_b[t, k] == self.b_b_prev[t+1][k]
        
        def State_Q_rule(model, t, k):
            return model.b_Q[t, k] == self.b_Q_prev[t+1][k]
        
        def State_q_rule(model):
            return sum(model.b_q[k]*(2**k) for k in range(self.Round[3] + 1)) == sum(self.b_q_prev[k]*(2**k) for k in range(self.Round[3] + 1)) + model.q_rt_next
        
        def State_b_rt_rule(model):
            return sum(model.b_b_rt[k]*(2**k) for k in range(self.Round[4] + 1)) == model.b_rt_next
        
        def State_q_rt_rule(model):
            return sum(model.b_q_rt[k]*(2**k) for k in range(self.Round[5] + 1)) == model.q_rt_next
        
        def State_E_rule(model):
            return sum(model.b_E[k]*(2**k) for k in range(self.Round[6] + 1)) == math.ceil(self.delta_E_0*E_0_partial[self.stage + 1])
        
        ## General Constraints
        
        def overbid_rule(model):
            return sum(model.b_q[k]*(2**k) for k in range(self.Round[3] + 1)) <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*self.b_b_rt_prev[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - self.b_b_rt_prev[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*self.b_b_prev[0][i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - self.b_b_prev[0][i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*self.b_b_prev[0][i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - self.b_b_prev[0][i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*self.b_b_rt_prev[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - self.b_b_rt_prev[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
             
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4

        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + sum(self.psi[l][1][k]*model.b_S[k] for k in range(self.Round[0]+1)) + sum(sum(self.psi[l][2][t][k]*model.b_b[t, k] for k in range(self.Round[1]+1)) for t in range(T - 1 - self.stage)) + sum(sum(self.psi[l][3][t][k]*model.b_Q[t, k] for k in range(self.Round[2]+1)) for t in range(T - 1 - self.stage)) + sum(self.psi[l][4][k]*model.b_q[k] for k in range(self.Round[3]+1)) + sum(self.psi[l][5][k]*model.b_b_rt[k] for k in range(self.Round[4]+1)) + sum(self.psi[l][6][k]*model.b_q_rt[k] for k in range(self.Round[5]+1)) + sum(self.psi[l][7][k]*model.b_E[k] for k in range(self.Round[6]+1))
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_SOC_UB = pyo.Constraint(rule = State_SOC_UB_rule)
        model.State_SOC_LB = pyo.Constraint(rule = State_SOC_LB_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
                    
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        
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
        
        State_var.append([pyo.value(self.b_S[k]) for k in range(self.Round[0] + 1)])
        State_var.append([[pyo.value(self.b_b[t, k]) for k in range(self.Round[1][t] + 1)] for t in range(self.T)])
        State_var.append([[pyo.value(self.b_Q[t, k]) for k in range(self.Round[2][t] + 1)] for t in range(self.T)])
        State_var.append([pyo.value(self.b_q[k]) for k in range(self.Round[3] + 1)])
        State_var.append([pyo.value(self.b_b_rt[k]) for k in range(self.Round[4] + 1)])
        State_var.append([pyo.value(self.b_q_rt[k]) for k in range(self.Round[5] + 1)])
        State_var.append([pyo.value(self.b_E[k]) for k in range(self.Round[6] + 1)])
        
        return State_var 

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

class fw_rt_bin_LP_relax(pyo.ConcreteModel): ## (Backward - Benders' Cut)

    def __init__(self, stage, b_prev, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        
        self.b_S_prev = b_prev[0]
        self.b_b_prev = b_prev[1]
        self.b_Q_prev = b_prev[2]
        self.b_q_prev = b_prev[3]
        self.b_b_rt_prev = b_prev[4]
        self.b_q_rt_prev = b_prev[5]
        self.b_E_prev = b_prev[6]
        
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self.Round = []
        
        self._BigM_setting()

        self._Bin_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                              

    def _Bin_setting(self):
        
        # S
        self.Round[0] = math.ceil(math.log2(S_max)) 
        # T_b
        self.Round[1] = math.ceil(math.log2(P_r)) 
        # T_Q
        self.Round[2] = math.ceil(math.log2(E_0_partial_max)) 
        # T_q
        self.Round[3] = math.ceil(math.log2(E_0_sum))
        # T_b_rt
        self.Round[4] = math.ceil(math.log2(P_r))
        # T_q_rt
        self.Round[5] = math.ceil(math.log2(1.2*E_0_partial_max + B))
        # T_E
        self.Round[6] = math.ceil(math.log2(1.2*E_0_partial_max))    

        self.bin_num =  math.ceil(math.log2(P_r) + 1)

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
        
        model.S_ROUND = pyo.RangeSet(0, self.Round[0]) 
        model.b_ROUND = pyo.RangeSet(0, self.Round[1])
        model.Q_ROUND = pyo.RangeSet(0, self.Round[2])
        model.q_ROUND = pyo.RangeSet(0, self.Round[3])
        model.b_rt_ROUND = pyo.RangeSet(0, self.Round[4])
        model.q_rt_ROUND = pyo.RangeSet(0, self.Round[5])
        model.E_ROUND = pyo.RangeSet(0, self.Round[6]) 
               
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_b_S = pyo.Var(model.S_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_b = pyo.Var(model.Z_TIME, model.b_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_Q = pyo.Var(model.Z_TIME, model.Q_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_q = pyo.Var(model.q_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_b_rt = pyo.Var(model.b_rt_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_q_rt = pyo.Var(model.q_rt_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_E = pyo.Var(model.E_ROUND, bounds = (0, 1), domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.b_S = pyo.Var(model.S_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.b_b = pyo.Var(model.TIME, model.b_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.b_Q = pyo.Var(model.TIME, model.Q_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.b_q = pyo.Var(model.q_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.b_b_rt = pyo.Var(model.b_rt_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.b_q_rt = pyo.Var(model.q_rt_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.b_E = pyo.Var(model.E_ROUND, bounds = (0, 1), domain = pyo.Reals)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_S_rule(model, k):
            return model.z_b_S[k] == self.b_S_prev[k]
        
        def auxiliary_b_rule(model, t, k):
            return model.z_b_b[t][k] == self.b_b_prev[t][k]
        
        def auxiliary_Q_rule(model, t, k):
            return model.z_b_Q[t][k] == self.b_Q_prev[t][k]
        
        def auxiliary_q_rule(model, k):
            return model.z_b_q[k] == self.b_q_prev[k]
        
        def auxiliary_b_rt_rule(model, k):
            return model.z_b_b_rt[k] == self.b_b_rt_prev[k]
        
        def auxiliary_q_rt_rule(model, k):
            return model.z_b_q_rt[k] == self.b_q_rt_prev[k]
        
        def auxiliary_E_rule(model, k):
            return model.z_b_E[k] == self.b_E_prev[k]        
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == -sum(model.z_b_b[0, k]*(2**k) for k in range(self.Round[1] + 1))
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= -sum(model.z_b_b[1, k]*(2**k) for k in range(self.Round[1] + 1))
        
        def da_Q_rule(model):
            return model.Q_da == sum(model.z_b_Q[0, k]*(2**k) for k in range(self.Round[2] + 1))
        
        def rt_b_rule(model):
            return model.b_rt == -sum(model.z_b_b_rt[k]*(2**k) for k in range(self.Round[4] + 1))
        
        def rt_q_rule(model):
            return model.q_rt == sum(model.z_b_q_rt[k]*(2**k) for k in range(self.Round[5] + 1))
        
        def rt_E_rule(model):
            return model.E_1 == sum(model.z_b_E[k]*(2**k) for k in range(self.Round[6] + 1))
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) == sum(model.z_b_S[k]*(2**k) for k in range(self.Round[0] + 1)) + v*model.c - (1/v)*model.d
    
        def State_SOC_UB_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) <= S_max
        
        def State_SOC_LB_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) >= S_min
                
        def State_b_rule(model, t, k):
            return model.b_b[t, k] == model.z_b_b[t+1][k]
        
        def State_Q_rule(model, t, k):
            return model.b_Q[t, k] == model.z_b_Q[t+1][k]
        
        def State_q_rule(model):
            return sum(model.b_q[k]*(2**k) for k in range(self.Round[3] + 1)) == sum(model.z_b_q[k]*(2**k) for k in range(self.Round[3] + 1)) + model.q_rt_next
        
        def State_b_rt_rule(model):
            return sum(model.b_b_rt[k]*(2**k) for k in range(self.Round[4] + 1)) == model.b_rt_next
        
        def State_q_rt_rule(model):
            return sum(model.b_q_rt[k]*(2**k) for k in range(self.Round[5] + 1)) == model.q_rt_next
        
        def State_E_rule(model):
            return sum(model.b_E[k]*(2**k) for k in range(self.Round[6] + 1)) == math.ceil(self.delta_E_0*E_0_partial[self.stage + 1])
        
        ## General Constraints
        
        def overbid_rule(model):
            return sum(model.b_q[k]*(2**k) for k in range(self.Round[3] + 1)) <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.z_b_b_rt[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.z_b_b_rt[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.z_b_b[0][i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.z_b_b[0][i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.z_b_b[0][i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.z_b_b[0][i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.z_b_b_rt[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.z_b_b_rt[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
             
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4

        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + sum(self.psi[l][1][k]*model.b_S[k] for k in range(self.Round[0]+1)) + sum(sum(self.psi[l][2][t][k]*model.b_b[t, k] for k in range(self.Round[1]+1)) for t in range(T - 1 - self.stage)) + sum(sum(self.psi[l][3][t][k]*model.b_Q[t, k] for k in range(self.Round[2]+1)) for t in range(T - 1 - self.stage)) + sum(self.psi[l][4][k]*model.b_q[k] for k in range(self.Round[3]+1)) + sum(self.psi[l][5][k]*model.b_b_rt[k] for k in range(self.Round[4]+1)) + sum(self.psi[l][6][k]*model.b_q_rt[k] for k in range(self.Round[5]+1)) + sum(self.psi[l][7][k]*model.b_E[k] for k in range(self.Round[6]+1))
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        model.auxiliary_S = pyo.Constraint(model.S_ROUND, rule = auxiliary_S_rule)
        model.auxiliary_b = pyo.Constraint(model.Z_TIME, model.b_ROUND, rule = auxiliary_b_rule)
        model.auxiliary_Q = pyo.Constraint(model.Z_TIME, model.Q_ROUND, rule = auxiliary_Q_rule)
        model.auxiliary_q = pyo.Constraint(model.q_ROUND, rule = auxiliary_q_rule)
        model.auxiliary_b_rt = pyo.Constraint(model.b_rt_ROUND, rule = auxiliary_b_rt_rule)
        model.auxiliary_q_rt = pyo.Constraint(model.q_rt_ROUND, rule = auxiliary_q_rt_rule)
        model.auxiliary_E = pyo.Constraint(model.E_ROUND, rule = auxiliary_E_rule)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_SOC_UB = pyo.Constraint(rule = State_SOC_UB_rule)
        model.State_SOC_LB = pyo.Constraint(rule = State_SOC_LB_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
                    
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        
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
                return [3*3600000*(T - self.stage), [0 for _ in range(self.Round[0]+1)], [[0 for _ in range(self.Round[1]+1)] for _ in range(len(self.T_b_prev))],
                        [[0 for _ in range(self.Round[2]+1)] for _ in range(len(self.T_Q_prev))], [0 for _ in range(self.Round[3]+1)], [0 for _ in range(self.Round[4]+1)], [0 for _ in range(self.Round[5]+1)], [0 for _ in range(self.Round[6]+1)]]
        
        psi = []
        psi.append(pyo.value(self.objective))
        
        psi.append([self.dual[self.auxiliary_S[k]] for k in range(self.Round[0]+1)])
        
        pi_T_b = []
        for i in range(len(self.T_b_prev)):
            pi_T_b.append([
                self.dual[self.auxiliary_b[i, k]] for k in range(self.Round[1]+1)
                ])
        psi.append(pi_T_b)
        
        pi_T_Q = []
        for i in range(len(self.T_Q_prev)):
            pi_T_Q.append([
                self.dual[self.auxiliary_Q[i, k]] for k in range(self.Round[2]+1)
                ])
        psi.append(pi_T_Q)
        
        psi.append([self.dual[self.auxiliary_q[k]] for k in range(self.Round[3]+1)])
        psi.append([self.dual[self.auxiliary_b_rt[k]] for k in range(self.Round[4]+1)])
        psi.append([self.dual[self.auxiliary_q_rt[k]] for k in range(self.Round[5]+1)])
        psi.append([self.dual[self.auxiliary_E[k]] for k in range(self.Round[6]+1)])
        
        return psi

class fw_rt_bin_Lagrangian(pyo.ConcreteModel): ## stage = 0, 1, ..., T-1 (Backward - Strengthened Benders' Cut)

    def __init__(self, stage, pi, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        self.pi = pi
        self.psi = psi
        
        self.P_da = P_da_partial
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self.Round = []
        
        self._BigM_setting()
        
        self._Bin_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                               

    def _Bin_setting(self):
        
        # S
        self.Round[0] = math.ceil(math.log2(S_max)) 
        # T_b
        self.Round[1] = math.ceil(math.log2(P_r))
        # T_Q
        self.Round[2] = math.ceil(math.log2(E_0_partial_max)) 
        # T_q
        self.Round[3] = math.ceil(math.log2(E_0_sum))
        # T_b_rt
        self.Round[4] = math.ceil(math.log2(P_r))
        # T_q_rt
        self.Round[5] = math.ceil(math.log2(1.2*E_0_partial_max + B))
        # T_E
        self.Round[6] = math.ceil(math.log2(1.2*E_0_partial_max))  
        
        self.bin_num =  math.ceil(math.log2(P_r) + 1)
        
    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, T - 1 - self.stage)
        
        model.TIME = pyo.RangeSet(0, T - 2 - self.stage)
        
        model.PSIRANGE = pyo.RangeSet(0, len(self.psi) - 1)
        
        model.S_ROUND = pyo.RangeSet(0, self.Round[0]) 
        model.b_ROUND = pyo.RangeSet(0, self.Round[1])
        model.Q_ROUND = pyo.RangeSet(0, self.Round[2])
        model.q_ROUND = pyo.RangeSet(0, self.Round[3])
        model.b_rt_ROUND = pyo.RangeSet(0, self.Round[4])
        model.q_rt_ROUND = pyo.RangeSet(0, self.Round[5])
        model.E_ROUND = pyo.RangeSet(0, self.Round[6])
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_b_S = pyo.Var(model.S_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_b = pyo.Var(model.Z_TIME, model.b_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_Q = pyo.Var(model.Z_TIME, model.Q_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_q = pyo.Var(model.q_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_b_rt = pyo.Var(model.b_rt_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_q_rt = pyo.Var(model.q_rt_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_E = pyo.Var(model.E_ROUND, bounds = (0, 1), domain = pyo.Reals)
        
        ## CTG fcn approx
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt_next = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt_next = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
        ## Real-Time operation 
        
        model.g = pyo.Var(domain = pyo.NonNegativeReals)
        model.c = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.E_1 = pyo.Var(domain = pyo.NonNegativeReals)
        
        ## State Vars
        
        model.b_S = pyo.Var(model.S_ROUND, domain = pyo.Binary)
        model.b_b = pyo.Var(model.TIME, model.b_ROUND, domain = pyo.Binary)
        model.b_Q = pyo.Var(model.TIME, model.Q_ROUND, domain = pyo.Binary)
        model.b_q = pyo.Var(model.q_ROUND, domain = pyo.Binary)
        model.b_b_rt = pyo.Var(model.b_rt_ROUND, domain = pyo.Binary)
        model.b_q_rt = pyo.Var(model.q_rt_ROUND, domain = pyo.Binary)
        model.b_E = pyo.Var(model.E_ROUND, domain = pyo.Binary)
        
        ## min, max reformulation Vars
        
        model.m_1 = pyo.Var(domain = pyo.Reals)
        model.m_2 = pyo.Var(domain = pyo.Reals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        
        # Experiment for LP relaxation
        """
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        """
        ## settlement_fcn_Vars
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
                   
        # Constraints   
        
        ## Connected to t-1 state 
        
        
        def da_bidding_price_rule(model):
            return model.b_da == -sum(model.z_b_b[0, k]*(2**k) for k in range(self.Round[1] + 1))
        
        def rt_bidding_price_next_rule(model):
            return model.b_rt_next <= -sum(model.z_b_b[1, k]*(2**k) for k in range(self.Round[1] + 1))
        
        def da_Q_rule(model):
            return model.Q_da == sum(model.z_b_Q[0, k]*(2**k) for k in range(self.Round[2] + 1))
        
        def rt_b_rule(model):
            return model.b_rt == -sum(model.z_b_b_rt[k]*(2**k) for k in range(self.Round[4] + 1))
        
        def rt_q_rule(model):
            return model.q_rt == sum(model.z_b_q_rt[k]*(2**k) for k in range(self.Round[5] + 1))
        
        def rt_E_rule(model):
            return model.E_1 == sum(model.z_b_E[k]*(2**k) for k in range(self.Round[6] + 1))
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) == sum(model.z_b_S[k]*(2**k) for k in range(self.Round[0] + 1)) + v*model.c - (1/v)*model.d
    
        def State_SOC_UB_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) <= S_max
        
        def State_SOC_LB_rule(model):
            return sum(model.b_S[k]*(2**k) for k in range(self.Round[0] + 1)) >= S_min
                
        def State_b_rule(model, t, k):
            return model.b_b[t, k] == model.z_b_b[t+1][k]
        
        def State_Q_rule(model, t, k):
            return model.b_Q[t, k] == model.z_b_Q[t+1][k]
        
        def State_q_rule(model):
            return sum(model.b_q[k]*(2**k) for k in range(self.Round[3] + 1)) == sum(model.z_b_q[k]*(2**k) for k in range(self.Round[3] + 1)) + model.q_rt_next
        
        def State_b_rt_rule(model):
            return sum(model.b_b_rt[k]*(2**k) for k in range(self.Round[4] + 1)) == model.b_rt_next
        
        def State_q_rt_rule(model):
            return sum(model.b_q_rt[k]*(2**k) for k in range(self.Round[5] + 1)) == model.q_rt_next
        
        def State_E_rule(model):
            return sum(model.b_E[k]*(2**k) for k in range(self.Round[6] + 1)) == math.ceil(self.delta_E_0*E_0_partial[self.stage + 1])
        
        ## General Constraints
        
        def overbid_rule(model):
            return sum(model.b_q[k]*(2**k) for k in range(self.Round[3] + 1)) <= E_0_sum
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.z_b_b_rt[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.z_b_b_rt[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.z_b_b[0][i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.z_b_b[0][i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.z_b_b[0][i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.z_b_b[0][i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.z_b_b_rt[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.z_b_b_rt[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
             
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4

        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + sum(self.psi[l][1][k]*model.b_S[k] for k in range(self.Round[0]+1)) + sum(sum(self.psi[l][2][t][k]*model.b_b[t, k] for k in range(self.Round[1]+1)) for t in range(T - 1 - self.stage)) + sum(sum(self.psi[l][3][t][k]*model.b_Q[t, k] for k in range(self.Round[2]+1)) for t in range(T - 1 - self.stage)) + sum(self.psi[l][4][k]*model.b_q[k] for k in range(self.Round[3]+1)) + sum(self.psi[l][5][k]*model.b_b_rt[k] for k in range(self.Round[4]+1)) + sum(self.psi[l][6][k]*model.b_q_rt[k] for k in range(self.Round[5]+1)) + sum(self.psi[l][7][k]*model.b_E[k] for k in range(self.Round[6]+1))
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
        model.rt_bidding_price = pyo.Constraint(rule = rt_bidding_price_next_rule)
        model.da_Q = pyo.Constraint(rule = da_Q_rule)
        model.rt_b = pyo.Constraint(rule = rt_b_rule)
        model.rt_q = pyo.Constraint(rule = rt_q_rule)
        model.rt_E = pyo.Constraint(rule = rt_E_rule)
        
        model.State_SOC = pyo.Constraint(rule = State_SOC_rule)
        model.State_SOC_UB = pyo.Constraint(rule = State_SOC_UB_rule)
        model.State_SOC_LB = pyo.Constraint(rule = State_SOC_LB_rule)
        model.State_b = pyo.Constraint(model.TIME, rule = State_b_rule)
        model.State_Q = pyo.Constraint(model.TIME, rule = State_Q_rule)
        model.State_q = pyo.Constraint(rule = State_q_rule)
        model.State_b_rt = pyo.Constraint(rule = State_b_rt_rule)
        model.State_q_rt = pyo.Constraint(rule = State_q_rt_rule)
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
                    
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
        
        model.value_fcn_approx = pyo.Constraint(model.PSIRANGE, rule = value_fcn_approx_rule)
                
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.theta 
                + model.f - (
                    sum(self.pi[0][k]*model.z_b_S[k] for k in range(self.Round[0]+1))
                    + sum(sum(self.pi[1][j][k]*model.z_T_b[j][k] for k in range(self.Round[1]+1)) for j in range(T - self.stage)) 
                    + sum(sum(self.pi[2][j][k]*model.z_T_Q[j][k] for k in range(self.Round[2]+1)) for j in range(T - self.stage)) 
                    + sum(self.pi[3][k]*model.z_T_q[k] for k in range(self.Round[3]+1)) 
                    + sum(self.pi[4][k]*model.z_T_b_rt[k] for k in range(self.Round[4]+1)) 
                    + sum(self.pi[5][k]*model.z_T_q_rt[k] for k in range(self.Round[5]+1)) 
                    + sum(self.pi[6][k]*model.z_T_E[k] for k in range(self.Round[6]+1))
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
            [pyo.value(self.z_b_S[k]) for k in range(self.Round[0]+1)],
            [[pyo.value(self.z_b_b[t][k]) for k in range(self.Round[1]+1)] for t in range(self.T - self.stage)],
            [[pyo.value(self.z_b_Q[t][k]) for k in range(self.Round[2]+1)] for t in range(self.T - self.stage)],
            [pyo.value(self.z_b_q[k]) for k in range(self.Round[3]+1)],
            [pyo.value(self.z_b_b_rt[k]) for k in range(self.Round[4]+1)],
            [pyo.value(self.z_b_q_rt[k]) for k in range(self.Round[5]+1)],
            [pyo.value(self.z_b_E[k]) for k in range(self.Round[6]+1)]
        ]
        
        return z


## stage = T

class fw_rt_last(pyo.ConcreteModel): 
    
    def __init__(self, T_prev, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_b_prev = T_prev[1]
        self.T_Q_prev = T_prev[2]
        self.T_q_prev = T_prev[3]
        self.T_b_rt_prev = T_prev[4]
        self.T_q_rt_prev = T_prev[5]
        self.T_E_prev = T_prev[6]

        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                              
 
    def build_model(self):
        
        model = self.model()
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)         
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        
        # Experiment for LP relaxation
        
        """
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        """
        
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == self.T_b_prev[0]
        
        def da_Q_rule(model):
            return model.Q_da == self.T_Q_prev[0]
        
        def rt_b_rule(model):
            return model.b_rt == self.T_b_rt_prev
        
        def rt_q_rule(model):
            return model.q_rt == self.T_q_rt_prev
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3

        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4
        
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
       
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
                         
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
  
class fw_rt_last_Alt(pyo.ConcreteModel): 
    
    def __init__(self, T_prev, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_b_prev = T_prev[1]
        self.T_Q_prev = T_prev[2]
        self.T_q_prev = T_prev[3]
        self.T_b_rt_prev = T_prev[4]
        self.T_q_rt_prev = T_prev[5]
        self.T_E_prev = T_prev[6]

        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                             

    def build_model(self):
        
        model = self.model()
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        #model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        #model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        #model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)     
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)             
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        
        model.m_4_1_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_1_minus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_minus = pyo.Var(domain = pyo.NonNegativeReals)  
        model.l_4_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.l_4_2 = pyo.Var(domain = pyo.NonNegativeReals)
   
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_1 = pyo.Var(domain = pyo.Binary)
        model.n_4_2 = pyo.Var(domain = pyo.Binary)
        
        #model.n_3 = pyo.Var(boudns = (0, 1), domain = pyo.Reals)
        #model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == self.T_b_prev[0]
        
        def da_Q_rule(model):
            return model.Q_da == self.T_Q_prev[0]
        
        def rt_b_rule(model):
            return model.b_rt == self.T_b_rt_prev
        
        def rt_q_rule(model):
            return model.q_rt == self.T_q_rt_prev
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3

        def minmax_rule_4_1(model):
            return model.m_4_1 == model.m_4_1_plus - model.m_4_1_minus
        
        def minmax_rule_4_2(model):
            return model.m_4_1_plus <= self.M_set_decomp[0][0]*(1 - model.n_4_1)
        
        def minmax_rule_4_3(model):
            return model.m_4_1_minus <= self.M_set_decomp[0][1]*model.n_4_1
        
        def minmax_rule_4_4(model):
            return model.m_4_2 == model.m_4_2_plus - model.m_4_2_minus
        
        def minmax_rule_4_5(model):
            return model.m_4_2_plus <= self.M_set_decomp[1][0]*(1 - model.n_4_2)
        
        def minmax_rule_4_6(model):
            return model.m_4_2_minus <= self.M_set_decomp[1][1]*model.n_4_2
        
        def minmax_rule_4_7(model):
            return model.m_4_1_plus - model.m_4_2_plus <= self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4_2_plus - model.m_4_1_plus <= self.M_set_decomp[1][0]*model.n_4

        def minmax_rule_4_9(model):
            return model.m_4 == model.m_4_1_plus + model.l_4_2 - model.l_4_1
        
        def minmax_rule_4_10(model):
            return model.l_4_1 <= model.m_4_1_plus       
        
        def minmax_rule_4_11(model):
            return model.l_4_1 <= self.M_set_decomp[0][0]*model.n_4
        
        def minmax_rule_4_12(model):
            return model.l_4_1 >= model.m_4_1_plus - self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_13(model):
            return model.l_4_2 <= model.m_4_2_plus       
        
        def minmax_rule_4_14(model):
            return model.l_4_2 <= self.M_set_decomp[1][0]*model.n_4
        
        def minmax_rule_4_15(model):
            return model.l_4_2 >= model.m_4_2_plus - self.M_set_decomp[1][0]*(1 - model.n_4)
        
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        model.minmax_4_9 = pyo.Constraint(rule = minmax_rule_4_9)
        model.minmax_4_10 = pyo.Constraint(rule = minmax_rule_4_10)
        model.minmax_4_11 = pyo.Constraint(rule = minmax_rule_4_11)
        model.minmax_4_12 = pyo.Constraint(rule = minmax_rule_4_12)
        model.minmax_4_13 = pyo.Constraint(rule = minmax_rule_4_13)
        model.minmax_4_14 = pyo.Constraint(rule = minmax_rule_4_14)
        model.minmax_4_15 = pyo.Constraint(rule = minmax_rule_4_15)
       
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
                         
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
           
    def __init__(self, T_prev, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_b_prev = T_prev[1]
        self.T_Q_prev = T_prev[2]
        self.T_q_prev = T_prev[3]
        self.T_b_rt_prev = T_prev[4]
        self.T_q_rt_prev = T_prev[5]
        self.T_E_prev = T_prev[6]

        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                             

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_b_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_rt = pyo.Var(domain = pyo.Binary)

        
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)  
        #model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)     
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        #model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)             
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_S(model):
            return model.z_S == self.S_prev
        
        def auxiliary_T_b(model, t):
            return model.z_T_b[t] == self.T_b_prev[t]
        
        def auxiliary_T_Q(model, t):
            return model.z_T_Q[t] == self.T_Q_prev[t]
        
        def auxiliary_T_q(model):
            return model.z_T_q == self.T_q_prev
        
        def auxiliary_T_b_rt(model):
            return model.z_T_b_rt == self.T_b_rt_prev
        
        def auxiliary_T_q_rt(model):
            return model.z_T_q_rt == self.T_q_rt_prev
        
        def auxiliary_T_E(model):
            return model.z_T_E == self.T_E_prev
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
       
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4 
  
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)  
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S)
        model.auxiliary_T_b = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_b)
        model.auxiliary_T_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_Q)
        model.auxiliary_T_q = pyo.Constraint(rule = auxiliary_T_q)
        model.auxiliary_T_b_rt = pyo.Constraint(rule = auxiliary_T_b_rt)
        model.auxiliary_T_q_rt = pyo.Constraint(rule = auxiliary_T_q_rt)
        model.auxiliary_T_E = pyo.Constraint(rule = auxiliary_T_E)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
    
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
               
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)    
        

        
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
                return [3*3600000, 0, [0 for _ in range(len(self.T_b_prev))],
                        [0 for _ in range(len(self.T_Q_prev))], 0, 0, 0, 0]
        
        psi = []
        psi.append(pyo.value(self.objective))
        psi.append(self.dual[self.auxiliary_S])
        
        pi_T_b = []
        for i in range(len(self.T_b_prev)):
            pi_T_b.append(self.dual[self.auxiliary_T_b[i]])
        psi.append(pi_T_b)
        
        pi_T_Q = []
        for i in range(len(self.T_Q_prev)):
            pi_T_Q.append(self.dual[self.auxiliary_T_Q[i]])
        psi.append(pi_T_Q)
        
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_b_rt])
        psi.append(self.dual[self.auxiliary_T_q_rt])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi  
      
class fw_rt_last_LP_relax_Alt(pyo.ConcreteModel): 
           
    def __init__(self, T_prev, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_b_prev = T_prev[1]
        self.T_Q_prev = T_prev[2]
        self.T_q_prev = T_prev[3]
        self.T_b_rt_prev = T_prev[4]
        self.T_q_rt_prev = T_prev[5]
        self.T_E_prev = T_prev[6]

        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                              

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals)
        model.z_T_b_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_rt = pyo.Var(domain = pyo.Binary)

        
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)  
        #model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)     
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        #model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)             
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        
        model.m_4_1_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_1_minus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_minus = pyo.Var(domain = pyo.NonNegativeReals)  
        model.l_4_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.l_4_2 = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_S(model):
            return model.z_S == self.S_prev
        
        def auxiliary_T_b(model, t):
            return model.z_T_b[t] == self.T_b_prev[t]
        
        def auxiliary_T_Q(model, t):
            return model.z_T_Q[t] == self.T_Q_prev[t]
        
        def auxiliary_T_q(model):
            return model.z_T_q == self.T_q_prev
        
        def auxiliary_T_b_rt(model):
            return model.z_T_b_rt == self.T_b_rt_prev
        
        def auxiliary_T_q_rt(model):
            return model.z_T_q_rt == self.T_q_rt_prev
        
        def auxiliary_T_E(model):
            return model.z_T_E == self.T_E_prev
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
        
        def minmax_rule_4_1(model):
            return model.m_4_1 == model.m_4_1_plus - model.m_4_1_minus
        
        def minmax_rule_4_2(model):
            return model.m_4_1_plus <= self.M_set_decomp[0][0]*(1 - model.n_4_1)
        
        def minmax_rule_4_3(model):
            return model.m_4_1_minus <= self.M_set_decomp[0][1]*model.n_4_1
        
        def minmax_rule_4_4(model):
            return model.m_4_2 == model.m_4_2_plus - model.m_4_2_minus
        
        def minmax_rule_4_5(model):
            return model.m_4_2_plus <= self.M_set_decomp[1][0]*(1 - model.n_4_2)
        
        def minmax_rule_4_6(model):
            return model.m_4_2_minus <= self.M_set_decomp[1][1]*model.n_4_2
        
        def minmax_rule_4_7(model):
            return model.m_4_1_plus - model.m_4_2_plus <= self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4_2_plus - model.m_4_1_plus <= self.M_set_decomp[1][0]*model.n_4

        def minmax_rule_4_9(model):
            return model.m_4 == model.m_4_1_plus + model.l_4_2 - model.l_4_1
        
        def minmax_rule_4_10(model):
            return model.l_4_1 <= model.m_4_1_plus       
        
        def minmax_rule_4_11(model):
            return model.l_4_1 <= self.M_set_decomp[0][0]*model.n_4
        
        def minmax_rule_4_12(model):
            return model.l_4_1 >= model.m_4_1_plus - self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_13(model):
            return model.l_4_2 <= model.m_4_2_plus       
        
        def minmax_rule_4_14(model):
            return model.l_4_2 <= self.M_set_decomp[1][0]*model.n_4
        
        def minmax_rule_4_15(model):
            return model.l_4_2 >= model.m_4_2_plus - self.M_set_decomp[1][0]*(1 - model.n_4)
                

  
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)   
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S)
        model.auxiliary_T_b = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_b)
        model.auxiliary_T_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_T_Q)
        model.auxiliary_T_q = pyo.Constraint(rule = auxiliary_T_q)
        model.auxiliary_T_b_rt = pyo.Constraint(rule = auxiliary_T_b_rt)
        model.auxiliary_T_q_rt = pyo.Constraint(rule = auxiliary_T_q_rt)
        model.auxiliary_T_E = pyo.Constraint(rule = auxiliary_T_E)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        model.minmax_4_9 = pyo.Constraint(rule = minmax_rule_4_9)
        model.minmax_4_10 = pyo.Constraint(rule = minmax_rule_4_10)
        model.minmax_4_11 = pyo.Constraint(rule = minmax_rule_4_11)
        model.minmax_4_12 = pyo.Constraint(rule = minmax_rule_4_12)
        model.minmax_4_13 = pyo.Constraint(rule = minmax_rule_4_13)
        model.minmax_4_14 = pyo.Constraint(rule = minmax_rule_4_14)
        model.minmax_4_15 = pyo.Constraint(rule = minmax_rule_4_15)        
               
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)    
        

        
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
                return [3*3600000, 0, [0 for _ in range(len(self.T_b_prev))],
                        [0 for _ in range(len(self.T_Q_prev))], 0, 0, 0, 0]
        
        psi = []
        psi.append(pyo.value(self.objective))
        psi.append(self.dual[self.auxiliary_S])
        
        pi_T_b = []
        for i in range(len(self.T_b_prev)):
            pi_T_b.append(self.dual[self.auxiliary_T_b[i]])
        psi.append(pi_T_b)
        
        pi_T_Q = []
        for i in range(len(self.T_Q_prev)):
            pi_T_Q.append(self.dual[self.auxiliary_T_Q[i]])
        psi.append(pi_T_Q)
        
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_b_rt])
        psi.append(self.dual[self.auxiliary_T_q_rt])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi 

             
class fw_rt_last_Lagrangian(pyo.ConcreteModel): ## stage = T (Backward - Strengthened Benders' Cut)
           
    def __init__(self, pi, delta):
        
        super().__init__()

        self.solved = False

        self.pi = pi
        
        self.stage = T - 1

        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                              

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(bounds = (S_min, S_max), domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, bounds = (0, 2*B), domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals, bounds = (0, E_0_sum), initialize = 0.0)
        model.z_T_b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage] + B), domain = pyo.Reals)
        model.z_T_E = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage]), domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        
        # Experiment for LP relaxation
        """
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        """
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
    
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4   
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)   
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
    
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
               
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)    
          
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.f - (self.pi[0]*model.z_S + sum(self.pi[1][j]*model.z_T_b[j] for j in range(1)) + sum(self.pi[2][j]*model.z_T_Q[j] for j in range(1)) + self.pi[3]*model.z_T_q + self.pi[4]*model.z_T_b_rt + self.pi[5]*model.z_T_q_rt + self.pi[6]*model.z_T_E)
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
            [pyo.value(self.z_T_b[t]) for t in range(self.T - self.stage)],
            [pyo.value(self.z_T_Q[t]) for t in range(self.T - self.stage)],
            pyo.value(self.z_T_q),
            pyo.value(self.z_T_b_rt),
            pyo.value(self.z_T_q_rt),
            pyo.value(self.z_T_E)
        ]
        
        return z

class fw_rt_last_Lagrangian_Alt(pyo.ConcreteModel): 
           
    def __init__(self, pi, delta):
        
        super().__init__()

        self.solved = False

        self.pi = pi
        
        self.stage = T - 1

        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])            
    
    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.z_T_b_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(domain = pyo.Reals)
        model.z_T_E = pyo.Var(domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        #model.n_rt = pyo.Var(domain = pyo.Binary)

        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)  
        #model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)     
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)
        #model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)             
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        
        model.m_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        
        model.m_4_1_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_1_minus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_plus = pyo.Var(domain = pyo.NonNegativeReals)
        model.m_4_2_minus = pyo.Var(domain = pyo.NonNegativeReals)  
        model.l_4_1 = pyo.Var(domain = pyo.NonNegativeReals)
        model.l_4_2 = pyo.Var(domain = pyo.NonNegativeReals)
   
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_1 = pyo.Var(domain = pyo.Binary)
        model.n_4_2 = pyo.Var(domain = pyo.Binary)
        
        #model.n_3 = pyo.Var(bouns = (0, 1), domain = pyo.Reals)
        #model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3

        def minmax_rule_4_1(model):
            return model.m_4_1 == model.m_4_1_plus - model.m_4_1_minus
        
        def minmax_rule_4_2(model):
            return model.m_4_1_plus <= self.M_set_decomp[0][0]*(1 - model.n_4_1)
        
        def minmax_rule_4_3(model):
            return model.m_4_1_minus <= self.M_set_decomp[0][1]*model.n_4_1
        
        def minmax_rule_4_4(model):
            return model.m_4_2 == model.m_4_2_plus - model.m_4_2_minus
        
        def minmax_rule_4_5(model):
            return model.m_4_2_plus <= self.M_set_decomp[1][0]*(1 - model.n_4_2)
        
        def minmax_rule_4_6(model):
            return model.m_4_2_minus <= self.M_set_decomp[1][1]*model.n_4_2
        
        def minmax_rule_4_7(model):
            return model.m_4_1_plus - model.m_4_2_plus <= self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4_2_plus - model.m_4_1_plus <= self.M_set_decomp[1][0]*model.n_4

        def minmax_rule_4_9(model):
            return model.m_4 == model.m_4_1_plus + model.l_4_2 - model.l_4_1
        
        def minmax_rule_4_10(model):
            return model.l_4_1 <= model.m_4_1_plus       
        
        def minmax_rule_4_11(model):
            return model.l_4_1 <= self.M_set_decomp[0][0]*model.n_4
        
        def minmax_rule_4_12(model):
            return model.l_4_1 >= model.m_4_1_plus - self.M_set_decomp[0][0]*(1 - model.n_4)
        
        def minmax_rule_4_13(model):
            return model.l_4_2 <= model.m_4_2_plus       
        
        def minmax_rule_4_14(model):
            return model.l_4_2 <= self.M_set_decomp[1][0]*model.n_4
        
        def minmax_rule_4_15(model):
            return model.l_4_2 >= model.m_4_2_plus - self.M_set_decomp[1][0]*(1 - model.n_4)
        
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)  
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
        model.minmax_4_9 = pyo.Constraint(rule = minmax_rule_4_9)
        model.minmax_4_10 = pyo.Constraint(rule = minmax_rule_4_10)
        model.minmax_4_11 = pyo.Constraint(rule = minmax_rule_4_11)
        model.minmax_4_12 = pyo.Constraint(rule = minmax_rule_4_12)
        model.minmax_4_13 = pyo.Constraint(rule = minmax_rule_4_13)
        model.minmax_4_14 = pyo.Constraint(rule = minmax_rule_4_14)
        model.minmax_4_15 = pyo.Constraint(rule = minmax_rule_4_15)               
               
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)    
          
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.f - (self.pi[0]*model.z_S + sum(self.pi[1][j]*model.z_T_b[j] for j in range(1)) + sum(self.pi[2][j]*model.z_T_Q[j] for j in range(1)) + self.pi[3]*model.z_T_q + self.pi[4]*model.z_T_b_rt + self.pi[5]*model.z_T_q_rt + self.pi[6]*model.z_T_E)
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

### State Var Binarized version    

class fw_rt_last_bin(pyo.ConcreteModel): 
    
    def __init__(self, b_prev, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.b_S_prev = b_prev[0]
        self.b_b_prev = b_prev[1]
        self.b_Q_prev = b_prev[2]
        self.b_q_prev = b_prev[3]
        self.b_b_rt_prev = b_prev[4]
        self.b_q_rt_prev = b_prev[5]
        self.b_E_prev = b_prev[6]
        
        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self.Round = []
        
        self._BigM_setting()
        
        self._Bin_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                              
 
    def _Bin_setting(self):
        
        # S
        self.Round[0] = math.ceil(math.log2(S_max)) 
        # T_b
        self.Round[1] = math.ceil(math.log2(P_r))
        # T_Q
        self.Round[2] = math.ceil(math.log2(E_0_partial_max)) 
        # T_q
        self.Round[3] = math.ceil(math.log2(E_0_sum))
        # T_b_rt
        self.Round[4] = math.ceil(math.log2(P_r))
        # T_q_rt
        self.Round[5] = math.ceil(math.log2(1.2*E_0_partial_max + B))
        # T_E
        self.Round[6] = math.ceil(math.log2(1.2*E_0_partial_max))  
        
        self.bin_num =  math.ceil(math.log2(P_r) + 1)
 
    def build_model(self):
        
        model = self.model()
        
        model.S_ROUND = pyo.RangeSet(0, self.Round[0]) 
        model.b_ROUND = pyo.RangeSet(0, self.Round[1])
        model.Q_ROUND = pyo.RangeSet(0, self.Round[2])
        model.q_ROUND = pyo.RangeSet(0, self.Round[3])
        model.b_rt_ROUND = pyo.RangeSet(0, self.Round[4])
        model.q_rt_ROUND = pyo.RangeSet(0, self.Round[5])
        model.E_ROUND = pyo.RangeSet(0, self.Round[6])
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)    
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        
        # Experiment for LP relaxation
        
        """
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        """
        
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == -sum(self.b_b_prev[0][k]*(2**k) for k in range(self.Round[1] + 1))
        
        def da_Q_rule(model):
            return model.Q_da == sum(self.b_Q_prev[0][k]*(2**k) for k in range(self.Round[2] + 1))
        
        def rt_b_rule(model):
            return model.b_rt == -sum(self.b_b_rt_prev[k]*(2**k) for k in range(self.Round[4] + 1))
        
        def rt_q_rule(model):
            return model.q_rt == sum(self.b_q_rt_prev[k]*(2**k) for k in range(self.Round[5] + 1))
        
        def rt_E_rule(model):
            return model.E_1 == sum(self.b_E_prev[k]*(2**k) for k in range(self.Round[6] + 1))
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == sum(self.b_S_prev[k]*(2**k) for k in range(self.Round[0] + 1)) + v*model.c - (1/v)*model.d
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*self.b_b_rt_prev[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - self.b_b_rt_prev[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*self.b_b_prev[0][i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - self.b_b_prev[0][i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*self.b_b_prev[0][i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - self.b_b_prev[0][i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*self.b_b_rt_prev[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - self.b_b_rt_prev[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
             
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4
        
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
                     
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
       
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)
                         
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
 
class fw_rt_last_bin_LP_relax(pyo.ConcreteModel): ## stage = T (Backward)
           
    def __init__(self, b_prev, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.b_S_prev = b_prev[0]
        self.b_b_prev = b_prev[1]
        self.b_Q_prev = b_prev[2]
        self.b_q_prev = b_prev[3]
        self.b_b_rt_prev = b_prev[4]
        self.b_q_rt_prev = b_prev[5]
        self.b_E_prev = b_prev[6]
        
        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self.Round = []
        
        self._BigM_setting()

        self._Bin_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                             

    def _Bin_setting(self):
        
        # S
        self.Round[0] = math.ceil(math.log2(S_max)) 
        # T_b
        self.Round[1] = math.ceil(math.log2(P_r)) 
        # T_Q
        self.Round[2] = math.ceil(math.log2(E_0_partial_max)) 
        # T_q
        self.Round[3] = math.ceil(math.log2(E_0_sum))
        # T_b_rt
        self.Round[4] = math.ceil(math.log2(P_r))
        # T_q_rt
        self.Round[5] = math.ceil(math.log2(1.2*E_0_partial_max + B))
        # T_E
        self.Round[6] = math.ceil(math.log2(1.2*E_0_partial_max))    

        self.bin_num =  math.ceil(math.log2(P_r) + 1)

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
        
        model.S_ROUND = pyo.RangeSet(0, self.Round[0]) 
        model.b_ROUND = pyo.RangeSet(0, self.Round[1])
        model.Q_ROUND = pyo.RangeSet(0, self.Round[2])
        model.q_ROUND = pyo.RangeSet(0, self.Round[3])
        model.b_rt_ROUND = pyo.RangeSet(0, self.Round[4])
        model.q_rt_ROUND = pyo.RangeSet(0, self.Round[5])
        model.E_ROUND = pyo.RangeSet(0, self.Round[6]) 
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_b_S = pyo.Var(model.S_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_b = pyo.Var(model.Z_TIME, model.b_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_Q = pyo.Var(model.Z_TIME, model.Q_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_q = pyo.Var(model.q_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_b_rt = pyo.Var(model.b_rt_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_q_rt = pyo.Var(model.q_rt_ROUND, bounds = (0, 1), domain = pyo.Reals)
        model.z_b_E = pyo.Var(model.E_ROUND, bounds = (0, 1), domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        #model.n_rt = pyo.Var(domain = pyo.Binary)     
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## auxiliary variable z
        
        def auxiliary_S_rule(model, k):
            return model.z_b_S[k] == self.b_S_prev[k]
        
        def auxiliary_b_rule(model, t, k):
            return model.z_b_b[t][k] == self.b_b_prev[t][k]
        
        def auxiliary_Q_rule(model, t, k):
            return model.z_b_Q[t][k] == self.b_Q_prev[t][k]
        
        def auxiliary_q_rule(model, k):
            return model.z_b_q[k] == self.b_q_prev[k]
        
        def auxiliary_b_rt_rule(model, k):
            return model.z_b_b_rt[k] == self.b_b_rt_prev[k]
        
        def auxiliary_q_rt_rule(model, k):
            return model.z_b_q_rt[k] == self.b_q_rt_prev[k]
        
        def auxiliary_E_rule(model, k):
            return model.z_b_E[k] == self.b_E_prev[k]        
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == -sum(model.z_b_b[0, k]*(2**k) for k in range(self.Round[1] + 1))
        
        def da_Q_rule(model):
            return model.Q_da == sum(model.z_b_Q[0, k]*(2**k) for k in range(self.Round[2] + 1))
        
        def rt_b_rule(model):
            return model.b_rt == -sum(model.z_b_b_rt[k]*(2**k) for k in range(self.Round[4] + 1))
        
        def rt_q_rule(model):
            return model.q_rt == sum(model.z_b_q_rt[k]*(2**k) for k in range(self.Round[5] + 1))
        
        def rt_E_rule(model):
            return model.E_1 == sum(model.z_b_E[k]*(2**k) for k in range(self.Round[6] + 1))
        
        ## State Variable transition Constraints
        
        def State_SOC_rule(model):
            return model.S == sum(model.z_b_S[k]*(2**k) for k in range(self.Round[0] + 1)) + v*model.c - (1/v)*model.d
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.z_b_b_rt[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.z_b_b_rt[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.z_b_b[0][i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.z_b_b[0][i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.z_b_b[0][i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.z_b_b[0][i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.z_b_b_rt[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.z_b_b_rt[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
             
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4
  
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)  
        
        model.auxiliary_S = pyo.Constraint(rule = auxiliary_S_rule)
        model.auxiliary_b = pyo.Constraint(model.Z_TIME, rule = auxiliary_b_rule)
        model.auxiliary_Q = pyo.Constraint(model.Z_TIME, rule = auxiliary_Q_rule)
        model.auxiliary_q = pyo.Constraint(rule = auxiliary_q_rule)
        model.auxiliary_b_rt = pyo.Constraint(rule = auxiliary_b_rt_rule)
        model.auxiliary_q_rt = pyo.Constraint(rule = auxiliary_q_rt_rule)
        model.auxiliary_E = pyo.Constraint(rule = auxiliary_E_rule)
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
                     
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
    
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
               
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)    
        

        
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
                return [3*3600000, 0, [0 for _ in range(len(self.T_b_prev))],
                        [0 for _ in range(len(self.T_Q_prev))], 0, 0, 0, 0]
        
        psi = []
        psi.append(pyo.value(self.objective))
        psi.append(self.dual[self.auxiliary_S])
        
        pi_T_b = []
        for i in range(len(self.T_b_prev)):
            pi_T_b.append(self.dual[self.auxiliary_T_b[i]])
        psi.append(pi_T_b)
        
        pi_T_Q = []
        for i in range(len(self.T_Q_prev)):
            pi_T_Q.append(self.dual[self.auxiliary_T_Q[i]])
        psi.append(pi_T_Q)
        
        psi.append(self.dual[self.auxiliary_T_q])
        psi.append(self.dual[self.auxiliary_T_b_rt])
        psi.append(self.dual[self.auxiliary_T_q_rt])
        psi.append(self.dual[self.auxiliary_T_E])
        
        return psi  
    
class fw_rt_last_bin_Lagrangian(pyo.ConcreteModel): ## stage = T (Backward - Strengthened Benders' Cut)
           
    def __init__(self, pi, delta):
        
        super().__init__()

        self.solved = False

        self.pi = pi
        
        self.stage = T - 1

        self.P_da = P_da_partial
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_price = [0, 0]
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81
            self.M_set[0] = (160 - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (- self.P_da[t] - self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])                              

    def build_model(self):
        
        model = self.model()
        
        model.Z_TIME = pyo.RangeSet(0, 0)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars
        
        ## auxiliary variable z
        
        model.z_S = pyo.Var(bounds = (S_min, S_max), domain = pyo.Reals)
        model.z_T_b = pyo.Var(model.Z_TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.z_T_Q = pyo.Var(model.Z_TIME, bounds = (0, 2*B), domain = pyo.Reals)
        model.z_T_q = pyo.Var(domain = pyo.Reals, bounds = (0, E_0_sum), initialize = 0.0)
        model.z_T_b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.z_T_q_rt = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage] + B), domain = pyo.Reals)
        model.z_T_E = pyo.Var(bounds = (0, 1.2*E_0_partial[self.stage]), domain = pyo.Reals)
        
        ## Bidding for next and current stage
        
        model.b_da = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.Q_da = pyo.Var(domain = pyo.NonNegativeReals)
        model.b_rt = pyo.Var(bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.BINARIZE, domain = pyo.Reals)
        
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
        model.m_3 = pyo.Var(domain = pyo.Reals)
        model.m_4 = pyo.Var(domain = pyo.Reals)
        model.m_4_1 = pyo.Var(domain = pyo.Reals)
        model.m_4_2 = pyo.Var(domain = pyo.Reals)
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        
        # Experiment for LP relaxation
        """
        model.n_rt = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.lamb = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)        
        model.nu = pyo.Var(model.BINARIZE, bounds = (0, 1), domain = pyo.Reals)
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        """
        ## settlement_fcn
        
        model.f_prime = pyo.Var(domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(domain = pyo.Binary)
        
        model.f = pyo.Var(domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to t-1 state 
        
        def da_bidding_price_rule(model):
            return model.b_da == model.z_T_b[0]
        
        def da_Q_rule(model):
            return model.Q_da == model.z_T_Q[0]
        
        def rt_b_rule(model):
            return model.b_rt == model.z_T_b_rt
        
        def rt_q_rule(model):
            return model.q_rt == model.z_T_q_rt
        
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
         
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model):
            return model.b_da >= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model):
            return model.b_da <= -sum(model.lamb[i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model):
            return model.b_rt >= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model):
            return model.b_rt <= -sum(model.nu[j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, j):
            return model.w[j] >= 0
        
        def binarize_rule_1_2(model, j):
            return model.w[j] <= model.u
        
        def binarize_rule_1_3(model, j):
            return model.w[j] <= M_gen[self.stage][0]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage][0]*(1 - model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage][0]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage][0]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage][0]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage][0]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage][0]*(1 - model.nu[i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model):
            return model.m_4_1 == -sum(model.w[j]*(2**j) for j in range(self.bin_num)) - (model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt)
            
        def dummy_rule_4_2(model):
            return model.m_4_2 == (model.m_1 - model.m_2)*self.P_rt + sum((model.h[i] - model.k[i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model):
            return model.m_1 <= model.Q_da
        
        def minmax_rule_1_2(model):
            return model.m_1 <= model.q_rt
        
        def minmax_rule_1_3(model):
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage][0]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage][0]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage][0]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c + M_gen[self.stage][1]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage][0]*model.n_3
    
        def minmax_rule_4_1(model):
            return model.m_4_3 >= model.m_4_1
        
        def minmax_rule_4_2(model):
            return model.m_4_3 >= model.m_4_2
        
        def minmax_rule_4_3(model):
            return model.m_4_3 <= model.m_4_1 + self.M_set[0]*(1 - model.n_4_3)
        
        def minmax_rule_4_4(model):
            return model.m_4_3 <= model.m_4_2 + self.M_set[1]*model.n_4_3
        
        def minmax_rule_4_5(model):
            return model.m_4 >= model.m_4_3 
        
        def minmax_rule_4_6(model):
            return model.m_4 >= 0
        
        def minmax_rule_4_7(model):
            return model.m_4 <= self.M_set_decomp[2][0]*(1 - model.n_4)
        
        def minmax_rule_4_8(model):
            return model.m_4 <= model.m_4_3 + self.M_set_decomp[2][1]*model.n_4   
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u 
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage][1]*model.n_sum
        
        def settlement_fcn_rule_3(model):
            return model.Q_sum >= epsilon*model.n_sum
        
        def settlement_fcn_rule_4(model):
            return model.f >= 0
        
        def settlement_fcn_rule_5(model):
            return model.f <= model.f_prime
        
        def settlement_fcn_rule_6(model):
            return model.f <= M_set_fcn*model.n_sum
        
        def settlement_fcn_rule_7(model):
            return model.f >= model.f_prime - M_set_fcn*(1 - model.n_sum)   
        
        model.da_bidding_price = pyo.Constraint(rule = da_bidding_price_rule)
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
        
        model.binarize_b_da_1 = pyo.Constraint(rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(rule = minmax_rule_3_4)
    
        model.minmax_4_1 = pyo.Constraint(rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(rule = minmax_rule_4_8)
               
        model.settlement_fcn = pyo.Constraint(rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(rule = settlement_fcn_rule_7)    
          
        # Obj Fcn
        
        def objective_rule(model):
            return (
                model.f - (self.pi[0]*model.z_S + sum(self.pi[1][j]*model.z_T_b[j] for j in range(1)) + sum(self.pi[2][j]*model.z_T_Q[j] for j in range(1)) + self.pi[3]*model.z_T_q + self.pi[4]*model.z_T_b_rt + self.pi[5]*model.z_T_q_rt + self.pi[6]*model.z_T_E)
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
            [pyo.value(self.z_T_b[t]) for t in range(self.T - self.stage)],
            [pyo.value(self.z_T_Q[t]) for t in range(self.T - self.stage)],
            pyo.value(self.z_T_q),
            pyo.value(self.z_T_b_rt),
            pyo.value(self.z_T_q_rt),
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
        
        self.lamb = []
        self.k = []
    
        self._build_model()
    
    def _build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, self.T - self.stage - 1)
        
        # Vars
        
        model.theta = pyo.Var(domain = pyo.Reals)
        
        model.pi_S = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.pi_T_b = pyo.Var(model.TIME, domain = pyo.Reals, initialize = 0.0)
        model.pi_T_Q = pyo.Var(model.TIME, domain = pyo.Reals, initialize = 0.0)
        model.pi_T_q = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.pi_T_b_rt = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.pi_T_q_rt = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.pi_T_E = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        
        model.y_S = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.y_T_b = pyo.Var(model.TIME, domain = pyo.Reals, initialize = 0.0)
        model.y_T_Q = pyo.Var(model.TIME, domain = pyo.Reals, initialize = 0.0)
        model.y_T_q = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.y_T_b_rt = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.y_T_q_rt = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        model.y_T_E = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        
        model.t = pyo.Var(domain = pyo.Reals, initialize = 0.0)
        
        # Constraints
        
        def L_1_norm_S_1_rule(model):
            return model.pi_S <= model.y_S
        
        def L_1_norm_S_2_rule(model):
            return -model.pi_S <= model.y_S
        
        def L_1_norm_T_b_1_rule(model, t):
            return model.pi_T_b[t] <= model.y_T_b[t]
        
        def L_1_norm_T_b_2_rule(model, t):
            return -model.pi_T_b[t] <= model.y_T_b[t]
        
        def L_1_norm_T_Q_1_rule(model, t):
            return model.pi_T_Q[t] <= model.y_T_Q[t]
        
        def L_1_norm_T_Q_2_rule(model, t):
            return -model.pi_T_Q[t] <= model.y_T_Q[t]
        
        def L_1_norm_T_q_1_rule(model):
            return model.pi_T_q <= model.y_T_q
        
        def L_1_norm_T_q_2_rule(model):
            return -model.pi_T_q <= model.y_T_q
        
        def L_1_norm_T_b_rt_1_rule(model):
            return model.pi_T_b_rt <= model.y_T_b_rt
        
        def L_1_norm_T_b_rt_2_rule(model):
            return -model.pi_T_b_rt <= model.y_T_b_rt
        
        def L_1_norm_T_q_rt_1_rule(model):
            return model.pi_T_q_rt <= model.y_T_q_rt
        
        def L_1_norm_T_q_rt_2_rule(model):
            return -model.pi_T_q_rt <= model.y_T_q_rt
        
        def L_1_norm_T_E_1_rule(model):
            return model.pi_T_E <= model.y_T_E
        
        def L_1_norm_T_E_2_rule(model):
            return -model.pi_T_E <= model.y_T_E
        
        def L_1_norm_rule(model):
            return model.t == model.y_S + sum(model.y_T_b[t] for t in range(self.T - self.stage)) + sum(model.y_T_Q[t] for t in range(self.T - self.stage)) + model.y_T_q + model.y_T_b_rt + model.y_T_q_rt + model.y_T_E
        
        model.L_1_norm_S_1 = pyo.Constraint(rule = L_1_norm_S_1_rule)
        model.L_1_norm_S_2 = pyo.Constraint(rule = L_1_norm_S_2_rule)
        model.L_1_norm_T_b_1 = pyo.Constraint(model.TIME, rule = L_1_norm_T_b_1_rule)
        model.L_1_norm_T_b_2 = pyo.Constraint(model.TIME, rule = L_1_norm_T_b_2_rule)
        model.L_1_norm_T_Q_1 = pyo.Constraint(model.TIME, rule = L_1_norm_T_Q_1_rule)
        model.L_1_norm_T_Q_2 = pyo.Constraint(model.TIME, rule = L_1_norm_T_Q_2_rule)
        model.L_1_norm_T_q_1 = pyo.Constraint(rule = L_1_norm_T_q_1_rule)
        model.L_1_norm_T_q_2 = pyo.Constraint(rule = L_1_norm_T_q_2_rule)
        model.L_1_norm_T_b_rt_1 = pyo.Constraint(rule = L_1_norm_T_b_rt_1_rule)
        model.L_1_norm_T_b_rt_2 = pyo.Constraint(rule = L_1_norm_T_b_rt_2_rule)
        model.L_1_norm_T_q_rt_1 = pyo.Constraint(rule = L_1_norm_T_q_rt_1_rule)
        model.L_1_norm_T_q_rt_2 = pyo.Constraint(rule = L_1_norm_T_q_rt_2_rule)
        model.L_1_norm_T_E_1 = pyo.Constraint(rule = L_1_norm_T_E_1_rule)
        model.L_1_norm_T_E_2 = pyo.Constraint(rule = L_1_norm_T_E_2_rule)
        
        model.L_1_norm = pyo.Constraint(rule = L_1_norm_rule)
        
        def initialize_theta_rule(model):
            return model.theta >= 0
        
        model.initialize_theta = pyo.Constraint(rule = initialize_theta_rule)
        
        model.dual_fcn_approx = pyo.ConstraintList() 
        
        # Obj Fcn
        
        def objective_rule(model):
            return (
                # L-1 norm regularization :
                # model.theta + model.t
                
                # L-2 norm regularization :
                model.theta + self.reg*((model.pi_S - self.pi[0])**2 + sum((model.pi_T_b[t] - self.pi[1][t])**2 for t in range(self.T - self.stage)) + sum((model.pi_T_Q[t] - self.pi[2][t])**2 for t in range(self.T - self.stage)) + (model.pi_T_q - self.pi[3])**2 + (model.pi_T_b_rt - self.pi[4])**2 + (model.pi_T_q_rt - self.pi[5])**2 + (model.pi_T_E - self.pi[6])**2)
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.minimize)

    def add_plane(self, coeff):
        
        lamb = coeff[0]
        k = coeff[1]
                
        model = self.model()
        
        model.dual_fcn_approx.add(model.theta >= lamb + k[0]*model.pi_S + sum(k[1][t]*model.pi_T_b[t] for t in range(self.T - self.stage)) + sum(k[2][t]*model.pi_T_Q[t] for t in range(self.T - self.stage)) + k[3]*model.pi_T_q + k[4]*model.pi_T_b_rt + k[5]*model.pi_T_q_rt + k[6]*model.pi_T_E)
    
    def solve(self):
        
        SOLVER.solve(self)
        self.solved = True
        
    def get_solution_value(self):
        
        self.solve()
        self.solved = True

        pi = [
            pyo.value(self.pi_S),
            [pyo.value(self.pi_T_b[t]) for t in range(self.T - self.stage)],
            [pyo.value(self.pi_T_Q[t]) for t in range(self.T - self.stage)],
            pyo.value(self.pi_T_q),
            pyo.value(self.pi_T_b_rt),
            pyo.value(self.pi_T_q_rt),
            pyo.value(self.pi_T_E)
        ]
        
        return pi
    
    def get_objective_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True

        return pyo.value(self.theta) 
   
class dual_approx_lev(pyo.ConcreteModel): ## Level method
    
    def __init__(self, stage, coeff, pi, level):
        
        super().__init__()
        
        self.solved = False
        
        self.stage = stage
        
        self.coeff = coeff
        self.pi = pi
        
        self.level = level
        
        self.T = T
        
        self.lamb = []
        self.k = []
        
        self._build_model()

    def _build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, self.T - self.stage - 1)
        
        # Vars
            
        model.pi_S = pyo.Var(domain = pyo.Reals)
        model.pi_T_b = pyo.Var(model.TIME, domain = pyo.Reals)
        model.pi_T_Q = pyo.Var(model.TIME, domain = pyo.Reals)
        model.pi_T_q = pyo.Var(domain = pyo.Reals)
        model.pi_T_b_rt = pyo.Var(domain = pyo.Reals)
        model.pi_T_q_rt = pyo.Var(domain = pyo.Reals)
        model.pi_T_E = pyo.Var(domain = pyo.Reals)

        # Constraints
        
        model.lev = pyo.Param(mutable = True, initialize = 0.0)
        
        model.dual_fcn_approx = pyo.ConstraintList() 
        
        # Obj Fcn
        
        def objective_rule(model):
            return (
                (model.pi_S - self.pi[0])**2 + sum((model.pi_T_b[t] - self.pi[1][t])**2 for t in range(self.T - self.stage)) + sum((model.pi_T_Q[t] - self.pi[2][t])**2 for t in range(self.T - self.stage)) + (model.pi_T_q - self.pi[3])**2 + (model.pi_T_b_rt - self.pi[4])**2 + (model.pi_T_q_rt - self.pi[5])**2 + (model.pi_T_E - self.pi[6])**2
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.minimize)
        
    def add_plane(self, coeff):
        
        lamb = coeff[0]
        k = coeff[1]
                
        model = self.model()
        
        model.dual_fcn_approx.add(model.lev >= lamb + k[0]*model.pi_S + sum(k[1][t]*model.pi_T_b[t] for t in range(self.T - self.stage)) + sum(k[2][t]*model.pi_T_Q[t] for t in range(self.T - self.stage)) + k[3]*model.pi_T_q + k[4]*model.pi_T_b_rt + k[5]*model.pi_T_q_rt + k[6]*model.pi_T_E)
    
    def solve(self):
        
        model = self.model()
        model.lev.set_value(self.level)
        print(f"level = {pyo.value(self.lev)}")
        SOLVER.solve(self)
        self.solved = True
        
    def get_solution_value(self):
        
        self.solve()
        self.solved = True

        pi = [
            pyo.value(self.pi_S),
            [pyo.value(self.pi_T_b[t]) for t in range(self.T - self.stage)],
            [pyo.value(self.pi_T_Q[t]) for t in range(self.T - self.stage)],
            pyo.value(self.pi_T_q),
            pyo.value(self.pi_T_b_rt),
            pyo.value(self.pi_T_q_rt),
            pyo.value(self.pi_T_E)
        ]
        
        return pi
    
    def get_objective_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True

        return pyo.value(self.objective) 
       
    
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

        
# T Stage Deterministic Equivalent Problem.

class T_stage_DEF_BigM(pyo.ConcreteModel):
    
    def __init__(self, scenario_tree):
        
        super().__init__()

        self.solved = False
        self.result = None
        
        self.scenario_tree = scenario_tree
        self.scenarios = self.scenario_tree.scenarios()
        self.num_scenarios = len(self.scenarios)
        self.node_scenarios = self.scenario_tree.node_scenarios()
        
        self.E_0 = E_0_partial
        self.P_da = P_da_partial
        
        self.delta_E = [[self.scenarios[k][t].param[0] for t in range(T)] for k in range(self.num_scenarios)]
        self.P_rt = [[self.scenarios[k][t].param[1] for t in range(T)] for k in range(self.num_scenarios)]
        self.delta_c = [[self.scenarios[k][t].param[2] for t in range(T)] for k in range(self.num_scenarios)]
        
        self.bin_num = 7
        
        self.T = T
        
        self.M_price = [
            [[[0, 0], [0, 0]] for t in range(self.T)] for k in range(self.num_scenarios)
        ]
        self.M_set = [
            [[0, 0] for t in range(self.T)] for k in range(self.num_scenarios)
        ]
        
        self.M_set_decomp = [
            [[[0, 0] for i in range(3)] for t in range(self.T)] for k in range(self.num_scenarios)
        ]
        
        self._BigM_setting()
 
        
    def _BigM_setting(self):
        
        for k in range(self.num_scenarios):
            
            for t in range(T):
                
                if self.P_da[t] >=0 and self.P_rt[k][t] >= 0:
                    
                    self.M_price[k][t][0][0] = 0
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = 0
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
                    self.M_set[k][t][0] = (160 + self.P_da[t] + 2*self.P_rt[k][t])*K[t]
                    self.M_set[k][t][1] = (80 + 2*self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][0] = (self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][1] = (80 + self.P_da[t] + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][0] = (self.P_rt[k][t] + 80)*K[t]
                    self.M_set_decomp[k][t][1][1] = (self.P_rt[k][t] + 80)*K[t]
                    self.M_set_decomp[k][t][2][0] = max(self.M_set_decomp[k][t][0][0], self.M_set_decomp[k][t][1][0])
                    self.M_set_decomp[k][t][2][1] = min(self.M_set_decomp[k][t][0][1], self.M_set_decomp[k][t][1][1])
        
                elif self.P_da[t] >=0 and self.P_rt[k][t] < 0:            
                    
                    self.M_price[k][t][0][0] = 0
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = -self.P_rt[k][t]
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
                    self.M_set[k][t][0] = (160 + self.P_da[t] - 2*self.P_rt[k][t])*K[t]
                    self.M_set[k][t][1] = (80 - 2*self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][0] = (-self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][1] = (80 + self.P_da[t] - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][0] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][1] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][2][0] = max(self.M_set_decomp[k][t][0][0], self.M_set_decomp[k][t][1][0])
                    self.M_set_decomp[k][t][2][1] = min(self.M_set_decomp[k][t][0][1], self.M_set_decomp[k][t][1][1])
        
                elif self.P_da[t] < 0 and self.P_rt[k][t] >= 0:
                    
                    self.M_price[k][t][0][0] = -self.P_da[t]
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = 0
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
                    self.M_set[k][t][0] = (160 + 2*self.P_rt[k][t])*K[t]
                    self.M_set[k][t][1] = (80 - self.P_da[t] + 2*self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][0] = (-self.P_da[t] + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][1] = (80 + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][0] = (80 + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][1] = (80 + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][2][0] = max(self.M_set_decomp[k][t][0][0], self.M_set_decomp[k][t][1][0])
                    self.M_set_decomp[k][t][2][1] = min(self.M_set_decomp[k][t][0][1], self.M_set_decomp[k][t][1][1])
        
                else:
                    
                    self.M_price[k][t][0][0] = -self.P_da[t]
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = -self.P_rt[k][t]
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
                    self.M_set[k][t][0] = (160 - 2*self.P_rt[k][t])*K[t]
                    self.M_set[k][t][1] = (80 - 2*self.P_da[t] - 2*self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][0] = (- self.P_da[t] - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][1] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][0] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][1] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][2][0] = max(self.M_set_decomp[k][t][0][0], self.M_set_decomp[k][t][1][0])
                    self.M_set_decomp[k][t][2][1] = min(self.M_set_decomp[k][t][0][1], self.M_set_decomp[k][t][1][1])                      
    
        
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T - 1)

        model.S_TIME = pyo.RangeSet(-1, T - 1)
        
        model.SCENARIO = pyo.RangeSet(0, self.num_scenarios - 1)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)

        # Vars
        
        ## day ahead
        
        model.b_da = pyo.Var(model.SCENARIO, model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_da = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
                
        ## real time
        
        ### Bidding & Market clearing
        
        model.b_rt = pyo.Var(model.SCENARIO, model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        ### Real-Time operation 
        
        model.g = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.SCENARIO, model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.SCENARIO, model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
                
        model.S = pyo.Var(model.SCENARIO, model.S_TIME, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        ### min, max reformulation Vars
        
        model.m_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        
        model.n_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_3 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_4_3 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        
        ### settlement_fcn_Vars
        
        model.f_prime = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        
                   
        # Constraints
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, k, t):
            return model.q_da[k, t] <= self.E_0[t] + B
        
        def da_overbid_rule(model, k):
            return sum(model.q_da[k, t] for t in range(self.T)) <= E_0_sum
        
        def da_market_clearing_1_rule(model, k, t):
            return model.b_da[k, t] - self.P_da[t] <= self.M_price[k][t][0][0]*(1 - model.n_da[k, t])
        
        def da_market_clearing_2_rule(model, k, t):
            return self.P_da[t] - model.b_da[k, t] <= self.M_price[k][t][0][1]*model.n_da[k, t]
        
        def da_market_clearing_3_rule(model, k, t):
            return model.Q_da[k, t] <= model.q_da[k, t]
        
        def da_market_clearing_4_rule(model, k, t):
            return model.Q_da[k, t] <= M_gen[t][0]*model.n_da[k, t]
        
        def da_market_clearing_5_rule(model, k, t):
            return model.Q_da[k, t] >= model.q_da[k, t] - M_gen[t][0]*(1 - model.n_da[k, t])        
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_price_rule(model, k, t):
            return model.b_rt[k, t] <= model.b_da[k, t]
        
        def rt_bidding_amount_rule_0(model, k):
            return model.q_rt[k, 0] <= B
        
        model.rt_bidding_amount = pyo.ConstraintList()
        
        for k in range(self.num_scenarios):
            
            for t in range(1, T):
                
                model.rt_bidding_amount.add(model.q_rt[k, t] <= B + self.E_0[t]*self.delta_E[k][t-1])
        
        #def rt_bidding_amount_rule_1(model, k):
        #    return model.q_rt[k, 1] <= B + self.E_0[1]*self.delta_E[k][0]
        
        #def rt_bidding_amount_rule_2(model, k):
        #    return model.q_rt[k, 2] <= B + self.E_0[2]*self.delta_E[k][1]
        
        #def rt_bidding_amount_rule_3(model, k):
        #    return model.q_rt[k, 3] <= B + self.E_0[3]*self.delta_E[k][2]  
        
        def rt_overbid_rule(model, k):
            return sum(model.q_rt[k, t] for t in range(self.T)) <= E_0_sum
        
        def SOC_initial_rule(model, k):
            return model.S[k, -1] == 0.5*S
        
        def SOC_rule(model, k, t):
            return model.S[k, t] == model.S[k, t - 1] + v*model.c[k, t] - (1/v)*model.d[k, t]
        
        def generation_rule_0(model, k):
            return model.g[k, 0] <= 0

        model.generation = pyo.ConstraintList()
        
        for k in range(self.num_scenarios):
                
            for t in range(1, T):
                
                model.generation.add(model.g[k, t] <= self.E_0[t]*self.delta_E[k][t-1])

        #def generation_rule_1(model, k):
        #    return model.g[k, 1] <= self.E_0[1]*self.delta_E[k][0]
        
        #def generation_rule_2(model, k):
        #    return model.g[k, 2] <= self.E_0[2]*self.delta_E[k][1]
        
        #def generation_rule_3(model, k):
        #    return model.g[k, 3] <= self.E_0[3]*self.delta_E[k][2]
        
        def charge_rule(model, k, t):
            return model.c[k, t] <= model.g[k, t]
        
        def electricity_supply_rule(model, k, t):
            return model.u[k, t] == model.g[k, t] + model.d[k, t] - model.c[k, t]
        
        def rt_market_clearing_rule_1(model, k, t):
            return model.b_rt[k, t] - self.P_rt[k][t] <= self.M_price[k][t][1][0]*(1 - model.n_rt[k, t])
        
        def rt_market_clearing_rule_2(model, k, t):
            return self.P_rt[k][t] - model.b_rt[k, t] <= self.M_price[k][t][1][1]*model.n_rt[k, t] 
        
        def rt_market_clearing_rule_3(model, k, t):
            return model.Q_rt[k, t] <= model.q_rt[k, t]
        
        def rt_market_clearing_rule_4(model, k, t):
            return model.Q_rt[k, t] <= M_gen[t][0]*model.n_rt[k, t]
        
        def rt_market_clearing_rule_5(model, k, t):
            return model.Q_rt[k, t] >= model.q_rt[k, t] - M_gen[t][0]*(1 - model.n_rt[k, t])
        
        def dispatch_rule(model, k, t):
            return model.Q_c[k, t] == (1 + self.delta_c[k][t])*model.Q_rt[k, t]
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
                
        def binarize_b_da_rule_1(model, k, t):
            return model.b_da[k, t] >= -sum(model.lamb[k, t, i]*(2**i) for i in range(self.bin_num)) - 0.5

        def binarize_b_da_rule_2(model, k, t):
            return model.b_da[k, t] <= -sum(model.lamb[k, t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, k, t):
            return model.b_rt[k, t] >= -sum(model.nu[k, t, j]*(2**j) for j in range(self.bin_num)) - 0.5

        def binarize_b_rt_rule_2(model, k, t):
            return model.b_rt[k, t] <= -sum(model.nu[k, t, j]*(2**j) for j in range(self.bin_num)) + 0.5

        def binarize_rule_1_1(model, k, t, j):
            return model.w[k, t, j] >= 0

        def binarize_rule_1_2(model, k, t, j):
            return model.w[k, t, j] <= model.u[k, t]

        def binarize_rule_1_3(model, k, t, j):
            return model.w[k, t, j] <= M_gen[t][0]*model.nu[k, t, j]

        def binarize_rule_1_4(model, k, t, j):
            return model.w[k, t, j] >= model.u[k, t] - M_gen[t][0]*(1 - model.nu[k, t, j])

        def binarize_rule_2_1(model, k, t, i):
            return model.h[k, t, i] >= 0

        def binarize_rule_2_2(model, k, t, i):
            return model.h[k, t, i] <= model.m_1[k, t]

        def binarize_rule_2_3(model, k, t, i):
            return model.h[k, t, i] <= M_gen[t][0]*model.lamb[k, t, i]

        def binarize_rule_2_4(model, k, t, i):
            return model.h[k, t, i] >= model.m_1[k, t] - M_gen[t][0]*(1 - model.lamb[k, t, i])

        def binarize_rule_3_1(model, k, t, i):
            return model.k[k, t, i] >= 0

        def binarize_rule_3_2(model, k, t, i):
            return model.k[k, t, i] <= model.m_2[k, t]

        def binarize_rule_3_3(model, k, t, i):
            return model.k[k, t, i] <= M_gen[t][0]*model.lamb[k, t, i]

        def binarize_rule_3_4(model, k, t, i):
            return model.k[k, t, i] >= model.m_2[k, t] - M_gen[t][0]*(1 - model.lamb[k, t, i])

        def binarize_rule_4_1(model, k, t, i):
            return model.o[k, t, i] >= 0

        def binarize_rule_4_2(model, k, t, i):
            return model.o[k, t, i] <= model.m_3[k, t]

        def binarize_rule_4_3(model, k, t, i):
            return model.o[k, t, i] <= M_gen[t][0]*model.nu[k, t, i]

        def binarize_rule_4_4(model, k, t, i):
            return model.o[k, t, i] >= model.m_3[k, t] - M_gen[t][0]*(1 - model.nu[k, t, i])      
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, k, t):
            return model.m_4_1[k, t] == -sum(model.w[k, t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[k, t]*P_da[t] + (model.u[k, t] - model.Q_da[k, t])*self.P_rt[k][t])
            
        def dummy_rule_4_2(model, k, t):
            return model.m_4_2[k, t] == (model.m_1[k, t] - model.m_2[k, t])*self.P_rt[k][t] + sum((model.h[k, t, i] - model.k[k, t, i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model, k, t):
            return model.m_1[k, t] <= model.Q_da[k, t]
        
        def minmax_rule_1_2(model, k, t):
            return model.m_1[k, t] <= model.q_rt[k, t]
        
        def minmax_rule_1_3(model, k, t):
            return model.m_1[k, t] >= model.Q_da[k, t] - (1 - model.n_1[k, t])*M_gen[t][0]
        
        def minmax_rule_1_4(model, k, t):
            return model.m_1[k, t] >= model.q_rt[k, t] - (model.n_1[k, t])*M_gen[t][0]
        
        def minmax_rule_2_1(model, k, t):
            return model.m_2[k, t] <= model.u[k, t]
        
        def minmax_rule_2_2(model, k, t):
            return model.m_2[k, t] <= model.q_rt[k, t]
        
        def minmax_rule_2_3(model, k, t):
            return model.m_2[k, t] >= model.u[k, t] - (1 - model.n_2[k, t])*M_gen[t][0]
        
        def minmax_rule_2_4(model, k, t):
            return model.m_2[k, t] >= model.q_rt[k, t] - model.n_2[k, t]*M_gen[t][0]
        
        def minmax_rule_3_1(model, k, t):
            return model.m_3[k, t] >= model.u[k, t] - model.Q_c[k, t]
        
        def minmax_rule_3_2(model, k, t):
            return model.m_3[k, t] >= 0
        
        def minmax_rule_3_3(model, k, t):
            return model.m_3[k, t] <= model.u[k, t] - model.Q_c[k, t] + M_gen[t][1]*(1 - model.n_3[k, t])
        
        def minmax_rule_3_4(model, k, t):
            return model.m_3[k, t] <= M_gen[t][0]*model.n_3[k, t]
        
        def minmax_rule_4_1(model, k, t):
            return model.m_4_3[k, t] >= model.m_4_1[k, t]
        
        def minmax_rule_4_2(model, k, t):
            return model.m_4_3[k, t] >= model.m_4_2[k, t]
        
        def minmax_rule_4_3(model, k, t):
            return model.m_4_3[k, t] <= model.m_4_1[k, t] + self.M_set[k][t][0]*(1 - model.n_4_3[k, t])
        
        def minmax_rule_4_4(model, k, t):
            return model.m_4_3[k, t] <= model.m_4_2[k, t] + self.M_set[k][t][1]*model.n_4_3[k, t]
        
        def minmax_rule_4_5(model, k, t):
            return model.m_4[k, t] >= model.m_4_3[k, t] 
        
        def minmax_rule_4_6(model, k, t):
            return model.m_4[k, t] >= 0
        
        def minmax_rule_4_7(model, k, t):
            return model.m_4[k, t] <= self.M_set_decomp[k][t][2][0]*(1 - model.n_4[k, t])
        
        def minmax_rule_4_8(model, k, t):
            return model.m_4[k, t] <= model.m_4_3[k, t] + self.M_set_decomp[k][t][2][1]*model.n_4[k, t]     
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, k, t):
            return model.f_prime[k, t] == model.Q_da[k, t]*P_da[t] + (model.u[k, t] - model.Q_da[k, t])*self.P_rt[k][t] + self.m_4[k, t] - self.P_rt[k][t]*model.m_3[k, t] - sum((2**j)*model.o[k, t, j] for j in range(self.bin_num)) + P_r*model.u[k, t] 
        
        def settlement_fcn_rule_1(model, k, t):
            return model.Q_sum[k, t] == model.Q_da[k, t] + model.Q_rt[k, t]
        
        def settlement_fcn_rule_2(model, k, t):
            return model.Q_sum[k, t] <= M_gen[t][1]*model.n_sum[k, t]
        
        def settlement_fcn_rule_3(model, k, t):
            return model.Q_sum[k, t] >= epsilon*model.n_sum[k, t]
        
        def settlement_fcn_rule_4(model, k, t):
            return model.f[k, t] >= 0
        
        def settlement_fcn_rule_5(model, k, t):
            return model.f[k, t] <= model.f_prime[k, t]
        
        def settlement_fcn_rule_6(model, k, t):
            return model.f[k, t] <= M_set_fcn*model.n_sum[k, t]
        
        def settlement_fcn_rule_7(model, k, t):
            return model.f[k, t] >= model.f_prime[k, t] - M_set_fcn*(1 - model.n_sum[k, t])        


        model.da_bidding_amount = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(model.SCENARIO, rule = da_overbid_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_5_rule)

        model.rt_bidding_price = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount_0 = pyo.Constraint(model.SCENARIO, rule = rt_bidding_amount_rule_0)

        model.rt_overbid = pyo.Constraint(model.SCENARIO, rule = rt_overbid_rule)
        model.SOC_initial = pyo.Constraint(model.SCENARIO, rule = SOC_initial_rule)
        model.SOC = pyo.Constraint(model.SCENARIO, model.TIME, rule = SOC_rule)
        model.generation_0 = pyo.Constraint(model.SCENARIO, rule = generation_rule_0)
        
        model.charge = pyo.Constraint(model.SCENARIO, model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.SCENARIO, model.TIME, rule = electricity_supply_rule)
        model.rt_market_clearing_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_5)

        model.dispatch = pyo.Constraint(model.SCENARIO, model.TIME, rule = dispatch_rule)

        model.binarize_b_da_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_8)
        
        model.settlement_fcn = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_7)

        ## Non-anticipativ
        
        
        model.NonAnticipativity = pyo.ConstraintList()
              
        for k in range(1, self.num_scenarios):
            
            for t in range(self.T):
                
                model.NonAnticipativity.add(model.b_da[k, t] == model.b_da[0, t])
                model.NonAnticipativity.add(model.q_da[k, t] == model.q_da[0, t])
                #model.NonAnticipativity.add(model.Q_da[k, t] == model.Q_da[0, t])
                #model.NonAnticipativity.add(model.n_da[k, t] == model.n_da[0, t])
                
            model.NonAnticipativity.add(model.b_rt[k, 0] == model.b_rt[0, 0])
            model.NonAnticipativity.add(model.q_rt[k, 0] == model.q_rt[0, 0])
           
        
        for (t, node_id), kset in self.node_scenarios.items():
            
            if len(kset) > 1:
                
                klist = list(kset)
                base = klist[0]
                
                for k in klist[1:]:
                    
                    model.NonAnticipativity.add(model.b_rt[k, t+1] == model.b_rt[base, t+1])
                    model.NonAnticipativity.add(model.q_rt[k, t+1] == model.q_rt[base, t+1])
                    model.NonAnticipativity.add(model.Q_rt[k, t] == model.Q_rt[base, t])
                    model.NonAnticipativity.add(model.n_rt[k, t] == model.n_rt[base, t])
                    
                    for i in range(self.bin_num):
                        #model.NonAnticipativity.add(model.lamb[k, t, i] == model.lamb[base, t, i])
                        #model.NonAnticipativity.add(model.nu[k, t, i] == model.nu[base, t, i])
                        #model.NonAnticipativity.add(model.w[k, t, i] == model.w[base, t, i])
                        #model.NonAnticipativity.add(model.h[k, t, i] == model.h[base, t, i])
                        #model.NonAnticipativity.add(model.k[k, t, i] == model.k[base, t, i])
                        #model.NonAnticipativity.add(model.o[k, t, i] == model.o[base, t, i])
                        pass
                        
                    model.NonAnticipativity.add(model.g[k, t] == model.g[base, t])
                    model.NonAnticipativity.add(model.c[k, t] == model.c[base, t])
                    model.NonAnticipativity.add(model.d[k, t] == model.d[base, t])
                    model.NonAnticipativity.add(model.u[k, t] == model.u[base, t])
                    #model.NonAnticipativity.add(model.Q_c[k, t] == model.Q_c[base, t])
                    #model.NonAnticipativity.add(model.S[k, t] == model.S[base, t])
                    #model.NonAnticipativity.add(model.m_1[k, t] == model.m_1[base, t])
                    #model.NonAnticipativity.add(model.m_2[k, t] == model.m_2[base, t])
                    #model.NonAnticipativity.add(model.m_3[k, t] == model.m_3[base, t])
                    #model.NonAnticipativity.add(model.m_4[k, t] == model.m_4[base, t])
                    #model.NonAnticipativity.add(model.m_4_1[k, t] == model.m_4_1[base, t])
                    #model.NonAnticipativity.add(model.m_4_2[k, t] == model.m_4_2[base, t])
                    #model.NonAnticipativity.add(model.m_4_3[k, t] == model.m_4_3[base, t])
                    #model.NonAnticipativity.add(model.n_1[k, t] == model.n_1[base, t])
                    #model.NonAnticipativity.add(model.n_2[k, t] == model.n_2[base, t])
                    #model.NonAnticipativity.add(model.n_3[k, t] == model.n_3[base, t])
                    #model.NonAnticipativity.add(model.n_4[k, t] == model.n_4[base, t])
                    #model.NonAnticipativity.add(model.n_4_1[k, t] == model.n_4_1[base, t])
                    #model.NonAnticipativity.add(model.n_4_2[k, t] == model.n_4_2[base, t])
                    #model.NonAnticipativity.add(model.n_4_3[k, t] == model.n_4_3[base, t])
                    #model.NonAnticipativity.add(model.f_prime[k, t] == model.f_prime[base, t])
                    #model.NonAnticipativity.add(model.Q_sum[k, t] == model.Q_sum[base, t])
                    #model.NonAnticipativity.add(model.n_sum[k, t] == model.n_sum[base, t])
                    #model.NonAnticipativity.add(model.f[k, t] == model.f[base, t])
        
        
        def objective_rule(model):
            return (
                (1/self.num_scenarios)*sum(sum(model.f[k, t] for t in range(T)) for k in range(self.num_scenarios))
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)
    
    
    def solve(self):
        
        self.build_model()
        self.result = SOLVER.solve(self, tee = True)
        print(f"{self.T} stage DEF solved")
        self.solved = True
    
    
    def get_objective_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)

class T_stage_DEF_Alt(pyo.ConcreteModel):
    
    def __init__(self, scenario_tree):
        
        super().__init__()

        self.solved = False
        self.result = None
        
        self.scenario_tree = scenario_tree
        self.scenarios = self.scenario_tree.scenarios()
        self.num_scenarios = len(self.scenarios)
        self.node_scenarios = self.scenario_tree.node_scenarios()
        
        self.E_0 = E_0_partial
        self.P_da = P_da_partial
        
        self.delta_E = [[self.scenarios[k][t].param[0] for t in range(T)] for k in range(self.num_scenarios)]
        self.P_rt = [[self.scenarios[k][t].param[1] for t in range(T)] for k in range(self.num_scenarios)]
        self.delta_c = [[self.scenarios[k][t].param[2] for t in range(T)] for k in range(self.num_scenarios)]
        
        self.bin_num = 7
        
        self.T = T
        
        self.M_price = [
            [[[0, 0], [0, 0]] for t in range(self.T)] for k in range(self.num_scenarios)
        ]
        self.M_set = [
            [[0, 0] for t in range(self.T)] for k in range(self.num_scenarios)
        ]
        
        self.M_set_decomp = [
            [[[0, 0] for i in range(3)] for t in range(self.T)] for k in range(self.num_scenarios)
        ]
        
        self._BigM_setting()
 
        
    def _BigM_setting(self):
        
        for k in range(self.num_scenarios):
            
            for t in range(T):
                
                if self.P_da[t] >=0 and self.P_rt[k][t] >= 0:
                    
                    self.M_price[k][t][0][0] = 0
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = 0
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
                    self.M_set[k][t][0] = (160 + self.P_da[t] + 2*self.P_rt[k][t])*K[t]
                    self.M_set[k][t][1] = (80 + 2*self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][0] = (self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][1] = (80 + self.P_da[t] + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][0] = (self.P_rt[k][t] + 80)*K[t]
                    self.M_set_decomp[k][t][1][1] = (self.P_rt[k][t] + 80)*K[t]
                    self.M_set_decomp[k][t][2][0] = max(self.M_set_decomp[k][t][0][0], self.M_set_decomp[k][t][1][0])
                    self.M_set_decomp[k][t][2][1] = min(self.M_set_decomp[k][t][0][1], self.M_set_decomp[k][t][1][1])
        
                elif self.P_da[t] >=0 and self.P_rt[k][t] < 0:            
                    
                    self.M_price[k][t][0][0] = 0
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = -self.P_rt[k][t]
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
                    self.M_set[k][t][0] = (160 + self.P_da[t] - 2*self.P_rt[k][t])*K[t]
                    self.M_set[k][t][1] = (80 - 2*self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][0] = (-self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][1] = (80 + self.P_da[t] - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][0] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][1] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][2][0] = max(self.M_set_decomp[k][t][0][0], self.M_set_decomp[k][t][1][0])
                    self.M_set_decomp[k][t][2][1] = min(self.M_set_decomp[k][t][0][1], self.M_set_decomp[k][t][1][1])
        
                elif self.P_da[t] < 0 and self.P_rt[k][t] >= 0:
                    
                    self.M_price[k][t][0][0] = -self.P_da[t]
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = 0
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
                    self.M_set[k][t][0] = (160 + 2*self.P_rt[k][t])*K[t]
                    self.M_set[k][t][1] = (80 - self.P_da[t] + 2*self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][0] = (-self.P_da[t] + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][1] = (80 + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][0] = (80 + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][1] = (80 + self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][2][0] = max(self.M_set_decomp[k][t][0][0], self.M_set_decomp[k][t][1][0])
                    self.M_set_decomp[k][t][2][1] = min(self.M_set_decomp[k][t][0][1], self.M_set_decomp[k][t][1][1])
        
                else:
                    
                    self.M_price[k][t][0][0] = -self.P_da[t]
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = -self.P_rt[k][t]
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
                    self.M_set[k][t][0] = (160 - 2*self.P_rt[k][t])*K[t]
                    self.M_set[k][t][1] = (80 - 2*self.P_da[t] - 2*self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][0] = (- self.P_da[t] - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][0][1] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][0] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][1][1] = (80 - self.P_rt[k][t])*K[t]
                    self.M_set_decomp[k][t][2][0] = max(self.M_set_decomp[k][t][0][0], self.M_set_decomp[k][t][1][0])
                    self.M_set_decomp[k][t][2][1] = min(self.M_set_decomp[k][t][0][1], self.M_set_decomp[k][t][1][1])                         
        
        
    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T - 1)

        model.S_TIME = pyo.RangeSet(-1, T - 1)
        
        model.SCENARIO = pyo.RangeSet(0, self.num_scenarios - 1)
        
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)

        # Vars
        
        ## day ahead
        
        model.b_da = pyo.Var(model.SCENARIO, model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_da = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
                
        ## real time
        
        ### Bidding & Market clearing
        
        model.b_rt = pyo.Var(model.SCENARIO, model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.q_rt = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_rt = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.SCENARIO, model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        ### Real-Time operation 
        
        model.g = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.SCENARIO, model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.SCENARIO, model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
                
        model.S = pyo.Var(model.SCENARIO, model.S_TIME, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        ### min, max reformulation Vars
        
        model.m_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.m_3 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        
        model.m_4_1_plus = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.m_4_1_minus = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.m_4_2_plus = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.m_4_2_minus = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)  
        model.l_4_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.l_4_2 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
   
        model.n_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_3 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_4_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        model.n_4_2 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        
        ### settlement_fcn_Vars
        
        model.f_prime = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Reals)
        
                   
        # Constraints
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, k, t):
            return model.q_da[k, t] <= self.E_0[t] + B
        
        def da_overbid_rule(model, k):
            return sum(model.q_da[k, t] for t in range(self.T)) <= E_0_sum
        
        def da_market_clearing_1_rule(model, k, t):
            return model.b_da[k, t] - self.P_da[t] <= self.M_price[k][t][0][0]*(1 - model.n_da[k, t])
        
        def da_market_clearing_2_rule(model, k, t):
            return self.P_da[t] - model.b_da[k, t] <= self.M_price[k][t][0][1]*model.n_da[k, t]
        
        def da_market_clearing_3_rule(model, k, t):
            return model.Q_da[k, t] <= model.q_da[k, t]
        
        def da_market_clearing_4_rule(model, k, t):
            return model.Q_da[k, t] <= M_gen[t][0]*model.n_da[k, t]
        
        def da_market_clearing_5_rule(model, k, t):
            return model.Q_da[k, t] >= model.q_da[k, t] - M_gen[t][0]*(1 - model.n_da[k, t])        
        
        ## Real-Time Market rules(especially for t = 0)
        
        def rt_bidding_price_rule(model, k, t):
            return model.b_rt[k, t] <= model.b_da[k, t]
        
        def rt_bidding_amount_rule_0(model, k):
            return model.q_rt[k, 0] <= B
        
        model.rt_bidding_amount = pyo.ConstraintList()
        
        for k in range(self.num_scenarios):
            
            for t in range(1, T):
                
                model.rt_bidding_amount.add(model.q_rt[k, t] <= B + self.E_0[t]*self.delta_E[k][t-1])
        
        #def rt_bidding_amount_rule_1(model, k):
        #    return model.q_rt[k, 1] <= B + self.E_0[1]*self.delta_E[k][0]
        
        #def rt_bidding_amount_rule_2(model, k):
        #    return model.q_rt[k, 2] <= B + self.E_0[2]*self.delta_E[k][1]
        
        #def rt_bidding_amount_rule_3(model, k):
        #    return model.q_rt[k, 3] <= B + self.E_0[3]*self.delta_E[k][2]  
        
        def rt_overbid_rule(model, k):
            return sum(model.q_rt[k, t] for t in range(self.T)) <= E_0_sum
        
        def SOC_initial_rule(model, k):
            return model.S[k, -1] == 0.5*S
        
        def SOC_rule(model, k, t):
            return model.S[k, t] == model.S[k, t - 1] + v*model.c[k, t] - (1/v)*model.d[k, t]
        
        def generation_rule_0(model, k):
            return model.g[k, 0] <= 0

        model.generation = pyo.ConstraintList()
        
        for k in range(self.num_scenarios):
                
            for t in range(1, T):
                
                model.generation.add(model.g[k, t] <= self.E_0[t]*self.delta_E[k][t-1])

        #def generation_rule_1(model, k):
        #    return model.g[k, 1] <= self.E_0[1]*self.delta_E[k][0]
        
        #def generation_rule_2(model, k):
        #    return model.g[k, 2] <= self.E_0[2]*self.delta_E[k][1]
        
        #def generation_rule_3(model, k):
        #    return model.g[k, 3] <= self.E_0[3]*self.delta_E[k][2]
        
        def charge_rule(model, k, t):
            return model.c[k, t] <= model.g[k, t]
        
        def electricity_supply_rule(model, k, t):
            return model.u[k, t] == model.g[k, t] + model.d[k, t] - model.c[k, t]
        
        def rt_market_clearing_rule_1(model, k, t):
            return model.b_rt[k, t] - self.P_rt[k][t] <= self.M_price[k][t][1][0]*(1 - model.n_rt[k, t])
        
        def rt_market_clearing_rule_2(model, k, t):
            return self.P_rt[k][t] - model.b_rt[k, t] <= self.M_price[k][t][1][1]*model.n_rt[k, t] 
        
        def rt_market_clearing_rule_3(model, k, t):
            return model.Q_rt[k, t] <= model.q_rt[k, t]
        
        def rt_market_clearing_rule_4(model, k, t):
            return model.Q_rt[k, t] <= M_gen[t][0]*model.n_rt[k, t]
        
        def rt_market_clearing_rule_5(model, k, t):
            return model.Q_rt[k, t] >= model.q_rt[k, t] - M_gen[t][0]*(1 - model.n_rt[k, t])
        
        def dispatch_rule(model, k, t):
            return model.Q_c[k, t] == (1 + self.delta_c[k][t])*model.Q_rt[k, t]
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
                
        def binarize_b_da_rule_1(model, k, t):
            return model.b_da[k, t] >= -sum(model.lamb[k, t, i]*(2**i) for i in range(self.bin_num)) - 0.5

        def binarize_b_da_rule_2(model, k, t):
            return model.b_da[k, t] <= -sum(model.lamb[k, t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, k, t):
            return model.b_rt[k, t] >= -sum(model.nu[k, t, j]*(2**j) for j in range(self.bin_num)) - 0.5

        def binarize_b_rt_rule_2(model, k, t):
            return model.b_rt[k, t] <= -sum(model.nu[k, t, j]*(2**j) for j in range(self.bin_num)) + 0.5

        def binarize_rule_1_1(model, k, t, j):
            return model.w[k, t, j] >= 0

        def binarize_rule_1_2(model, k, t, j):
            return model.w[k, t, j] <= model.u[k, t]

        def binarize_rule_1_3(model, k, t, j):
            return model.w[k, t, j] <= M_gen[t][0]*model.nu[k, t, j]

        def binarize_rule_1_4(model, k, t, j):
            return model.w[k, t, j] >= model.u[k, t] - M_gen[t][0]*(1 - model.nu[k, t, j])

        def binarize_rule_2_1(model, k, t, i):
            return model.h[k, t, i] >= 0

        def binarize_rule_2_2(model, k, t, i):
            return model.h[k, t, i] <= model.m_1[k, t]

        def binarize_rule_2_3(model, k, t, i):
            return model.h[k, t, i] <= M_gen[t][0]*model.lamb[k, t, i]

        def binarize_rule_2_4(model, k, t, i):
            return model.h[k, t, i] >= model.m_1[k, t] - M_gen[t][0]*(1 - model.lamb[k, t, i])

        def binarize_rule_3_1(model, k, t, i):
            return model.k[k, t, i] >= 0

        def binarize_rule_3_2(model, k, t, i):
            return model.k[k, t, i] <= model.m_2[k, t]

        def binarize_rule_3_3(model, k, t, i):
            return model.k[k, t, i] <= M_gen[t][0]*model.lamb[k, t, i]

        def binarize_rule_3_4(model, k, t, i):
            return model.k[k, t, i] >= model.m_2[k, t] - M_gen[t][0]*(1 - model.lamb[k, t, i])

        def binarize_rule_4_1(model, k, t, i):
            return model.o[k, t, i] >= 0

        def binarize_rule_4_2(model, k, t, i):
            return model.o[k, t, i] <= model.m_3[k, t]

        def binarize_rule_4_3(model, k, t, i):
            return model.o[k, t, i] <= M_gen[t][0]*model.nu[k, t, i]

        def binarize_rule_4_4(model, k, t, i):
            return model.o[k, t, i] >= model.m_3[k, t] - M_gen[t][0]*(1 - model.nu[k, t, i])      
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, k, t):
            return model.m_4_1[k, t] == -sum(model.w[k, t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[k, t]*P_da[t] + (model.u[k, t] - model.Q_da[k, t])*self.P_rt[k][t])
            
        def dummy_rule_4_2(model, k, t):
            return model.m_4_2[k, t] == (model.m_1[k, t] - model.m_2[k, t])*self.P_rt[k][t] + sum((model.h[k, t, i] - model.k[k, t, i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_1_1(model, k, t):
            return model.m_1[k, t] <= model.Q_da[k, t]
        
        def minmax_rule_1_2(model, k, t):
            return model.m_1[k, t] <= model.q_rt[k, t]
        
        def minmax_rule_1_3(model, k, t):
            return model.m_1[k, t] >= model.Q_da[k, t] - (1 - model.n_1[k, t])*M_gen[t][0]
        
        def minmax_rule_1_4(model, k, t):
            return model.m_1[k, t] >= model.q_rt[k, t] - (model.n_1[k, t])*M_gen[t][0]
        
        def minmax_rule_2_1(model, k, t):
            return model.m_2[k, t] <= model.u[k, t]
        
        def minmax_rule_2_2(model, k, t):
            return model.m_2[k, t] <= model.q_rt[k, t]
        
        def minmax_rule_2_3(model, k, t):
            return model.m_2[k, t] >= model.u[k, t] - (1 - model.n_2[k, t])*M_gen[t][0]
        
        def minmax_rule_2_4(model, k, t):
            return model.m_2[k, t] >= model.q_rt[k, t] - model.n_2[k, t]*M_gen[t][0]
        
        def minmax_rule_3_1(model, k, t):
            return model.m_3[k, t] >= model.u[k, t] - model.Q_c[k, t]
        
        def minmax_rule_3_2(model, k, t):
            return model.m_3[k, t] >= 0
        
        def minmax_rule_3_3(model, k, t):
            return model.m_3[k, t] <= model.u[k, t] - model.Q_c[k, t] + M_gen[t][1]*(1 - model.n_3[k, t])
        
        def minmax_rule_3_4(model, k, t):
            return model.m_3[k, t] <= M_gen[t][0]*model.n_3[k, t]

        def minmax_rule_4_1(model, k, t):
            return model.m_4_1[k, t] == model.m_4_1_plus[k, t] - model.m_4_1_minus[k, t]
        
        def minmax_rule_4_2(model, k, t):
            return model.m_4_1_plus[k, t] <= self.M_set_decomp[k][t][0][0]*(1 - model.n_4_1[k, t])
        
        def minmax_rule_4_3(model, k, t):
            return model.m_4_1_minus[k, t] <= self.M_set_decomp[k][t][0][1]*model.n_4_1[k, t]
        
        def minmax_rule_4_4(model, k, t):
            return model.m_4_2[k, t] == model.m_4_2_plus[k, t] - model.m_4_2_minus[k, t]
        
        def minmax_rule_4_5(model, k, t):
            return model.m_4_2_plus[k, t] <= self.M_set_decomp[k][t][1][0]*(1 - model.n_4_2[k, t])
        
        def minmax_rule_4_6(model, k, t):
            return model.m_4_2_minus[k, t] <= self.M_set_decomp[k][t][1][1]*model.n_4_2[k, t]
        
        def minmax_rule_4_7(model, k, t):
            return model.m_4_1_plus[k, t] - model.m_4_2_plus[k, t] <= self.M_set_decomp[k][t][0][0]*(1 - model.n_4[k, t])
        
        def minmax_rule_4_8(model, k, t):
            return model.m_4_2_plus[k, t] - model.m_4_1_plus[k, t] <= self.M_set_decomp[k][t][1][0]*model.n_4[k, t]

        def minmax_rule_4_9(model, k, t):
            return model.m_4[k, t] == model.m_4_1_plus[k, t] + model.l_4_2[k, t] - model.l_4_1[k, t]
        
        def minmax_rule_4_10(model, k, t):
            return model.l_4_1[k, t] <= model.m_4_1_plus[k, t]       
        
        def minmax_rule_4_11(model, k, t):
            return model.l_4_1[k, t] <= self.M_set_decomp[k][t][0][0]*model.n_4[k, t]
        
        def minmax_rule_4_12(model, k, t):
            return model.l_4_1[k, t] >= model.m_4_1_plus[k, t] - self.M_set_decomp[k][t][0][0]*(1 - model.n_4[k, t])
        
        def minmax_rule_4_13(model, k, t):
            return model.l_4_2[k, t] <= model.m_4_2_plus[k, t]       
        
        def minmax_rule_4_14(model, k, t):
            return model.l_4_2[k, t] <= self.M_set_decomp[k][t][1][0]*model.n_4[k, t]
        
        def minmax_rule_4_15(model, k, t):
            return model.l_4_2[k, t] >= model.m_4_2_plus[k, t] - self.M_set_decomp[k][t][1][0]*(1 - model.n_4[k, t])
            
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, k, t):
            return model.f_prime[k, t] == model.Q_da[k, t]*P_da[t] + (model.u[k, t] - model.Q_da[k, t])*self.P_rt[k][t] + self.m_4[k, t] - self.P_rt[k][t]*model.m_3[k, t] - sum((2**j)*model.o[k, t, j] for j in range(self.bin_num)) + P_r*model.u[k, t] 
        
        def settlement_fcn_rule_1(model, k, t):
            return model.Q_sum[k, t] == model.Q_da[k, t] + model.Q_rt[k, t]
        
        def settlement_fcn_rule_2(model, k, t):
            return model.Q_sum[k, t] <= M_gen[t][1]*model.n_sum[k, t]
        
        def settlement_fcn_rule_3(model, k, t):
            return model.Q_sum[k, t] >= epsilon*model.n_sum[k, t]
        
        def settlement_fcn_rule_4(model, k, t):
            return model.f[k, t] >= 0
        
        def settlement_fcn_rule_5(model, k, t):
            return model.f[k, t] <= model.f_prime[k, t]
        
        def settlement_fcn_rule_6(model, k, t):
            return model.f[k, t] <= M_set_fcn*model.n_sum[k, t]
        
        def settlement_fcn_rule_7(model, k, t):
            return model.f[k, t] >= model.f_prime[k, t] - M_set_fcn*(1 - model.n_sum[k, t])        


        model.da_bidding_amount = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(model.SCENARIO, rule = da_overbid_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_5_rule)

        model.rt_bidding_price = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount_0 = pyo.Constraint(model.SCENARIO, rule = rt_bidding_amount_rule_0)

        model.rt_overbid = pyo.Constraint(model.SCENARIO, rule = rt_overbid_rule)
        model.SOC_initial = pyo.Constraint(model.SCENARIO, rule = SOC_initial_rule)
        model.SOC = pyo.Constraint(model.SCENARIO, model.TIME, rule = SOC_rule)
        model.generation_0 = pyo.Constraint(model.SCENARIO, rule = generation_rule_0)
        
        model.charge = pyo.Constraint(model.SCENARIO, model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.SCENARIO, model.TIME, rule = electricity_supply_rule)
        model.rt_market_clearing_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = rt_market_clearing_rule_5)

        model.dispatch = pyo.Constraint(model.SCENARIO, model.TIME, rule = dispatch_rule)

        model.binarize_b_da_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.SCENARIO, model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = dummy_rule_4_2)
        model.minmax_1_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_1)
        model.minmax_2_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_2)
        model.minmax_2_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_3)
        model.minmax_2_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_4)
        model.minmax_3_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_8)
        model.minmax_4_9 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_9)
        model.minmax_4_10 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_10)
        model.minmax_4_11 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_11)
        model.minmax_4_12 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_12)
        model.minmax_4_13 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_13)
        model.minmax_4_14 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_14)
        model.minmax_4_15 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_4_15)
        
        model.settlement_fcn = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule_7)

        ## Non-anticipativ
        
        
        model.NonAnticipativity = pyo.ConstraintList()
              
        for k in range(1, self.num_scenarios):
            
            for t in range(self.T):
                
                model.NonAnticipativity.add(model.b_da[k, t] == model.b_da[0, t])
                model.NonAnticipativity.add(model.q_da[k, t] == model.q_da[0, t])
                #model.NonAnticipativity.add(model.Q_da[k, t] == model.Q_da[0, t])
                #model.NonAnticipativity.add(model.n_da[k, t] == model.n_da[0, t])
                
            model.NonAnticipativity.add(model.b_rt[k, 0] == model.b_rt[0, 0])
            model.NonAnticipativity.add(model.q_rt[k, 0] == model.q_rt[0, 0])
           
        
        for (t, node_id), kset in self.node_scenarios.items():
            
            if len(kset) > 1:
                
                klist = list(kset)
                base = klist[0]
                
                for k in klist[1:]:
                    
                    model.NonAnticipativity.add(model.b_rt[k, t+1] == model.b_rt[base, t+1])
                    model.NonAnticipativity.add(model.q_rt[k, t+1] == model.q_rt[base, t+1])
                    model.NonAnticipativity.add(model.Q_rt[k, t] == model.Q_rt[base, t])
                    model.NonAnticipativity.add(model.n_rt[k, t] == model.n_rt[base, t])
                    
                    for i in range(self.bin_num):
                        #model.NonAnticipativity.add(model.lamb[k, t, i] == model.lamb[base, t, i])
                        #model.NonAnticipativity.add(model.nu[k, t, i] == model.nu[base, t, i])
                        #model.NonAnticipativity.add(model.w[k, t, i] == model.w[base, t, i])
                        #model.NonAnticipativity.add(model.h[k, t, i] == model.h[base, t, i])
                        #model.NonAnticipativity.add(model.k[k, t, i] == model.k[base, t, i])
                        #model.NonAnticipativity.add(model.o[k, t, i] == model.o[base, t, i])
                        pass
                        
                    model.NonAnticipativity.add(model.g[k, t] == model.g[base, t])
                    model.NonAnticipativity.add(model.c[k, t] == model.c[base, t])
                    model.NonAnticipativity.add(model.d[k, t] == model.d[base, t])
                    model.NonAnticipativity.add(model.u[k, t] == model.u[base, t])
                    #model.NonAnticipativity.add(model.Q_c[k, t] == model.Q_c[base, t])
                    #model.NonAnticipativity.add(model.S[k, t] == model.S[base, t])
                    #model.NonAnticipativity.add(model.m_1[k, t] == model.m_1[base, t])
                    #model.NonAnticipativity.add(model.m_2[k, t] == model.m_2[base, t])
                    #model.NonAnticipativity.add(model.m_3[k, t] == model.m_3[base, t])
                    #model.NonAnticipativity.add(model.m_4[k, t] == model.m_4[base, t])
                    #model.NonAnticipativity.add(model.m_4_1[k, t] == model.m_4_1[base, t])
                    #model.NonAnticipativity.add(model.m_4_2[k, t] == model.m_4_2[base, t])
                    #model.NonAnticipativity.add(model.m_4_3[k, t] == model.m_4_3[base, t])
                    #model.NonAnticipativity.add(model.n_1[k, t] == model.n_1[base, t])
                    #model.NonAnticipativity.add(model.n_2[k, t] == model.n_2[base, t])
                    #model.NonAnticipativity.add(model.n_3[k, t] == model.n_3[base, t])
                    #model.NonAnticipativity.add(model.n_4[k, t] == model.n_4[base, t])
                    #model.NonAnticipativity.add(model.n_4_1[k, t] == model.n_4_1[base, t])
                    #model.NonAnticipativity.add(model.n_4_2[k, t] == model.n_4_2[base, t])
                    #model.NonAnticipativity.add(model.n_4_3[k, t] == model.n_4_3[base, t])
                    #model.NonAnticipativity.add(model.f_prime[k, t] == model.f_prime[base, t])
                    #model.NonAnticipativity.add(model.Q_sum[k, t] == model.Q_sum[base, t])
                    #model.NonAnticipativity.add(model.n_sum[k, t] == model.n_sum[base, t])
                    #model.NonAnticipativity.add(model.f[k, t] == model.f[base, t])
        
        
        def objective_rule(model):
            return (
                (1/self.num_scenarios)*sum(sum(model.f[k, t] for t in range(T)) for k in range(self.num_scenarios))
            )
        
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.maximize)
  
    
    def solve(self):
        
        self.build_model()
        self.result = SOLVER.solve(self, tee = True)
        print(f"{self.T} stage DEF solved")
        self.solved = True
    
    
    def get_objective_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)


# SDDiP Algorithm
                                 
class SDDiPModel:
        
    def __init__(
        self, 
        STAGE = T, 
        scenario_tree = None, 
        forward_scenario_num = 3, 
        backward_branch = 3, 
        max_iter = 20, 
        alpha = 0.95, 
        cut_mode = 'B',
        BigM_mode = 'BigM',
        exp_mode = False):

        ## STAGE = -1, 0, ..., STAGE - 1
        ## forward_scenarios = M
        ## baackward_scenarios = N_t

        self.STAGE = STAGE
        self.scenario_tree = scenario_tree
        self.M = forward_scenario_num
        self.N_t = backward_branch
        self.alpha = alpha
        self.cut_mode = cut_mode
        self.BigM_mode = BigM_mode
        
        self.iteration = 0
        
        self.gap = 1
        
        self.max_iter = max_iter
        
        self.LB = [-np.inf]
        self.UB = [np.inf]

        self.forward_solutions = [  ## T(-1), ..., T(T - 2)
            [] for _ in range(self.STAGE)
        ]
        
        self.psi = [[] for _ in range(self.STAGE)] ## t = {0 -> -1}, ..., {T - 1 -> T - 2}
        
        self._initialize_psi()

    def _initialize_psi(self):
        
        for t in range(self.STAGE): ## psi(-1), ..., psi(T - 2)
            self.psi[t].append([3*3600000*(T - t), 0, [0 for _ in range(self.STAGE - t)], [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0])

    def sample_scenarios(self):

        all_scenarios = self.scenario_tree.scenarios()
        
        scenario_probs = []
        
        for path in all_scenarios:
            
            p = 1.0
            for node in path:
                
                p *= node.prob
            
            scenario_probs.append(p)
        
        chosen_scenarios = random.choices(all_scenarios, weights = scenario_probs, k = self.M)
        
        scenarios = []
        
        for scenario_object in chosen_scenarios:
            
            scenario = []
            for node in scenario_object:
                
                scenario.append(node.param)
            
            scenarios.append(scenario)
        
        return scenarios
       
    def forward_pass(self, scenarios):
        
        f = []
        
        for k in range(self.M):
        
            scenario = scenarios[k]
            
            fw_da_subp = fw_da(self.psi[0])
            fw_da_state = fw_da_subp.get_state_solutions()
            
            self.forward_solutions[0].append(fw_da_state)
            
            state = fw_da_state
            
            f_scenario = 0
            
            for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                
                if self.BigM_mode == 'BigM':
                    fw_rt_subp = fw_rt(t, state, self.psi[t+1], scenario[t])
                   
                elif self.BigM_mode == 'Alt': 
                    fw_rt_subp = fw_rt_Alt(t, state, self.psi[t+1], scenario[t])
                
                state = fw_rt_subp.get_state_solutions()
                self.forward_solutions[t+1].append(state)
                f_scenario += fw_rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            if self.BigM_mode == 'BigM':
                fw_rt_last_subp = fw_rt_last(state, scenario[self.STAGE-1])
             
            elif self.BigM_mode == 'Alt':
                fw_rt_last_subp = fw_rt_last_Alt(state, scenario[self.STAGE-1])
            f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
            
            f.append(f_scenario)
        
        mu_hat = np.mean(f)
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        #self.LB.append(mu_hat - z_alpha_half * (sigma_hat / np.sqrt(self.M))) 
        self.LB.append(mu_hat) 
        
        return mu_hat

    def inner_product(self, t, pi, sol):
        
        return sum(pi[i]*sol[i] for i in [0, 3, 4, 5, 6]) + sum(sum(pi[i][j]*sol[i][j] for j in range(self.STAGE - t)) for i in [1, 2])

    def backward_pass(self):
                
        ## t = {T-1 -> T-2}
        
        BL = ['B']
        SBL = ['SB', 'L-sub', 'L-lev']
        
        v_sum = 0 
        pi_mean = [0, [0], [0], 0, 0, 0, 0]
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
        
        for j in range(self.N_t): 
            
            delta = stage_params[T - 1][j]      
            
            if self.BigM_mode == 'BigM':
                fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, delta)
          
            elif self.BigM_mode == 'Alt':
                fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax_Alt(prev_solution, delta)

            
            psi_sub = fw_rt_last_LP_relax_subp.get_cut_coefficients()
            
            pi_mean[0] += psi_sub[1]/self.N_t
            pi_mean[1][0] += psi_sub[2][0]/self.N_t
            pi_mean[2][0] += psi_sub[3][0]/self.N_t
            pi_mean[3] += psi_sub[4]/self.N_t
            pi_mean[4] += psi_sub[5]/self.N_t
            pi_mean[5] += psi_sub[6]/self.N_t
            pi_mean[6] += psi_sub[7]/self.N_t
            
            if self.cut_mode in BL:
                
                v_sum += psi_sub[0]
            
            elif self.cut_mode in SBL:
                
                if self.BigM_mode == 'BigM':
                    fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian([psi_sub[i] for i in range(1, 8)], delta)
                  
                elif self.BigM_mode == 'Alt':
                    fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian_Alt([psi_sub[i] for i in range(1, 8)], delta)   
                    
                v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
            
        if self.cut_mode in BL:   
                 
            #v = v_sum/self.N_t - sum(pi_mean[i]*prev_solution[i] for i in [0, 3, 4, 5, 6]) - sum(pi_mean[j][0]*prev_solution[j][0] for j in [1, 2])
            v = v_sum/self.N_t - self.inner_product(self.STAGE - 1, pi_mean, prev_solution)
        
        elif self.cut_mode in SBL:
            
            v = v_sum/self.N_t
            
        cut_coeff = []
        
        cut_coeff.append(v)
        
        for i in range(7):
            cut_coeff.append(pi_mean[i])
        
        self.psi[T-1].append(cut_coeff)
        
        #print(f"last_stage_cut_coeff = {self.psi[T-1]}")
        
        ## t = {T-2 -> T-3}, ..., {0 -> -1}
        for t in range(self.STAGE - 2, -1, -1): 
                
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
            
            for j in range(self.N_t):
                
                delta = stage_params[t][j]
                
                if self.BigM_mode == 'BigM':
                    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[t+1], delta)
               
                elif self.BigM_mode == 'Alt':
                    fw_rt_LP_relax_subp = fw_rt_LP_relax_Alt(t, prev_solution, self.psi[t+1], delta)         
                                        
                psi_sub = fw_rt_LP_relax_subp.get_cut_coefficients()
                
                pi_mean[0] += psi_sub[1]/self.N_t
                
                for i in range(self.STAGE - t):
                    pi_mean[1][i] += psi_sub[2][i]/self.N_t
                    pi_mean[2][i] += psi_sub[3][i]/self.N_t
                    
                pi_mean[3] += psi_sub[4]/self.N_t
                pi_mean[4] += psi_sub[5]/self.N_t
                pi_mean[5] += psi_sub[6]/self.N_t
                pi_mean[6] += psi_sub[7]/self.N_t
                
                if self.cut_mode in BL:
                    
                    v_sum += psi_sub[0]
                    
                elif self.cut_mode in SBL:
                    
                    if self.BigM_mode == 'BigM':
                        fw_rt_Lagrangian_subp = fw_rt_Lagrangian(t, [psi_sub[i] for i in range(1, 8)], self.psi[t+1], delta)
                 
                    elif self.BigM_mode == 'Alt':
                        fw_rt_Lagrangian_subp = fw_rt_Lagrangian_Alt(t, [psi_sub[i] for i in range(1, 8)], self.psi[t+1], delta)

                    v_sum += fw_rt_Lagrangian_subp.get_objective_value()
            
            if self.cut_mode in BL:
                
                #v = v_sum/self.N_t - sum(pi_mean[i]*prev_solution[i] for i in [0, 3, 4, 5, 6]) - sum(sum(pi_mean[i][j]*prev_solution[i][j] for j in range(self.STAGE - t)) for i in [1, 2])
                v = v_sum/self.N_t - self.inner_product(t, pi_mean, prev_solution)
        
            if self.cut_mode in SBL:
                
                v = v_sum/self.N_t
        
            cut_coeff = []
            
            cut_coeff.append(v)
            
            for i in range(7):
                cut_coeff.append(pi_mean[i])
        
            self.psi[t].append(cut_coeff)
            #print(f"stage {t - 1} _cut_coeff = {self.psi[t]}")

        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0])
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value())) 

    def backward_pass_Lagrangian(self):
                
        ## t = {T-1 -> T-2}
        
        v_sum = 0 
        pi_mean = [0, [0], [0], 0, 0, 0, 0]
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
        
        print(f"-----Solving Dual stage = {self.STAGE - 1}-----")
        
        for j in range(self.N_t): 
            
            delta = stage_params[T - 1][j]      
            
            fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, delta)
            
            pi_LP = fw_rt_last_LP_relax_subp.get_cut_coefficients()[1:]
            
            pi = pi_LP
            pi_min = pi_LP
            
            reg = 0.000001
            lev = 0.9
            gap = 1       
            lamb = 0
            k = [0, [0], [0], 0, 0, 0, 0]
            l = 10000000
                        
            dual_subp_sub_last = dual_approx_sub(self.STAGE - 1, reg, pi_LP)
            dual_subp_lev_last = dual_approx_lev(self.STAGE - 1, reg, pi_LP, l)
            
            fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, delta)
            
            L = fw_rt_last_Lag_subp.get_objective_value()
            z = fw_rt_last_Lag_subp.get_auxiliary_value()
            
            Lag_iter = 1
                        
            pi_minobj = 10000000
            
            while gap >= 0.0001:            
                
                lamb = L + self.inner_product(self.STAGE - 1, pi, z)
                
                for l in [0, 3, 4, 5, 6]:
                    
                    k[l] = prev_solution[l] - z[l]
                
                for l in [1, 2]:
                    
                    k[l][0] = prev_solution[l][0] - z[l][0]
                
                dual_coeff = [lamb, k]
                                
                if self.cut_mode == 'L-sub':
                    
                    dual_subp_sub_last.add_plane(dual_coeff)
                    pi = dual_subp_sub_last.get_solution_value()
                    obj = dual_subp_sub_last.get_objective_value()
                    
                elif self.cut_mode == 'L-lev':
                                                            
                    dual_subp_sub_last.add_plane(dual_coeff)
                    dual_subp_lev_last.add_plane(dual_coeff)
                                        
                    f_lb = dual_subp_sub_last.get_objective_value()
                    f_ub = pi_minobj
                                        
                    l = f_lb + lev*(f_ub - f_lb)
                    
                    print(f"f_lb = {f_lb}, f_ub = {f_ub}, l = {l}")
                    
                    dual_subp_lev_last.level = l
                    dual_subp_lev_last.pi = pi_min
                                        
                    pi = dual_subp_lev_last.get_solution_value()
                    
                    obj = f_lb
                
                fw_rt_last_Lag_subp = fw_rt_last_Lagrangian(pi, delta)
                
                L = fw_rt_last_Lag_subp.get_objective_value()
                z = fw_rt_last_Lag_subp.get_auxiliary_value()
                                
                pi_obj = L + self.inner_product(self.STAGE - 1, pi, prev_solution)
                                
                if pi_obj < pi_minobj:
                    
                    pi_minobj = pi_obj
                    pi_min = pi
                
                gap = (pi_obj - obj)/pi_obj
                                        
                #print(f"k = {k}, \npi = {pi} \n, \ngap = {gap}, \npi_obj = {pi_obj}, \nobj = {obj}")
                                              
                Lag_iter += 1
            
            print(f"gap = {gap}, pi = {pi}")
            
            pi_mean[0] += pi[0]/self.N_t
            pi_mean[1][0] += pi[1][0]/self.N_t
            pi_mean[2][0] += pi[2][0]/self.N_t
            pi_mean[3] += pi[3]/self.N_t
            pi_mean[4] += pi[4]/self.N_t
            pi_mean[5] += pi[5]/self.N_t
            pi_mean[6] += pi[6]/self.N_t
            
            v_sum += L
                
        v = v_sum/self.N_t
            
        cut_coeff = []
        
        cut_coeff.append(v)
        
        for i in range(7):
            cut_coeff.append(pi_mean[i])
        
        self.psi[self.STAGE - 1].append(cut_coeff)
                
        #print(f"last_stage_cut_coeff = {self.psi[T-1]}")
        
        ## t = {T-2 -> T-3}, ..., {0 -> -1}
        
        for t in range(self.STAGE - 2, -1, -1): 
            
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
            
            print(f"-----Solving Dual stage = {t}-----")
            
            for j in range(self.N_t):
                                
                delta = stage_params[t][j]
                
                fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[t+1], delta)

                pi_LP = fw_rt_LP_relax_subp.get_cut_coefficients()[1:]
                
                pi = pi_LP
                pi_min = pi_LP
                                
                reg = 0.000001
                lev = 0.5
                gap = 1
                lamb = 0
                k = [0, [0 for _ in range(self.STAGE - t)], [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
                l = 10000000*(self.STAGE - t)
                      
                dual_subp_sub = dual_approx_sub(t, reg, pi_LP)
                
                fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[t+1], delta)
                
                L = fw_rt_Lag_subp.get_objective_value()
                z = fw_rt_Lag_subp.get_auxiliary_value()    
                                
                Lag_iter = 1
                
                dual_coeff_list = []
                
                pi_minobj = 10000000*(self.STAGE - t)
                
                while gap >= 0.0001:
                    
                    lamb = L + self.inner_product(t, pi, z)
                    
                    for l in [0, 3, 4, 5, 6]:
                        
                        k[l] = prev_solution[l] - z[l]
                        
                    for l in [1, 2]:
                        
                        for i in range(self.STAGE - t):
                            
                            k[l][i] = prev_solution[l][i] - z[l][i]
                                        
                    dual_coeff = [lamb, k]
                    dual_coeff_list.append(dual_coeff) 
                                        
                    if self.cut_mode == 'L-sub' or Lag_iter == 1:
                        dual_subp_sub.add_plane(dual_coeff)
                        pi = dual_subp_sub.get_solution_value()
                        obj = dual_subp_sub.get_objective_value()
                    
                    elif self.cut_mode == 'L-lev':
                        
                        dual_subp_sub.add_plane(dual_coeff)
                        f_lb = dual_subp_sub.get_objective_value()
                        f_ub = pi_minobj
                        
                        l = f_lb + lev*(f_ub - f_lb)
                        
                        dual_subp_lev = dual_approx_lev(t, dual_coeff_list, pi, l)

                        pi = dual_subp_lev.get_solution_value()
                        
                        obj = f_lb
                                        
                    fw_rt_Lag_subp = fw_rt_Lagrangian(t, pi, self.psi[t+1], delta)
                    
                    L = fw_rt_Lag_subp.get_objective_value()
                    z = fw_rt_Lag_subp.get_auxiliary_value()
                    
                    pi_obj = L + self.inner_product(t, pi, prev_solution)
                    
                    if pi_obj < pi_minobj:
                    
                        pi_minobj = pi_obj
                        pi_min = pi
                        
                    gap = (pi_obj - obj)/pi_obj
                    
                    #print(f"k = {k} , \npi = {pi} , \ngap = {gap}, \npi_obj = {pi_obj}, \nobj = {obj}")
                                                        
                    Lag_iter += 1                
                
                print(f"gap = {gap}, pi = {pi}")
                    
                pi_mean[0] += pi[0]/self.N_t
                
                for i in range(self.STAGE - t):
                    pi_mean[1][i] += pi[1][i]/self.N_t
                    pi_mean[2][i] += pi[2][i]/self.N_t
                    
                pi_mean[3] += pi[3]/self.N_t
                pi_mean[4] += pi[4]/self.N_t
                pi_mean[5] += pi[5]/self.N_t
                pi_mean[6] += pi[6]/self.N_t
                                
                v_sum += L
                            
            v = v_sum/self.N_t
        
            cut_coeff = []
            
            cut_coeff.append(v)
            
            for i in range(7):
                cut_coeff.append(pi_mean[i])
        
            self.psi[t].append(cut_coeff)
            #print(f"stage {t - 1} _cut_coeff = {self.psi[t]}")

        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0])
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value()))   
         
    def stopping_criterion(self, tol = 0.01):
      
        if self.iteration >= self.max_iter:
            return True
        self.gap = (self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]
        return False

    def run_sddip(self):
        
        while not self.stopping_criterion():
            self.iteration += 1
            print(f"\n=== Iteration {self.iteration} ===")

            scenarios = self.sample_scenarios()

            self.forward_pass(scenarios)

            print(f"  LB for iter = {self.iteration} updated to: {self.LB[self.iteration]:.4f}")
            
            if self.cut_mode == 'B' or self.cut_mode == 'SB':
                self.backward_pass()
            
            elif self.cut_mode == 'L-sub' or self.cut_mode == 'L-lev':
                
                if self.gap > 0.5 or self.iteration <= 2:
                    self.backward_pass()
                
                else:
                    self.backward_pass_Lagrangian()
                
                
            else:
                print("Not a proposed cut")
                break
            
            print(f"  UB for iter = {self.iteration} updated to: {self.UB[self.iteration]:.4f}")
            
        print("\nSDDiP complete.")
        print(f"Final LB = {self.LB[self.iteration]:.4f}, UB = {self.UB[self.iteration]:.4f}, gap = {(self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]:.4f}")
        print(f"LB list = {self.LB}, UB list = {self.UB}")


if __name__ == "__main__":
    
    num_iter = 25
    scenario_branch = 4  ## N_t
    fw_scenario_num = 10  ## M
    
    num_iter_list = [10, 25, 50]
    scenario_branch_list = [2, 4, 7, 10]
    fw_scenario_num_list = [2, 4, 6, 8, 10]
    BigM_mode_list = ['BigM', 'Alt']
    
    stage_params = []
    
    
    for t in range(9, 9 + T):
        
        stage_param = scenario_generator.sample_multiple_delta(t, scenario_branch)
        
        stage_params.append(stage_param)     
    
    
    stage_params = [
        [[1.0568037492471882, 99.2150319380103, 0], [0.8541591674818009, 103.45161675891347, 0], [0.9212841857860108, -12.6821832148197, 0], [0.8436906609947701, 108.65916144510042, 0]], 
        [[0.944393934353801, 106.16613527143109, 0], [0.9980446516218301, 132.19415678199823, 0], [0.8226665907126034, 95.40992777649764, 0], [0.9761652480607177, 109.37295079949263, 0]], 
        [[0.9639480113042348, 100.46593285560922, 0], [1.0031398798406639, 103.78275063792817, 0], [0.8093512104476058, 102.33300251027543, 0], [1.0991221913686902, 118.59169266739974, 0]], 
        [[1.087466326285433, 100.32338793425832, 0], [1.159245374394473, 103.67644682218588, -0.19179979971204353], [1.12102933753565, 110.01092585228216, 0], [0.8617548399782096, 152.37489352218722, 0]], 
        [[1.0834976219212336, 104.07271226957103, 0], [0.8634498605081226, 118.03221736803673, 0], [1.1243335742671112, 103.20854856981401, 0], [0.9114198128210322, 101.19156069226295, 0]], 
        [[0.8572519085283472, 101.31222172327678, 0], [0.928104724382114, 122.24819178552463, 0], [0.8545139078847096, 108.5435698896676, 0], [0.8626342190157117, 105.17070251683006, 0]], 
        [[0.9387870765915632, -40, 0], [1.1314970540447034, 105.46379701442571, 0], [1.0724074653195903, 129.26308660133995, 0], [1.1740660716616722, 126.37538413434159, 0]], 
        [[0.811680967361771, 165.77340414277157, 0], [0.9903501561327369, 107.24395081093166, 0], [0.9303225486946357, 101.52293987784844, 0], [1.073680500657586, 105.91863994492752, 0]], 
        [[1.0096636550616571, 106.49119243927669, 0], [1.1769362403042851, 113.00487937251063, 0], [0.8426279475355245, 103.2266055200999, 0], [0.8716942875836529, 124.43592953940643, 0]], 
        [[0.8833331326752389, 84.25117087896008, 0], [1.0492024939789395, 105.95747450552334, 0], [0.8059872474874955, 164.91212360360043, 0], [1.075847955832046, 102.54032593542846, 0]]
        ]
    
    ScenarioTree1 = RecombiningScenarioTree(T, scenario_branch, stage_params)

    
    def Comparison(BigM_mode):
        """
        if BigM_mode == 'BigM':
            DEF_1 = T_stage_DEF_BigM(ScenarioTree1)

        elif BigM_mode == 'Alt':
            DEF_1 = T_stage_DEF_Alt(ScenarioTree1)
        
        DEF_start_time = time.time()
        DEF_obj = DEF_1.get_objective_value()
        DEF_end_time = time.time()
        time_DEF = DEF_end_time - DEF_start_time
        """
        sddip_1 = SDDiPModel(
            max_iter=num_iter,
            scenario_tree=ScenarioTree1,
            forward_scenario_num=fw_scenario_num,
            backward_branch=scenario_branch,
            cut_mode='SB',
            BigM_mode=BigM_mode,
        )

        SDDiP_1_start_time = time.time()
        sddip_1.run_sddip()
        SDDiP_1_end_time = time.time()
        time_SDDiP_1 = SDDiP_1_end_time - SDDiP_1_start_time


        LB_1_list = sddip_1.LB[:num_iter] 
        UB_1_list = sddip_1.UB[:num_iter]
        
        gap_SDDiP_1 = (sddip_1.UB[sddip_1.iteration] - sddip_1.LB[sddip_1.iteration])/sddip_1.UB[sddip_1.iteration]
        
        sddip_2 = SDDiPModel(
            max_iter=num_iter,
            scenario_tree=ScenarioTree1,
            forward_scenario_num=fw_scenario_num,
            backward_branch=scenario_branch,
            cut_mode='L-sub',
            BigM_mode=BigM_mode,
        )
        
        SDDiP_2_start_time = time.time()
        sddip_2.run_sddip()
        SDDiP_2_end_time = time.time()
        time_SDDiP_2 = SDDiP_2_end_time - SDDiP_2_start_time

        LB_2_list = sddip_2.LB[:num_iter]
        UB_2_list = sddip_2.UB[:num_iter]
        
        gap_SDDiP_2 = (sddip_2.UB[sddip_2.iteration] - sddip_2.LB[sddip_2.iteration])/sddip_2.UB[sddip_2.iteration]
        
        """
        sddip_3 = SDDiPModel(
            max_iter=num_iter,
            scenario_tree=ScenarioTree1,
            forward_scenario_num=fw_scenario_num,
            backward_branch=scenario_branch,
            cut_mode='SB',
            BigM_mode=BigM_mode,
            binary_mode=True
        )

        SDDiP_3_start_time = time.time()
        sddip_3.run_sddip()
        SDDiP_3_end_time = time.time()
        time_SDDiP_3 = SDDiP_3_end_time - SDDiP_3_start_time

        LB_3_list = sddip_3.LB[:num_iter]
        UB_3_list = sddip_3.UB[:num_iter]
        
        gap_SDDiP_3 = (sddip_3.UB[sddip_3.iteration] - sddip_3.LB[sddip_3.iteration])/sddip_3.UB[sddip_3.iteration]
        """
        plt.figure(figsize=(7,5))

        iterations = range(num_iter)
        
        plt.plot(iterations, LB_1_list, label="LB (SB)", marker='o', color='tab:blue')
        plt.plot(iterations, UB_1_list, label="UB (SB)", marker='^', color='tab:blue', linestyle='--')
        plt.fill_between(iterations, LB_1_list, UB_1_list, alpha=0.1, color='tab:blue')
        
        plt.plot(iterations, LB_2_list, label="LB (l-sub)", marker='o', color='tab:orange')
        plt.plot(iterations, UB_2_list, label="UB (l-sub)", marker='^', color='tab:orange', linestyle='--')
        plt.fill_between(iterations, LB_2_list, UB_2_list, alpha=0.1, color='tab:orange')
        """
        plt.plot(iterations, LB_3_list, label="LB (SB)", marker='o', color='tab:green')
        plt.plot(iterations, UB_3_list, label="UB (SB-Alt)", marker='^', color='tab:green', linestyle='--')
        plt.fill_between(iterations, LB_3_list, UB_3_list, alpha=0.1, color='tab:green')
        """
        
        #plt.axhline(y=DEF_obj, color='black', linestyle='--', label='DEF_obj')
        
        plt.xlabel('Iteration')
        plt.ylabel('Bound')
        plt.legend()
        
        plt.ylim(0, 10800000*T)
        
        plt.show()

        #print(f"Solving T-stage DEF took {time_DEF:.2f} seconds.\n")
        print(f"SDDiP (cut='SB') for {BigM_mode} took {time_SDDiP_1:.2f} seconds.\n")
        print(f"SDDiP (cut='L-sub') for {BigM_mode} took {time_SDDiP_2:.2f} seconds.\n")
        #print(f"SDDiP (cut='SB-Alt') took {time_SDDiP_3:.2f} seconds.\n")   
             
        print(f"SDDiP optimality for {BigM_mode} gap for SB = {gap_SDDiP_1:.4f}")
        print(f"SDDiP optimality for {BigM_mode} gap for L-sub = {gap_SDDiP_2:.4f}")    
        #print(f"SDDiP optimality gap for SB-Alt = {gap_SDDiP_3:.4f}")    
        
    for mode in ['BigM']:    
        Comparison(mode)
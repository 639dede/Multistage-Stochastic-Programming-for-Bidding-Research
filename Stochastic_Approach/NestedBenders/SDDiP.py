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

from Scenarios import Scenario

solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

SOLVER.options['TimeLimit'] = 1000
#SOLVER.options['MIPGap'] = 1e-4

assert SOLVER.available(), f"Solver {solver} is available."

# Generate Scenario

# SDDiP Model

Total_time = 18000

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

T = 24

dual_tolerance = 1e-7
tol = 1e-5
Node_num = 60
Lag_iter_UB = 500

P_da_minus_mode = True
P_rt_minus_mode = True

E_0 = Scenario.E_0

if T <= 10:

    E_0_partial = [E_0[t] for t in range(9, 19)]
    
elif T == 24:
    
    E_0_partial = E_0

E_0_sum = 0
E_0_partial_max = max(E_0_partial)

for t in range(len(E_0_partial)):
    E_0_sum += E_0_partial[t]


if T <= 10:
    
    if P_da_minus_mode == True:    
        P_da_partial = [
            -40.04237333851853, # T = 9
            -70.02077847557203, # T = 10
            180.6656414585262,  # T = 11
            -30.223343222316,   # T = 12
            01.1443379704008,   # T = 13
            69.23105760780754,  # T = 14
            -61.57109079814273, # T = 15
            -43.93082813230762, # T = 16
            -38.80926230668373, # T = 17
            124.39462311589915  # T = 18
            ]
        
    else:
        P_da_partial = [
            140.04237333851853, # T = 9
            170.02077847557203, # T = 10
            180.6656414585262,  # T = 11
            160.223343222316,   # T = 12
            101.1443379704008,  # T = 13
            89.23105760780754,  # T = 14
            161.57109079814273, # T = 15
            143.93082813230762, # T = 16
            138.80926230668373, # T = 17
            124.39462311589915  # T = 18
            ]
    

elif T == 24:
    
    if P_da_minus_mode == True:
        P_da_partial = [
            52.83429128347213,        # t = 0
            66.22347892347328,        # t = 1
            80.98439238472342,        # t = 2
            112.78293487312834,       # t = 3
            125.32984719238412,       # t = 4
            -38.98473284912348,       # t = 5
            -49.12834712389432,       # t = 6
            -21.48723847293847,       # t = 7 
            -10.32948327481234,       # t = 8 
            0.0000000000000000,       # t = 9
            -40.04237333851853,       # t = 10
            90.02077847557203,       # t = 11
            -70.6656414585262,        # t = 12
            -60.223343222316,         # t = 13
            141.1443379704008,        # t = 14
            -39.23105760780754,       # t = 15
            -25.57109079814273,       # t = 16
            143.93082813230762,       # t = 17
            -38.80926230668373,       # t = 18
            124.39462311589915,       # t = 19
            108.92834719847234,       # t = 20
            88.49238471239842,        # t = 21
            69.18234718234713,        # t = 22
            54.32847239482347         # t = 23
            ]
    
    else:
        P_da_partial = [
            152.83429128347213,        # t = 0
            166.22347892347328,        # t = 1
            180.98439238472342,        # t = 2
            172.78293487312834,       # t = 3
            155.32984719238412,       # t = 4
            168.98473284912348,       # t = 5
            169.12834712389432,       # t = 6
            171.48723847293847,       # t = 7 
            180.32948327481234,       # t = 8 
            153.48329481234872,       # t = 9
            150.04237333851853,       # t = 10
            170.02077847557203,       # t = 11
            180.6656414585262,        # t = 12
            160.223343222316,         # t = 13
            141.1443379704008,        # t = 14
            139.23105760780754,       # t = 15
            125.57109079814273,       # t = 16
            143.93082813230762,       # t = 17
            178.80926230668373,       # t = 18
            154.39462311589915,       # t = 19
            158.92834719847234,       # t = 20
            168.49238471239842,        # t = 21
            169.18234718234713,        # t = 22
            154.32847239482347         # t = 23
            ]
        
    
K = [1.23*E_0_partial[t] + 1.02*B for t in range(T)]

M_gen = [[1.02*K[t], 2*K[t]] for t in range(T)]


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

    def __init__(self, stage, T_prev, psi, delta):
        
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
        
        self.P_da = P_da_partial
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
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81           
        
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

    def __init__(self, stage, T_prev, psi, delta, exp):
        
        super().__init__()

        self.solved = False
        
        self.exp = exp
        
        self.stage = stage
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_o_prev = T_prev[2]
        self.T_b_prev = T_prev[3]
        self.T_q_prev = T_prev[4]
        self.T_E_prev = T_prev[5]
        
        self.psi = psi
        
        self.P_da = P_da_partial
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
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81    
        
  
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
        
        ## experiment mode
        
        def experiment_mode_1(model):
            return model.m_1 == 0
        
        def experiment_mode_2(model):
            return model.m_1 == model.Q_da - model.Q_c
        
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
        
        if self.exp == 0:
            model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
            model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
            model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
            model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        
        elif self.exp == 1:
            model.minmax_exp_1 = pyo.Constraint(rule = experiment_mode_1)
            
        elif self.exp ==2:
            model.minmax_exp_2 = pyo.Constraint(rule = experiment_mode_2)
        
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
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81    
        
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
    
    def __init__(self, T_prev, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_o_prev = T_prev[2]
        self.T_b_prev = T_prev[3]
        self.T_q_prev = T_prev[4]
        self.T_E_prev = T_prev[5]

        self.P_da = P_da_partial
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
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81                          
 
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
           
    def __init__(self, T_prev, delta, exp):
        
        super().__init__()

        self.solved = False
        
        self.exp = exp
        
        self.stage = T - 1
        
        self.S_prev = T_prev[0]
        self.T_Q_prev = T_prev[1]
        self.T_o_prev = T_prev[2]
        self.T_b_prev = T_prev[3]
        self.T_q_prev = T_prev[4]
        self.T_E_prev = T_prev[5]

        self.P_da = P_da_partial
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
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81    

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
  
        ## experiment mode
        
        def experiment_mode_1(model):
            return model.m_1 == 0
        
        def experiment_mode_2(model):
            return model.m_1 == model.Q_da - model.Q_c
  
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
        
        if self.exp == 0:
            model.minmax_1_1 = pyo.Constraint(rule = minmax_rule_1_1)
            model.minmax_1_2 = pyo.Constraint(rule = minmax_rule_1_2)
            model.minmax_1_3 = pyo.Constraint(rule = minmax_rule_1_3)
            model.minmax_1_4 = pyo.Constraint(rule = minmax_rule_1_4)
        
        elif self.exp == 1:
            model.minmax_exp_1 = pyo.Constraint(rule = experiment_mode_1)
            
        elif self.exp ==2:
            model.minmax_exp_2 = pyo.Constraint(rule = experiment_mode_2)
        
        model.minmax_2_1 = pyo.Constraint(rule = minmax_rule_2_1)
        
        model.imbalance_over = pyo.Constraint(rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(rule = imbalance_under_rule) 
                      
        if self.exp == 1:
            model.exp_1 = pyo.Constraint(rule = experiment_mode_1)
        
        elif self.exp == 2:
            model.exp_2 = pyo.Constraint(rule = experiment_mode_2)
                      
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
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_price[0] = 1
            self.M_price[1] = self.P_rt + 81

        else:
            
            self.M_price[0] = -self.P_rt + 1
            self.M_price[1] = self.P_rt + 81    
        
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
        model.pi_Q = pyo.Var(model.TIME, domain = pyo.Reals)
        model.pi_o = pyo.Var(domain = pyo.Reals)
        model.pi_b = pyo.Var(domain = pyo.Reals)
        model.pi_q = pyo.Var(domain = pyo.Reals)
        model.pi_E = pyo.Var(domain = pyo.Reals)

        # Constraints
        
        model.lev = pyo.Param(mutable = True, initialize = 0.0)
        
        model.dual_fcn_approx = pyo.ConstraintList() 
        
        # Obj Fcn
        
        def objective_rule(model):
            return (
                (model.pi_S - self.pi[0])**2 
                + sum((model.pi_Q[t] - self.pi[1][t])**2 for t in range(self.T - self.stage)) 
                + (model.pi_o - self.pi[2])**2 
                + (model.pi_b - self.pi[3])**2 
                + (model.pi_q - self.pi[4])**2 
                + (model.pi_E - self.pi[5])**2
            )
            
        model.objective = pyo.Objective(rule = objective_rule, sense = pyo.minimize)
        
    def add_plane(self, coeff):
        
        lamb = coeff[0]
        k = coeff[1]
                
        model = self.model()
        
        model.dual_fcn_approx.add(model.lev >= (
            lamb 
            + k[0]*model.pi_S 
            + sum(k[1][t]*model.pi_Q[t] for t in range(self.T - self.stage)) 
            + k[2]*model.pi_o
            + k[3]*model.pi_b
            + k[4]*model.pi_q
            + k[5]*model.pi_E))
    
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

class T_stage_DEF(pyo.ConcreteModel):
    
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
        
        self.delta_E = [
            [self.scenarios[k][t].param[0] for t in range(T)] 
            for k in range(self.num_scenarios)
            ]
        self.P_rt = [
            [self.scenarios[k][t].param[1] for t in range(T)] 
            for k in range(self.num_scenarios)
            ]
        self.delta_c = [
            [self.scenarios[k][t].param[2] for t in range(T)] 
            for k in range(self.num_scenarios)
            ]
                
        self.P_abs = [
            [max(self.P_rt[k][t] - self.P_da[t], 0) for t in range(T)]
            for k in range(self.num_scenarios)
        ]        
                
        self.T = T
        
        self.M_price = [
            [[[0, 0], [0, 0]] for t in range(self.T)] for k in range(self.num_scenarios)
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
  
                elif self.P_da[t] >=0 and self.P_rt[k][t] < 0:            
                    
                    self.M_price[k][t][0][0] = 0
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = -self.P_rt[k][t]
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80

                elif self.P_da[t] < 0 and self.P_rt[k][t] >= 0:
                    
                    self.M_price[k][t][0][0] = -self.P_da[t]
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = 0
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80
 
                else:
                    
                    self.M_price[k][t][0][0] = -self.P_da[t]
                    self.M_price[k][t][0][1] = self.P_da[t] + 80
                    self.M_price[k][t][1][0] = -self.P_rt[k][t]
                    self.M_price[k][t][1][1] = self.P_rt[k][t] + 80

    def build_model(self):
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T - 1)

        model.S_TIME = pyo.RangeSet(-1, T - 1)
        
        model.SCENARIO = pyo.RangeSet(0, self.num_scenarios - 1)
        
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
        
        model.n_1 = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.Binary)
        
        ## Imbalance Penerty Vars
        
        model.phi_over = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        model.phi_under = pyo.Var(model.SCENARIO, model.TIME, domain = pyo.NonNegativeReals)
        
        ### settlement_fcn_Vars
        
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
        
        def rt_bidding_amount_rule_0(model, k):
            return model.q_rt[k, 0] <= B
        
        model.rt_bidding_amount = pyo.ConstraintList()
        
        for k in range(self.num_scenarios):
            
            for t in range(1, T):
                
                model.rt_bidding_amount.add(model.q_rt[k, t] <= B + self.E_0[t]*self.delta_E[k][t-1])
        
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
        
        def minmax_rule_1_1(model, k, t):
            return model.m_1[k, t] >= model.Q_da[k, t] - model.u[k, t]
        
        def minmax_rule_1_2(model, k, t):
            return model.m_1[k, t] >= 0
        
        def minmax_rule_1_3(model, k, t):
            return model.m_1[k, t] <= model.Q_da[k, t] - model.u[k, t] + M_gen[t][0]*(1 - model.n_1[k, t])
        
        def minmax_rule_1_4(model, k, t):
            return model.m_1[k, t] <= M_gen[t][0]*model.n_1[k, t]
        
        def minmax_rule_2_1(model, k, t):
            return model.m_2[k, t] == model.m_1[k, t]*self.P_abs[k][t]
        
        ### Imbalance Penalty
        
        def imbalance_over_rule(model, k, t):
            return model.u[k, t] - model.Q_c[k, t] <= model.phi_over[k, t]
        
        def imbalance_under_rule(model, k, t):
            return model.Q_c[k, t] - model.u[k, t] <= model.phi_under[k, t]
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, k, t):
            return model.f[k, t] == (
                model.Q_da[k, t]*P_da_partial[t] 
                + (model.u[k, t] - model.Q_da[k, t])*self.P_rt[k][t] 
                + self.m_2[k, t] 
                - gamma_over*model.phi_over[k, t]
                - gamma_under*model.phi_under[k, t]
                )
            
        model.da_bidding_amount = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(model.SCENARIO, rule = da_overbid_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.SCENARIO, model.TIME, rule = da_market_clearing_5_rule)

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
        
        model.minmax_1_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_1_4)
        model.minmax_2_1 = pyo.Constraint(model.SCENARIO, model.TIME, rule = minmax_rule_2_1)

        model.imbalance_over = pyo.Constraint(model.SCENARIO, model.TIME, rule = imbalance_over_rule)
        model.imbalance_under = pyo.Constraint(model.SCENARIO, model.TIME, rule = imbalance_under_rule)

        model.settlement_fcn = pyo.Constraint(model.SCENARIO, model.TIME, rule = settlement_fcn_rule)

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
                        
                    model.NonAnticipativity.add(model.g[k, t] == model.g[base, t])
                    model.NonAnticipativity.add(model.c[k, t] == model.c[base, t])
                    model.NonAnticipativity.add(model.d[k, t] == model.d[base, t])
                    model.NonAnticipativity.add(model.u[k, t] == model.u[base, t])
        
        
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
        stage_params = None, 
        forward_scenario_num = 3, 
        backward_branch = 3, 
        max_iter = 20, 
        alpha = 0.95, 
        cut_mode = 'B',
        plot = True,
        tol = 0.001,
        K = 1
        ):

        ## STAGE = -1, 0, ..., STAGE - 1
        ## forward_scenarios(last iteration) = M
        ## baackward_scenarios = N_t

        self.STAGE = STAGE
        self.stage_params = stage_params
        self.M = forward_scenario_num
        self.N_t = backward_branch
        self.alpha = alpha
        self.cut_mode = cut_mode
        self.plot = plot
        self.tol = tol
        self.K = K
        
        self.iteration = 0
        
        self.start_time = time.time()
        self.running_time = 0
        
        self.Lag_elapsed_time_list = []
        self.Total_Lag_iter_list = []
        
        self.gap = 1
        
        self.max_iter = max_iter
        
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
            
            fw_da_subp = fw_da(self.psi[0])
            fw_da_state = fw_da_subp.get_state_solutions()
            
            b_da = [pyo.value(fw_da_subp.b_da[t]) for t in range(self.STAGE)]
            Q_da = [pyo.value(fw_da_subp.Q_da[t]) for t in range(self.STAGE)]
            local_b_da.append(b_da)
            local_Q_da.append(Q_da)
            
            self.forward_solutions[0].append(fw_da_state)
            
            state = fw_da_state
            
            f_scenario = fw_da_subp.get_settlement_fcn_value()
            
            for t in range(self.STAGE - 1): ## t = 0, ..., T-2
                
                fw_rt_subp = fw_rt(t, state, self.psi[t+1], scenario[t])
                
                state = fw_rt_subp.get_state_solutions()
                
                local_b_rt.append((k,t,pyo.value(fw_rt_subp.b_rt)))
                local_Q_rt.append((k,t,pyo.value(fw_rt_subp.Q_rt)))
                
                self.forward_solutions[t+1].append(state)
                f_scenario += fw_rt_subp.get_settlement_fcn_value()
            
            ## t = T-1
            
            fw_rt_last_subp = fw_rt_last(state, scenario[self.STAGE-1])

            f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
            
            local_b_rt.append((k,self.STAGE-1,pyo.value(fw_rt_last_subp.b_rt)))
            local_Q_rt.append((k,self.STAGE-1,pyo.value(fw_rt_last_subp.Q_rt)))
            
            f.append(f_scenario)
        
        if self.iteration == self.max_iter + 1:

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
        SBL = ['SB', 'L-sub', 'L-lev', 'hyb']
        
        v_sum = 0 
        pi_mean = [0, [0], 0, 0, 0, 0]
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
        
        for j in range(self.N_t): 
            
            delta = stage_params[T - 1][j]      
            
            fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, delta, 0)
            
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
                
                fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian([psi_sub[i] for i in range(1, 7)], delta)

                v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
            
        if self.cut_mode in BL:   
                 
            v = v_sum/self.N_t - self.inner_product(self.STAGE - 1, pi_mean, prev_solution)
        
        elif self.cut_mode in SBL:
            
            v = v_sum/self.N_t
            
        cut_coeff = []
        
        cut_coeff.append(v)
        
        for i in range(6):
            cut_coeff.append(pi_mean[i])
        
        self.psi[T-1].append(cut_coeff)
        
        #print(f"last_stage_cut_coeff = {self.psi[T-1]}")
        
        ## t = {T-2 -> T-3}, ..., {0 -> -1}
        for t in range(self.STAGE - 2, -1, -1): 
                
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
            
            for j in range(self.N_t):
                
                delta = stage_params[t][j]
                
                fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[t+1], delta, 0)
   
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
                    
                    fw_rt_Lagrangian_subp = fw_rt_Lagrangian(t, [psi_sub[i] for i in range(1, 7)], self.psi[t+1], delta)

                    v_sum += fw_rt_Lagrangian_subp.get_objective_value()
            
            if self.cut_mode in BL:
                
                v = v_sum/self.N_t - self.inner_product(t, pi_mean, prev_solution)
        
            if self.cut_mode in SBL:
                
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
        
        self.UB.append(pyo.value(fw_da_for_UB.get_objective_value())) 

    def backward_pass_Lagrangian(self):
                
        ## t = {T-1 -> T-2}
        
        v_sum = 0 
        pi_mean = [0, [0], 0, 0, 0, 0]
        
        Lag_elapsed_time = 0
        Total_Lag_iter = 0
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
                
        for j in range(self.N_t): 
            
            delta = stage_params[T - 1][j]      
            
            fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, delta, 0)
            
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
                                
                if self.cut_mode == 'L-sub':
                    
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
                
        #print(f"last_stage_cut_coeff = {self.psi[T-1]}")
        
        ## t = {T-2 -> T-3}, ..., {0 -> -1}
        
        for t in range(self.STAGE - 2, -1, -1): 
            
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
                        
            for j in range(self.N_t):
                                
                delta = stage_params[t][j]
                
                fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[t+1], delta, 0)

                pi_LP = fw_rt_LP_relax_subp.get_cut_coefficients()[1:]
                
                pi = pi_LP
                pi_min = pi_LP
                                
                lev = 0.9
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
                                        
                    if self.cut_mode == 'L-sub':
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
            #print(f"stage {t - 1} _cut_coeff = {self.psi[t]}")

        self.forward_solutions = [
            [] for _ in range(self.STAGE)
        ]
        
        fw_da_for_UB = fw_da(self.psi[0])
        
        self.Lag_elapsed_time_list.append(Lag_elapsed_time)
        self.Total_Lag_iter_list.append(Total_Lag_iter)
        
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

    def stopping_criterion(self, tol = 1e-3):
        
        self.gap = (self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]
        
        self.running_time = time.time() - self.start_time
        
        print(f"run_time = {self.running_time}, criteria = {Total_time/self.K}")
        
        if self.plot == True:    
            if self.iteration >= self.max_iter:
                return True
        
        else:
            if self.iteration <= 3:
                baseline = 1e9
                
            elif self.iteration >= 4:
                baseline = self.UB[self.iteration-3]
            
            if (
                (baseline - self.UB[self.iteration])/self.UB[self.iteration] < self.tol 
                or self.running_time > Total_time/self.K
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
                scenarios = self.sample_scenarios(self.M)
                
            else:
                scenarios = self.sample_scenarios(2)

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
    
    if T <= 4:
        num_iter = 30
        
    elif T >= 4 and T <= 8:
        num_iter = 60
        
    elif T >= 10:
        num_iter = 20
    
    scenario_branch = 4  ## N_t
    fw_scenario_num = 200  ## M
    
    plot_mode = False
    
    stage_params = []
    
    if T <= 10:
        
        for t in range(9, 9 + T):
            
            """
            stage_param = scenario_generator.sample_multiple_delta(t, scenario_branch)
            
            stage_params.append(stage_param)  
            """
            
            if P_rt_minus_mode == True:
            
                stage_params = [
                [[1.0568037492471882, -75.2150319380103, 0], [0.8541591674818009, 103.45161675891347, 0], [0.9212841857860108, -12.6821832148197, 0], [0.8436906609947701, 108.65916144510042, 0]], 
                [[0.944393934353801, 106.16613527143109, 0], [0.9980446516218301, 132.19415678199823, 0], [0.8226665907126034, 95.40992777649764, 0], [0.9761652480607177, 109.37295079949263, 0]], 
                [[0.9639480113042348, -70.46593285560922, 0], [1.0031398798406639, -63.78275063792817, 0], [0.8093512104476058, -2.33300251027543, 0], [1.0991221913686902, 118.59169266739974, 0]], 
                [[1.087466326285433, -60.32338793425832, 0], [1.159245374394473, -3.67644682218588, -0.19179979971204353], [1.12102933753565, -40.01092585228216, 0], [0.8617548399782096, 152.37489352218722, 0]], 
                [[1.0834976219212336, 104.07271226957103, 0], [0.8634498605081226, -18.03221736803673, 0], [1.1243335742671112, -53.20854856981401, 0], [0.9114198128210322, 101.19156069226295, 0]], 
                [[0.8572519085283472, 101.31222172327678, 0], [0.928104724382114, -22.24819178552463, 0], [0.8545139078847096, -70.5435698896676, 0], [0.8626342190157117, 105.17070251683006, 0]], 
                [[0.9387870765915632, -40, 0], [1.1314970540447034, 105.46379701442571, 0], [1.0724074653195903, 129.26308660133995, 0], [1.1740660716616722, 126.37538413434159, 0]], 
                [[0.811680967361771, -65.77340414277157, 0], [0.9903501561327369, 107.24395081093166, 0], [0.9303225486946357, -71.52293987784844, 0], [1.073680500657586, 105.91863994492752, 0]], 
                [[1.0096636550616571, 106.49119243927669, 0], [1.1769362403042851, 113.00487937251063, 0], [0.8426279475355245, -73.2266055200999, 0], [0.8716942875836529, -24.43592953940643, 0]], 
                [[0.8833331326752389, 84.25117087896008, 0], [1.0492024939789395, 105.95747450552334, 0], [0.8059872474874955, 164.91212360360043, 0], [1.075847955832046, 102.54032593542846, 0]]
                ]
            
            else:
                
                stage_params = [
                [[1.0568037492471882, 99.2150319380103, 0], [0.8541591674818009, 103.45161675891347, 0], [0.9212841857860108, 112.6821832148197, 0], [0.8436906609947701, 108.65916144510042, 0]], 
                [[0.944393934353801, 106.16613527143109, 0], [0.9980446516218301, 132.19415678199823, 0], [0.8226665907126034, 95.40992777649764, 0], [0.9761652480607177, 109.37295079949263, 0]], 
                [[0.9639480113042348, 170.46593285560922, 0], [1.0031398798406639, 163.78275063792817, 0], [0.8093512104476058, 102.33300251027543, 0], [1.0991221913686902, 118.59169266739974, 0]], 
                [[1.087466326285433, 160.32338793425832, 0], [1.159245374394473, 103.67644682218588, -0.19179979971204353], [1.12102933753565, 140.01092585228216, 0], [0.8617548399782096, 152.37489352218722, 0]], 
                [[1.0834976219212336, 104.07271226957103, 0], [0.8634498605081226, 118.03221736803673, 0], [1.1243335742671112, 153.20854856981401, 0], [0.9114198128210322, 101.19156069226295, 0]], 
                [[0.8572519085283472, 101.31222172327678, 0], [0.928104724382114, 122.24819178552463, 0], [0.8545139078847096, 170.5435698896676, 0], [0.8626342190157117, 105.17070251683006, 0]], 
                [[0.9387870765915632, 140, 0], [1.1314970540447034, 105.46379701442571, 0], [1.0724074653195903, 129.26308660133995, 0], [1.1740660716616722, 126.37538413434159, 0]], 
                [[0.811680967361771, 165.77340414277157, 0], [0.9903501561327369, 107.24395081093166, 0], [0.9303225486946357, 101.52293987784844, 0], [1.073680500657586, 105.91863994492752, 0]], 
                [[1.0096636550616571, 106.49119243927669, 0], [1.1769362403042851, 113.00487937251063, 0], [0.8426279475355245, 103.2266055200999, 0], [0.8716942875836529, 124.43592953940643, 0]], 
                [[0.8833331326752389, 84.25117087896008, 0], [1.0492024939789395, 105.95747450552334, 0], [0.8059872474874955, 164.91212360360043, 0], [1.075847955832046, 102.54032593542846, 0]]
                ]
            
            #ScenarioTree1 = RecombiningScenarioTree(T, scenario_branch, stage_params)
            
    elif T == 24:
        
        for t in range(T):
            
            if P_rt_minus_mode == True:    
                
                stage_params = [
                    [[1.0215916878707407, 80.78063274505286, 0], [0.913198717630274, 207.15368016517033, 0], [0.8834311252191159, 112.47748126297796, 0], [0.9093990290952852, 104.16773575789867, 0]],
                    [[1.1488431388110705, 130.9160409043475, 0], [1.065076595008393, 107.24674977480271, 0], [0.8196148184043662, 154.03597880917974, 0], [1.056322550250875, 130.5887198443035, 0]],
                    [[0.8382805679787786, 161.66092232270626, 0], [0.9072248713716513, 144.28413611903053, 0], [1.008446512641973, 122.0270351143196, 0], [1.0762045957212751, 150.74339164683713, 0]],
                    [[1.1914322160053217, 119.40454637923602, 0], [1.193322380728992, 130.3142120174733, 0], [1.1234266258049317, 111.230438943751, 0], [0.913473226771982, 151.25455717947463, 0]],
                    [[0.9806408592150446, 117.68811785393548, 0], [1.1579324579530106, 145.43799591219667, 0], [1.0753349724707235, 132.98835900203508, 0], [0.9302061644925295, 107.2725599508661, 0]],
                    [[1.0328008590207247, 104.2448741445248, 0], [0.8897749313427393, 150.9105776976912, 0.5620083635886923], [0.884279376272486, 144.33114812030516, 0], [1.1427580174945196, 153.27114703688127, 0]],
                    [[1.0419324778937955, 89.49134638951436, 0], [0.8399441852929602, 143.1041652905437, 0], [1.1088398149147847, -58.1343491710808, 0], [1.1124797099474808, -70.17352907311863, 0]],
                    [[1.0404812228728273, 187.43198044293445, 0], [0.9606678628866206, 153.54564830510174, 0], [1.1024340464905058, 148.57787436913443, 0], [1.1226162597769818, -83.24495101198171, 0]],
                    [[0.9347148687381786, 108.46472427885557, 0], [0.9920412478402375, 141.5702662836685, 0], [1.020324603203949, -54.87733374658288, 0], [0.8438221282456536, 136.71228452714038, 0]],
                    [[1.19164263223995, 141.85190820423819, 0], [0.8248244059889797, -40.2505909394309, 0], [0.9098962152318385, 153.33715894430296, 0], [0.9340122362746441, 118.19139405788043, 0.12240362302166444]],
                    [[1.1821557322096625, 124.15893383969049, 0], [1.1035953416269042, -62.94491703408116, 0], [1.039434444581245, 135.3373986267369, -0.3718429526822882], [0.8996312360574763, 155.48471886063214, 0]],
                    [[0.8464797880021295, -40.83018294386079, 0], [1.1919607033884003, 125.00152614176613, 0], [0.9860733556794887, -42.61271770412862, 0], [0.8133068329067613, 159.6656920917638, 0]],
                    [[1.0021363463719195, 119.87546641833352, 0], [1.1777970592002465, 128.6422485692858, 0], [0.814082811047605, 139.32314602782463, 0], [1.1460402827779228, -52.52770259929176, 0]],
                    [[0.8206351825907535, -42.93625642779996, 0], [1.0363732899855291, 133.7275935505178, 0], [0.8240914258609248, 135.68421048930225, 0], [0.9403565021945437, 122.54326915837518, 0]],
                    [[0.918712259653239, 131.85794253397466, 0], [1.188237378119679, -56.8532769134401, 0], [1.1616356067497748, 138.89768755162373, 0], [1.1797580281083941, 140.59570014596747, 0]],
                    [[0.9733791108916049, 147.19917007286438, 0], [0.9880022815913414, 144.98208445831798, 0], [1.1834931792106, -70.6078697006711, 0], [1.0425587972099644, -29.15288294852678, 0]],
                    [[0.8343515073585218, 141.30319115794194, 0], [1.0999285823780465, 147.26211796024688, 0], [0.9223317131098161, 148.83421599302298, 0], [1.1913610430318116, 139.14098675605615, 0]],
                    [[1.1321858667927096, 89.65675763916045, 0], [1.0879435940828859, 141.206567233732, 0], [0.8850163096182874, 146.12531608973316, 0], [0.8498997084837714, 140.58573887134324, 0]],
                    [[1.0857077759999743, 141.683763069225, 0], [1.1863095342242285, -32.0067508244964, 0], [1.0691546869491797, 141.25004016178457, 0], [0.9054617522388662, 140.17527106547365, 0]],
                    [[1.1840944219266767, 113.65705462932182, 0], [1.0680634482160214, 121.02956495657905, 0], [0.9009198607461647, 129.5521662600149, 0], [1.0654860271095627, 130.06505208633504, 0]],
                    [[0.8723540804845644, 170.3702932938346, 0], [1.051904252827458, 151.50240042834247, 0], [0.8214806038830954, 90.76440746230455, 0], [1.1139277934008909, 102.96393647635523, 0]],
                    [[1.1296385651730396, 134.60261763268483, 0], [0.8249847922150656, 106.84217246452607, 0], [1.0769263303296945, 121.90347693000534, 0], [0.857235079695519, 152.3212855748905, 0]],
                    [[0.8261830968053797, 158.75750983854374, 0], [1.1541271626992855, 147.49360537847625, 0], [0.8075457115100096, 109.91956945322873, -0.40742964944915894], [0.9742769980442817, 150.92485054893862, 0]],
                    [[0, 216.86335042940675, 0], [0, 131.74841739827423, 0], [0, 153.5741300362661, 0], [0, 142.61852676386806, 0]],
                ]
            
            else:
                
                stage_params = [
                    [[1.0215916878707407, 80.78063274505286, 0], [0.913198717630274, 207.15368016517033, 0], [0.8834311252191159, 112.47748126297796, 0], [0.9093990290952852, 104.16773575789867, 0]],
                    [[1.1488431388110705, 130.9160409043475, 0], [1.065076595008393, 107.24674977480271, 0], [0.8196148184043662, 154.03597880917974, 0], [1.056322550250875, 130.5887198443035, 0]],
                    [[0.8382805679787786, 161.66092232270626, 0], [0.9072248713716513, 144.28413611903053, 0], [1.008446512641973, 122.0270351143196, 0], [1.0762045957212751, 150.74339164683713, 0]],
                    [[1.1914322160053217, 119.40454637923602, 0], [1.193322380728992, 130.3142120174733, 0], [1.1234266258049317, 111.230438943751, 0], [0.913473226771982, 151.25455717947463, 0]],
                    [[0.9806408592150446, 117.68811785393548, 0], [1.1579324579530106, 145.43799591219667, 0], [1.0753349724707235, 132.98835900203508, 0], [0.9302061644925295, 107.2725599508661, 0]],
                    [[1.0328008590207247, 104.2448741445248, 0], [0.8897749313427393, 150.9105776976912, 0.5620083635886923], [0.884279376272486, 144.33114812030516, 0], [1.1427580174945196, 153.27114703688127, 0]],
                    [[1.0419324778937955, 89.49134638951436, 0], [0.8399441852929602, 143.1041652905437, 0], [1.1088398149147847, 158.1343491710808, 0], [1.1124797099474808, 170.17352907311863, 0]],
                    [[1.0404812228728273, 187.43198044293445, 0], [0.9606678628866206, 153.54564830510174, 0], [1.1024340464905058, 148.57787436913443, 0], [1.1226162597769818, 183.2449510119817, 0]],
                    [[0.9347148687381786, 108.46472427885557, 0], [0.9920412478402375, 141.5702662836685, 0], [1.020324603203949, 154.87733374658288, 0], [0.8438221282456536, 136.71228452714038, 0]],
                    [[1.19164263223995, 141.85190820423819, 0], [0.8248244059889797, 140.2505909394309, 0], [0.9098962152318385, 153.33715894430296, 0], [0.9340122362746441, 118.19139405788043, 0.12240362302166444]],
                    [[1.1821557322096625, 124.15893383969049, 0], [1.1035953416269042, 162.94491703408116, 0], [1.039434444581245, 135.3373986267369, -0.3718429526822882], [0.8996312360574763, 155.48471886063214, 0]],
                    [[0.8464797880021295, 140.8301829438608, 0], [1.1919607033884003, 125.00152614176613, 0], [0.9860733556794887, 142.61271770412862, 0], [0.8133068329067613, 159.6656920917638, 0]],
                    [[1.0021363463719195, 119.87546641833352, 0], [1.1777970592002465, 128.6422485692858, 0], [0.814082811047605, 139.32314602782463, 0], [1.1460402827779228, 152.52770259929176, 0]],
                    [[0.8206351825907535, 142.93625642779996, 0], [1.0363732899855291, 133.7275935505178, 0], [0.8240914258609248, 135.68421048930225, 0], [0.9403565021945437, 122.54326915837518, 0]],
                    [[0.918712259653239, 131.85794253397466, 0], [1.188237378119679, 156.8532769134401, 0], [1.1616356067497748, 138.89768755162373, 0], [1.1797580281083941, 140.59570014596747, 0]],
                    [[0.9733791108916049, 147.19917007286438, 0], [0.9880022815913414, 144.98208445831798, 0], [1.1834931792106, 170.6078697006711, 0], [1.0425587972099644, 129.15288294852678, 0]],
                    [[0.8343515073585218, 141.30319115794194, 0], [1.0999285823780465, 147.26211796024688, 0], [0.9223317131098161, 148.83421599302298, 0], [1.1913610430318116, 139.14098675605615, 0]],
                    [[1.1321858667927096, 89.65675763916045, 0], [1.0879435940828859, 141.206567233732, 0], [0.8850163096182874, 146.12531608973316, 0], [0.8498997084837714, 140.58573887134324, 0]],
                    [[1.0857077759999743, 141.683763069225, 0], [1.1863095342242285, 132.0067508244964, 0], [1.0691546869491797, 141.25004016178457, 0], [0.9054617522388662, 140.17527106547365, 0]],
                    [[1.1840944219266767, 113.65705462932182, 0], [1.0680634482160214, 121.02956495657905, 0], [0.9009198607461647, 129.5521662600149, 0], [1.0654860271095627, 130.06505208633504, 0]],
                    [[0.8723540804845644, 170.3702932938346, 0], [1.051904252827458, 151.50240042834247, 0], [0.8214806038830954, 90.76440746230455, 0], [1.1139277934008909, 102.96393647635523, 0]],
                    [[1.1296385651730396, 134.60261763268483, 0], [0.8249847922150656, 106.84217246452607, 0], [1.0769263303296945, 121.90347693000534, 0], [0.857235079695519, 152.3212855748905, 0]],
                    [[0.8261830968053797, 158.75750983854374, 0], [1.1541271626992855, 147.49360537847625, 0], [0.8075457115100096, 109.91956945322873, -0.40742964944915894], [0.9742769980442817, 150.92485054893862, 0]],
                    [[0, 216.86335042940675, 0], [0, 131.74841739827423, 0], [0, 153.5741300362661, 0], [0, 142.61852676386806, 0]],
                ]
                 
                        
    P_rt_per_hour = [ [branch[1] for branch in stage] for stage in stage_params ]     
    
    ## Plot Day Ahead and Real Time Price values
    
    """
    if T <= 10:
        hours = np.arange(10)
        
    elif T == 24:
        hours = np.arange(24)
    
    plt.figure(figsize=(12, 6))
    
    P_rt_curves = list(zip(*P_rt_per_hour))  

    for curve in P_rt_curves:
        plt.plot(hours, curve, color='royalblue', alpha=0.4, linewidth=1)

    plt.plot(hours, P_da_partial, color='black', linewidth=2.5, label='P_da')

    plt.title("Day-Ahead vs Real-Time Prices")
    plt.xlabel("Hour (t)")
    plt.ylabel("Price")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)  
    plt.legend()
    
    plt.ylim(-P_r - 20, P_max + 40)
    
    plt.tight_layout()
    plt.show()
    """
        
    def convergence_Comparison(plot = True):
        
        #DEF_1 = T_stage_DEF(ScenarioTree1)
    
        sddip_1 = SDDiPModel(
                max_iter=num_iter,
                stage_params=stage_params,
                forward_scenario_num=fw_scenario_num,
                backward_branch=scenario_branch,
                cut_mode='SB',
                plot=plot,
                tol=tol,
                K=Node_num
            )
        
        sddip_2 = SDDiPModel(
                max_iter=num_iter,
                stage_params=stage_params,
                forward_scenario_num=fw_scenario_num,
                backward_branch=scenario_branch,
                cut_mode='L-sub',
                plot=plot,
                tol=tol,
                K=Node_num
            )
        
        sddip_3 = SDDiPModel(
                max_iter=num_iter,
                stage_params=stage_params,
                forward_scenario_num=fw_scenario_num,
                backward_branch=scenario_branch,
                cut_mode='hyb',
                plot=plot,
                tol=tol,
                K=Node_num
            )
        
        """
        DEF_start_time = time.time()
        DEF_obj = DEF_1.get_objective_value()
        DEF_end_time = time.time()
        time_DEF = DEF_end_time - DEF_start_time
        """

        SDDiP_1_start_time = time.time()
        sddip_1.run_sddip()
        SDDiP_1_end_time = time.time()
        time_SDDiP_1 = SDDiP_1_end_time - SDDiP_1_start_time
        
        LB_1_list = sddip_1.LB 
        UB_1_list = sddip_1.UB
        
        gap_SDDiP_1 = (sddip_1.UB[sddip_1.iteration] - sddip_1.LB[sddip_1.iteration])/sddip_1.UB[sddip_1.iteration]
        
        
        SDDiP_2_start_time = time.time()
        sddip_2.run_sddip()
        SDDiP_2_end_time = time.time()
        time_SDDiP_2 = SDDiP_2_end_time - SDDiP_2_start_time
        
        """
        Lag_time_2 = sddip_2.Lag_elapsed_time_list
        Lag_iter_2 = sddip_2.Total_Lag_iter_list
        
        plt.figure(figsize=(10, 5))
        plt.plot(Lag_time_2, marker='o', linestyle='-', linewidth=2, color='orange')
        plt.title("Lag_time", fontsize=14)
        plt.xlabel("Backward Pass Iteration", fontsize=12)
        plt.ylabel("Time (seconds)", fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        plt.show()

        plt.figure(figsize=(10, 5))
        
        plt.plot(Lag_iter_2, marker='s', linestyle='--', linewidth=2, color='blue')
        plt.title("L-lev Cut Total Iterations", fontsize=14)
        plt.xlabel("Backward Pass Iteration", fontsize=12)
        plt.ylabel("Number of Iterations", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        """
        
        LB_2_list = sddip_2.LB
        UB_2_list = sddip_2.UB
        
        gap_SDDiP_2 = (sddip_2.UB[sddip_2.iteration] - sddip_2.LB[sddip_2.iteration])/sddip_2.UB[sddip_2.iteration]
        
        
        SDDiP_3_start_time = time.time()
        sddip_3.run_sddip()
        SDDiP_3_end_time = time.time()
        time_SDDiP_3 = SDDiP_3_end_time - SDDiP_3_start_time

        LB_3_list = sddip_3.LB
        UB_3_list = sddip_3.UB
        
        gap_SDDiP_3 = (sddip_3.UB[sddip_3.iteration] - sddip_3.LB[sddip_3.iteration])/sddip_3.UB[sddip_3.iteration]        
        
        ## Plot SDDiP results
        
        if plot:    
            plt.figure(figsize=(7,5))
            
            iterations = range(num_iter+2)
            
            plt.plot(iterations, LB_1_list, label=f"LB ({sddip_1.cut_mode})", marker='o', color='tab:blue')
            plt.plot(iterations, UB_1_list, label=f"UB ({sddip_1.cut_mode})", marker='^', color='tab:blue', linestyle='--')
            plt.fill_between(iterations, LB_1_list, UB_1_list, alpha=0.1, color='tab:blue')
            
            plt.plot(iterations, LB_2_list, label=f"LB ({sddip_2.cut_mode})", marker='o', color='tab:orange')
            plt.plot(iterations, UB_2_list, label=f"UB ({sddip_2.cut_mode})", marker='^', color='tab:orange', linestyle='--')
            plt.fill_between(iterations, LB_2_list, UB_2_list, alpha=0.1, color='tab:orange')
        
            plt.plot(iterations, LB_3_list, label=f"LB ({sddip_3.cut_mode})", marker='o', color='tab:green')
            plt.plot(iterations, UB_3_list, label=f"UB ({sddip_3.cut_mode})", marker='^', color='tab:green', linestyle='--')
            plt.fill_between(iterations, LB_3_list, UB_3_list, alpha=0.1, color='tab:green')
            
            #plt.axhline(y=DEF_obj, color='black', linestyle='--', label='DEF_obj')
            
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
        
        #print(f"Solving T-stage DEF took {time_DEF:.2f} seconds.\n")
        print(f"SDDiP1 took {time_SDDiP_1:.2f} seconds.\n")
        print(f"SDDiP2 took {time_SDDiP_2:.2f} seconds.\n")
        print(f"SDDiP3 took {time_SDDiP_3:.2f} seconds.\n")
             
        print(f"SDDiP1 optimality gap = {gap_SDDiP_1:.4f}")
        print(f"SDDiP2 optimality gap = {gap_SDDiP_2:.4f}")
        print(f"SDDiP3 optimality gap = {gap_SDDiP_3:.4f}")      
        
        #print(f"DEF obj = {DEF_obj}")
        print(f"SDDiP1 final LB = {LB_1_list[-1]}, UB = {UB_1_list[-1]}")
        print(f"SDDiP2 final LB = {LB_2_list[-1]}, UB = {UB_2_list[-1]}")
        print(f"SDDiP3 final LB = {LB_3_list[-1]}, UB = {UB_3_list[-1]}")
        
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
                title="SDDiP 2 solution plot"
                )       
            
    convergence_Comparison(plot_mode)
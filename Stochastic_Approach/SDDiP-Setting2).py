import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import random
import time
import math
import concurrent.futures
import pandas as pd


scenarios_path = os.path.join(os.path.dirname(__file__), "Scenarios")
sys.path.append(scenarios_path)

from Scenarios import Scenario

solver='gurobi'
   
SOLVER=pyo.SolverFactory(solver)
SOLVER.options['TimeLimit'] = 20

assert SOLVER.available(), f"Solver {solver} is available."

# Generate Scenario

# SDDiP Model

T = 24

#E_0 = Scenario.E_0
E_0_daily = Scenario.E_0_daily
delta_E_daily = Scenario.delta_E_daily
P_da_daily = Scenario.day_ahead_prices_daily
P_rt_daily = Scenario.real_time_prices_daily

print(len(E_0_daily), len(delta_E_daily), len(P_da_daily), len(P_rt_daily))

print(
    f"""
E_0 = {E_0_daily[15]} 
delta_E = {delta_E_daily[15]}
P_da = {P_da_daily[15]}
P_rt = {P_rt_daily[15]}
    """
)

"""
scenario_generator = Scenario.Setting2_scenario(E_0)

E_0_sum = 0

for t in range(T):
    E_0_sum += E_0[t]

exp_P_da = scenario_generator.exp_P_da
exp_P_rt = scenario_generator.exp_P_rt
"""

C = 21022.1
S = C*3
B = C*1
S_min = 0.1*S
S_max = 0.9*S
P_r = 80
P_max = 270
P_c = 22.05
v = 0.95


#K = [1.22*E_0[t] + 1.01*B for t in range(T)]

M_price = 500
#M_gen = [4*K[t] for t in range(T)]
M_set_fcn = 15000000

epsilon = 0.0000000000000000001

# Deterministic Problem for EEV & WS

class Deterministic_Setting(pyo.ConcreteModel):
    
    def __init__(self, E_0, P_da, P_rt, delta_E, scenario):
        
        super().__init__()

        self.solved = False

        self.E_0 = E_0
        self.P_da = P_da
        
        self.scenario = scenario
        
        self.delta_E = delta_E
        self.P_rt = P_rt
        self.delta_c = []
                
        self.T = T
        self.bin_num = 7
        
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(self.T)]
        
        self.M_set = [[0, 0] for t in range(T)]
        self.M_set_decomp = [[[0, 0] for i in range(3)] for t in range(T)]

        self.E_0_sum = 0
        
        for t in range(T):
            self.E_0_sum += E_0[t]
        
        self._extract_parameters()
        self._BigM_setting()
    
    def _BigM_setting(self):
        
        self.M_gen = [5*self.K[t] for t in range(T)]
        
        for t in range(self.T):   
             
            if self.P_da[t] >=0 and self.P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] + 2*self.P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 + 2*self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (self.P_rt[t] + 80)*self.K[t]
                self.M_set_decomp[t][1][1] = (self.P_rt[t] + 80)*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] >=0 and self.P_rt[t] < 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] - 2*self.P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - 2*self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (-self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] < 0 and self.P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + 2*self.P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - self.P_da[t] + 2*self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (-self.P_da[t] + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            else:
                
                self.M_set[t][0] = (160 - 2*self.P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - 2*self.P_da[t] - 2*self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (- self.P_da[t] - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1]) 
        
    def _extract_parameters(self):
        
        for t in range(self.T):
            
            #self.delta_E.append(self.scenario[t][0])
            #self.P_rt.append(self.scenario[t][1])
            self.delta_c.append(self.scenario[t][2])
            
    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= self.E_0[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= self.E_0_sum
        
        def da_market_clearing_1_rule(model, t):
            return model.b_da[t] - self.P_da[t] <= M_price*(1 - model.n_da[t])
        
        def da_market_clearing_2_rule(model, t):
            return self.P_da[t] - model.b_da[t] <= M_price*model.n_da[t]
        
        def da_market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def da_market_clearing_4_rule(model, t):
            return model.Q_da[t] <= self.M_gen[t]*model.n_da[t]
        
        def da_market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_da[t])
        
        def da_State_SOC_rule(model):
            return model.S[-1] == 0.5*S
        
        def da_State_SOC_last_rule(model):
            return model.S[23] == 0.5*S
        
        ## Real-Time Market rules
        
        def rt_E_rule(model, t):
            return model.E_1[t] == self.E_0[t]*self.delta_E[t]
        
        def rt_bidding_price_rule(model, t):
            return model.b_rt[t] <= model.b_da[t]
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= model.E_1[t] + B
        
        def rt_overbid_rule(model):
            return sum(model.q_rt[t] for t in range(self.T)) <= self.E_0_sum
        
        def rt_market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.P_rt[t] <= M_price*(1 - model.n_rt[t])
        
        def rt_market_clearing_rule_2(model, t):
            return self.P_rt[t] - model.b_rt[t] <= M_price*model.n_rt[t] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[t]*(1 - model.n_rt[t])
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == (1 + self.delta_c[t])*model.Q_rt[t]
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.m_4_1_1[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.m_4_1_1[t] - self.M_gen[t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.P_rt[t])
    
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.P_rt[t] + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, t):
            return model.f_prime[t] == model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.P_rt[t] + self.m_4[t] - (self.P_rt[t] - model.b_da[t])*model.m_3[t] + P_r*model.u[t] + P_c*(model.m_6[t] - model.m_7[t] + model.m_8[t])
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.TIME, rule = da_market_clearing_5_rule)
        model.da_State_SOC = pyo.Constraint(rule = da_State_SOC_rule)
        model.da_State_SOC_last = pyo.Constraint(rule = da_State_SOC_last_rule)
        
        model.rt_E = pyo.Constraint(model.TIME, rule = rt_E_rule)
        model.rt_bidding_price = pyo.Constraint(model.TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)
        model.rt_market_clearing_1 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)

        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.TIME, rule = dummy_rule_4_2)
        
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn = pyo.Constraint(model.TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def get_objective_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)

    def get_objective_part_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
        
        part1 = sum(pyo.value(self.Q_da[t] * self.P_da[t] + (self.u[t] - self.Q_da[t]) * self.P_rt[t]) for t in range(self.T))
        part2_1 = sum(pyo.value(self.m_4_1[t]) for t in range(self.T))
        part2_2 = sum(pyo.value(self.m_4_2[t]) for t in range(self.T))
        part2 = sum(pyo.value(self.m_4[t]) for t in range(self.T))
        part3 = sum(pyo.value(-(self.P_rt[t] - self.b_da[t]) * self.m_3[t]) for t in range(self.T))
        part4 = sum(pyo.value(P_c * (self.m_6[t] - self.m_7[t] + self.m_8[t])) for t in range(self.T))
        part5 = sum(pyo.value(P_r * self.u[t]) for t in range(self.T))
        
        return [part1, part2_1, part2_2, part2, part3, part4, part5]
        
    def get_b_da_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.b_da[t]) for t in range(self.T)] 

    def get_b_rt_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.b_rt[t]) for t in range(self.T)] 

    def get_q_da_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.q_da[t]) for t in range(self.T)] 
  
    def get_q_rt_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.q_rt[t]) for t in range(self.T)]   
    
    def get_Q_da_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.Q_da[t]) for t in range(self.T)] 

    def get_Q_rt_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.Q_rt[t]) for t in range(self.T)] 

    def get_Q_c_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.Q_c[t]) for t in range(self.T)]    

    def get_u_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.u[t]) for t in range(self.T)] 

    def get_S_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.S[t]) for t in range(self.T)] 
    
class Deterministic_Setting_withoutESS(pyo.ConcreteModel):
    
    def __init__(self, E_0, P_da, P_rt, delta_E, scenario):
        
        super().__init__()

        self.solved = False

        self.E_0 = E_0
        self.P_da = P_da
        
        self.scenario = scenario
        
        self.delta_E = delta_E
        self.P_rt = P_rt
        self.delta_c = []
        
        self.T = T
        self.bin_num = 7
        
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(self.T)]
        
        self.M_set = [[0, 0] for t in range(T)]
        self.M_set_decomp = [[[0, 0] for i in range(3)] for t in range(T)]

        self.E_0_sum = 0
        
        for t in range(T):
            self.E_0_sum += E_0[t]
        
        self._extract_parameters()
        self._BigM_setting()
    
    def _BigM_setting(self):
        
        self.M_gen = [4*self.K[t] for t in range(T)]
        
        for t in range(self.T):   
             
            if self.P_da[t] >=0 and self.P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] + 2*self.P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 + 2*self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (self.P_rt[t] + 80)*self.K[t]
                self.M_set_decomp[t][1][1] = (self.P_rt[t] + 80)*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] >=0 and self.P_rt[t] < 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] - 2*self.P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - 2*self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (-self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] < 0 and self.P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + 2*self.P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - self.P_da[t] + 2*self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (-self.P_da[t] + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 + self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            else:
                
                self.M_set[t][0] = (160 - 2*self.P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - 2*self.P_da[t] - 2*self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (- self.P_da[t] - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 - self.P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1]) 
        
    def _extract_parameters(self):
        
        for t in range(self.T):
            
            #self.delta_E.append(self.scenario[t][0])
            #self.P_rt.append(self.scenario[t][1])
            self.delta_c.append(self.scenario[t][2])
            
    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= self.E_0[t]
        
        def da_market_clearing_1_rule(model, t):
            return model.b_da[t] - self.P_da[t] <= M_price*(1 - model.n_da[t])
        
        def da_market_clearing_2_rule(model, t):
            return self.P_da[t] - model.b_da[t] <= M_price*model.n_da[t]
        
        def da_market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def da_market_clearing_4_rule(model, t):
            return model.Q_da[t] <= self.M_gen[t]*model.n_da[t]
        
        def da_market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_da[t])
        
        def da_State_SOC_rule(model):
            return model.S[-1] == 0.5*S
        
        ## Real-Time Market rules
        
        def rt_E_rule(model, t):
            return model.E_1[t] == self.E_0[t]*self.delta_E[t]
        
        def rt_bidding_price_rule(model, t):
            return model.b_rt[t] <= model.b_da[t]
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= model.E_1[t]
        
        def rt_market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.P_rt[t] <= M_price*(1 - model.n_rt[t])
        
        def rt_market_clearing_rule_2(model, t):
            return self.P_rt[t] - model.b_rt[t] <= M_price*model.n_rt[t] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[t]*(1 - model.n_rt[t])
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == (1 + self.delta_c[t])*model.Q_rt[t]
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def without_ESS_1_rule(model, t):
            return model.c[t] == 0
        
        def without_ESS_2_rule(model, t):
            return model.d[t] == 0
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.m_4_1_1[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.m_4_1_1[t] - self.M_gen[t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.P_rt[t])
    
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.P_rt[t] + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C+S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C+S) + self.M_gen[t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]


        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, t):
            return model.f_prime[t] == model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.P_rt[t] + self.m_4[t] - (self.P_rt[t] - model.b_da[t])*model.m_3[t] + P_r*model.u[t] + P_c*(model.m_6[t] - model.m_7[t] + model.m_8[t])
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.TIME, rule = da_market_clearing_5_rule)
        model.da_State_SOC = pyo.Constraint(rule = da_State_SOC_rule)
        model.rt_E = pyo.Constraint(model.TIME, rule = rt_E_rule)
        model.rt_bidding_price = pyo.Constraint(model.TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule)
        model.rt_market_clearing_1 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        model.without_ESS_1 = pyo.Constraint(model.TIME, rule = without_ESS_1_rule)
        model.without_ESS_2 = pyo.Constraint(model.TIME, rule = without_ESS_2_rule)

        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.TIME, rule = dummy_rule_4_2)
        
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn = pyo.Constraint(model.TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def get_objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
    
        return pyo.value(self.objective)

    def get_objective_part_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
    
        part1 = sum(pyo.value(self.Q_da[t] * self.P_da[t] + (self.u[t] - self.Q_da[t]) * self.P_rt[t]) for t in range(self.T))
        part2_1 = sum(pyo.value(self.m_4_1[t]) for t in range(self.T))
        part2_2 = sum(pyo.value(self.m_4_2[t]) for t in range(self.T))
        part2 = sum(pyo.value(self.m_4[t]) for t in range(self.T))
        part3 = sum(pyo.value(-(self.P_rt[t] - self.b_da[t]) * self.m_3[t]) for t in range(self.T))
        part4 = sum(pyo.value(P_c * (self.m_6[t] - self.m_7[t] + self.m_8[t])) for t in range(self.T))
        part5 = sum(pyo.value(P_r * self.u[t]) for t in range(self.T))
        
        return [part1, part2_1, part2_2, part2, part3, part4, part5]
      
    def get_b_da_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.b_da[t]) for t in range(self.T)] 

    def get_b_rt_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.b_rt[t]) for t in range(self.T)] 

    def get_q_da_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.q_da[t]) for t in range(self.T)] 
  
    def get_q_rt_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.q_rt[t]) for t in range(self.T)]   
    
    def get_Q_da_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.Q_da[t]) for t in range(self.T)] 

    def get_Q_rt_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.Q_rt[t]) for t in range(self.T)] 

    def get_Q_c_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.Q_c[t]) for t in range(self.T)]    

    def get_u_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
    
        return [pyo.value(self.u[t]) for t in range(self.T)] 


# Subproblems for Rolling Horizon

class Rolling_da(pyo.ConcreteModel): # t = -2
    
    def __init__(self, E_0, exp_P_da, exp_P_rt):
        
        super().__init__()

        self.solved = False
        
        self.E_0 = E_0
        
        self.E_0_sum = 0
        
        for t in range(T):
            self.E_0_sum += self.E_0[t]
        
        self.exp_P_da = exp_P_da
        
        self.exp_delta_E = [0 for t in range(T)]
        self.exp_P_rt = exp_P_rt
        self.exp_delta_c = [0 for t in range(T)]
        
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(T)]
        
        self.M_set = [[0, 0] for t in range(T)]
        self.M_set_decomp = [[[0, 0] for i in range(3)] for t in range(T)]

        self.T = T
        self.bin_num = 7
        
        self.sol_b_da = []
        self.sol_q_da = []
        
        self._BigM_setting()
        self._solve()
        self._solution_value()
    
    def _BigM_setting(self):
     
        self.M_gen = [5*self.K[t] for t in range(T)] 
        
        for t in range(self.T):   
             
            if self.exp_P_da[t] >=0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + self.exp_P_da[t] + 2*self.exp_P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 + 2*self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_da[t] + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (self.exp_P_rt[t] + 80)*self.K[t]
                self.M_set_decomp[t][1][1] = (self.exp_P_rt[t] + 80)*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.exp_P_da[t] >=0 and self.exp_P_rt[t] < 0:
                
                self.M_set[t][0] = (160 + self.exp_P_da[t] - 2*self.exp_P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - 2*self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (-self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_da[t] - self.exp_P_rt)*self.K[t]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt)*self.K[t]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt)*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.exp_P_da[t] < 0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + 2*self.exp_P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - self.exp_P_da[t] + 2*self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (-self.exp_P_da[t] + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            else:
                
                self.M_set[t][0] = (160 - 2*self.exp_P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - 2*self.exp_P_da[t] - 2*self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (- self.exp_P_da[t] - self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 - self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1]) 
        
    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= self.E_0[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= self.E_0_sum
        
        def da_market_clearing_1_rule(model, t):
            return model.b_da[t] - self.exp_P_da[t] <= M_price*(1 - model.n_da[t])
        
        def da_market_clearing_2_rule(model, t):
            return self.exp_P_da[t] - model.b_da[t] <= M_price*model.n_da[t]
        
        def da_market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def da_market_clearing_4_rule(model, t):
            return model.Q_da[t] <= self.M_gen[t]*model.n_da[t]
        
        def da_market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_da[t])
        
        def da_State_SOC_rule(model):
            return model.S[-1] == 0.5*S
        
        ## Real-Time Market rules
        
        def rt_E_rule(model, t):
            return model.E_1[t] == self.E_0[t]
        
        def rt_bidding_price_rule(model, t):
            return model.b_rt[t] <= model.b_da[t]
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= model.E_1[t] + B
        
        def rt_overbid_rule(model):
            return sum(model.q_rt[t] for t in range(self.T)) <= self.E_0_sum
        
        def rt_market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= M_price*(1 - model.n_rt[t])
        
        def rt_market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= M_price*model.n_rt[t] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[t]*(1 - model.n_rt[t])
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def last_SOC_rule(model):
            return model.S[T-1] == 0.5*S
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.u[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.u[t] - self.M_gen[t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.exp_P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t])
            
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.exp_P_rt[t] + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, t):
            return model.f_prime[t] == model.Q_da[t]*self.exp_P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] + self.m_4[t] - self.exp_P_rt[t]*model.m_3[t] - sum((2**j)*model.o[t, j] for j in range(self.bin_num)) + P_r*model.u[t] + P_c*(model.m_6[t] - model.m_7[t] + model.m_8[t]) 
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_overbid = pyo.Constraint(rule = da_overbid_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.TIME, rule = da_market_clearing_5_rule)
        model.da_State_SOC = pyo.Constraint(rule = da_State_SOC_rule)
        model.rt_E = pyo.Constraint(model.TIME, rule = rt_E_rule)
        model.rt_bidding_price = pyo.Constraint(model.TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)
        model.rt_market_clearing_1 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        model.last_SOC = pyo.Constraint(rule = last_SOC_rule)
        
        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.TIME, rule = dummy_rule_4_2)     
           
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn = pyo.Constraint(model.TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def _solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def _solution_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
        
        for t in range(T):
            self.sol_b_da.append(pyo.value(self.b_da[t]))
            self.sol_q_da.append(pyo.value(self.q_da[t]))

class Rolling_rt_init(pyo.ConcreteModel): # t = -1

    def __init__(self, da, E_0, P_da, exp_P_rt):
        
        super().__init__()
        
        self.solved = False
        self.stage = 0
        
        self.b_da_prev = da[0]
        self.q_da_prev = da[1]
        
        self.E_0 = E_0
        
        self.E_0_sum = 0
        
        for t in range(T):
            self.E_0_sum += self.E_0[t]
            
        self.P_da = P_da
        
        self.exp_delta_E = [0 for t in range(T)]
        self.exp_P_rt = exp_P_rt
        self.exp_delta_c = [0 for t in range(T)]
        
        self.delta_E = 0
        
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(T)]
        
        self.M_set = [[0, 0] for t in range(T)]
        self.M_set_decomp = [[[0, 0] for i in range(3)] for t in range(T)]

        self.T = T
        self.bin_num = 7
        
        self.sol_Q_da = []
        
        self.sol_b_rt = 0
        self.sol_q_rt = 0
        self.sol_S = 0
        self.sol_E_1 = 0
        self.sol_T_q = 0
        
        self._BigM_setting()
        self._solve()
        self._solution_value()
 
    def _BigM_setting(self):        
        
        self.M_gen = [5*self.K[t] for t in range(T)] 

        for t in range(self.T-self.stage):   
             
            if self.P_da[t] >=0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (self.exp_P_rt[t] + 80)*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (self.exp_P_rt[t] + 80)*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] >=0 and self.exp_P_rt[t] < 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (-self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] < 0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - self.P_da[t] + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (-self.P_da[t] + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            else:
                
                self.M_set[t][0] = (160 - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - 2*self.P_da[t] - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (- self.P_da[t] - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1]) 
      
    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1-self.stage)
        model.RT_BID_TIME = pyo.RangeSet(1, T-1-self.stage)
        model.E_1_TIME = pyo.RangeSet(2, T-1-self.stage)
        model.TIME_ESS = pyo.RangeSet(-1, T-1-self.stage)
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.T_q_next = pyo.Var(bounds = (0, self.E_0_sum), domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to previous solutions
        
        def prev_1_rule(model, t):
            return model.b_da[t] == self.b_da_prev[t]
        
        def prev_2_rule(model, t):
            return model.q_da[t] == self.q_da_prev[t]
        
        ## Day-Ahead Market Rules
        
        def da_market_clearing_1_rule(model, t):
            return model.b_da[t] - self.P_da[t] <= M_price*(1 - model.n_da[t])
        
        def da_market_clearing_2_rule(model, t):
            return self.P_da[t] - model.b_da[t] <= M_price*model.n_da[t]
        
        def da_market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def da_market_clearing_4_rule(model, t):
            return model.Q_da[t] <= self.M_gen[t]*model.n_da[t]
        
        def da_market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_da[t])
        
        def da_State_SOC_rule(model):
            return model.S[-1] == 0.5*S
        
        ## Real-Time Market rules
        
        def rt_E_rule(model, t):
            return model.E_1[t] == self.E_0[t]
        
        def rt_bidding_price_rule(model, t):
            return model.b_rt[t] <= model.b_da[t]
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= model.E_1[t] + B
        
        def rt_overbid_rule(model):
            return sum(model.q_rt[t] for t in range(self.T)) <= self.E_0_sum
        
        def rt_market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= M_price*(1 - model.n_rt[t])
        
        def rt_market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= M_price*model.n_rt[t] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[t]*(1 - model.n_rt[t])
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def last_SOC_rule(model):
            return model.S[T-1] == 0.5*S   
             
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.u[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.u[t] - self.M_gen[t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t])
            
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.exp_P_rt[t] + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, t):
            return model.f_prime[t] == model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] + self.m_4[t] - self.exp_P_rt[t]*model.m_3[t] - sum((2**j)*model.o[t, j] for j in range(self.bin_num)) + P_r*model.u[t] + P_c*(model.m_6[t] - model.m_7[t] + model.m_8[t]) 
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.prev_1 = pyo.Constraint(model.TIME, rule = prev_1_rule)
        model.prev_2 = pyo.Constraint(model.TIME, rule = prev_2_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.TIME, rule = da_market_clearing_5_rule)
        model.da_State_SOC = pyo.Constraint(rule = da_State_SOC_rule)
        model.rt_E = pyo.Constraint(model.TIME, rule = rt_E_rule)
        model.rt_bidding_price = pyo.Constraint(model.TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule)
        model.rt_overbid = pyo.Constraint(rule = rt_overbid_rule)
        model.rt_market_clearing_1 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        model.last_SOC = pyo.Constraint(rule = last_SOC_rule)

        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.TIME, rule = dummy_rule_4_2)
        
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn = pyo.Constraint(model.TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def _solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def _solution_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
        
        for t in range(T):
            self.sol_Q_da.append(pyo.value(self.Q_da[t]))
        
        self.sol_b_rt = pyo.value(self.b_rt[0])
        self.sol_q_rt = pyo.value(self.q_rt[0])
        self.sol_S = 0.5*S
        self.sol_E_1 = 0   
            
    def get_objective_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
    
        return pyo.value(self.f[0])    
                  
class Rolling_rt(pyo.ConcreteModel): # t = 0, ..., 22
    
    def __init__(
        self, t, da, b_rt, q_rt, S, E_1, T_q, E_0, P_da, P_rt, delta_E, exp_P_rt, delta
        ):
        
        super().__init__()

        self.solved = False
        self.stage = t
        
        self.b_da_prev = da[0]
        self.Q_da_prev = da[1]
        self.b_rt_prev = b_rt
        self.q_rt_prev = q_rt
        self.S_prev = S
        self.E_1_prev = E_1
        self.T_q = T_q
        
        self.E_0 = E_0
        
        self.E_0_sum = 0
        
        for t in range(T):
            self.E_0_sum += self.E_0[t]
        
        self.P_da = [P_da[t] for t in range(self.stage, T)]
        
        self.exp_P_rt = [exp_P_rt[t] for t in range(self.stage, T)]
        
        self.delta_E = delta_E
        self.P_rt = P_rt
        self.delta_c = delta[2]
        
        self.K = [1.3*E_0[t] + 1.1*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(T)]
        
        self.M_set = [[0, 0] for t in range(T-self.stage)]
        self.M_set_decomp = [[[0, 0] for i in range(3)] for t in range(T-self.stage)]

        self.T = T
        self.bin_num = 7

        self.sol_b_rt = 0
        self.sol_q_rt = 0
        self.sol_S = 0
        self.sol_E_1 = 0
        self.sol_T_q = 0
        
        self._BigM_setting()
        self._solve()
        self._solution_value()
    
    def _BigM_setting(self):        
        
        self.M_gen = [5 * self.K[t] for t in range(self.T)]

        fixed_price = 500  # uniform price term

        for t in range(self.T - self.stage):
            stage_index = t + self.stage
            K_val = self.K[stage_index]

            self.M_set[t][0] = fixed_price * K_val
            self.M_set[t][1] = fixed_price * K_val

            self.M_set_decomp[t][0][0] = fixed_price * K_val
            self.M_set_decomp[t][0][1] = fixed_price * K_val
            self.M_set_decomp[t][1][0] = fixed_price * K_val
            self.M_set_decomp[t][1][1] = fixed_price * K_val
            self.M_set_decomp[t][2][0] = fixed_price * K_val
            self.M_set_decomp[t][2][1] = fixed_price * K_val
        
        """        
        if self.P_da[0] >=0 and self.P_rt >= 0:
            
            self.M_set[0][0] = (160 + self.P_da[0] + 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 + 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_da[0] + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (self.P_rt + 80)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (self.P_rt + 80)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        elif self.P_da[0] >=0 and self.P_rt < 0:
            
            self.M_set[0][0] = (160 + self.P_da[0] - 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (-self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_da[0] - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        elif self.P_da[0] < 0 and self.P_rt >= 0:
            
            self.M_set[0][0] = (160 + 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - self.P_da[0] + 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (-self.P_da[0] + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        else:
            
            self.M_set[0][0] = (160 - 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - 2*self.P_da[0] - 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (- self.P_da[0] - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1]) 
        
        for t in range(1, self.T-self.stage):   
             
            if self.P_da[t] >=0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (self.exp_P_rt[t] + 80)*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (self.exp_P_rt[t] + 80)*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] >=0 and self.exp_P_rt[t] < 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (-self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] < 0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - self.P_da[t] + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (-self.P_da[t] + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            else:
                
                self.M_set[t][0] = (160 - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - 2*self.P_da[t] - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (- self.P_da[t] - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1]) 
                """
            
    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1-self.stage)
        model.RT_BID_TIME = pyo.RangeSet(1, T-1-self.stage)
        model.E_1_TIME = pyo.RangeSet(2, T-1-self.stage)
        model.TIME_ESS = pyo.RangeSet(-1, T-1-self.stage)
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.T_q_next = pyo.Var(bounds = (0, self.E_0_sum), domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to previous solutions
        
        def prev_1_rule(model, t):
            return model.b_da[t] == self.b_da_prev[t]
        
        def prev_2_rule(model, t):
            return model.Q_da[t] == self.Q_da_prev[t]
        
        def prev_3_rule(model):
            return model.b_rt[0] == self.b_rt_prev
        
        def prev_4_rule(model):
            return model.q_rt[0] == self.q_rt_prev
        
        def prev_5_rule(model):
            return model.E_1[0] == self.E_1_prev
                
        def prev_State_SOC_rule(model):
            return model.S[-1] == self.S_prev
        
        ## Real-Time Market rules
        
        def rt_next_E_rule(model):
            return model.E_1[1] == self.E_0[self.stage+1]*self.delta_E
        
        def rt_E_rule(model, t):
            return model.E_1[t] == self.E_0[self.stage+t]
        
        def rt_bidding_price_rule(model, t):
            return model.b_rt[t] <= model.b_da[t]
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= model.E_1[t] + B
        
        def rt_overbid_state_rule(model):
            return self.T_q + sum(model.q_rt[t] for t in range(1, T-self.stage)) <= self.E_0_sum
        
        def rt_overbid_sum_rule(model):
            return self.T_q + model.q_rt[1] == model.T_q_next
        
        def rt_prev_market_clearing_rule_1(model):
            return model.b_rt[0] - self.P_rt <= M_price*(1 - model.n_rt[0])
        
        def rt_prev_market_clearing_rule_2(model):
            return self.P_rt - model.b_rt[0] <= M_price*model.n_rt[0] 
        
        def rt_market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= M_price*(1 - model.n_rt[t])
        
        def rt_market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= M_price*model.n_rt[t] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[self.stage+t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[self.stage+t]*(1 - model.n_rt[t])
      
        def dispatch_prev_rule(model):
            return model.Q_c[0] == (1 + self.delta_c)*model.Q_rt[0]      
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def last_SOC_rule(model):
            return model.S[T-1-self.stage] == 0.5*S
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.u[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[self.stage+t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.u[t] - self.M_gen[self.stage+t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[self.stage+t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[self.stage+t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[self.stage+t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[self.stage+t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[self.stage+t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[self.stage+t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def init_dummy_rule_4_1(model):
            return model.m_4_1[0] == -sum(model.w[0, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[0]*self.P_da[0] + (model.u[0] - model.Q_da[0])*self.P_rt)
            
        def init_dummy_rule_4_2(model):
            return model.m_4_2[0] == (model.m_1[0] - model.m_2[0])*self.P_rt + sum((model.h[0, i] - model.k[0, i])*(2**i) for i in range(self.bin_num))
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t])
            
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.exp_P_rt[t] + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[self.stage + t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[self.stage + t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[self.stage + t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[self.stage + t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[self.stage + t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[self.stage + t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[self.stage + t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[self.stage + t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[self.stage + t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[self.stage + t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[self.stage + t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[self.stage + t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[self.stage + t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[self.stage + t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[self.stage + t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[self.stage + t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[self.stage + t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[self.stage + t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[self.stage + t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[self.stage + t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[self.stage + t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[self.stage + t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_current_rule(model):
            return model.f_prime[0] == model.Q_da[0]*self.P_da[0] + (model.u[0] - model.Q_da[0])*self.P_rt + self.m_4[0] - self.P_rt*model.m_3[0] - sum((2**j)*model.o[0, j] for j in range(self.bin_num)) + P_r*model.u[0] + P_c*(model.m_6[0] - model.m_7[0] + model.m_8[0])
        
        def settlement_fcn_rule(model, t):
            return model.f_prime[t] == model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] + self.m_4[t] - self.exp_P_rt[t]*model.m_3[t] - sum((2**j)*model.o[t, j] for j in range(self.bin_num)) + P_r*model.u[t] + P_c*(model.m_6[t] - model.m_7[t] + model.m_8[t])
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[self.stage+t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.prev_1 = pyo.Constraint(model.TIME, rule = prev_1_rule)
        model.prev_2 = pyo.Constraint(model.TIME, rule = prev_2_rule)
        model.prev_3 = pyo.Constraint(rule = prev_3_rule)
        model.prev_4 = pyo.Constraint(rule = prev_4_rule)
        model.prev_5 = pyo.Constraint(rule = prev_5_rule)
        model.prev_State_SOC = pyo.Constraint(rule = prev_State_SOC_rule)
        model.rt_next_E = pyo.Constraint(rule = rt_next_E_rule)
        model.rt_E = pyo.Constraint(model.E_1_TIME, rule = rt_E_rule)
        model.rt_bidding_price = pyo.Constraint(model.RT_BID_TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount = pyo.Constraint(model.RT_BID_TIME, rule = rt_bidding_amount_rule)
        model.rt_overbid_state = pyo.Constraint(rule = rt_overbid_state_rule)
        model.rt_overbid_sum = pyo.Constraint(rule = rt_overbid_sum_rule)
        model.rt_prev_market_clearing_1 = pyo.Constraint(rule = rt_prev_market_clearing_rule_1)
        model.rt_prev_market_clearing_2 = pyo.Constraint(rule = rt_prev_market_clearing_rule_2)
        model.rt_market_clearing_1 = pyo.Constraint(model.RT_BID_TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.RT_BID_TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch_prev = pyo.Constraint(rule = dispatch_prev_rule)
        model.dispatch = pyo.Constraint(model.RT_BID_TIME, rule = dispatch_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        #model.last_SOC = pyo.Constraint(rule = last_SOC_rule)

        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
    
        model.init_dummy_4_1 = pyo.Constraint(rule = init_dummy_rule_4_1)
        model.init_dummy_4_2 = pyo.Constraint(rule = init_dummy_rule_4_2)        
        model.dummy_4_1 = pyo.Constraint(model.RT_BID_TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.RT_BID_TIME, rule = dummy_rule_4_2)
        
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn_current = pyo.Constraint(rule = settlement_fcn_current_rule)
        model.settlement_fcn = pyo.Constraint(model.RT_BID_TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def _solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def _solution_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
        
        self.sol_b_rt = pyo.value(self.b_rt[1])
        self.sol_q_rt = pyo.value(self.q_rt[1])
        self.sol_S = pyo.value(self.S[0])
        self.sol_E_1 = pyo.value(self.E_1[1])
        self.sol_T_q = pyo.value(self.T_q_next)
            
    def get_objective_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
    
        return pyo.value(self.f[0])    

class Rolling_rt_last(pyo.ConcreteModel): # t = 23
    
    def __init__(self, da, b_rt, q_rt, S, E_1, E_0, P_da, P_rt, delta_E, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T-1
        
        self.b_da_prev = da[0]
        self.Q_da_prev = da[1]
        self.b_rt_prev = b_rt
        self.q_rt_prev = q_rt    
        self.S_prev = S
        self.E_1_prev = E_1
        
        self.E_0 = E_0
        
        self.P_da = P_da[self.stage]
        
        self.delta_E = delta_E
        self.P_rt = P_rt
        self.delta_c = delta[2]      
          
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(T)]    
              
        self.M_set = [[0, 0]]
        self.M_set_decomp = [[[0, 0] for i in range(3)]]

        self.T = T
        self.bin_num = 7
        
        self._BigM_setting()
        self._solve()
    
    def _BigM_setting(self):
    
        self.M_gen = [5*self.K[t] for t in range(T)] 
        
        if self.P_da >=0 and self.P_rt >= 0:
            
            self.M_set[0][0] = (160 + self.P_da + 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 + 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_da + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (self.P_rt + 80)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (self.P_rt + 80)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        elif self.P_da >=0 and self.P_rt < 0:
            
            self.M_set[0][0] = (160 + self.P_da - 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (-self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_da - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        elif self.P_da < 0 and self.P_rt >= 0:
            
            self.M_set[0][0] = (160 + 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - self.P_da + 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (-self.P_da + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        else:
            
            self.M_set[0][0] = (160 - 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - 2*self.P_da - 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (- self.P_da - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])    

    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1-self.stage) # 0
        model.TIME_ESS = pyo.RangeSet(-1, T-1-self.stage) # -1, 0
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1) 
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to previous solutions
        
        def prev_1_rule(model, t):
            return model.b_da[t] == self.b_da_prev[t]
        
        def prev_2_rule(model, t):
            return model.Q_da[t] == self.Q_da_prev[t]
        
        def prev_3_rule(model):
            return model.b_rt[0] == self.b_rt_prev
        
        def prev_4_rule(model):
            return model.q_rt[0] == self.q_rt_prev
        
        def prev_5_rule(model):
            return model.E_1[0] == self.E_1_prev
        
        ## Day-Ahead Market Rules
        
        def prev_State_SOC_rule(model):
            return model.S[-1] == self.S_prev
        
        ## Real-Time Market rules
        
        def rt_prev_market_clearing_rule_1(model):
            return model.b_rt[0] - self.P_rt <= M_price*(1 - model.n_rt[0])
        
        def rt_prev_market_clearing_rule_2(model):
            return self.P_rt - model.b_rt[0] <= M_price*model.n_rt[0] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[self.stage+t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[self.stage+t]*(1 - model.n_rt[t])
      
        def dispatch_prev_rule(model):
            return model.Q_c[0] == (1 + self.delta_c)*model.Q_rt[0]      
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def last_SOC_rule(model):
            return model.S[T-1-self.stage] == 0.5*S
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.u[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[self.stage+t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.u[t] - self.M_gen[self.stage+t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[self.stage+t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[self.stage+t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[self.stage+t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[self.stage+t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[self.stage+t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[self.stage+t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.P_da + (model.u[t] - model.Q_da[t])*self.P_rt)
        
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.P_rt + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[self.stage + t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[self.stage + t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[self.stage + t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[self.stage + t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[self.stage + t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[self.stage + t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[self.stage + t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[self.stage + t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[self.stage + t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[self.stage + t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[self.stage + t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[self.stage + t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[self.stage + t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[self.stage + t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[self.stage + t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[self.stage + t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[self.stage + t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[self.stage + t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[self.stage + t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[self.stage + t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[self.stage + t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[self.stage + t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_current_rule(model):
            return model.f_prime[0] == model.Q_da[0]*self.P_da + (model.u[0] - model.Q_da[0])*self.P_rt + self.m_4[0] - self.P_rt*model.m_3[0] - sum((2**j)*model.o[0, j] for j in range(self.bin_num)) + P_c*(model.m_6[0] - model.m_7[0] + model.m_8[0]) 
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[self.stage+t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.prev_1 = pyo.Constraint(model.TIME, rule = prev_1_rule)
        model.prev_2 = pyo.Constraint(model.TIME, rule = prev_2_rule)
        model.prev_3 = pyo.Constraint(rule = prev_3_rule)
        model.prev_4 = pyo.Constraint(rule = prev_4_rule)
        model.prev_5 = pyo.Constraint(rule = prev_5_rule)
        model.prev_State_SOC = pyo.Constraint(rule = prev_State_SOC_rule)
        model.rt_prev_market_clearing_1 = pyo.Constraint(rule = rt_prev_market_clearing_rule_1)
        model.rt_prev_market_clearing_2 = pyo.Constraint(rule = rt_prev_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch_prev = pyo.Constraint(rule = dispatch_prev_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        #model.last_SOC = pyo.Constraint(rule = last_SOC_rule)

        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.TIME, rule = dummy_rule_4_2)

        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn_current = pyo.Constraint(rule = settlement_fcn_current_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def _solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
            
    def get_objective_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
    
        return pyo.value(self.f[0])    



class Rolling_da_ESSx(pyo.ConcreteModel): # t = -2
    
    def __init__(self, E_0, exp_P_da, exp_P_rt):
        
        super().__init__()

        self.solved = False
        
        self.E_0 = E_0
        
        self.E_0_sum = 0
        
        for t in range(T):
            self.E_0_sum += self.E_0[t]
        
        self.exp_P_da = exp_P_da
        
        self.exp_delta_E = [0 for t in range(T)]
        self.exp_P_rt = exp_P_rt
        self.exp_delta_c = [0 for t in range(T)]
        
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(T)]
        
        self.M_set = [[0, 0] for t in range(T)]
        self.M_set_decomp = [[[0, 0] for i in range(3)] for t in range(T)]

        self.T = T
        self.bin_num = 7
        
        self.sol_b_da = []
        self.sol_q_da = []
        
        self._BigM_setting()
        self._solve()
        self._solution_value()
    
    def _BigM_setting(self):
     
        self.M_gen = [5*self.K[t] for t in range(T)] 
        
        for t in range(self.T):   
             
            if self.exp_P_da[t] >=0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + self.exp_P_da[t] + 2*self.exp_P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 + 2*self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_da[t] + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (self.exp_P_rt[t] + 80)*self.K[t]
                self.M_set_decomp[t][1][1] = (self.exp_P_rt[t] + 80)*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.exp_P_da[t] >=0 and self.exp_P_rt[t] < 0:
                
                self.M_set[t][0] = (160 + self.exp_P_da[t] - 2*self.exp_P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - 2*self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (-self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_da[t] - self.exp_P_rt)*self.K[t]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt)*self.K[t]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt)*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.exp_P_da[t] < 0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + 2*self.exp_P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - self.exp_P_da[t] + 2*self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (-self.exp_P_da[t] + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 + self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            else:
                
                self.M_set[t][0] = (160 - 2*self.exp_P_rt[t])*self.K[t]
                self.M_set[t][1] = (80 - 2*self.exp_P_da[t] - 2*self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][0] = (- self.exp_P_da[t] - self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][0][1] = (80 - self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt[t])*self.K[t]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1]) 
        
    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        model.TIME_ESS = pyo.RangeSet(-1, T-1)
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Day-Ahead Market Rules
        
        def da_bidding_amount_rule(model, t):
            return model.q_da[t] <= self.E_0[t]
        
        def da_market_clearing_1_rule(model, t):
            return model.b_da[t] - self.exp_P_da[t] <= M_price*(1 - model.n_da[t])
        
        def da_market_clearing_2_rule(model, t):
            return self.exp_P_da[t] - model.b_da[t] <= M_price*model.n_da[t]
        
        def da_market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def da_market_clearing_4_rule(model, t):
            return model.Q_da[t] <= self.M_gen[t]*model.n_da[t]
        
        def da_market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_da[t])
        
        def da_State_SOC_rule(model):
            return model.S[-1] == 0.5*S
        
        def without_ESS_1_rule(model, t):
            return model.c[t] == 0
        
        def without_ESS_2_rule(model, t):
            return model.d[t] == 0
        
        ## Real-Time Market rules
        
        def rt_E_rule(model, t):
            return model.E_1[t] == self.E_0[t]
        
        def rt_bidding_price_rule(model, t):
            return model.b_rt[t] <= model.b_da[t]
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= model.E_1[t]
        
        def rt_market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= M_price*(1 - model.n_rt[t])
        
        def rt_market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= M_price*model.n_rt[t] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[t]*(1 - model.n_rt[t])
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def last_SOC_rule(model):
            return model.S[T-1] == 0.5*S
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.u[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.u[t] - self.M_gen[t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.exp_P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t])
            
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.exp_P_rt[t] + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, t):
            return model.f_prime[t] == model.Q_da[t]*self.exp_P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] + self.m_4[t] - self.exp_P_rt[t]*model.m_3[t] - sum((2**j)*model.o[t, j] for j in range(self.bin_num)) + P_r*model.u[t] + P_c*(model.m_6[t] - model.m_7[t] + model.m_8[t]) 
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.da_bidding_amount = pyo.Constraint(model.TIME, rule = da_bidding_amount_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.TIME, rule = da_market_clearing_5_rule)
        model.da_State_SOC = pyo.Constraint(rule = da_State_SOC_rule)
        model.rt_E = pyo.Constraint(model.TIME, rule = rt_E_rule)
        model.rt_bidding_price = pyo.Constraint(model.TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule)
        model.rt_market_clearing_1 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        model.last_SOC = pyo.Constraint(rule = last_SOC_rule)
        model.without_ESS_1 = pyo.Constraint(model.TIME, rule = without_ESS_1_rule)
        model.without_ESS_2 = pyo.Constraint(model.TIME, rule = without_ESS_2_rule)
        
        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.TIME, rule = dummy_rule_4_2)
        
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn = pyo.Constraint(model.TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def _solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def _solution_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
        
        for t in range(T):
            self.sol_b_da.append(pyo.value(self.b_da[t]))
            self.sol_q_da.append(pyo.value(self.q_da[t]))

class Rolling_rt_init_ESSx(pyo.ConcreteModel): # t = -1

    def __init__(self, da, E_0, P_da, exp_P_rt):
        
        super().__init__()
        
        self.solved = False
        self.stage = 0
        
        self.b_da_prev = da[0]
        self.q_da_prev = da[1]
        
        self.E_0 = E_0
        
        self.E_0_sum = 0
        
        for t in range(T):
            self.E_0_sum += self.E_0[t]
            
        self.P_da = P_da
        
        self.exp_delta_E = [0 for t in range(T)]
        self.exp_P_rt = exp_P_rt
        self.exp_delta_c = [0 for t in range(T)]
        
        self.delta_E = 0
        
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(T)]
        
        self.M_set = [[0, 0] for t in range(T)]
        self.M_set_decomp = [[[0, 0] for i in range(3)] for t in range(T)]

        self.T = T
        self.bin_num = 7
        
        self.sol_Q_da = []
        
        self.sol_b_rt = 0
        self.sol_q_rt = 0
        self.sol_S = 0
        self.sol_E_1 = 0
        self.sol_T_q = 0
        
        self._BigM_setting()
        self._solve()
        self._solution_value()
 
    def _BigM_setting(self):        
        
        self.M_gen = [5*self.K[t] for t in range(T)] 

        for t in range(self.T-self.stage):   
             
            if self.P_da[t] >=0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (self.exp_P_rt[t] + 80)*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (self.exp_P_rt[t] + 80)*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] >=0 and self.exp_P_rt[t] < 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (-self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] < 0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - self.P_da[t] + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (-self.P_da[t] + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            else:
                
                self.M_set[t][0] = (160 - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - 2*self.P_da[t] - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (- self.P_da[t] - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1]) 
      
    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1-self.stage)
        model.RT_BID_TIME = pyo.RangeSet(1, T-1-self.stage)
        model.E_1_TIME = pyo.RangeSet(2, T-1-self.stage)
        model.TIME_ESS = pyo.RangeSet(-1, T-1-self.stage)
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.T_q_next = pyo.Var(bounds = (0, self.E_0_sum), initialize = 0.0, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to previous solutions
        
        def prev_1_rule(model, t):
            return model.b_da[t] == self.b_da_prev[t]
        
        def prev_2_rule(model, t):
            return model.q_da[t] == self.q_da_prev[t]
        
        ## Day-Ahead Market Rules
        
        def da_market_clearing_1_rule(model, t):
            return model.b_da[t] - self.P_da[t] <= M_price*(1 - model.n_da[t])
        
        def da_market_clearing_2_rule(model, t):
            return self.P_da[t] - model.b_da[t] <= M_price*model.n_da[t]
        
        def da_market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def da_market_clearing_4_rule(model, t):
            return model.Q_da[t] <= self.M_gen[t]*model.n_da[t]
        
        def da_market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_da[t])
        
        def da_State_SOC_rule(model):
            return model.S[-1] == 0.5*S
        
        ## Real-Time Market rules
        
        def rt_E_rule(model, t):
            return model.E_1[t] == self.E_0[t]
        
        def rt_bidding_price_rule(model, t):
            return model.b_rt[t] <= model.b_da[t]
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= model.E_1[t]

        
        def rt_market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= M_price*(1 - model.n_rt[t])
        
        def rt_market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= M_price*model.n_rt[t] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[t]*(1 - model.n_rt[t])
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def last_SOC_rule(model):
            return model.S[T-1] == 0.5*S   
             
        def without_ESS_1_rule(model, t):
            return model.c[t] == 0
        
        def without_ESS_2_rule(model, t):
            return model.d[t] == 0          
           
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.u[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.u[t] - self.M_gen[t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t])
            
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.exp_P_rt[t] + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
     
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model, t):
            return model.f_prime[t] == model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] + self.m_4[t] - self.exp_P_rt[t]*model.m_3[t] - sum((2**j)*model.o[t, j] for j in range(self.bin_num)) + P_r*model.u[t] + P_c*(model.m_6[t] - model.m_7[t] + model.m_8[t]) 
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.prev_1 = pyo.Constraint(model.TIME, rule = prev_1_rule)
        model.prev_2 = pyo.Constraint(model.TIME, rule = prev_2_rule)
        model.da_market_clearing_1 = pyo.Constraint(model.TIME, rule = da_market_clearing_1_rule)
        model.da_market_clearing_2 = pyo.Constraint(model.TIME, rule = da_market_clearing_2_rule)
        model.da_market_clearing_3 = pyo.Constraint(model.TIME, rule = da_market_clearing_3_rule)
        model.da_market_clearing_4 = pyo.Constraint(model.TIME, rule = da_market_clearing_4_rule)
        model.da_market_clearing_5 = pyo.Constraint(model.TIME, rule = da_market_clearing_5_rule)
        model.da_State_SOC = pyo.Constraint(rule = da_State_SOC_rule)
        model.rt_E = pyo.Constraint(model.TIME, rule = rt_E_rule)
        model.rt_bidding_price = pyo.Constraint(model.TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount = pyo.Constraint(model.TIME, rule = rt_bidding_amount_rule)
        model.rt_market_clearing_1 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch = pyo.Constraint(model.TIME, rule = dispatch_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        model.last_SOC = pyo.Constraint(rule = last_SOC_rule)
        model.without_ESS_1 = pyo.Constraint(model.TIME, rule = without_ESS_1_rule)
        model.without_ESS_2 = pyo.Constraint(model.TIME, rule = without_ESS_2_rule)
        
        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.TIME, rule = dummy_rule_4_2)
        
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn = pyo.Constraint(model.TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def _solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def _solution_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
        
        for t in range(T):
            self.sol_Q_da.append(pyo.value(self.Q_da[t]))
        
        self.sol_b_rt = pyo.value(self.b_rt[0])
        self.sol_q_rt = pyo.value(self.q_rt[0])
        self.sol_S = 0.5*S
        self.sol_E_1 = 0   
            
    def get_objective_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
    
        return pyo.value(self.f[0])    
                  
class Rolling_rt_ESSx(pyo.ConcreteModel): # t = 0, ..., 22
    
    def __init__(
        self, t, da, b_rt, q_rt, S, E_1, T_q, E_0, P_da, P_rt, delta_E, exp_P_rt, delta
        ):
        
        super().__init__()

        self.solved = False
        self.stage = t
        
        self.b_da_prev = da[0]
        self.Q_da_prev = da[1]
        self.b_rt_prev = b_rt
        self.q_rt_prev = q_rt
        self.S_prev = S
        self.E_1_prev = E_1
        self.T_q = T_q
        
        self.E_0 = E_0
        
        self.E_0_sum = 0
        
        for t in range(T):
            self.E_0_sum += self.E_0[t]
        
        self.P_da = [P_da[t] for t in range(self.stage, T)]
        
        self.exp_P_rt = [exp_P_rt[t] for t in range(self.stage, T)]
        
        self.delta_E = delta_E
        self.P_rt = P_rt
        self.delta_c = delta[2]
        
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(T)]
        
        self.M_set = [[0, 0] for t in range(T-self.stage)]
        self.M_set_decomp = [[[0, 0] for i in range(3)] for t in range(T-self.stage)]

        self.T = T
        self.bin_num = 7

        self.sol_b_rt = 0
        self.sol_q_rt = 0
        self.sol_S = 0
        self.sol_E_1 = 0
        self.sol_T_q = 0
        
        self._BigM_setting()
        self._solve()
        self._solution_value()
    
    def _BigM_setting(self):        
        
        self.M_gen = [5*self.K[t] for t in range(T)] 
        
        if self.P_da[0] >=0 and self.P_rt >= 0:
            
            self.M_set[0][0] = (160 + self.P_da[0] + 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 + 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_da[0] + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (self.P_rt + 80)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (self.P_rt + 80)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        elif self.P_da[0] >=0 and self.P_rt < 0:
            
            self.M_set[0][0] = (160 + self.P_da[0] - 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (-self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_da[0] - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        elif self.P_da[0] < 0 and self.P_rt >= 0:
            
            self.M_set[0][0] = (160 + 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - self.P_da[0] + 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (-self.P_da[0] + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        else:
            
            self.M_set[0][0] = (160 - 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - 2*self.P_da[0] - 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (- self.P_da[0] - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1]) 
        
        for t in range(1, self.T-self.stage):   
             
            if self.P_da[t] >=0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (self.exp_P_rt[t] + 80)*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (self.exp_P_rt[t] + 80)*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] >=0 and self.exp_P_rt[t] < 0:
                
                self.M_set[t][0] = (160 + self.P_da[t] - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (-self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.P_da[t] - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt)*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            elif self.P_da[t] < 0 and self.exp_P_rt[t] >= 0:
                
                self.M_set[t][0] = (160 + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - self.P_da[t] + 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (-self.P_da[t] + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 + self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1])

            else:
                
                self.M_set[t][0] = (160 - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set[t][1] = (80 - 2*self.P_da[t] - 2*self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][0] = (- self.P_da[t] - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][0][1] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][0] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][1][1] = (80 - self.exp_P_rt[t])*self.K[t+self.stage]
                self.M_set_decomp[t][2][0] = max(self.M_set_decomp[t][0][0], self.M_set_decomp[t][1][0])
                self.M_set_decomp[t][2][1] = min(self.M_set_decomp[t][0][1], self.M_set_decomp[t][1][1]) 
            
    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1-self.stage)
        model.RT_BID_TIME = pyo.RangeSet(1, T-1-self.stage)
        model.E_1_TIME = pyo.RangeSet(2, T-1-self.stage)
        model.TIME_ESS = pyo.RangeSet(-1, T-1-self.stage)
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1)
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.T_q_next = pyo.Var(bounds = (0, self.E_0_sum), initialize = 0.0, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to previous solutions
        
        def prev_1_rule(model, t):
            return model.b_da[t] == self.b_da_prev[t]
        
        def prev_2_rule(model, t):
            return model.Q_da[t] == self.Q_da_prev[t]
        
        def prev_3_rule(model):
            return model.b_rt[0] == self.b_rt_prev
        
        def prev_4_rule(model):
            return model.q_rt[0] == self.q_rt_prev
        
        def prev_5_rule(model):
            return model.E_1[0] == self.E_1_prev
                
        def prev_State_SOC_rule(model):
            return model.S[-1] == self.S_prev
        
        ## Real-Time Market rules
        
        def rt_next_E_rule(model):
            return model.E_1[1] == self.E_0[self.stage+1]*self.delta_E
        
        def rt_E_rule(model, t):
            return model.E_1[t] == self.E_0[self.stage+t]
        
        def rt_bidding_price_rule(model, t):
            return model.b_rt[t] <= model.b_da[t]
        
        def rt_bidding_amount_rule(model, t):
            return model.q_rt[t] <= model.E_1[t]
        
        def rt_overbid_sum_rule(model):
            return self.T_q + model.q_rt[1] == model.T_q_next
        
        def rt_prev_market_clearing_rule_1(model):
            return model.b_rt[0] - self.P_rt <= M_price*(1 - model.n_rt[0])
        
        def rt_prev_market_clearing_rule_2(model):
            return self.P_rt - model.b_rt[0] <= M_price*model.n_rt[0] 
        
        def rt_market_clearing_rule_1(model, t):
            return model.b_rt[t] - self.exp_P_rt[t] <= M_price*(1 - model.n_rt[t])
        
        def rt_market_clearing_rule_2(model, t):
            return self.exp_P_rt[t] - model.b_rt[t] <= M_price*model.n_rt[t] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[self.stage+t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[self.stage+t]*(1 - model.n_rt[t])
      
        def dispatch_prev_rule(model):
            return model.Q_c[0] == (1 + self.delta_c)*model.Q_rt[0]      
        
        def dispatch_rule(model, t):
            return model.Q_c[t] == model.Q_rt[t]
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def last_SOC_rule(model):
            return model.S[T-1-self.stage] == 0.5*S
        
        def without_ESS_1_rule(model, t):
            return model.c[t] == 0
        
        def without_ESS_2_rule(model, t):
            return model.d[t] == 0
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.u[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[self.stage+t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.u[t] - self.M_gen[self.stage+t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[self.stage+t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[self.stage+t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[self.stage+t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[self.stage+t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[self.stage+t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[self.stage+t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def init_dummy_rule_4_1(model):
            return model.m_4_1[0] == -sum(model.w[0, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[0]*self.P_da[0] + (model.u[0] - model.Q_da[0])*self.P_rt)
            
        def init_dummy_rule_4_2(model):
            return model.m_4_2[0] == (model.m_1[0] - model.m_2[0])*self.P_rt + sum((model.h[0, i] - model.k[0, i])*(2**i) for i in range(self.bin_num))
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t])
            
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.exp_P_rt[t] + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        
        
        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[self.stage + t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[self.stage + t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[self.stage + t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[self.stage + t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[self.stage + t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[self.stage + t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[self.stage + t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[self.stage + t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[self.stage + t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[self.stage + t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[self.stage + t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[self.stage + t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[self.stage + t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[self.stage + t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[self.stage + t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[self.stage + t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[self.stage + t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[self.stage + t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[self.stage + t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[self.stage + t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[self.stage + t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[self.stage + t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        ## Settlement fcn
        
        def settlement_fcn_current_rule(model):
            return model.f_prime[0] == model.Q_da[0]*self.P_da[0] + (model.u[0] - model.Q_da[0])*self.P_rt + self.m_4[0] - self.P_rt*model.m_3[0] - sum((2**j)*model.o[0, j] for j in range(self.bin_num)) + P_r*model.u[0] + P_c*(model.m_6[0] - model.m_7[0] + model.m_8[0])
        
        def settlement_fcn_rule(model, t):
            return model.f_prime[t] == model.Q_da[t]*self.P_da[t] + (model.u[t] - model.Q_da[t])*self.exp_P_rt[t] + self.m_4[t] - self.exp_P_rt[t]*model.m_3[t] - sum((2**j)*model.o[t, j] for j in range(self.bin_num)) + P_r*model.u[t] + P_c*(model.m_6[t] - model.m_7[t] + model.m_8[t])
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[self.stage+t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.prev_1 = pyo.Constraint(model.TIME, rule = prev_1_rule)
        model.prev_2 = pyo.Constraint(model.TIME, rule = prev_2_rule)
        model.prev_3 = pyo.Constraint(rule = prev_3_rule)
        model.prev_4 = pyo.Constraint(rule = prev_4_rule)
        model.prev_5 = pyo.Constraint(rule = prev_5_rule)
        model.prev_State_SOC = pyo.Constraint(rule = prev_State_SOC_rule)
        model.rt_next_E = pyo.Constraint(rule = rt_next_E_rule)
        model.rt_E = pyo.Constraint(model.E_1_TIME, rule = rt_E_rule)
        model.rt_bidding_price = pyo.Constraint(model.RT_BID_TIME, rule = rt_bidding_price_rule)
        model.rt_bidding_amount = pyo.Constraint(model.RT_BID_TIME, rule = rt_bidding_amount_rule)
        model.rt_overbid_sum = pyo.Constraint(rule = rt_overbid_sum_rule)
        model.rt_prev_market_clearing_1 = pyo.Constraint(rule = rt_prev_market_clearing_rule_1)
        model.rt_prev_market_clearing_2 = pyo.Constraint(rule = rt_prev_market_clearing_rule_2)
        model.rt_market_clearing_1 = pyo.Constraint(model.RT_BID_TIME, rule = rt_market_clearing_rule_1)
        model.rt_market_clearing_2 = pyo.Constraint(model.RT_BID_TIME, rule = rt_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch_prev = pyo.Constraint(rule = dispatch_prev_rule)
        model.dispatch = pyo.Constraint(model.RT_BID_TIME, rule = dispatch_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        model.last_SOC = pyo.Constraint(rule = last_SOC_rule)
        model.without_ESS_1 = pyo.Constraint(model.TIME, rule = without_ESS_1_rule)
        model.without_ESS_2 = pyo.Constraint(model.TIME, rule = without_ESS_2_rule)        

        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
    
        model.init_dummy_4_1 = pyo.Constraint(rule = init_dummy_rule_4_1)
        model.init_dummy_4_2 = pyo.Constraint(rule = init_dummy_rule_4_2)        
        model.dummy_4_1 = pyo.Constraint(model.RT_BID_TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.RT_BID_TIME, rule = dummy_rule_4_2)
        
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn_current = pyo.Constraint(rule = settlement_fcn_current_rule)
        model.settlement_fcn = pyo.Constraint(model.RT_BID_TIME, rule = settlement_fcn_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def _solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
        
    def _solution_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
        
        self.sol_b_rt = pyo.value(self.b_rt[1])
        self.sol_q_rt = pyo.value(self.q_rt[1])
        self.sol_S = pyo.value(self.S[0])
        self.sol_E_1 = pyo.value(self.E_1[1])
        self.sol_T_q = pyo.value(self.T_q_next)
            
    def get_objective_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
    
        return pyo.value(self.f[0])    

class Rolling_rt_last_ESSx(pyo.ConcreteModel): # t = 23
    
    def __init__(self, da, b_rt, q_rt, S, E_1, E_0, P_da, P_rt, delta_E, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = T-1
        
        self.b_da_prev = da[0]
        self.Q_da_prev = da[1]
        self.b_rt_prev = b_rt
        self.q_rt_prev = q_rt    
        self.S_prev = S
        self.E_1_prev = E_1
        
        self.E_0 = E_0
        
        self.P_da = P_da[self.stage]
        
        self.delta_E = delta_E
        self.P_rt = P_rt
        self.delta_c = delta[2]      
          
        self.K = [1.22*E_0[t] + 1.01*B for t in range(T)]
        
        self.M_gen = [0 for _ in range(T)]    
              
        self.M_set = [[0, 0]]
        self.M_set_decomp = [[[0, 0] for i in range(3)]]

        self.T = T
        self.bin_num = 7
        
        self._BigM_setting()
        self._solve()
    
    def _BigM_setting(self):
    
        self.M_gen = [5*self.K[t] for t in range(T)] 
        
        if self.P_da >=0 and self.P_rt >= 0:
            
            self.M_set[0][0] = (160 + self.P_da + 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 + 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_da + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (self.P_rt + 80)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (self.P_rt + 80)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        elif self.P_da >=0 and self.P_rt < 0:
            
            self.M_set[0][0] = (160 + self.P_da - 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (-self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_da - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        elif self.P_da < 0 and self.P_rt >= 0:
            
            self.M_set[0][0] = (160 + 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - self.P_da + 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (-self.P_da + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 + self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])

        else:
            
            self.M_set[0][0] = (160 - 2*self.P_rt)*self.K[self.stage]
            self.M_set[0][1] = (80 - 2*self.P_da - 2*self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][0] = (- self.P_da - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][0][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][0] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][1][1] = (80 - self.P_rt)*self.K[self.stage]
            self.M_set_decomp[0][2][0] = max(self.M_set_decomp[0][0][0], self.M_set_decomp[0][1][0])
            self.M_set_decomp[0][2][1] = min(self.M_set_decomp[0][0][1], self.M_set_decomp[0][1][1])    

    def build_model(self):    
        
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1-self.stage) # 0
        model.TIME_ESS = pyo.RangeSet(-1, T-1-self.stage) # -1, 0
        model.BINARIZE = pyo.RangeSet(0, self.bin_num - 1) 
        
        # Vars

        model.b_da = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        model.b_rt = pyo.Var(model.TIME, bounds = (-P_r, 0), domain = pyo.Reals)
        
        model.q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_da = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.n_da = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_rt = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.lamb = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)        
        model.nu = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Binary)
        
        model.w = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.h = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.k = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        model.o = pyo.Var(model.TIME, model.BINARIZE, domain = pyo.Reals)
        
        model.g = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.d = pyo.Var(model.TIME, bounds = (0, B), domain = pyo.Reals)
        model.u = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.TIME_ESS, bounds = (S_min, S_max), domain = pyo.NonNegativeReals)

        model.E_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        
        model.m_1 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2 = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.m_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_1_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_2_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_2_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_1_1 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_2 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_4_3 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_5 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_6 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_7 = pyo.Var(model.TIME, domain = pyo.Reals)
        model.m_8 = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.n_E = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_1_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_2_2_2 = pyo.Var(model.TIME, domain = pyo.Binary)        
        model.n_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_4_1_1 = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.n_4_3 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_5 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_6 = pyo.Var(model.TIME, domain = pyo.Binary)
        model.n_c = pyo.Var(model.TIME, domain = pyo.Binary)
                
        model.f_prime = pyo.Var(model.TIME, domain = pyo.Reals)
        
        model.Q_sum = pyo.Var(model.TIME, domain = pyo.NonNegativeReals)
        model.n_sum = pyo.Var(model.TIME, domain = pyo.Binary)
        
        model.f = pyo.Var(model.TIME, domain = pyo.Reals)
        
        # Constraints
        
        ## Connected to previous solutions
        
        def prev_1_rule(model, t):
            return model.b_da[t] == self.b_da_prev[t]
        
        def prev_2_rule(model, t):
            return model.Q_da[t] == self.Q_da_prev[t]
        
        def prev_3_rule(model):
            return model.b_rt[0] == self.b_rt_prev
        
        def prev_4_rule(model):
            return model.q_rt[0] == self.q_rt_prev
        
        def prev_5_rule(model):
            return model.E_1[0] == self.E_1_prev
        
        ## Day-Ahead Market Rules
        
        def prev_State_SOC_rule(model):
            return model.S[-1] == self.S_prev
        
        ## Real-Time Market rules
        
        def rt_prev_market_clearing_rule_1(model):
            return model.b_rt[0] - self.P_rt <= M_price*(1 - model.n_rt[0])
        
        def rt_prev_market_clearing_rule_2(model):
            return self.P_rt - model.b_rt[0] <= M_price*model.n_rt[0] 
        
        def rt_market_clearing_rule_3(model, t):
            return model.Q_rt[t] <= model.q_rt[t]
        
        def rt_market_clearing_rule_4(model, t):
            return model.Q_rt[t] <= self.M_gen[self.stage+t]*model.n_rt[t]
        
        def rt_market_clearing_rule_5(model, t):
            return model.Q_rt[t] >= model.q_rt[t] - self.M_gen[self.stage+t]*(1 - model.n_rt[t])
      
        def dispatch_prev_rule(model):
            return model.Q_c[0] == (1 + self.delta_c)*model.Q_rt[0]      
        
        def generation_rule(model, t):
            return model.g[t] <= model.E_1[t]
        
        def charge_rule(model, t):
            return model.c[t] <= model.g[t]
        
        def electricity_supply_rule(model, t):
            return model.u[t] == model.g[t] + model.d[t] - model.c[t]

        def State_SOC_rule(model, t):
            return model.S[t] == model.S[t-1] + v*model.c[t] - (1/v)*model.d[t]
        
        def last_SOC_rule(model):
            return model.S[T-1-self.stage] == 0.5*S
        
        def without_ESS_1_rule(model, t):
            return model.c[t] == 0
        
        def without_ESS_2_rule(model, t):
            return model.d[t] == 0
        
        ## f(t) MIP reformulation
        
        ### 1. Bilinear -> MIP by binarize b_da, b_rt
        
        def binarize_b_da_rule_1(model, t):
            return model.b_da[t] >= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) - 0.5 
        
        def binarize_b_da_rule_2(model, t):
            return model.b_da[t] <= -sum(model.lamb[t, i]*(2**i) for i in range(self.bin_num)) + 0.5

        def binarize_b_rt_rule_1(model, t):
            return model.b_rt[t] >= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) - 0.5 
        
        def binarize_b_rt_rule_2(model, t):
            return model.b_rt[t] <= -sum(model.nu[t, j]*(2**j) for j in range(self.bin_num)) + 0.5
        
        def binarize_rule_1_1(model, t, j):
            return model.w[t, j] >= 0
        
        def binarize_rule_1_2(model, t, j):
            return model.w[t, j] <= model.u[t]
        
        def binarize_rule_1_3(model, t, j):
            return model.w[t, j] <= self.M_gen[self.stage+t]*model.nu[t, j]
        
        def binarize_rule_1_4(model, t, j):
            return model.w[t, j] >= model.u[t] - self.M_gen[self.stage+t]*(1-model.nu[t, j])
        
        def binarize_rule_2_1(model, t, i):
            return model.h[t, i] >= 0
        
        def binarize_rule_2_2(model, t, i):
            return model.h[t, i] <= model.m_1[t]
        
        def binarize_rule_2_3(model, t, i):
            return model.h[t, i] <= self.M_gen[self.stage+t]*model.lamb[t, i]
        
        def binarize_rule_2_4(model, t, i):
            return model.h[t, i] >= model.m_1[t] - self.M_gen[self.stage+t]*(1 - model.lamb[t, i])
        
        def binarize_rule_3_1(model, t, i):
            return model.k[t, i] >= 0
        
        def binarize_rule_3_2(model, t, i):
            return model.k[t, i] <= model.m_2[t]
        
        def binarize_rule_3_3(model, t, i):
            return model.k[t, i] <= self.M_gen[self.stage+t]*model.lamb[t, i]
        
        def binarize_rule_3_4(model, t, i):
            return model.k[t, i] >= model.m_2[t] - self.M_gen[self.stage+t]*(1 - model.lamb[t, i])        
        
        def binarize_rule_4_1(model, t, i):
            return model.o[t, i] >= 0
        
        def binarize_rule_4_2(model, t, i):
            return model.o[t, i] <= model.m_3[t]
        
        def binarize_rule_4_3(model, t, i):
            return model.o[t, i] <= self.M_gen[self.stage+t]*model.nu[t, i]
        
        def binarize_rule_4_4(model, t, i):
            return model.o[t, i] >= model.m_3[t] - self.M_gen[self.stage+t]*(1 - model.nu[t, i])        
        
        ### 2. Minmax -> MIP
        
        def dummy_rule_4_1(model, t):
            return model.m_4_1[t] == -sum(model.w[t, j]*(2**j) for j in range(self.bin_num)) - (model.Q_da[t]*self.P_da + (model.u[t] - model.Q_da[t])*self.P_rt)
        
        def dummy_rule_4_2(model, t):
            return model.m_4_2[t] == (model.m_1[t] - model.m_2[t])*self.P_rt + sum((model.h[t, i] - model.k[t, i])*(2**i) for i in range(self.bin_num))
        

        def minmax_rule_4_1_1_1(model, t):
            return model.m_4_1_1[t] <= model.u[t]
        
        def minmax_rule_4_1_1_2(model, t):
            return model.m_4_1_1[t] <= model.Q_c[t]
        
        def minmax_rule_4_1_1_3(model, t):
            return model.m_4_1_1[t] >= model.u[t] - self.M_gen[self.stage + t]*(1 - model.n_4_1_1[t])
        
        def minmax_rule_4_1_1_4(model, t):
            return model.m_4_1_1[t] >= model.Q_c[t] - self.M_gen[self.stage + t]*model.n_4_1_1[t]
  
        def minmax_rule_1_1(model, t):
            return model.m_1[t] <= model.Q_da[t]
        
        def minmax_rule_1_2(model, t):
            return model.m_1[t] <= model.q_rt[t]
        
        def minmax_rule_1_3(model, t):
            return model.m_1[t] >= model.Q_da[t] - (1 - model.n_1[t])*self.M_gen[self.stage + t]
        
        def minmax_rule_1_4(model, t):
            return model.m_1[t] >= model.q_rt[t] - (model.n_1[t])*self.M_gen[self.stage + t]
        
        
        def minmax_rule_2(model, t):
            return model.m_2[t] == model.m_2_1[t] - model.m_2_3[t] + model.m_2_4[t]
        
        
        def minmax_rule_2_E_1(model, t):
            return model.Q_c[t] - model.Q_da[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_E_2(model, t):
            return model.Q_da[t] - model.Q_c[t] <= self.M_gen[self.stage + t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_3_1(model, t):
            return model.m_2_3[t] >= 0
        
        def minmax_rule_2_3_2(model, t):
            return model.m_2_3[t] <= model.m_2_1[t]
        
        def minmax_rule_2_3_3(model, t):
            return model.m_2_3[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_3_4(model, t):
            return model.m_2_3[t] >= model.m_2_1[t] - self.M_gen[self.stage + t]*(1 - model.n_E[t])
        
        
        def minmax_rule_2_4_1(model, t):
            return model.m_2_4[t] >= 0
        
        def minmax_rule_2_4_2(model, t):
            return model.m_2_4[t] <= model.m_2_2[t]
        
        def minmax_rule_2_4_3(model, t):
            return model.m_2_4[t] <= self.M_gen[self.stage + t]*model.n_E[t]
        
        def minmax_rule_2_4_4(model, t):
            return model.m_2_4[t] >= model.m_2_2[t] - self.M_gen[self.stage + t]*(1 - model.n_E[t])
            
        
        def minmax_rule_2_1_1_a(model, t):
            return model.m_2_1_1[t] <= model.u[t]
        
        def minmax_rule_2_1_1_b(model, t):
            return model.m_2_1_1[t] <= model.Q_da[t]
        
        def minmax_rule_2_1_1_c(model, t):
            return model.m_2_1_1[t] >= model.u[t] - self.M_gen[self.stage + t]*(1 - model.n_2_1_1[t])
        
        def minmax_rule_2_1_1_d(model, t):
            return model.m_2_1_1[t] >= model.Q_da[t] - self.M_gen[self.stage + t]*model.n_2_1_1[t]
        
        def minmax_rule_2_1_2_a(model, t):
            return model.m_2_1_2[t] >= model.m_2_1_1[t]
        
        def minmax_rule_2_1_2_b(model, t):
            return model.m_2_1_2[t] >= model.Q_c[t]
        
        def minmax_rule_2_1_2_c(model, t):
            return model.m_2_1_2[t] <= model.m_2_1_1[t] + self.M_gen[self.stage + t]*(1 - model.n_2_1_2[t])
        
        def minmax_rule_2_1_2_d(model, t):
            return model.m_2_1_2[t] <= model.Q_c[t] + self.M_gen[self.stage + t]*model.n_2_1_2[t]
        
        def minmax_rule_2_1_a(model, t):
            return model.m_2_1[t] <= model.m_2_1_2[t]
        
        def minmax_rule_2_1_b(model, t):
            return model.m_2_1[t] <= model.q_rt[t]
        
        def minmax_rule_2_1_c(model, t):
            return model.m_2_1[t] >= model.m_2_1_2[t] - self.M_gen[self.stage + t]*(1 - model.n_2_1[t])
        
        def minmax_rule_2_1_d(model, t):
            return model.m_2_1[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*model.n_2_1[t]
        
        
        def minmax_rule_2_2_1_a(model, t):
            return model.m_2_2_1[t] >= model.u[t]
        
        def minmax_rule_2_2_1_b(model, t):
            return model.m_2_2_1[t] >= model.Q_da[t]
        
        def minmax_rule_2_2_1_c(model, t):
            return model.m_2_2_1[t] <= model.u[t] + self.M_gen[self.stage + t]*(1 - model.n_2_2_1[t])
        
        def minmax_rule_2_2_1_d(model, t):
            return model.m_2_2_1[t] <= model.Q_da[t] + self.M_gen[self.stage + t]*model.n_2_2_1[t]
        
        def minmax_rule_2_2_2_a(model, t):
            return model.m_2_2_2[t] <= model.m_2_2_1[t]
        
        def minmax_rule_2_2_2_b(model, t):
            return model.m_2_2_2[t] <= model.Q_c[t]
        
        def minmax_rule_2_2_2_c(model, t):
            return model.m_2_2_2[t] >= model.m_2_2_1[t] - self.M_gen[self.stage + t]*(1 - model.n_2_2_2[t])
        
        def minmax_rule_2_2_2_d(model, t):
            return model.m_2_2_2[t] >= model.Q_c[t] - self.M_gen[self.stage + t]*model.n_2_2_2[t]
        
        def minmax_rule_2_2_a(model, t):
            return model.m_2_2[t] <= model.m_2_2_2[t]
        
        def minmax_rule_2_2_b(model, t):
            return model.m_2_2[t] <= model.q_rt[t]
        
        def minmax_rule_2_2_c(model, t):
            return model.m_2_2[t] >= model.m_2_2_2[t] - self.M_gen[self.stage + t]*(1 - model.n_2_2[t])
        
        def minmax_rule_2_2_d(model, t):
            return model.m_2_2[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*model.n_2_2[t]
        
        
        def minmax_rule_3_1(model, t):
            return model.m_3[t] >= model.u[t] - model.Q_c[t] - 0.08*(C + S)
        
        def minmax_rule_3_2(model, t):
            return model.m_3[t] >= 0
        
        def minmax_rule_3_3(model, t):
            return model.m_3[t] <= model.u[t] - model.Q_c[t] - 0.08*(C + S) + self.M_gen[self.stage + t]*(1 - model.n_3[t])
        
        def minmax_rule_3_4(model, t):
            return model.m_3[t] <= self.M_gen[self.stage + t]*model.n_3[t]
             
        def minmax_rule_4_1(model, t):
            return model.m_4_3[t] >= model.m_4_1[t]
        
        def minmax_rule_4_2(model, t):
            return model.m_4_3[t] >= model.m_4_2[t]
        
        def minmax_rule_4_3(model, t):
            return model.m_4_3[t] <= model.m_4_1[t] + self.M_set[t][0]*(1 - model.n_4_3[t])
        
        def minmax_rule_4_4(model, t):
            return model.m_4_3[t] <= model.m_4_2[t] + self.M_set[t][1]*model.n_4_3[t]
        
        def minmax_rule_4_5(model, t):
            return model.m_4[t] >= model.m_4_3[t] 
        
        def minmax_rule_4_6(model, t):
            return model.m_4[t] >= 0
        
        def minmax_rule_4_7(model, t):
            return model.m_4[t] <= self.M_set_decomp[t][2][0]*(1 - model.n_4[t])
        
        def minmax_rule_4_8(model, t):
            return model.m_4[t] <= model.m_4_3[t] + self.M_set_decomp[t][2][1]*model.n_4[t]
        
        
        def minmax_rule_5_1(model, t):
            return model.m_5[t] <= model.q_da[t]
        
        def minmax_rule_5_2(model, t):
            return model.m_5[t] <= model.q_rt[t]
        
        def minmax_rule_5_3(model, t):
            return model.m_5[t] >= model.q_da[t] - self.M_gen[self.stage + t]*(1 - model.n_5[t])
        
        def minmax_rule_5_4(model, t):
            return model.m_5[t] >= model.q_rt[t] - self.M_gen[self.stage + t]*(model.n_5[t])
        
        def minmax_rule_6_1(model, t):
            return model.m_6[t] <= model.m_5[t]
        
        def minmax_rule_6_2(model, t):
            return model.m_6[t] <= model.u[t]
        
        def minmax_rule_6_3(model, t):
            return model.m_6[t] >= model.m_5[t] - self.M_gen[self.stage + t]*(1 - model.n_6[t])
        
        def minmax_rule_6_4(model, t):
            return model.m_6[t] >= model.u[t] - self.M_gen[self.stage + t]*model.n_6[t]
        
        
        def bin_c_rule_1(model, t):
            return model.q_rt[t] - model.Q_c[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def bin_c_rule_2(model, t):
            return model.Q_c[t] - model.q_rt[t] <= self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        
        def minmax_rule_7_1(model, t):
            return model.m_7[t] >= 0
        
        def minmax_rule_7_2(model, t):
            return model.m_7[t] <= model.m_5[t]
        
        def minmax_rule_7_3(model, t):
            return model.m_7[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def minmax_rule_7_4(model, t):
            return model.m_7[t] >= model.m_5[t] - self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        def minmax_rule_8_1(model, t):
            return model.m_8[t] >= 0
        
        def minmax_rule_8_2(model, t):
            return model.m_8[t] <= model.m_6[t]
        
        def minmax_rule_8_3(model, t):
            return model.m_8[t] <= self.M_gen[self.stage + t]*model.n_c[t]
        
        def minmax_rule_8_4(model, t):
            return model.m_8[t] >= model.m_6[t] - self.M_gen[self.stage + t]*(1 - model.n_c[t])
        
        
        ## Settlement fcn
        
        def settlement_fcn_current_rule(model):
            return model.f_prime[0] == model.Q_da[0]*self.P_da + (model.u[0] - model.Q_da[0])*self.P_rt + self.m_4[0] - self.P_rt*model.m_3[0] - sum((2**j)*model.o[0, j] for j in range(self.bin_num)) + P_c*(model.m_6[0] - model.m_7[0] + model.m_8[0]) 
        
        def settlement_fcn_rule_1(model, t):
            return model.Q_sum[t] == model.Q_da[t] + model.Q_rt[t]
        
        def settlement_fcn_rule_2(model, t):
            return model.Q_sum[t] <= self.M_gen[self.stage+t]*model.n_sum[t]
        
        def settlement_fcn_rule_3(model, t):
            return model.Q_sum[t] >= epsilon*model.n_sum[t]
        
        def settlement_fcn_rule_4(model, t):
            return model.f[t] >= 0
        
        def settlement_fcn_rule_5(model, t):
            return model.f[t] <= model.f_prime[t]
        
        def settlement_fcn_rule_6(model, t):
            return model.f[t] <= M_set_fcn*model.n_sum[t]
        
        def settlement_fcn_rule_7(model, t):
            return model.f[t] >= model.f_prime[t] - M_set_fcn*(1 - model.n_sum[t])
        
        model.prev_1 = pyo.Constraint(model.TIME, rule = prev_1_rule)
        model.prev_2 = pyo.Constraint(model.TIME, rule = prev_2_rule)
        model.prev_3 = pyo.Constraint(rule = prev_3_rule)
        model.prev_4 = pyo.Constraint(rule = prev_4_rule)
        model.prev_5 = pyo.Constraint(rule = prev_5_rule)
        model.prev_State_SOC = pyo.Constraint(rule = prev_State_SOC_rule)
        model.rt_prev_market_clearing_1 = pyo.Constraint(rule = rt_prev_market_clearing_rule_1)
        model.rt_prev_market_clearing_2 = pyo.Constraint(rule = rt_prev_market_clearing_rule_2)
        model.rt_market_clearing_3 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_3)
        model.rt_market_clearing_4 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_4)
        model.rt_market_clearing_5 = pyo.Constraint(model.TIME, rule = rt_market_clearing_rule_5)
        model.dispatch_prev = pyo.Constraint(rule = dispatch_prev_rule)
        model.generation = pyo.Constraint(model.TIME, rule = generation_rule)
        model.charge = pyo.Constraint(model.TIME, rule = charge_rule)
        model.electricity_supply = pyo.Constraint(model.TIME, rule = electricity_supply_rule)
        model.State_SOC = pyo.Constraint(model.TIME, rule = State_SOC_rule)
        model.last_SOC = pyo.Constraint(rule = last_SOC_rule)
        model.without_ESS_1 = pyo.Constraint(model.TIME, rule = without_ESS_1_rule)
        model.without_ESS_2 = pyo.Constraint(model.TIME, rule = without_ESS_2_rule)

        model.binarize_b_da_1 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_1)
        model.binarize_b_da_2 = pyo.Constraint(model.TIME, rule = binarize_b_da_rule_2)
        model.binarize_b_rt_1 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_1)
        model.binarize_b_rt_2 = pyo.Constraint(model.TIME, rule = binarize_b_rt_rule_2)                
        model.binarize_1_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_1)
        model.binarize_1_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_2)
        model.binarize_1_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_3)       
        model.binarize_1_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_1_4)
        model.binarize_2_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_1)
        model.binarize_2_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_2)
        model.binarize_2_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_3)
        model.binarize_2_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_2_4)
        model.binarize_3_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_1)
        model.binarize_3_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_2)
        model.binarize_3_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_3)
        model.binarize_3_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_3_4)
        model.binarize_4_1 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_1)
        model.binarize_4_2 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_2)
        model.binarize_4_3 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_3)
        model.binarize_4_4 = pyo.Constraint(model.TIME, model.BINARIZE, rule = binarize_rule_4_4)
        
        model.dummy_4_1 = pyo.Constraint(model.TIME, rule = dummy_rule_4_1)
        model.dummy_4_2 = pyo.Constraint(model.TIME, rule = dummy_rule_4_2)
        
        model.minmax_4_1_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_1)
        model.minmax_4_1_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_2)
        model.minmax_4_1_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_3)
        model.minmax_4_1_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1_1_4)
        
        model.minmax_1_1 = pyo.Constraint(model.TIME, rule = minmax_rule_1_1)
        model.minmax_1_2 = pyo.Constraint(model.TIME, rule = minmax_rule_1_2)
        model.minmax_1_3 = pyo.Constraint(model.TIME, rule = minmax_rule_1_3)
        model.minmax_1_4 = pyo.Constraint(model.TIME, rule = minmax_rule_1_4)
        
        model.minmax_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2)
        
        model.minmax_2_E_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_1)
        model.minmax_2_E_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_E_2)
        
        model.minmax_2_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_1)
        model.minmax_2_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_2)
        model.minmax_2_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_3)
        model.minmax_2_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_3_4)
        
        model.minmax_2_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_1)
        model.minmax_2_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_2)
        model.minmax_2_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_3)
        model.minmax_2_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_2_4_4)
        
        model.minmax_2_1_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_a)
        model.minmax_2_1_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_b)
        model.minmax_2_1_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_c)
        model.minmax_2_1_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_1_d)
        
        model.minmax_2_1_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_a)
        model.minmax_2_1_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_b)
        model.minmax_2_1_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_c)
        model.minmax_2_1_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_2_d)
        
        model.minmax_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_a)
        model.minmax_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_b)
        model.minmax_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_c)
        model.minmax_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_1_d)
        
        model.minmax_2_2_1_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_a)
        model.minmax_2_2_1_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_b)
        model.minmax_2_2_1_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_c)
        model.minmax_2_2_1_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_1_d)
        
        model.minmax_2_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_a)
        model.minmax_2_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_b)
        model.minmax_2_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_c)
        model.minmax_2_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_2_d)
        
        model.minmax_2_2_a = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_a)
        model.minmax_2_2_b = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_b)
        model.minmax_2_2_c = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_c)
        model.minmax_2_2_d = pyo.Constraint(model.TIME, rule = minmax_rule_2_2_d)
        
        model.minmax_3_1 = pyo.Constraint(model.TIME, rule = minmax_rule_3_1)
        model.minmax_3_2 = pyo.Constraint(model.TIME, rule = minmax_rule_3_2)
        model.minmax_3_3 = pyo.Constraint(model.TIME, rule = minmax_rule_3_3)
        model.minmax_3_4 = pyo.Constraint(model.TIME, rule = minmax_rule_3_4)
        model.minmax_4_1 = pyo.Constraint(model.TIME, rule = minmax_rule_4_1)
        model.minmax_4_2 = pyo.Constraint(model.TIME, rule = minmax_rule_4_2)
        model.minmax_4_3 = pyo.Constraint(model.TIME, rule = minmax_rule_4_3)
        model.minmax_4_4 = pyo.Constraint(model.TIME, rule = minmax_rule_4_4)
        model.minmax_4_5 = pyo.Constraint(model.TIME, rule = minmax_rule_4_5)
        model.minmax_4_6 = pyo.Constraint(model.TIME, rule = minmax_rule_4_6)
        model.minmax_4_7 = pyo.Constraint(model.TIME, rule = minmax_rule_4_7)
        model.minmax_4_8 = pyo.Constraint(model.TIME, rule = minmax_rule_4_8)

        model.minmax_5_1 = pyo.Constraint(model.TIME, rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(model.TIME, rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(model.TIME, rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(model.TIME, rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(model.TIME, rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(model.TIME, rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(model.TIME, rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(model.TIME, rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(model.TIME, rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(model.TIME, rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(model.TIME, rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(model.TIME, rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(model.TIME, rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(model.TIME, rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(model.TIME, rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(model.TIME, rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(model.TIME, rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(model.TIME, rule=minmax_rule_8_4)

        model.settlement_fcn_current = pyo.Constraint(rule = settlement_fcn_current_rule)
        model.settlement_fcn_1 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_1)
        model.settlement_fcn_2 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_2)
        model.settlement_fcn_3 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_3)
        model.settlement_fcn_4 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_4)
        model.settlement_fcn_5 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_5)
        model.settlement_fcn_6 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_6)
        model.settlement_fcn_7 = pyo.Constraint(model.TIME, rule = settlement_fcn_rule_7)
                
        # Obj Fcn

        model.objective = pyo.Objective(
            expr = sum(model.f[t] for t in model.TIME), 
            sense = pyo.maximize
            )
    
    def _solve(self):
        
        self.build_model()
        SOLVER.solve(self)
        self.solved = True
            
    def get_objective_value(self):
        
        if not self.solved:
            self._solve()
            self.solved = True
    
        return pyo.value(self.f[0])    


"""
       
# Subproblems for SDDiP

## stage = -1

class fw_da(pyo.ConcreteModel): 
    
    def __init__(self, psi):
        
        super().__init__()

        self.solved = False
        self.psi = psi
        self.T = T
        
        self.P_da = exp_P_da
        
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
            return model.q_da[t] <= E_0[t] + B
        
        def da_overbid_rule(model):
            return sum(model.q_da[t] for t in range(self.T)) <= E_0_sum
        
        def market_clearing_1_rule(model, t):
            return model.b_da[t] - self.P_da[t] <= M_price*(1 - model.n_da[t])
        
        def market_clearing_2_rule(model, t):
            return self.P_da[t] - model.b_da[t] <= M_price*model.n_da[t]
        
        def market_clearing_3_rule(model, t):
            return model.Q_da[t] <= model.q_da[t]
        
        def market_clearing_4_rule(model, t):
            return model.Q_da[t] <= M_gen[t]*model.n_da[t]
        
        def market_clearing_5_rule(model, t):
            return model.Q_da[t] >= model.q_da[t] - M_gen[t]*(1 - model.n_da[t])
        
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
        
        self.P_da = exp_P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        model.m_5 = pyo.Var(domain = pyo.Reals)
        model.m_6 = pyo.Var(domain = pyo.Reals)
        model.m_7 = pyo.Var(domain = pyo.Reals)
        model.m_8 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        model.n_5 = pyo.Var(domain = pyo.Binary)
        model.n_6 = pyo.Var(domain = pyo.Binary)
        model.n_c = pyo.Var(domain = pyo.Binary)
        
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
            return model.T_E == self.delta_E_0*E_0[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.08*(C + S)
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.08*(C + S) + self.M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= self.M_gen[self.stage]*model.n_3
             
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
        
        
        def minmax_rule_5_1(model):
            return model.m_5 <= model.q_da
        
        def minmax_rule_5_2(model):
            return model.m_5 <= model.q_rt
        
        def minmax_rule_5_3(model):
            return model.m_5 >= model.q_da - M_gen[self.stage]*(1 - model.n_5)
        
        def minmax_rule_5_4(model):
            return model.m_5 >= model.q_rt - M_gen[self.stage]*(model.n_5)
        
        def minmax_rule_6_1(model):
            return model.m_6 <= model.m_5
        
        def minmax_rule_6_2(model):
            return model.m_6 <= model.u
        
        def minmax_rule_6_3(model):
            return model.m_6 >= model.m_5 - M_gen[self.stage]*(1 - model.n_6)
        
        def minmax_rule_6_4(model):
            return model.m_6 >= model.u - M_gen[self.stage]*model.n_6
        
        
        def bin_c_rule_1(model):
            return model.q_rt - model.Q_c <= M_gen[self.stage]*model.n_c
        
        def bin_c_rule_2(model):
            return model.Q_c - model.q_rt <= M_gen[self.stage]*(1 - model.n_c)
        
        
        def minmax_rule_7_1(model):
            return model.m_7 >= 0
        
        def minmax_rule_7_2(model):
            return model.m_7 <= model.m_5
        
        def minmax_rule_7_3(model):
            return model.m_7 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_7_4(model):
            return model.m_7 >= model.m_5 - M_gen[self.stage]*(1 - model.n_c)
        
        def minmax_rule_8_1(model):
            return model.m_8 >= 0
        
        def minmax_rule_8_2(model):
            return model.m_8 <= model.m_6
        
        def minmax_rule_8_3(model):
            return model.m_8 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_8_4(model):
            return model.m_8 >= model.m_6 - M_gen[self.stage]*(1 - model.n_c)

        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u + P_c*(model.m_6 - model.m_7 + model.m_8)
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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
        
        model.minmax_5_1 = pyo.Constraint(rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(rule=minmax_rule_8_4)
        
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
        
        self.P_da = exp_P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
            return model.T_E == self.delta_E_0*E_0[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.12*C
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.12*C + M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage]*model.n_3

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
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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
        
        self.P_da = exp_P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
                
        self.T = T
        self.bin_num = 7

        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        model.m_5 = pyo.Var(domain = pyo.Reals)
        model.m_6 = pyo.Var(domain = pyo.Reals)
        model.m_7 = pyo.Var(domain = pyo.Reals)
        model.m_8 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_5 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_6 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_c = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
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
            return model.T_E == self.delta_E_0*E_0[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.08*(C + S)
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.08*(C + S) + self.M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= self.M_gen[self.stage]*model.n_3
             
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
        
        
        def minmax_rule_5_1(model):
            return model.m_5 <= model.q_da
        
        def minmax_rule_5_2(model):
            return model.m_5 <= model.q_rt
        
        def minmax_rule_5_3(model):
            return model.m_5 >= model.q_da - M_gen[self.stage]*(1 - model.n_5)
        
        def minmax_rule_5_4(model):
            return model.m_5 >= model.q_rt - M_gen[self.stage]*(model.n_5)
        
        def minmax_rule_6_1(model):
            return model.m_6 <= model.m_5
        
        def minmax_rule_6_2(model):
            return model.m_6 <= model.u
        
        def minmax_rule_6_3(model):
            return model.m_6 >= model.m_5 - M_gen[self.stage]*(1 - model.n_6)
        
        def minmax_rule_6_4(model):
            return model.m_6 >= model.u - M_gen[self.stage]*model.n_6
        
        
        def bin_c_rule_1(model):
            return model.q_rt - model.Q_c <= M_gen[self.stage]*model.n_c
        
        def bin_c_rule_2(model):
            return model.Q_c - model.q_rt <= M_gen[self.stage]*(1 - model.n_c)
        
        
        def minmax_rule_7_1(model):
            return model.m_7 >= 0
        
        def minmax_rule_7_2(model):
            return model.m_7 <= model.m_5
        
        def minmax_rule_7_3(model):
            return model.m_7 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_7_4(model):
            return model.m_7 >= model.m_5 - M_gen[self.stage]*(1 - model.n_c)
        
        def minmax_rule_8_1(model):
            return model.m_8 >= 0
        
        def minmax_rule_8_2(model):
            return model.m_8 <= model.m_6
        
        def minmax_rule_8_3(model):
            return model.m_8 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_8_4(model):
            return model.m_8 >= model.m_6 - M_gen[self.stage]*(1 - model.n_c)
        
                
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u + P_c*(model.m_6 - model.m_7 + model.m_8)
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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
        
        model.minmax_5_1 = pyo.Constraint(rule = minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(rule = minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(rule = minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(rule = minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(rule = minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(rule = minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(rule = minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(rule = minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(rule = bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(rule = bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(rule = minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(rule = minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(rule = minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(rule = minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(rule = minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(rule = minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(rule = minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(rule = minmax_rule_8_4)
        
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
        SOLVER.solve(self)
        self.solved = True
        
    def get_cut_coefficients(self):
        if not self.solved:
            self.solve()
            self.solved = True
        
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
        
        self.P_da = exp_P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
            return model.T_E == self.delta_E_0*E_0[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.12*C
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.12*C + M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage]*model.n_3
        
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
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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
        # Solve the model and store the results. Make sure SOLVER is correctly defined, e.g.,
        # SOLVER = pyo.SolverFactory('glpk')
        self.solver_results = SOLVER.solve(self, tee=False)
        # Only mark as solved if the termination condition is optimal.
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
        
        self.P_da = exp_P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        model.m_5 = pyo.Var(domain = pyo.Reals)
        model.m_6 = pyo.Var(domain = pyo.Reals)
        model.m_7 = pyo.Var(domain = pyo.Reals)
        model.m_8 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        model.n_5 = pyo.Var(domain = pyo.Binary)
        model.n_6 = pyo.Var(domain = pyo.Binary)
        model.n_c = pyo.Var(domain = pyo.Binary)
        
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
            return model.T_E == self.delta_E_0*E_0[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.08*(C + S)
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.08*(C + S) + self.M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= self.M_gen[self.stage]*model.n_3
             
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
        
        
        def minmax_rule_5_1(model):
            return model.m_5 <= model.q_da
        
        def minmax_rule_5_2(model):
            return model.m_5 <= model.q_rt
        
        def minmax_rule_5_3(model):
            return model.m_5 >= model.q_da - M_gen[self.stage]*(1 - model.n_5)
        
        def minmax_rule_5_4(model):
            return model.m_5 >= model.q_rt - M_gen[self.stage]*(model.n_5)
        
        def minmax_rule_6_1(model):
            return model.m_6 <= model.m_5
        
        def minmax_rule_6_2(model):
            return model.m_6 <= model.u
        
        def minmax_rule_6_3(model):
            return model.m_6 >= model.m_5 - M_gen[self.stage]*(1 - model.n_6)
        
        def minmax_rule_6_4(model):
            return model.m_6 >= model.u - M_gen[self.stage]*model.n_6
        
        
        def bin_c_rule_1(model):
            return model.q_rt - model.Q_c <= M_gen[self.stage]*model.n_c
        
        def bin_c_rule_2(model):
            return model.Q_c - model.q_rt <= M_gen[self.stage]*(1 - model.n_c)
        
        
        def minmax_rule_7_1(model):
            return model.m_7 >= 0
        
        def minmax_rule_7_2(model):
            return model.m_7 <= model.m_5
        
        def minmax_rule_7_3(model):
            return model.m_7 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_7_4(model):
            return model.m_7 >= model.m_5 - M_gen[self.stage]*(1 - model.n_c)
        
        def minmax_rule_8_1(model):
            return model.m_8 >= 0
        
        def minmax_rule_8_2(model):
            return model.m_8 <= model.m_6
        
        def minmax_rule_8_3(model):
            return model.m_8 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_8_4(model):
            return model.m_8 >= model.m_6 - M_gen[self.stage]*(1 - model.n_c)
        
        
        ## Approximated value fcn
        
        def value_fcn_approx_rule(model, l):
            return model.theta <= self.psi[l][0] + self.psi[l][1]*model.S + sum(self.psi[l][2][t]*model.T_b[t] for t in range(T - 1 - self.stage)) + sum(self.psi[l][3][t]*model.T_Q[t] for t in range(T - 1 - self.stage)) + self.psi[l][4]*model.T_q + self.psi[l][5]*model.T_b_rt + self.psi[l][6]*model.T_q_rt + self.psi[l][7]*model.T_E
        
        ## Settlement fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u + P_c*(model.m_6 - model.m_7 + model.m_8)
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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

        model.minmax_5_1 = pyo.Constraint(rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(rule=minmax_rule_8_4)
        
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

class fw_rt_Lagrangian_Alt(pyo.ConcreteModel): 

    def __init__(self, stage, pi, psi, delta):
        
        super().__init__()

        self.solved = False
        
        self.stage = stage
        self.pi = pi
        self.psi = psi
        
        self.P_da = exp_P_da
        self.delta_E_0 = delta[0]
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
            return model.T_E == self.delta_E_0*E_0[self.stage + 1]
        
        ## General Constraints
        
        def overbid_rule(model):
            return model.T_q <= E_0_sum
        
        def next_q_rt_rule(model):
            return model.q_rt_next <= self.delta_E_0*E_0[self.stage + 1] + B
        
        def generation_rule(model):
            return model.g <= model.E_1
        
        def charge_rule(model):
            return model.c <= model.g
        
        def electricity_supply_rule(model):
            return model.u == model.g + model.d - model.c
        
        
        def market_clearing_rule_1(model):
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.12*C
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.12*C + M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage]*model.n_3
          
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
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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

        self.P_da = exp_P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()
        
    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        model.m_5 = pyo.Var(domain = pyo.Reals)
        model.m_6 = pyo.Var(domain = pyo.Reals)
        model.m_7 = pyo.Var(domain = pyo.Reals)
        model.m_8 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        model.n_5 = pyo.Var(domain = pyo.Binary)
        model.n_6 = pyo.Var(domain = pyo.Binary)
        model.n_c = pyo.Var(domain = pyo.Binary)
        
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
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.08*(C + S)
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.08*(C + S) + self.M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= self.M_gen[self.stage]*model.n_3
             
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
        
        
        def minmax_rule_5_1(model):
            return model.m_5 <= model.q_da
        
        def minmax_rule_5_2(model):
            return model.m_5 <= model.q_rt
        
        def minmax_rule_5_3(model):
            return model.m_5 >= model.q_da - M_gen[self.stage]*(1 - model.n_5)
        
        def minmax_rule_5_4(model):
            return model.m_5 >= model.q_rt - M_gen[self.stage]*(model.n_5)
        
        def minmax_rule_6_1(model):
            return model.m_6 <= model.m_5
        
        def minmax_rule_6_2(model):
            return model.m_6 <= model.u
        
        def minmax_rule_6_3(model):
            return model.m_6 >= model.m_5 - M_gen[self.stage]*(1 - model.n_6)
        
        def minmax_rule_6_4(model):
            return model.m_6 >= model.u - M_gen[self.stage]*model.n_6
        
        
        def bin_c_rule_1(model):
            return model.q_rt - model.Q_c <= M_gen[self.stage]*model.n_c
        
        def bin_c_rule_2(model):
            return model.Q_c - model.q_rt <= M_gen[self.stage]*(1 - model.n_c)
        
        
        def minmax_rule_7_1(model):
            return model.m_7 >= 0
        
        def minmax_rule_7_2(model):
            return model.m_7 <= model.m_5
        
        def minmax_rule_7_3(model):
            return model.m_7 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_7_4(model):
            return model.m_7 >= model.m_5 - M_gen[self.stage]*(1 - model.n_c)
        
        def minmax_rule_8_1(model):
            return model.m_8 >= 0
        
        def minmax_rule_8_2(model):
            return model.m_8 <= model.m_6
        
        def minmax_rule_8_3(model):
            return model.m_8 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_8_4(model):
            return model.m_8 >= model.m_6 - M_gen[self.stage]*(1 - model.n_c)
        
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u + P_c*(model.m_6 - model.m_7 + model.m_8)
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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
       
        model.minmax_5_1 = pyo.Constraint(rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(rule=minmax_rule_8_4)       
       
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

        self.P_da = exp_P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.12*C
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.12*C + M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage]*model.n_3

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
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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

        self.P_da = exp_P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        model.m_5 = pyo.Var(domain = pyo.Reals)
        model.m_6 = pyo.Var(domain = pyo.Reals)
        model.m_7 = pyo.Var(domain = pyo.Reals)
        model.m_8 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_2 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_4_3 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_5 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_6 = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        model.n_c = pyo.Var(bounds = (0, 1), domain = pyo.Reals)
        
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
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.08*(C + S)
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.08*(C + S) + self.M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= self.M_gen[self.stage]*model.n_3
             
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
        
        
        def minmax_rule_5_1(model):
            return model.m_5 <= model.q_da
        
        def minmax_rule_5_2(model):
            return model.m_5 <= model.q_rt
        
        def minmax_rule_5_3(model):
            return model.m_5 >= model.q_da - M_gen[self.stage]*(1 - model.n_5)
        
        def minmax_rule_5_4(model):
            return model.m_5 >= model.q_rt - M_gen[self.stage]*(model.n_5)
        
        def minmax_rule_6_1(model):
            return model.m_6 <= model.m_5
        
        def minmax_rule_6_2(model):
            return model.m_6 <= model.u
        
        def minmax_rule_6_3(model):
            return model.m_6 >= model.m_5 - M_gen[self.stage]*(1 - model.n_6)
        
        def minmax_rule_6_4(model):
            return model.m_6 >= model.u - M_gen[self.stage]*model.n_6
        
        
        def bin_c_rule_1(model):
            return model.q_rt - model.Q_c <= M_gen[self.stage]*model.n_c
        
        def bin_c_rule_2(model):
            return model.Q_c - model.q_rt <= M_gen[self.stage]*(1 - model.n_c)
        
        
        def minmax_rule_7_1(model):
            return model.m_7 >= 0
        
        def minmax_rule_7_2(model):
            return model.m_7 <= model.m_5
        
        def minmax_rule_7_3(model):
            return model.m_7 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_7_4(model):
            return model.m_7 >= model.m_5 - M_gen[self.stage]*(1 - model.n_c)
        
        def minmax_rule_8_1(model):
            return model.m_8 >= 0
        
        def minmax_rule_8_2(model):
            return model.m_8 <= model.m_6
        
        def minmax_rule_8_3(model):
            return model.m_8 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_8_4(model):
            return model.m_8 >= model.m_6 - M_gen[self.stage]*(1 - model.n_c)
  
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u + P_c*(model.m_6 - model.m_7 + model.m_8)
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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
               
        model.minmax_5_1 = pyo.Constraint(rule = minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(rule = minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(rule = minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(rule = minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(rule = minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(rule = minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(rule = minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(rule = minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(rule = bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(rule = bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(rule = minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(rule = minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(rule = minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(rule = minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(rule = minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(rule = minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(rule = minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(rule = minmax_rule_8_4)
               
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
        SOLVER.solve(self)
        self.solved = True

    def get_objective_value(self):
        
        if not self.solved:
            self.solve()
            self.solved = True 
        
        return(pyo.value(self.objective))

    def get_cut_coefficients(self):
        
        if not self.solved:
            self.solve()
            self.solved = True
        
        psi = []
        
        psi.append(pyo.value(self.objective))
        
        psi.append(self.dual[self.auxiliary_S])
        
        pi_T_b = []
        pi_T_b.append(self.dual[self.auxiliary_T_b[0]])
        psi.append(pi_T_b)
        
        pi_T_Q = []
        pi_T_Q.append(self.dual[self.auxiliary_T_Q[0]])  
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

        self.P_da = exp_P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.12*C
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.12*C + M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage]*model.n_3
        
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
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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

        self.P_da = exp_P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7

        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
        model.m_4_3 = pyo.Var(domain = pyo.Reals)
        model.m_5 = pyo.Var(domain = pyo.Reals)
        model.m_6 = pyo.Var(domain = pyo.Reals)
        model.m_7 = pyo.Var(domain = pyo.Reals)
        model.m_8 = pyo.Var(domain = pyo.Reals)
        
        model.n_1 = pyo.Var(domain = pyo.Binary)
        model.n_2 = pyo.Var(domain = pyo.Binary)
        model.n_3 = pyo.Var(domain = pyo.Binary)
        model.n_4 = pyo.Var(domain = pyo.Binary)
        model.n_4_3 = pyo.Var(domain = pyo.Binary)
        model.n_5 = pyo.Var(domain = pyo.Binary)
        model.n_6 = pyo.Var(domain = pyo.Binary)
        model.n_c = pyo.Var(domain = pyo.Binary)
        
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
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.08*(C + S)
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.08*(C + S) + self.M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= self.M_gen[self.stage]*model.n_3
             
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
        
        
        def minmax_rule_5_1(model):
            return model.m_5 <= model.q_da
        
        def minmax_rule_5_2(model):
            return model.m_5 <= model.q_rt
        
        def minmax_rule_5_3(model):
            return model.m_5 >= model.q_da - M_gen[self.stage]*(1 - model.n_5)
        
        def minmax_rule_5_4(model):
            return model.m_5 >= model.q_rt - M_gen[self.stage]*(model.n_5)
        
        def minmax_rule_6_1(model):
            return model.m_6 <= model.m_5
        
        def minmax_rule_6_2(model):
            return model.m_6 <= model.u
        
        def minmax_rule_6_3(model):
            return model.m_6 >= model.m_5 - M_gen[self.stage]*(1 - model.n_6)
        
        def minmax_rule_6_4(model):
            return model.m_6 >= model.u - M_gen[self.stage]*model.n_6
        
        
        def bin_c_rule_1(model):
            return model.q_rt - model.Q_c <= M_gen[self.stage]*model.n_c
        
        def bin_c_rule_2(model):
            return model.Q_c - model.q_rt <= M_gen[self.stage]*(1 - model.n_c)
        
        
        def minmax_rule_7_1(model):
            return model.m_7 >= 0
        
        def minmax_rule_7_2(model):
            return model.m_7 <= model.m_5
        
        def minmax_rule_7_3(model):
            return model.m_7 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_7_4(model):
            return model.m_7 >= model.m_5 - M_gen[self.stage]*(1 - model.n_c)
        
        def minmax_rule_8_1(model):
            return model.m_8 >= 0
        
        def minmax_rule_8_2(model):
            return model.m_8 <= model.m_6
        
        def minmax_rule_8_3(model):
            return model.m_8 <= M_gen[self.stage]*model.n_c
        
        def minmax_rule_8_4(model):
            return model.m_8 >= model.m_6 - M_gen[self.stage]*(1 - model.n_c)
        
        ### settlement_fcn
        
        def settlement_fcn_rule(model):
            return model.f_prime == model.Q_da*self.P_da[self.stage] + (model.u - model.Q_da)*self.P_rt + self.m_4 - self.P_rt*model.m_3 - sum((2**j)*model.o[j] for j in range(self.bin_num)) + P_r*model.u + P_c*(model.m_6 - model.m_7 + model.m_8)
        
        def settlement_fcn_rule_1(model):
            return model.Q_sum == model.Q_da + model.Q_rt
        
        def settlement_fcn_rule_2(model):
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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
               
        model.minmax_5_1 = pyo.Constraint(rule=minmax_rule_5_1)
        model.minmax_5_2 = pyo.Constraint(rule=minmax_rule_5_2)
        model.minmax_5_3 = pyo.Constraint(rule=minmax_rule_5_3)
        model.minmax_5_4 = pyo.Constraint(rule=minmax_rule_5_4)

        model.minmax_6_1 = pyo.Constraint(rule=minmax_rule_6_1)
        model.minmax_6_2 = pyo.Constraint(rule=minmax_rule_6_2)
        model.minmax_6_3 = pyo.Constraint(rule=minmax_rule_6_3)
        model.minmax_6_4 = pyo.Constraint(rule=minmax_rule_6_4)

        model.bin_c_1 = pyo.Constraint(rule=bin_c_rule_1)
        model.bin_c_2 = pyo.Constraint(rule=bin_c_rule_2)

        model.minmax_7_1 = pyo.Constraint(rule=minmax_rule_7_1)
        model.minmax_7_2 = pyo.Constraint(rule=minmax_rule_7_2)
        model.minmax_7_3 = pyo.Constraint(rule=minmax_rule_7_3)
        model.minmax_7_4 = pyo.Constraint(rule=minmax_rule_7_4)

        model.minmax_8_1 = pyo.Constraint(rule=minmax_rule_8_1)
        model.minmax_8_2 = pyo.Constraint(rule=minmax_rule_8_2)
        model.minmax_8_3 = pyo.Constraint(rule=minmax_rule_8_3)
        model.minmax_8_4 = pyo.Constraint(rule=minmax_rule_8_4)
               
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

class fw_rt_last_Lagrangian_Alt(pyo.ConcreteModel): 
           
    def __init__(self, pi, delta):
        
        super().__init__()

        self.solved = False

        self.pi = pi
        
        self.stage = T - 1

        self.P_da = exp_P_da
        self.P_rt = delta[1]
        self.delta_c = delta[2]
        
        self.T = T
        self.bin_num = 7
        
        self.M_set = [0, 0]
        
        self.M_set_decomp = [[0, 0] for i in range(3)]
        
        self._BigM_setting()

    def _BigM_setting(self):
        
        if self.P_da[self.stage] >=0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[1][1] = (self.P_rt + 80)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] >=0 and self.P_rt < 0:
            
            self.M_set[0] = (160 + self.P_da[self.stage] - 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_da[self.stage] - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 - self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        elif self.P_da[self.stage] < 0 and self.P_rt >= 0:
            
            self.M_set[0] = (160 + 2*self.P_rt)*K[self.stage]
            self.M_set[1] = (80 - self.P_da[self.stage] + 2*self.P_rt)*K[self.stage]
            self.M_set_decomp[0][0] = (-self.P_da[self.stage] + self.P_rt)*K[self.stage]
            self.M_set_decomp[0][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][0] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[1][1] = (80 + self.P_rt)*K[self.stage]
            self.M_set_decomp[2][0] = max(self.M_set_decomp[0][0], self.M_set_decomp[1][0])
            self.M_set_decomp[2][1] = min(self.M_set_decomp[0][1], self.M_set_decomp[1][1])

        else:
            
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
            return model.b_rt - self.P_rt <= M_price*(1 - model.n_rt)
        
        def market_clearing_rule_2(model):
            return self.P_rt - model.b_rt <= M_price*model.n_rt 
        
        def market_clearing_rule_3(model):
            return model.Q_rt <= model.q_rt
        
        def market_clearing_rule_4(model):
            return model.Q_rt <= M_gen[self.stage]*model.n_rt
        
        def market_clearing_rule_5(model):
            return model.Q_rt >= model.q_rt - M_gen[self.stage]*(1 - model.n_rt)
        
        
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
            return model.w[j] <= M_gen[self.stage]*model.nu[j]
        
        def binarize_rule_1_4(model, j):
            return model.w[j] >= model.u - M_gen[self.stage]*(1-model.nu[j])
        
        def binarize_rule_2_1(model, i):
            return model.h[i] >= 0
        
        def binarize_rule_2_2(model, i):
            return model.h[i] <= model.m_1
        
        def binarize_rule_2_3(model, i):
            return model.h[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_2_4(model, i):
            return model.h[i] >= model.m_1 - M_gen[self.stage]*(1 - model.lamb[i])
        
        def binarize_rule_3_1(model, i):
            return model.k[i] >= 0
        
        def binarize_rule_3_2(model, i):
            return model.k[i] <= model.m_2
        
        def binarize_rule_3_3(model, i):
            return model.k[i] <= M_gen[self.stage]*model.lamb[i]
        
        def binarize_rule_3_4(model, i):
            return model.k[i] >= model.m_2 - M_gen[self.stage]*(1 - model.lamb[i])        
        
        def binarize_rule_4_1(model, i):
            return model.o[i] >= 0
        
        def binarize_rule_4_2(model, i):
            return model.o[i] <= model.m_3
        
        def binarize_rule_4_3(model, i):
            return model.o[i] <= M_gen[self.stage]*model.nu[i]
        
        def binarize_rule_4_4(model, i):
            return model.o[i] >= model.m_3 - M_gen[self.stage]*(1 - model.nu[i])        
        
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
            return model.m_1 >= model.Q_da - (1 - model.n_1)*M_gen[self.stage]
        
        def minmax_rule_1_4(model):
            return model.m_1 >= model.q_rt - (model.n_1)*M_gen[self.stage]
        
        def minmax_rule_2_1(model):
            return model.m_2 <= model.u
        
        def minmax_rule_2_2(model):
            return model.m_2 <= model.q_rt
        
        def minmax_rule_2_3(model):
            return model.m_2 >= model.u - (1 - model.n_2)*M_gen[self.stage]
        
        def minmax_rule_2_4(model):
            return model.m_2 >= model.q_rt - model.n_2*M_gen[self.stage]
        
        def minmax_rule_3_1(model):
            return model.m_3 >= model.u - model.Q_c - 0.12*C
        
        def minmax_rule_3_2(model):
            return model.m_3 >= 0
        
        def minmax_rule_3_3(model):
            return model.m_3 <= model.u - model.Q_c - 0.12*C + M_gen[self.stage]*(1 - model.n_3)
        
        def minmax_rule_3_4(model):
            return model.m_3 <= M_gen[self.stage]*model.n_3

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
            return model.Q_sum <= M_gen[self.stage]*model.n_sum
        
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

"""


# Perfect Forecasting

class PerfectForecasting:
    
    def __init__(
        self, 
        E_0_daily, 
        P_da_daily,
        P_rt_daily,
        delta_E_daily
        ):

        self.STAGE = T
        self.E_0_daily = E_0_daily
        self.P_da_daily = P_da_daily
        self.P_rt_daily = P_rt_daily
        self.delta_E_daily = delta_E_daily
        
        self.f_ESSo = []
        self.f_ESSx = []
        
        self.P_ESSo = 0
        self.P_ESSx = 0
        
        self.f_ESSo_f_P = []
        self.f_ESSo_f_V = []
        self.f_ESSo_f_E = []
        self.f_ESSo_f_VE = []
        self.f_ESSo_f_Im = []
        self.f_ESSo_f_C = []
        self.f_ESSo_f_REC = []

        self.f_ESSx_f_P = []
        self.f_ESSx_f_V = []
        self.f_ESSx_f_E = []
        self.f_ESSx_f_VE = []
        self.f_ESSx_f_Im = []
        self.f_ESSx_f_C = []
        self.f_ESSx_f_REC = []
        
        self.P_ESSo_f_P = 0
        self.P_ESSo_f_V = 0
        self.P_ESSo_f_E = 0
        self.P_ESSo_f_VE = 0
        self.P_ESSo_f_Im = 0
        self.P_ESSo_f_C = 0
        self.P_ESSO_f_REC = 0
        
        self.P_ESSx_f_P = 0
        self.P_ESSx_f_V = 0
        self.P_ESSx_f_E = 0        
        self.P_ESSx_f_VE = 0
        self.P_ESSx_f_Im = 0
        self.P_ESSx_f_C = 0
        self.P_ESSx_f_REC = 0
        
        self.b_da_ESSo = []
        self.b_rt_ESSo = []
        self.q_da_ESSo = []
        self.q_rt_ESSo = []
        self.Q_da_ESSo = []
        self.Q_rt_ESSo = []
        self.Q_c_ESSo = []
        self.u_ESSo = []
        self.S_ESSo = []

        self.b_da_ESSx = []
        self.b_rt_ESSx = []
        self.q_da_ESSx = []
        self.q_rt_ESSx = []
        self.Q_da_ESSx = []
        self.Q_rt_ESSx = []
        self.Q_c_ESSx = []
        self.u_ESSx = []

    def sample_scenarios(self):

        scenarios = []
        
        for E_0 in self.E_0_daily:
            scenario_generator = Scenario.Setting2_scenario(E_0)
            scenario = scenario_generator.scenario()
            scenarios.append(scenario)
            
        return scenarios

    def perfect_forecasting(self, scenarios):

        for k, E_0 in enumerate(self.E_0_daily):
            
            print(f"Perfect forecasting for scenario {k} solving...")
            
            P_da = self.P_da_daily[k]
            P_rt = self.P_rt_daily[k]
            delta_E = self.delta_E_daily[k]
            
            scenario = scenarios[k]
            
            Det_ESSo = Deterministic_Setting(E_0, P_da, P_rt, delta_E, scenario)
            Det_ESSx = Deterministic_Setting_withoutESS(E_0, P_da, P_rt, delta_E, scenario)
                
            f_scenario_ESSo = Det_ESSo.get_objective_value()
            f_scenario_ESSx = Det_ESSx.get_objective_value()
            
            (
                f_scenario_ESSo_f_P,
                f_scenario_ESSo_f_V,
                f_scenario_ESSo_f_E,
                f_scenario_ESSo_f_VE,
                f_scenario_ESSo_f_Im,
                f_scenario_ESSo_f_C,
                f_scenario_ESSo_f_REC,
            ) = Det_ESSo.get_objective_part_value()

            (
                f_scenario_ESSx_f_P,
                f_scenario_ESSx_f_V,
                f_scenario_ESSx_f_E,
                f_scenario_ESSx_f_VE,
                f_scenario_ESSx_f_Im,
                f_scenario_ESSx_f_C,
                f_scenario_ESSx_f_REC,
            ) = Det_ESSx.get_objective_part_value()

            self.f_ESSo.append(f_scenario_ESSo)
            self.f_ESSx.append(f_scenario_ESSx)

            self.f_ESSo_f_P.append(f_scenario_ESSo_f_P)
            self.f_ESSo_f_V.append(f_scenario_ESSo_f_V)
            self.f_ESSo_f_E.append(f_scenario_ESSo_f_E)
            self.f_ESSo_f_VE.append(f_scenario_ESSo_f_VE)
            self.f_ESSo_f_Im.append(f_scenario_ESSo_f_Im)
            self.f_ESSo_f_C.append(f_scenario_ESSo_f_C)
            self.f_ESSo_f_REC.append(f_scenario_ESSo_f_REC)

            self.f_ESSx_f_P.append(f_scenario_ESSx_f_P)
            self.f_ESSx_f_V.append(f_scenario_ESSx_f_V)
            self.f_ESSx_f_E.append(f_scenario_ESSx_f_E)
            self.f_ESSx_f_VE.append(f_scenario_ESSx_f_VE)
            self.f_ESSx_f_Im.append(f_scenario_ESSx_f_Im)
            self.f_ESSx_f_C.append(f_scenario_ESSx_f_C)
            self.f_ESSx_f_REC.append(f_scenario_ESSx_f_REC)
            
            self.b_da_ESSo.append(Det_ESSo.get_b_da_value())
            self.b_rt_ESSo.append(Det_ESSo.get_b_rt_value())
            self.q_da_ESSo.append(Det_ESSo.get_q_da_value())
            self.q_rt_ESSo.append(Det_ESSo.get_q_rt_value())
            self.Q_da_ESSo.append(Det_ESSo.get_Q_da_value())
            self.Q_rt_ESSo.append(Det_ESSo.get_Q_rt_value())
            self.Q_c_ESSo.append(Det_ESSo.get_Q_c_value())
            self.u_ESSo.append(Det_ESSo.get_u_value())
            self.S_ESSo.append(Det_ESSo.get_S_value())

            self.b_da_ESSx.append(Det_ESSx.get_b_da_value())
            self.b_rt_ESSx.append(Det_ESSx.get_b_rt_value())
            self.q_da_ESSx.append(Det_ESSx.get_q_da_value())
            self.q_rt_ESSx.append(Det_ESSx.get_q_rt_value())
            self.Q_da_ESSx.append(Det_ESSx.get_Q_da_value())
            self.Q_rt_ESSx.append(Det_ESSx.get_Q_rt_value())
            self.Q_c_ESSx.append(Det_ESSx.get_Q_c_value())
            self.u_ESSx.append(Det_ESSx.get_u_value())

        self.P_ESSo = np.mean(self.f_ESSo) 
        self.P_ESSx = np.mean(self.f_ESSx)
        
        self.P_ESSo_f_P = np.mean(self.f_ESSo_f_P)
        self.P_ESSo_f_V = np.mean(self.f_ESSo_f_V)
        self.P_ESSo_f_E = np.mean(self.f_ESSo_f_E)
        self.P_ESSo_f_VE = np.mean(self.f_ESSo_f_VE)
        self.P_ESSo_f_Im = np.mean(self.f_ESSo_f_Im)
        self.P_ESSo_f_C = np.mean(self.f_ESSo_f_C)
        self.P_ESSo_f_REC = np.mean(self.f_ESSo_f_REC)

        self.P_ESSx_f_P = np.mean(self.f_ESSx_f_P)
        self.P_ESSx_f_V = np.mean(self.f_ESSx_f_V)
        self.P_ESSx_f_E = np.mean(self.f_ESSx_f_E)
        self.P_ESSx_f_VE = np.mean(self.f_ESSx_f_VE)
        self.P_ESSx_f_Im = np.mean(self.f_ESSx_f_Im)
        self.P_ESSx_f_C = np.mean(self.f_ESSx_f_C)
        self.P_ESSx_f_REC = np.mean(self.f_ESSx_f_REC)

    def run_perfect_forecasting(self):
                
        scenarios = self.sample_scenarios()

        print(f"\n=== Perfect forecasting ===")
        self.perfect_forecasting(scenarios)
        
        print("\nPerfect forecasting complete.")
        print(f"P_ESSo = {self.P_ESSo}")
        print(f"P_ESSx = {self.P_ESSx}")
        
        print(f"P_ESSo_f_P = {self.P_ESSo_f_P}")
        print(f"P_ESSo_f_V = {self.P_ESSo_f_V}")
        print(f"P_ESSo_f_E = {self.P_ESSo_f_E}")
        print(f"P_ESSo_f_VE = {self.P_ESSo_f_VE}")
        print(f"P_ESSo_f_Im = {self.P_ESSo_f_Im}")
        print(f"P_ESSo_f_C = {self.P_ESSo_f_C}")
        print(f"P_ESSo_f_REC = {self.P_ESSo_f_REC}")

        print(f"P_ESSx_f_P = {self.P_ESSx_f_P}")
        print(f"P_ESSx_f_V = {self.P_ESSx_f_V}")
        print(f"P_ESSx_f_E = {self.P_ESSx_f_E}")
        print(f"P_ESSx_f_VE = {self.P_ESSx_f_VE}")
        print(f"P_ESSx_f_Im = {self.P_ESSx_f_Im}")
        print(f"P_ESSx_f_C = {self.P_ESSx_f_C}")
        print(f"P_ESSx_f_REC = {self.P_ESSx_f_REC}")

"""   
class PerfectForecastingComp:
    
    def __init__(
        self, 
        STAGE = 24 ,
        scenario_num = 30, 
        ):

        self.STAGE = STAGE
        self.M = scenario_num
        
        self.P = 0

    def sample_P_da(self):
        
        scenario_generator = Scenario.Setting2_scenario(E_0)
        
        P_da_list = scenario_generator.sample_multiple_P_da(self.M)
        
        return P_da_list
    
    def sample_scenarios(self):

        scenarios = []
        
        for _ in range(self.M):
            scenario_generator = Scenario.Setting2_scenario(E_0)
            scenario = scenario_generator.scenario()
            scenarios.append(scenario)
            
        return scenarios

    def perfect_forecasting(self, P_da_list, scenarios):
        
        f = []
        
        for k in range(self.M):
            
            print(f"Perfect forecasting for scenario {k} solving...")
            
            scenario = scenarios[k]
            P_da = P_da_list[k]
            
            Det_ESSo = Deterministic_Setting(E_0, P_da, scenario)
                
            f_scenario = Det_ESSo.get_objective_value()
                        
            f.append(f_scenario)

        self.P = np.mean(f) 

    def run_perfect_forecasting(self):
                
        scenarios = self.sample_scenarios()
        P_da_list = self.sample_P_da()

        print(f"\n=== Perfect forecasting ===")
        self.perfect_forecasting(P_da_list, scenarios)
        
        print("\nPerfect forecasting complete.")
        print(f"P = {self.P}")     
"""  
   
# RollingHorizon

class RollingHorizon:
    
    def __init__(
        self, 
        E_0_daily,
        P_da_daily,
        P_rt_daily,
        delta_E_daily,
        ):

        self.STAGE = 24
        
        self.E_0_daily = E_0_daily
        self.P_da_daily = P_da_daily
        self.P_rt_daily = P_rt_daily
        self.delta_E_daily = delta_E_daily
        
        self.M = len(self.E_0_daily)
        
        self.R_ESSo = 0
        self.R_ESSx = 0

    def expected_P(self, E_0):
        
        scenario_generator = Scenario.Setting2_scenario(E_0)
        
        exp_P_da = scenario_generator.exp_P_da
        exp_P_rt = scenario_generator.exp_P_rt
        
        return exp_P_da, exp_P_rt

    def sample_scenarios(self):

        scenarios = []
        
        for E_0 in self.E_0_daily:
            scenario_generator = Scenario.Setting2_scenario(E_0)
            scenario = scenario_generator.scenario()
            scenarios.append(scenario)
            
        return scenarios

    def rolling_horizon_ESSo(self, scenarios):
        
        f = []
        
        for k in range(self.M):
            
            print(f"Rolling for scenario ESSo {k} solving...")
            
            scenario = scenarios[k]
            E_0 = self.E_0_daily[k]
            P_da = self.P_da_daily[k]
            P_rt = self.P_rt_daily[k]
            delta_E = self.delta_E_daily[k]
                    
            exp_P_da, exp_P_rt = self.expected_P(E_0)
            
            roll_da_subp = Rolling_da(E_0, exp_P_da, exp_P_rt)
            b_da = roll_da_subp.sol_b_da
            q_da = roll_da_subp.sol_q_da
            
            roll_rt_init_subp = Rolling_rt_init([b_da, q_da], E_0, P_da, exp_P_rt)
            
            Q_da = roll_rt_init_subp.sol_Q_da
            b_rt = roll_rt_init_subp.sol_b_rt
            q_rt = roll_rt_init_subp.sol_q_rt
            S = roll_rt_init_subp.sol_S
            E_1 = 0
            T_q = q_rt
                        
            f_scenario = 0
            
            for t in range(self.STAGE - 1): ## t = 0, ..., 22
                
                print(f"rolling {t} solving..")
                
                roll_rt_subp = Rolling_rt(
                    t, 
                    [[b_da[i] for i in range(t, T)], [Q_da[i] for i in range(t, T)]], 
                    b_rt,
                    q_rt,
                    S,
                    E_1,
                    T_q,
                    E_0,
                    P_da,
                    P_rt[t],
                    delta_E[t+1],
                    exp_P_rt,
                    scenario[t]
                    )

                b_rt = roll_rt_subp.sol_b_rt
                q_rt = roll_rt_subp.sol_q_rt
                S = roll_rt_subp.sol_S
                E_1 = roll_rt_subp.sol_E_1
                T_q = roll_rt_subp.sol_T_q
                                
                f_scenario += roll_rt_subp.get_objective_value()
            
            print("rolling 23 solving")
            
            # t = 23
            roll_rt_last_subp = Rolling_rt_last(
                [[b_da[T - 1]], [Q_da[T - 1]]],
                b_rt,
                q_rt,
                S,
                E_1, 
                E_0,
                P_da,
                P_rt[T - 1],
                delta_E[T - 1],
                scenario[self.STAGE-1]
                )
                
            f_scenario += roll_rt_last_subp.get_objective_value()
            
            print(f_scenario)
            
            f.append(f_scenario)

        self.R_ESSo = np.mean(f) 

    def rolling_horizon_ESSx(self, scenarios):
        
        f = []
        
        for k in range(self.M):
            
            print(f"Rolling for scenario ESSx {k} solving...")
            
            scenario = scenarios[k]
            E_0 = self.E_0_daily[k]
            P_da = self.P_da_daily[k]
            P_rt = self.P_rt_daily[k]
            delta_E = self.delta_E_daily[k]
            
            exp_P_da, exp_P_rt = self.expected_P(E_0)
            
            roll_da_subp = Rolling_da_ESSx(E_0, exp_P_da, exp_P_rt)
            b_da = roll_da_subp.sol_b_da
            q_da = roll_da_subp.sol_q_da
            
            roll_rt_init_subp = Rolling_rt_init_ESSx([b_da, q_da], E_0, P_da, exp_P_rt)
            
            Q_da = roll_rt_init_subp.sol_Q_da
            b_rt = roll_rt_init_subp.sol_b_rt
            q_rt = roll_rt_init_subp.sol_q_rt
            S = roll_rt_init_subp.sol_S
            E_1 = 0
            T_q = q_rt
                        
            f_scenario = 0
            
            for t in range(self.STAGE - 1): ## t = 0, ..., 22
                
                roll_rt_subp = Rolling_rt_ESSx(
                    t, 
                    [[b_da[i] for i in range(t, T)], [Q_da[i] for i in range(t, T)]], 
                    b_rt,
                    q_rt,
                    S,
                    E_1,
                    T_q,
                    E_0,
                    P_da,
                    P_rt[t],
                    delta_E[t+1],
                    exp_P_rt,
                    scenario[t]
                    )

                b_rt = roll_rt_subp.sol_b_rt
                q_rt = roll_rt_subp.sol_q_rt
                S = roll_rt_subp.sol_S
                E_1 = roll_rt_subp.sol_E_1
                T_q = roll_rt_subp.sol_T_q
                                
                f_scenario += roll_rt_subp.get_objective_value()
            
            # t = 23
            roll_rt_last_subp = Rolling_rt_last_ESSx(
                [[b_da[T - 1]], [Q_da[T - 1]]],
                b_rt,
                q_rt,
                S,
                E_1, 
                E_0,
                P_da,
                P_rt[T - 1],
                delta_E[T - 1],
                scenario[self.STAGE-1]
                )
                
            f_scenario += roll_rt_last_subp.get_objective_value()
            
            print(f_scenario)
            
            f.append(f_scenario)

        self.R_ESSx = np.mean(f) 

    def run_rolling(self):

        scenarios = self.sample_scenarios()

        print(f"\n=== Rolling Horizon ===")
        self.rolling_horizon_ESSx(scenarios)
        self.rolling_horizon_ESSo(scenarios)
        
        print("\nRolling Horizon complete.")
        print(f"R_ESSo = {self.R_ESSo}")
        print(f"R_ESSx = {self.R_ESSx}")
        
# SDDiPModel

"""                          
class SDDiPModel:
        
    def __init__(
        self, 
        STAGE = 24 ,
        forward_scenario_num = 5, 
        backward_branch = 10, 
        max_iter = 2, 
        alpha = 0.95, 
        cut_mode = 'B', 
        bigM_mode = 'bigM',
        ):

        ## STAGE = -1, 0, ..., STAGE - 1
        ## forward_scenarios = M
        ## baackward_scenarios = N_t

        self.STAGE = STAGE
        self.M = forward_scenario_num
        self.N_t = backward_branch
        self.alpha = alpha
        self.cut_mode = cut_mode
        self.bigM_mode = bigM_mode
        
        self.iteration = 0
        
        self.max_iter = max_iter
        
        self.P = [0]
        self.R = [0]
        self.LB = [-np.inf]
        self.UB = [np.inf]

        self.forward_solutions = [  ## T(-1), ..., T(22)
            [] for _ in range(self.STAGE)
        ]
        
        self.psi = [[] for _ in range(self.STAGE)] ## t = {0 -> -1}, ..., {23 -> 22}
        
        self._initialize_psi()

    def _initialize_psi(self):
        
        for t in range(self.STAGE): ## psi(-1), ..., psi(22)
            self.psi[t].append([4000000*(24 - t), 0, [0 for _ in range(self.STAGE - t)], [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0])

    def sample_scenarios(self):

        scenarios = []
        
        for _ in range(self.M):
            scenario_generator = Scenario.Setting1_scenario(E_0)
            scenario = scenario_generator.scenario()
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
            
            for t in range(self.STAGE - 1): ## t = 0, ..., 22
                
                if self.bigM_mode == 'bigM':
                    fw_rt_subp = fw_rt(t, state, self.psi[t+1], scenario[t])
                    
                elif self.bigM_mode == 'Alt':
                    fw_rt_subp = fw_rt_Alt(t, state, self.psi[t+1], scenario[t])
                
                state = fw_rt_subp.get_state_solutions()
                self.forward_solutions[t+1].append(state)
                f_scenario += fw_rt_subp.get_settlement_fcn_value()
            
            ## t = 23
            if self.bigM_mode == 'bigM':
                fw_rt_last_subp = fw_rt_last(state, scenario[self.STAGE-1])
            
            elif self.bigM_mode == 'Alt':    
                fw_rt_last_subp = fw_rt_last_Alt(state, scenario[self.STAGE-1]) 
                
            f_scenario += fw_rt_last_subp.get_settlement_fcn_value()
            
            f.append(f_scenario)
        
        mu_hat = np.mean(f)
        sigma_hat = np.std(f, ddof=1)  

        z_alpha_half = 1.96  
        
        self.LB.append(mu_hat) 
        
        return mu_hat

    def inner_product(self, t, pi, sol):
        
        return sum(pi[i]*sol[i] for i in [0, 3, 4, 5, 6]) + sum(sum(pi[i][j]*sol[i][j] for j in range(self.STAGE - t)) for i in [1, 2])
                     
    def backward_pass(self):
        
        scenario_generator = Scenario.Setting1_scenario(E_0)
        
        ## t = {23 -> 22}
        
        v_sum = 0 
        pi_mean = [0, [0], [0], 0, 0, 0, 0]
        
        prev_solution = self.forward_solutions[self.STAGE - 1][0]
        
        deltas = scenario_generator.sample_multiple_delta(self.STAGE - 1, self.N_t) 
        
        for j in range(self.N_t): 
            
            delta = deltas[j]   
            
            if self.bigM_mode == 'bigM':
                fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax(prev_solution, delta)
           
            elif self.bigM_mode == 'Alt':    
                fw_rt_last_LP_relax_subp = fw_rt_last_LP_relax_Alt(prev_solution, delta)
                
            psi_sub = fw_rt_last_LP_relax_subp.get_cut_coefficients()
            
            pi_mean[0] += psi_sub[1]/self.N_t
            pi_mean[1][0] += psi_sub[2][0]/self.N_t
            pi_mean[2][0] += psi_sub[3][0]/self.N_t
            pi_mean[3] += psi_sub[4]/self.N_t
            pi_mean[4] += psi_sub[5]/self.N_t
            pi_mean[5] += psi_sub[6]/self.N_t
            pi_mean[6] += psi_sub[7]/self.N_t
            
            if self.cut_mode == 'B':
                
                v_sum += psi_sub[0]
            
            elif self.cut_mode == 'SB':
                
                if self.bigM_mode == 'bigM':
                    fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian([psi_sub[i] for i in range(1, 8)], delta)
              
                elif self.bigM_mode == 'Alt':
                    fw_rt_last_Lagrangian_subp = fw_rt_last_Lagrangian_Alt([psi_sub[i] for i in range(1, 8)], delta)
                
                v_sum += fw_rt_last_Lagrangian_subp.get_objective_value()
            
        if self.cut_mode == 'B':   
                 
            v = v_sum/self.N_t - self.inner_product(self.STAGE - 1, pi_mean, prev_solution)
        
        elif self.cut_mode == 'SB':
            
            v = v_sum/self.N_t
            
        cut_coeff = []
        
        cut_coeff.append(v)
        
        for i in range(7):
            cut_coeff.append(pi_mean[i])
        
        self.psi[23].append(cut_coeff)
        
        #print(f"last_stage_cut_coeff = {self.psi[23]}")
        
        ## t = {22 -> 21}, ..., {0 -> -1}
        for t in range(self.STAGE - 2, -1, -1): 
                
            v_sum = 0 
            pi_mean = [0, [0 for _ in range(self.STAGE - t)], [0 for _ in range(self.STAGE - t)], 0, 0, 0, 0]
            
            prev_solution = self.forward_solutions[t][0]
                        
            scenario_generator = Scenario.Setting1_scenario(E_0)
            
            deltas = scenario_generator.sample_multiple_delta(t, self.N_t) 
            
            for j in range(self.N_t):
                
                delta = deltas[j]
                
                if self.bigM_mode == 'bigM':
                    fw_rt_LP_relax_subp = fw_rt_LP_relax(t, prev_solution, self.psi[t+1], delta)
             
                elif self.bigM_mode == 'Alt':
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
                
                if self.cut_mode == 'B':
                    
                    v_sum += psi_sub[0]
                    
                elif self.cut_mode == 'SB':
                    
                    if self.bigM_mode == 'bigM':
                        fw_rt_Lagrangian_subp = fw_rt_Lagrangian(t, [psi_sub[i] for i in range(1, 8)], self.psi[t+1], delta)
                 
                    elif self.bigM_mode == 'Alt':
                        fw_rt_Lagrangian_subp = fw_rt_Lagrangian_Alt(t, [psi_sub[i] for i in range(1, 8)], self.psi[t+1], delta)
                    
                    
                    v_sum += fw_rt_Lagrangian_subp.get_objective_value()
            
            if self.cut_mode == 'B':
                
                v = v_sum/self.N_t - self.inner_product(self.STAGE - 1, pi_mean, prev_solution)
        
            if self.cut_mode == 'SB':
                
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
        
    def stopping_criterion(self, tol = 100000):
      
        if self.iteration >= self.max_iter:
            return True
        gap = self.UB[self.iteration] - self.LB[self.iteration]
        return (gap < tol)

    def run_sddip(self):
        
        while not self.stopping_criterion():
            self.iteration += 1
            print(f"\n=== Iteration {self.iteration} ===")

            scenarios = self.sample_scenarios()

            self.forward_pass(scenarios)

            print(f"  LB for iter = {self.iteration} updated to: {self.LB[self.iteration]:.4f}")
            
            self.backward_pass()

            print(f"  UB for iter = {self.iteration} updated to: {self.UB[self.iteration]:.4f}")

        print("\nSDDiP complete.")
        print(f"Final LB = {self.LB[self.iteration]:.4f}, UB = {self.UB[self.iteration]:.4f}, gap = {(self.UB[self.iteration] - self.LB[self.iteration])/self.UB[self.iteration]:.4f}")
        print(f"LB list = {self.LB}, UB list = {self.UB}")
"""

if __name__ == "__main__":
    
    num_iter = 25
    
    scenario_branch = 5  ## N_t
    fw_scenario_num = 10  ## M
    perfect_scenario_num = 20
    rolling_scenario_num = 20
    
    """ 
    # PF vs Roll vs SDDP
    
    perfect = PerfectForecastingComp(scenario_num = perfect_scenario_num)
    perfect.run_perfect_forecasting()
    
    rolling = RollingHorizon(scenario_num = rolling_scenario_num)
    
    rolling_start_time = time.time()
    rolling.run_rolling()
    rolling_end_time = time.time()
    avg_time_rolling = (rolling_end_time - rolling_start_time)/rolling_scenario_num
    
    perfect_obj = perfect.P
    rolling_obj = rolling.R
    
         
    sddip_B = SDDiPModel(
        max_iter=num_iter,
        forward_scenario_num=fw_scenario_num,
        backward_branch=scenario_branch,
        cut_mode='B',
        bigM_mode='Alt',
    )
    
    sddip_SB = SDDiPModel(
        max_iter=num_iter,
        forward_scenario_num=fw_scenario_num,
        backward_branch=scenario_branch,
        cut_mode='SB',
        bigM_mode='Alt',
    )
    
   
    SDDiP_B_start_time = time.time()
    sddip_B.run_sddip()
    SDDiP_B_end_time = time.time()
    time_SDDiP_B = SDDiP_B_end_time - SDDiP_B_start_time


    LB_B_list = sddip_B.LB[:num_iter] 
    UB_B_list = sddip_B.UB[:num_iter]
    
    gap_SDDiP_B = (sddip_B.UB[sddip_B.iteration] - sddip_B.LB[sddip_B.iteration])/sddip_B.UB[sddip_B.iteration]
    
    SDDiP_SB_start_time = time.time()
    sddip_SB.run_sddip()
    SDDiP_SB_end_time = time.time()
    time_SDDiP_SB = SDDiP_SB_end_time - SDDiP_SB_start_time

    LB_SB_list = sddip_SB.LB[:num_iter]
    UB_SB_list = sddip_SB.UB[:num_iter]
    
    gap_SDDiP_SB = (sddip_SB.UB[sddip_SB.iteration] - sddip_SB.LB[sddip_SB.iteration])/sddip_SB.UB[sddip_SB.iteration]

    iterations = range(num_iter)

    plt.plot(iterations, LB_B_list, label="LB (B)", marker='o', color='tab:blue')
    plt.plot(iterations, UB_B_list, label="UB (B)", marker='^', color='tab:blue', linestyle='--')
    plt.fill_between(iterations, LB_B_list, UB_B_list, alpha=0.1, color='tab:blue')
    
    plt.plot(iterations, LB_SB_list, label="LB (SB)", marker='o', color='tab:orange')
    plt.plot(iterations, UB_SB_list, label="UB (SB)", marker='^', color='tab:orange', linestyle='--')
    plt.fill_between(iterations, LB_SB_list, UB_SB_list, alpha=0.1, color='tab:orange')
    
    plt.axhline(y=perfect_obj, color='brown', linestyle='--', label='PerfectForecasting_obj')

    plt.axhline(y=rolling_obj, color='black', linestyle='--', label='RollingHorizon_obj')


    plt.xlabel('Iteration')
    plt.ylabel('Bound')
    plt.title('PF vs Rolling Horizon vs SDDP')
    plt.legend()
    
    plt.ylim(0, 50000000)
    
    plt.show()



    print(f"Rolling Horizon took {avg_time_rolling:.2f} seconds on average.\n")
    
    print(f"SDDiP (cut='B') took {time_SDDiP_B:.2f} seconds.\n")
    print(f"SDDiP (cut='SB') took {time_SDDiP_SB:.2f} seconds.\n")
    
    print(f"SDDiP optimality gap for B = {gap_SDDiP_B:.4f}")
    print(f"SDDiP optimality gap for SB = {gap_SDDiP_SB:.4f}")    

    """
    
    # ESSx vs ESSo
    
    E_0_daily = E_0_daily[:2]  # ~215
    P_da_daily = P_da_daily[:2]  # ~215
    P_rt_daily = P_rt_daily[:2]  # ~215
    delta_E_daily = delta_E_daily[:2]  # ~215
    
    rolling = RollingHorizon(E_0_daily, P_da_daily, P_rt_daily, delta_E_daily)
    
    rolling.run_rolling()
     
    """
    perfect = PerfectForecasting(E_0_daily, P_da_daily, P_rt_daily, delta_E_daily)
    
    perfect.run_perfect_forecasting()
    
    f_ESSo = perfect.f_ESSo
    f_ESSx = perfect.f_ESSx
    
    perfect_obj_ESS = perfect.P_ESSo
    perfect_obj_ESSx = perfect.P_ESSx
    
    b_da_ESSo_list = perfect.b_da_ESSo
    b_rt_ESSo_list = perfect.b_rt_ESSo
    q_da_ESSo_list = perfect.q_da_ESSo
    q_rt_ESSo_list = perfect.q_rt_ESSo
    Q_da_ESSo_list = perfect.Q_da_ESSo
    Q_rt_ESSo_list = perfect.Q_rt_ESSo
    Q_c_ESSo_list = perfect.Q_c_ESSo
    u_ESSo_list = perfect.u_ESSo
    S_ESSo_list = perfect.S_ESSo

    b_da_ESSx_list = perfect.b_da_ESSx
    b_rt_ESSx_list = perfect.b_rt_ESSx
    q_da_ESSx_list = perfect.q_da_ESSx
    q_rt_ESSx_list = perfect.q_rt_ESSx
    Q_da_ESSx_list = perfect.Q_da_ESSx
    Q_rt_ESSx_list = perfect.Q_rt_ESSx
    Q_c_ESSx_list = perfect.Q_c_ESSx
    u_ESSx_list = perfect.u_ESSx
    
    daily_dates = pd.date_range(start="2024-03-01", periods=len(u_ESSo_list), freq='D')
    omit_dates = [pd.Timestamp("2024-12-19"), pd.Timestamp("2025-01-11"), pd.Timestamp("2025-02-15")]
    filtered_daily_dates = daily_dates[~daily_dates.isin(omit_dates)]
    
    filtered = lambda lst: [lst[i] for i, dt in enumerate(daily_dates) if dt not in omit_dates]

    filtered_b_da_ESSo_list = filtered(b_da_ESSo_list)
    filtered_b_rt_ESSo_list = filtered(b_rt_ESSo_list)
    filtered_q_da_ESSo_list = filtered(q_da_ESSo_list)
    filtered_q_rt_ESSo_list = filtered(q_rt_ESSo_list)
    filtered_Q_da_ESSo_list = filtered(Q_da_ESSo_list)
    filtered_Q_rt_ESSo_list = filtered(Q_rt_ESSo_list)
    filtered_Q_c_ESSo_list = filtered(Q_c_ESSo_list)
    filtered_u_ESSo_list = filtered(u_ESSo_list)
    filtered_S_ESSo_list = filtered(S_ESSo_list)

    filtered_b_da_ESSx_list = filtered(b_da_ESSx_list)
    filtered_b_rt_ESSx_list = filtered(b_rt_ESSx_list)
    filtered_q_da_ESSx_list = filtered(q_da_ESSx_list)
    filtered_q_rt_ESSx_list = filtered(q_rt_ESSx_list)
    filtered_Q_da_ESSx_list = filtered(Q_da_ESSx_list)
    filtered_Q_rt_ESSx_list = filtered(Q_rt_ESSx_list)
    filtered_Q_c_ESSx_list = filtered(Q_c_ESSx_list)
    filtered_u_ESSx_list = filtered(u_ESSx_list)
    
    num_hours = len(u_ESSo_list[0])
    hourly_timestamps = []
    
    for dt in filtered_daily_dates:
        
        hourly_timestamps.extend(pd.date_range(start=dt, periods=num_hours, freq='H'))
        
    flat = lambda lst: [val for sublist in lst for val in sublist]    
        
    df_ESSo = pd.DataFrame({
        "date": hourly_timestamps,
        "b_da_ESSo": flat(filtered_b_da_ESSo_list),
        "b_rt_ESSo": flat(filtered_b_rt_ESSo_list),
        "q_da_ESSo": flat(filtered_q_da_ESSo_list),
        "q_rt_ESSo": flat(filtered_q_rt_ESSo_list),
        "Q_da_ESSo": flat(filtered_Q_da_ESSo_list),
        "Q_rt_ESSo": flat(filtered_Q_rt_ESSo_list),
        "Q_c_ESSo": flat(filtered_Q_c_ESSo_list),
        "u_ESSo": flat(filtered_u_ESSo_list),
        "S_ESSo": flat(filtered_S_ESSo_list),
    })

    df_ESSx = pd.DataFrame({
        "date": hourly_timestamps,
        "b_da_ESSx": flat(filtered_b_da_ESSx_list),
        "b_rt_ESSx": flat(filtered_b_rt_ESSx_list),
        "q_da_ESSx": flat(filtered_q_da_ESSx_list),
        "q_rt_ESSx": flat(filtered_q_rt_ESSx_list),
        "Q_da_ESSx": flat(filtered_Q_da_ESSx_list),
        "Q_rt_ESSx": flat(filtered_Q_rt_ESSx_list),
        "Q_c_ESSx": flat(filtered_Q_c_ESSx_list),
        "u_ESSx": flat(filtered_u_ESSx_list),
    })
    
    df_ESSo.to_csv("hourly_values_ESSo.csv", index=False)
    df_ESSx.to_csv("hourly_values_ESSx.csv", index=False)
    
    
    
    for i, day in enumerate(filtered_daily_dates):
        
        hours = pd.date_range(start=day, periods=num_hours, freq='H')
        plt.figure(figsize=(10, 6))
        plt.plot(hours, filtered_u_ESSo_list[i], marker='x', label='u_ESS')
        plt.plot(hours, filtered_Q_c_ESSo_list[i], marker='x', label='Q_c_ESS')
        plt.plot(hours, filtered_Q_da_ESSo_list[i], marker='x', label='Q_da_ESS')
        plt.xlabel("Time")
        plt.ylabel("u value")
        plt.title(f"Hourly u values for {day.strftime('%Y-%m-%d')}")
        plt.xticks(hours, [h.strftime("%H:%M") for h in hours], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    """
    
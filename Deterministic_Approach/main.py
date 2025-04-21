import pyomo.environ as pyo
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

solver='gurobi'
SOLVER=pyo.SolverFactory(solver)

SOLVER.options['NonConvex'] = 2 


assert SOLVER.available(), f"Solver {solver} is available."

# CONSTANTS

# Day-Ahead price

directory_path_da = '.\Deterministic_Approach\모의 실시간시장 가격\하루전'

files = os.listdir(directory_path_da)
csv_files = [file for file in files if file.endswith('.csv')]

def process_da_file(file_path):
    df = pd.read_csv(file_path)
    data = df.loc[3:27, df.columns[2]]  
    return data.tolist()

day_ahead_prices = []


for csv_file in csv_files:
    file_path = os.path.join(directory_path_da, csv_file)
    processed_data = process_da_file(file_path)
    day_ahead_prices.append(processed_data)


# Real-Time price

directory_path_rt = '.\Deterministic_Approach\모의 실시간시장 가격\실시간 임시'

files_rt = os.listdir(directory_path_rt)

csv_files_rt = [file for file in files_rt if file.endswith('.csv')]

def process_rt_file(file_path):

    df = pd.read_csv(file_path)
    data = df.iloc[3:99, 2]  
    reshaped_data = data.values.reshape(-1, 4).mean(axis=1)
    return reshaped_data

real_time_prices = []

for xlsx_file in csv_files_rt:
    file_path = os.path.join(directory_path_rt, xlsx_file)
    processed_data = process_rt_file(file_path)
    real_time_prices.append(processed_data)


# E_0

file_path_E_0 = '.\Deterministic_Approach\jeju_forecast.csv' 

df_E_0 = pd.read_csv(file_path_E_0)

df_E_0['timestamp'] = pd.to_datetime(df_E_0['timestamp'])

df_E_0['hour'] = df_E_0['timestamp'].dt.hour

average_forecast = df_E_0.groupby('hour')['gen_forecast'].mean()

E_0_mean = []

for i in average_forecast:
    E_0_mean.append(i)

E_0_sum = 0

for i in E_0_mean:
    E_0_sum += i

# Other Params

C = 560
S = 1680
B = 560
S_min = 168
S_max = 1512
P_r = 80
M_price = 1000
M_gen = 5000
M_set = 50000
T = 24
v = 0.95


# Scenario generation

def generate_scenario(n):
    np.random.seed(n)
    P_da=[]
    P_rt=[]
    E_0=[]
    E_1=[]
    U=[]
    I=np.random.binomial(1, 0.05, size = T)
    Un=np.random.uniform(0, 1, T)
    for t in range(T):
        P_da.append(day_ahead_prices[n][t])
        P_rt.append(real_time_prices[n][t])
        E_0.append(E_0_mean[t])
        E_1.append(E_0[t] * np.random.uniform(0.9, 1.1))
        U.append(I[t]*Un[t])
    scenario = [P_da, P_rt, E_0, E_1, U]
    return scenario

scenarios = []

for n in range(len(real_time_prices)):
    s = generate_scenario(n)
    scenarios.append(s) 


del scenarios[90]

class deterministic_setting_1(pyo.ConcreteModel):
    def __init__ (self, n, init_SoC):
        super().__init__("Deterministic_Setting1")
        
        self.solved = False        
        self.n = n        
        self.init_SoC = init_SoC
        
        self.scenario = scenarios[n]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]
        
        self.b_da_values = []
        self.b_rt_values = []
        self.q_da_values = []
        self.q_rt_values = []
        self.u_values = []
        self.g_values = []
        self.c_values = []
        self.d_values = []
        self.S_values = []
        self.Q_da_values = []
        self.Q_c_values = []
        
    def build_model(self):
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.ESSTIME = pyo.RangeSet(0, T)
        
        model.b_da = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.b_rt = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.g = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.d = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.TIME, bounds=(0, S+C), domain=pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.ESSTIME, bounds=(S_min, S_max), domain=pyo.Reals) 
        
        model.y_da = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_rt = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.Q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        # Linearization Real Variables
        model.m1_V = pyo.Var(model.TIME, bounds=(-2*M_gen, 2*M_gen), domain=pyo.Reals)
        model.m1_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m2_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m3_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m4_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m5_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m6_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m1_Im = pyo.Var(model.TIME, bounds=(-2*M_gen, 2*M_gen), domain=pyo.Reals)
        model.m2_Im = pyo.Var(model.TIME, bounds=(-2*M_price, 2*M_price), domain=pyo.Reals)
        model.S1_V = pyo.Var(model.TIME, domain=pyo.Reals)
        model.S1_E = pyo.Var(model.TIME, bounds=(-2*M_price, 2*M_price), domain=pyo.Reals)
        model.S2_E = pyo.Var(model.TIME, bounds=(-2*M_gen, 2*M_gen), domain=pyo.Reals)
        model.S1_Im = pyo.Var(model.TIME, domain=pyo.Reals)

        # Linearization Binary Variables
        model.n1_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n4_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n5_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n6_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n7_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n8_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_F = pyo.Var(model.TIME, domain=pyo.Binary)
        
        #부가정산금
        model.f_max = pyo.Var(model.TIME, domain=pyo.Reals)

        #Constraints

        if hasattr(model, 'constrs'):
            model.del_component('constrs')
            model.del_component('constrs_index')
        
        model.constrs = pyo.ConstraintList()
    
        for t in range(T):
            # q_da, q_rt constraint
            model.constrs.add(model.q_da[t] <= 1.1*self.E_0[t] + B)
            model.constrs.add(model.q_rt[t] <= 1.1*self.E_1[t] + B)
            
            # Demand response
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M_price * (1-model.y_da[t]))
            model.constrs.add(self.P_da[t] - model.b_da[t] <= M_price * model.y_da[t])
            model.constrs.add(model.Q_da[t] == model.y_da[t] * model.q_da[t])
        
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M_price * (1-model.y_rt[t]))
            model.constrs.add(self.P_rt[t] - model.b_rt[t] <= M_price * model.y_rt[t])
            model.constrs.add(model.Q_rt[t] == model.y_rt[t] * model.q_rt[t]) 
            
            model.constrs.add(model.Q_c[t] == model.Q_rt[t]*self.U[t])
 
            # b_rt <= b_da
            model.constrs.add(model.b_rt[t] <= model.b_da[t])
            
            # ESS operation
            model.constrs.add(model.S[t+1] == model.S[t] + v*model.c[t] - (model.d[t])/v)
            model.constrs.add(model.u[t] == model.g[t] + (model.d[t])/v - v*model.c[t])
            model.constrs.add(model.g[t] <= self.E_1[t])
            model.constrs.add(model.c[t] <= model.g[t])
            
            #f_V constraint
            model.constrs.add(model.S1_V[t] == model.b_rt[t] * model.m1_V[t] - model.Q_da[t] * self.P_da[t] - model.m1_V[t] * self.P_rt[t] + self.P_rt[t] * model.Q_da[t])
            model.constrs.add(model.m1_V[t] <= model.u[t])
            model.constrs.add(model.m1_V[t] <= model.Q_c[t])
            model.constrs.add(model.m1_V[t] >= model.u[t] - M_gen * (1 - model.n1_V[t]))
            model.constrs.add(model.m1_V[t] >= model.Q_c[t] - M_gen * model.n1_V[t])
    

            # f_E linearization constraints
            model.constrs.add(model.S1_E[t] == self.P_rt[t] - model.b_da[t])

            model.constrs.add(model.m1_E[t] <= model.Q_da[t])
            model.constrs.add(model.m1_E[t] <= model.q_rt[t])
            model.constrs.add(model.m1_E[t] >= model.Q_da[t] - M_gen * (1 - model.n1_E[t]))
            model.constrs.add(model.m1_E[t] >= model.q_rt[t] - M_gen * model.n1_E[t])
            
            model.constrs.add(model.Q_da[t] - model.Q_c[t] <= M_gen * model.y_E[t])
            model.constrs.add(model.Q_c[t] - model.Q_da[t] <= M_gen * (1-model.y_E[t]))

            model.constrs.add(model.m2_E[t] <= model.u[t])
            model.constrs.add(model.m2_E[t] <= model.Q_da[t])
            model.constrs.add(model.m2_E[t] >= model.u[t] - M_gen * (1 - model.n2_E[t]))
            model.constrs.add(model.m2_E[t] >= model.Q_da[t] - M_gen * model.n2_E[t])

            model.constrs.add(model.m3_E[t] >= model.m2_E[t])
            model.constrs.add(model.m3_E[t] >= model.Q_c[t])
            model.constrs.add(model.m3_E[t] <= model.m2_E[t] + M_gen * (1 - model.n3_E[t]))
            model.constrs.add(model.m3_E[t] <= model.Q_c[t] + M_gen * model.n3_E[t])

            model.constrs.add(model.m4_E[t] <= model.m3_E[t])
            model.constrs.add(model.m4_E[t] <= model.q_rt[t])
            model.constrs.add(model.m4_E[t] >= model.m3_E[t] - M_gen * (1 - model.n4_E[t]))
            model.constrs.add(model.m4_E[t] >= model.q_rt[t] - M_gen * model.n4_E[t])
        
            model.constrs.add(model.m5_E[t] >= model.u[t])
            model.constrs.add(model.m5_E[t] >= model.Q_da[t])
            model.constrs.add(model.m5_E[t] <= model.u[t] + M_gen * (1 - model.n5_E[t]))
            model.constrs.add(model.m5_E[t] <= model.Q_da[t] + M_gen * model.n5_E[t])

            model.constrs.add(model.m6_E[t] <= model.m5_E[t])
            model.constrs.add(model.m6_E[t] <= model.Q_c[t])
            model.constrs.add(model.m6_E[t] <= model.q_rt[t])
            model.constrs.add(model.m6_E[t] >= model.m5_E[t] - M_gen * (1-model.n6_E[t]))
            model.constrs.add(model.m6_E[t] >= model.Q_c[t] - M_gen * (1-model.n7_E[t]))            
            model.constrs.add(model.m6_E[t] >= model.q_rt[t] - M_gen * (1-model.n8_E[t]))    
            model.constrs.add(model.n6_E[t]+model.n7_E[t]+model.n8_E[t]==1)  
            
            model.constrs.add(model.S2_E[t] == model.m1_E[t]-model.y_E[t]*model.m4_E[t]-(1-model.y_E[t])*model.m6_E[t])
                 
            #f_max linearization constraints
            model.constrs.add(model.f_max[t] >= model.S1_V[t])
            model.constrs.add(model.f_max[t] >= model.S1_E[t]*model.S2_E[t]) 
            model.constrs.add(model.f_max[t] >= 0)
            model.constrs.add(model.f_max[t] <= model.S1_V[t] + M_set*(1-model.n1_F[t]))
            model.constrs.add(model.f_max[t] <= model.S1_E[t]*model.S2_E[t] + M_set*(1-model.n2_F[t]))
            model.constrs.add(model.f_max[t] <= M_set*(1-model.n3_F[t]))
            model.constrs.add(model.n1_F[t]+model.n2_F[t]+model.n3_F[t]==1)        
        
            # f_Im linearization constraints
            model.constrs.add(model.S1_Im[t] == (model.u[t] - model.Q_c[t]) - 0.12 * C)

            model.constrs.add(model.m1_Im[t] >= model.S1_Im[t])
            model.constrs.add(model.m1_Im[t] >= 0)
            model.constrs.add(model.m1_Im[t] <= model.S1_Im[t] + M_gen * (1 - model.n1_Im[t]))
            model.constrs.add(model.m1_Im[t] <= M_gen * model.n1_Im[t])

            model.constrs.add(model.m2_Im[t] >= self.P_rt[t] - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] >= - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] <= self.P_rt[t] - model.b_rt[t] + M_price * (1 - model.n2_Im[t]))
            model.constrs.add(model.m2_Im[t] <= - model.b_rt[t] + M_price * model.n2_Im[t])
        
        # General Constraints
        
        model.constrs.add(model.S[0] == self.init_SoC)
        model.constrs.add(model.S[24] == self.init_SoC)
        model.constrs.add(sum(model.q_da[t] for t in range(24)) == E_0_sum)
        model.constrs.add(sum(model.q_rt[t] for t in range(24)) == E_0_sum)  
              
        # Objective Function
            
        model.objective = pyo.Objective(expr=sum(self.P_da[t] * model.Q_da[t] + self.P_rt[t] * (model.u[t] - model.Q_da[t]) + model.f_max[t] + (- model.m1_Im[t] * model.m2_Im[t]) + model.u[t] * P_r for t in model.TIME), sense=pyo.maximize)

    def solve(self):
        self.build_model()
        SOLVER.solve(self)
        print(f"problem{n} solved")
        self.solved = True
        
    def report(self):
        if not self.solved:
            self.solve()
            self.solved = True            
        print(f"\noptimal value = {pyo.value(self.objective)}")

    def optimal_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        for t in range(T):
            self.b_da_values.append(pyo.value(self.b_da[t]))
            self.b_rt_values.append(pyo.value(self.b_rt[t]))
            self.q_da_values.append(pyo.value(self.q_da[t]))
            self.q_rt_values.append(pyo.value(self.q_rt[t]))
            self.u_values.append(pyo.value(self.u[t]))
            self.g_values.append(pyo.value(self.g[t]))
            self.c_values.append(pyo.value(self.c[t]))
            self.d_values.append(pyo.value(self.d[t]))
            self.S_values.append(pyo.value(self.S[t]))     
            self.Q_da_values.append(pyo.value(self.Q_da[t]))
            self.Q_c_values.append(pyo.value(self.Q_c[t]))
            
    def objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        return pyo.value(self.objective)
    
    def solve_with_fixed_vars(self, b_da_values, b_rt_values, q_da_values, q_rt_values, g_values, c_values, d_values, u_values):
        
        self.build_model()   
        model = self.model()
        
        for t in range(T):
            model.b_da[t].fix(b_da_values[t])
            model.b_rt[t].fix(b_rt_values[t])
            model.q_da[t].fix(q_da_values[t])
            model.q_rt[t].fix(q_rt_values[t])
            model.g[t].fix(g_values[t])
            model.c[t].fix(c_values[t])
            model.d[t].fix(d_values[t])
            model.u[t].fix(u_values[t])
                                
        return self.objective_value()

class deterministic_setting_2(pyo.ConcreteModel):
    def __init__ (self, n, init_SoC):
        super().__init__("Deterministic_Setting2")
        
        self.solved = False        
        self.n = n        
        self.scenario = scenarios[n]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]
        self.init_Soc = init_SoC
        
        self.b_da_values = []
        self.b_rt_values = []
        self.q_da_values = []
        self.q_rt_values = []
        self.u_values = []
        self.g_values = []
        self.c_values = []
        self.d_values = []
        self.S_values = []
        self.Q_da_values = []
        self.Q_c_values = []
        
    def build_model(self):
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.ESSTIME = pyo.RangeSet(0, T)
        
        model.b_da = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.b_rt = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.g = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.d = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.ESSTIME, bounds=(S_min, S_max), domain=pyo.Reals) 
        
        model.y_da = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_rt = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_S = pyo.Var(model.TIME, domain=pyo.Binary)
        
        model.Q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        # Linearization Real Variables
        model.m1_V = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m1_E = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m1_Im = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.m2_Im = pyo.Var(model.TIME, domain=pyo.Reals)
        model.S1_V = pyo.Var(model.TIME, domain=pyo.Reals)
        model.S1_E = pyo.Var(model.TIME, domain=pyo.Reals, initialize = 0)
        model.S1_Im = pyo.Var(model.TIME, domain=pyo.Reals)

        # Linearization Binary Variables
        model.n1_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_F = pyo.Var(model.TIME, domain=pyo.Binary)
        
        #부가정산금
        model.f_max = pyo.Var(model.TIME, domain=pyo.Reals)


        #Constraints

        if hasattr(model, 'constrs'):
            model.del_component('constrs')
            model.del_component('constrs_index')
        
        model.constrs = pyo.ConstraintList()
    
        for t in range(T):
            # q_da, q_rt constraint
            model.constrs.add(model.q_da[t] <= 1.1*self.E_0[t] + B)
            model.constrs.add(model.q_rt[t] <= 1.1*self.E_1[t] + B)
            
            # Demand response
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M_price * (1-model.y_da[t]))
            model.constrs.add(self.P_da[t] - model.b_da[t] <= M_price * model.y_da[t])
            model.constrs.add(model.Q_da[t] == model.y_da[t] * model.q_da[t])
        
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M_price * (1-model.y_rt[t]))
            model.constrs.add(self.P_rt[t] - model.b_rt[t] <= M_price * model.y_rt[t])
            model.constrs.add(model.Q_rt[t] == model.y_rt[t]*model.q_rt[t]) 
            
            model.constrs.add(model.Q_c[t] == model.Q_rt[t]*self.U[t])
 
            # b_rt <= b_da
            model.constrs.add(model.b_rt[t] <= model.b_da[t])
            
            # ESS operation
            model.constrs.add(model.S[t+1] == model.S[t] + v*model.c[t] - (model.d[t])/v)
            model.constrs.add(model.u[t] == model.g[t] + (model.d[t])/v - v*model.c[t])
            model.constrs.add(model.g[t] <= self.E_1[t])
            model.constrs.add(model.c[t] <= model.g[t])
            
            #f_V constraint
            model.constrs.add(model.S1_V[t] == model.b_rt[t] * model.u[t] - model.Q_da[t] * self.P_da[t] - model.u[t] * self.P_rt[t] + self.P_rt[t] * model.Q_da[t])
        
            model.constrs.add(model.m1_V[t] >= model.S1_V[t])
            model.constrs.add(model.m1_V[t] >= 0)
            model.constrs.add(model.m1_V[t] <= model.S1_V[t] + M_set * (1 - model.n1_V[t]))
            model.constrs.add(model.m1_V[t] <= M_set * model.n1_V[t])

            # f_E linearization constraints
            model.constrs.add(model.S1_E[t] == self.P_rt[t] - model.b_da[t])

            model.constrs.add(model.m1_E[t] >= (model.Q_da[t] - model.u[t])*model.S1_E[t])
            model.constrs.add(model.m1_E[t] >= 0)
            model.constrs.add(model.m1_E[t] <= (model.Q_da[t] - model.u[t])*model.S1_E[t] + M_set * (1 - model.n1_E[t]))
            model.constrs.add(model.m1_E[t] <= M_set * model.n1_E[t])
            
            # f_max linearization constraints
            model.constrs.add(model.f_max[t] >= model.m1_V[t])
            model.constrs.add(model.f_max[t] >= model.m1_E[t]) 
            model.constrs.add(model.f_max[t] >= 0)
            model.constrs.add(model.f_max[t] <= model.m1_V[t] + M_set*(1-model.n1_F[t]))
            model.constrs.add(model.f_max[t] <= model.m1_E[t] + M_set*(1-model.n2_F[t]))
            model.constrs.add(model.f_max[t] <= M_set*(1-model.n3_F[t]))
            model.constrs.add(model.n1_F[t]+model.n2_F[t]+model.n3_F[t]==1)
        
            # f_Im linearization constraints
            model.constrs.add(model.S1_Im[t] == (model.u[t] - model.Q_c[t]) - 0.12 * C)

            model.constrs.add(model.m1_Im[t] >= model.S1_Im[t])
            model.constrs.add(model.m1_Im[t] >= 0)
            model.constrs.add(model.m1_Im[t] <= model.S1_Im[t] + M_gen * (1 - model.n1_Im[t]))
            model.constrs.add(model.m1_Im[t] <= M_gen * model.n1_Im[t])

            model.constrs.add(model.m2_Im[t] >= self.P_rt[t] - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] >= - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] <= self.P_rt[t] - model.b_rt[t] + M_price * (1 - model.n2_Im[t]))
            model.constrs.add(model.m2_Im[t] <= - model.b_rt[t] + M_price * model.n2_Im[t])

        # General Constraints
        
        model.constrs.add(model.S[0] == self.init_Soc)
        model.constrs.add(model.S[24] == self.init_Soc)
        model.constrs.add(sum(model.q_da[t] for t in range(24)) == E_0_sum)
        model.constrs.add(sum(model.q_rt[t] for t in range(24)) == E_0_sum)

        # Objective Function
        model.objective = pyo.Objective(expr=sum(self.P_da[t] * model.Q_da[t] + self.P_rt[t] * (model.u[t] - model.Q_da[t]) + model.f_max[t] + (- model.m1_Im[t] * model.m2_Im[t]) + model.u[t] * P_r for t in model.TIME), sense=pyo.maximize)
    
    def solve(self):
        self.build_model()
        SOLVER.solve(self)
        print(f"problem{n} solved.")
        self.solved = True
        
    def report(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        print(f"\noptimal value = {pyo.value(self.objective)}")
        

        if not self.solved:
            self.solve()
            self.solved = True
        for t in range(T):
            self.S_values.append(pyo.value(self.S[t]))
        return self.S_values

    def optimal_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        for t in range(T):
            self.b_da_values.append(pyo.value(self.b_da[t]))
            self.b_rt_values.append(pyo.value(self.b_rt[t]))
            self.q_da_values.append(pyo.value(self.q_da[t]))
            self.q_rt_values.append(pyo.value(self.q_rt[t]))
            self.u_values.append(pyo.value(self.u[t]))
            self.g_values.append(pyo.value(self.g[t]))
            self.c_values.append(pyo.value(self.c[t]))
            self.d_values.append(pyo.value(self.d[t]))
            self.S_values.append(pyo.value(self.S[t]))     
            self.Q_da_values.append(pyo.value(self.Q_da[t]))
            self.Q_c_values.append(pyo.value(self.Q_c[t]))
      
    def objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        return pyo.value(self.objective)
 
class deterministic_setting_3(pyo.ConcreteModel):
    def __init__ (self, n, init_SoC):
        super().__init__("Deterministic_Setting1")
        
        self.solved = False        
        self.n = n        
        self.init_SoC = init_SoC
        
        self.scenario = scenarios[n]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]
        
        self.b_da_values = []
        self.b_rt_values = []
        self.q_da_values = []
        self.q_rt_values = []
        self.u_values = []
        self.g_values = []
        self.c_values = []
        self.d_values = []
        self.S_values = []
        self.Q_da_values = []
        self.Q_c_values = []
        
    def build_model(self):
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.ESSTIME = pyo.RangeSet(0, T)
        
        model.b_da = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.b_rt = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.g = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.d = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.TIME, bounds=(0, S+C), domain=pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.ESSTIME, bounds=(S_min, S_max), domain=pyo.Reals) 
        
        model.y_da = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_rt = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_S = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.Q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        # Linearization Real Variables
        model.m1_V = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m2_V = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m1_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m2_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m3_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m4_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m5_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m6_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m1_Im = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m2_Im = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.S1_V = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.S1_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.S2_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.S1_Im = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)

        # Linearization Binary Variables
        model.n1_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n4_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n5_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n6_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n7_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n8_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_F = pyo.Var(model.TIME, domain=pyo.Binary)
        
        #부가정산금
        model.f_max = pyo.Var(model.TIME, domain=pyo.Reals)

        #Constraints

        if hasattr(model, 'constrs'):
            model.del_component('constrs')
            model.del_component('constrs_index')
        
        model.constrs = pyo.ConstraintList()
    
        for t in range(T):
            # q_da, q_rt constraint
            model.constrs.add(model.q_da[t] <= 1.1*self.E_0[t] + B)
            model.constrs.add(model.q_rt[t] <= 1.1*self.E_1[t] + B)
            
            # Demand response
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M_price * (1-model.y_da[t]))
            model.constrs.add(self.P_da[t] - model.b_da[t] <= M_price * model.y_da[t])
            model.constrs.add(model.Q_da[t] == model.y_da[t] * model.q_da[t])
        
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M_price * (1-model.y_rt[t]))
            model.constrs.add(self.P_rt[t] - model.b_rt[t] <= M_price * model.y_rt[t])
            model.constrs.add(model.Q_rt[t] == model.y_rt[t] * model.q_rt[t]) 
            
            model.constrs.add(model.Q_c[t] == model.Q_rt[t]*self.U[t])
 
            # b_rt <= b_da
            model.constrs.add(model.b_rt[t] <= model.b_da[t])
            
            # ESS operation
            model.constrs.add(model.S[t+1] == model.S[t] + v*model.c[t] - (model.d[t])/v)
            model.constrs.add(model.u[t] == model.g[t] + (model.d[t])/v - v*model.c[t])
            model.constrs.add(model.g[t] <= self.E_1[t])
            model.constrs.add(model.c[t] <= model.g[t])
            
            #f_V constraint
            model.constrs.add(model.S1_V[t] == model.b_rt[t] * model.m1_V[t] - model.Q_da[t] * self.P_da[t] - model.m1_V[t] * self.P_rt[t] + self.P_rt[t] * model.Q_da[t])
            model.constrs.add(model.m1_V[t] <= model.u[t])
            model.constrs.add(model.m1_V[t] <= model.Q_c[t])
            model.constrs.add(model.m1_V[t] >= model.u[t] - M_gen * (1 - model.n1_V[t]))
            model.constrs.add(model.m1_V[t] >= model.Q_c[t] - M_gen * model.n1_V[t])
    

            # f_E linearization constraints
            model.constrs.add(model.S1_E[t] == self.P_rt[t] - model.b_da[t])
            
            model.constrs.add(model.Q_da[t] - model.Q_c[t] <= M_gen * model.y_E[t])
            model.constrs.add(model.Q_c[t] - model.Q_da[t] <= M_gen * (1-model.y_E[t]))

            model.constrs.add(model.m2_E[t] <= model.u[t])
            model.constrs.add(model.m2_E[t] <= model.Q_da[t])
            model.constrs.add(model.m2_E[t] >= model.u[t] - M_gen * (1 - model.n2_E[t]))
            model.constrs.add(model.m2_E[t] >= model.Q_da[t] - M_gen * model.n2_E[t])

            model.constrs.add(model.m3_E[t] >= model.m2_E[t])
            model.constrs.add(model.m3_E[t] >= model.Q_c[t])
            model.constrs.add(model.m3_E[t] <= model.m2_E[t] + M_gen * (1 - model.n3_E[t]))
            model.constrs.add(model.m3_E[t] <= model.Q_c[t] + M_gen * model.n3_E[t])
        
            model.constrs.add(model.m5_E[t] >= model.u[t])
            model.constrs.add(model.m5_E[t] >= model.Q_da[t])
            model.constrs.add(model.m5_E[t] <= model.u[t] + M_gen * (1 - model.n5_E[t]))
            model.constrs.add(model.m5_E[t] <= model.Q_da[t] + M_gen * model.n5_E[t])

            model.constrs.add(model.m6_E[t] <= model.m5_E[t])
            model.constrs.add(model.m6_E[t] <= model.Q_c[t])
            model.constrs.add(model.m6_E[t] >= model.m5_E[t] - M_gen * (1-model.n6_E[t]))
            model.constrs.add(model.m6_E[t] >= model.Q_c[t] - M_gen * model.n6_E[t])              
            
            model.constrs.add(model.S2_E[t] == model.Q_da[t]-model.y_E[t]*model.m3_E[t]-(1-model.y_E[t])*model.m6_E[t])
                 
            #f_max linearization constraints
            model.constrs.add(model.f_max[t] >= model.S1_V[t])
            model.constrs.add(model.f_max[t] >= model.S1_E[t]*model.S2_E[t]) 
            model.constrs.add(model.f_max[t] >= 0)
            model.constrs.add(model.f_max[t] <= model.S1_V[t] + M_set*(1-model.n1_F[t]))
            model.constrs.add(model.f_max[t] <= model.S1_E[t]*model.S2_E[t] + M_set*(1-model.n2_F[t]))
            model.constrs.add(model.f_max[t] <= M_set*(1-model.n3_F[t]))
            model.constrs.add(model.n1_F[t]+model.n2_F[t]+model.n3_F[t]==1)        
        
            # f_Im linearization constraints
            model.constrs.add(model.S1_Im[t] == (model.u[t] - model.Q_c[t]) - 0.12 * C)

            model.constrs.add(model.m1_Im[t] >= model.S1_Im[t])
            model.constrs.add(model.m1_Im[t] >= 0)
            model.constrs.add(model.m1_Im[t] <= model.S1_Im[t] + M_gen * (1 - model.n1_Im[t]))
            model.constrs.add(model.m1_Im[t] <= M_gen * model.n1_Im[t])

            model.constrs.add(model.m2_Im[t] >= self.P_rt[t] - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] >= - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] <= self.P_rt[t] - model.b_rt[t] + M_price * (1 - model.n2_Im[t]))
            model.constrs.add(model.m2_Im[t] <= - model.b_rt[t] + M_price * model.n2_Im[t])
        
        # General Constraints
        
        model.constrs.add(model.S[0] == self.init_SoC)
        model.constrs.add(model.S[24] == self.init_SoC)
        model.constrs.add(sum(model.q_da[t] for t in range(24)) == E_0_sum)
        model.constrs.add(sum(model.q_rt[t] for t in range(24)) == E_0_sum)  
              
        # Objective Function
            
        model.objective = pyo.Objective(expr=sum(self.P_da[t] * model.Q_da[t] + self.P_rt[t] * (model.u[t] - model.Q_da[t]) + model.f_max[t] + (- model.m1_Im[t] * model.m2_Im[t]) + model.u[t] * P_r for t in model.TIME), sense=pyo.maximize)

    def solve(self):
        self.build_model()
        SOLVER.solve(self)
        print(f"problem{n} solved")
        self.solved = True
        
    def report(self):
        if not self.solved:
            self.solve()
            self.solved = True            
        print(f"\noptimal value = {pyo.value(self.objective)}")

    def optimal_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        for t in range(T):
            self.b_da_values.append(pyo.value(self.b_da[t]))
            self.b_rt_values.append(pyo.value(self.b_rt[t]))
            self.q_da_values.append(pyo.value(self.q_da[t]))
            self.q_rt_values.append(pyo.value(self.q_rt[t]))
            self.u_values.append(pyo.value(self.u[t]))
            self.g_values.append(pyo.value(self.g[t]))
            self.c_values.append(pyo.value(self.c[t]))
            self.d_values.append(pyo.value(self.d[t]))
            self.S_values.append(pyo.value(self.S[t]))     
            self.Q_da_values.append(pyo.value(self.Q_da[t]))
            self.Q_c_values.append(pyo.value(self.Q_c[t]))
            
    def objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        return pyo.value(self.objective)
    
    def solve_with_fixed_vars(self, b_da_values, b_rt_values, q_da_values, q_rt_values, g_values, c_values, d_values, u_values):
        
        self.build_model()   
        model = self.model()
        
        for t in range(T):
            model.b_da[t].fix(b_da_values[t])
            model.b_rt[t].fix(b_rt_values[t])
            model.q_da[t].fix(q_da_values[t])
            model.q_rt[t].fix(q_rt_values[t])
            model.g[t].fix(g_values[t])
            model.c[t].fix(c_values[t])
            model.d[t].fix(d_values[t])
            model.u[t].fix(u_values[t])
                                
        return self.objective_value()

class deterministic_setting_4(pyo.ConcreteModel):
    def __init__ (self, n, init_SoC):
        super().__init__("Deterministic_Setting1")
        
        self.solved = False        
        self.n = n        
        self.init_SoC = init_SoC
        
        self.scenario = scenarios[n]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]
        
        self.b_da_values = []
        self.b_rt_values = []
        self.q_da_values = []
        self.q_rt_values = []
        self.u_values = []
        self.g_values = []
        self.c_values = []
        self.d_values = []
        self.S_values = []
        self.Q_da_values = []
        self.Q_c_values = []
        
    def build_model(self):
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.ESSTIME = pyo.RangeSet(0, T)
        
        model.b_da = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.b_rt = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.g = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.d = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.TIME, bounds=(0, S+C), domain=pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.ESSTIME, bounds=(S_min, S_max), domain=pyo.Reals) 
        
        model.y_da = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_rt = pyo.Var(model.TIME, domain=pyo.Binary)
        model.Q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        # Linearization Real Variables
        model.m1_V = pyo.Var(model.TIME, bounds=(-M_gen, M_gen),domain=pyo.Reals)
        model.m1_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m4_E = pyo.Var(model.TIME, domain=pyo.Reals)
        model.m1_Im = pyo.Var(model.TIME, bounds=(-M_gen, M_gen), domain=pyo.Reals)
        model.m2_Im = pyo.Var(model.TIME, bounds=(-M_price, M_price), domain=pyo.Reals)
        model.S1_V = pyo.Var(model.TIME, domain=pyo.Reals)
        model.S1_E = pyo.Var(model.TIME, bounds=(-M_price, M_price), domain=pyo.Reals)
        model.S2_E = pyo.Var(model.TIME, bounds=(-M_gen, M_gen), domain=pyo.Reals)
        model.S1_Im = pyo.Var(model.TIME, domain=pyo.Reals)

        # Linearization Binary Variables
        model.n1_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n4_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_F = pyo.Var(model.TIME, domain=pyo.Binary)
        
        #부가정산금
        model.f_max = pyo.Var(model.TIME, domain=pyo.Reals)

        #Constraints

        if hasattr(model, 'constrs'):
            model.del_component('constrs')
            model.del_component('constrs_index')
        
        model.constrs = pyo.ConstraintList()
    
        for t in range(T):
            # q_da, q_rt constraint
            model.constrs.add(model.q_da[t] <= 1.1*self.E_0[t] + B)
            model.constrs.add(model.q_rt[t] <= 1.1*self.E_1[t] + B)
            
            # Demand response
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M_price * (1-model.y_da[t]))
            model.constrs.add(self.P_da[t] - model.b_da[t] <= M_price * model.y_da[t])
            model.constrs.add(model.Q_da[t] == model.y_da[t] * model.q_da[t])
        
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M_price * (1-model.y_rt[t]))
            model.constrs.add(self.P_rt[t] - model.b_rt[t] <= M_price * model.y_rt[t])
            model.constrs.add(model.Q_rt[t] == model.y_rt[t] * model.q_rt[t]) 
            
            model.constrs.add(model.Q_c[t] == model.Q_rt[t]*self.U[t])
 
            # b_rt <= b_da
            model.constrs.add(model.b_rt[t] <= model.b_da[t])
            
            # ESS operation
            model.constrs.add(model.S[t+1] == model.S[t] + v*model.c[t] - (model.d[t])/v)
            model.constrs.add(model.u[t] == model.g[t] + (model.d[t])/v - v*model.c[t])
            model.constrs.add(model.g[t] <= self.E_1[t])
            model.constrs.add(model.c[t] <= model.g[t])
            
            #f_V constraint
            model.constrs.add(model.S1_V[t] == model.b_rt[t] * model.m1_V[t] - model.Q_da[t] * self.P_da[t] - model.m1_V[t] * self.P_rt[t] + self.P_rt[t] * model.Q_da[t])
            model.constrs.add(model.m1_V[t] <= model.u[t])
            model.constrs.add(model.m1_V[t] <= model.Q_c[t])
            model.constrs.add(model.m1_V[t] >= model.u[t] - M_gen * (1 - model.n1_V[t]))
            model.constrs.add(model.m1_V[t] >= model.Q_c[t] - M_gen * model.n1_V[t])
    

            # f_E linearization constraints
            model.constrs.add(model.S1_E[t] == self.P_rt[t] - model.b_da[t])

            model.constrs.add(model.m1_E[t] <= model.Q_da[t])
            model.constrs.add(model.m1_E[t] <= model.q_rt[t])
            model.constrs.add(model.m1_E[t] >= model.Q_da[t] - M_gen * (1 - model.n1_E[t]))
            model.constrs.add(model.m1_E[t] >= model.q_rt[t] - M_gen * model.n1_E[t])

            model.constrs.add(model.m4_E[t] <= model.u[t])
            model.constrs.add(model.m4_E[t] <= model.q_rt[t])
            model.constrs.add(model.m4_E[t] >= model.u[t] - M_gen * (1 - model.n4_E[t]))
            model.constrs.add(model.m4_E[t] >= model.q_rt[t] - M_gen * model.n4_E[t])
            
            model.constrs.add(model.S2_E[t] == model.m1_E[t]-model.m4_E[t])
                 
            #f_max linearization constraints
            model.constrs.add(model.f_max[t] >= model.S1_V[t])
            model.constrs.add(model.f_max[t] >= model.S1_E[t]*model.S2_E[t]) 
            model.constrs.add(model.f_max[t] >= 0)
            model.constrs.add(model.f_max[t] <= model.S1_V[t] + M_set*(1-model.n1_F[t]))
            model.constrs.add(model.f_max[t] <= model.S1_E[t]*model.S2_E[t] + M_set*(1-model.n2_F[t]))
            model.constrs.add(model.f_max[t] <= M_set*(1-model.n3_F[t]))
            model.constrs.add(model.n1_F[t]+model.n2_F[t]+model.n3_F[t]==1)        
        
            # f_Im linearization constraints
            model.constrs.add(model.S1_Im[t] == (model.u[t] - model.Q_c[t]) - 0.12 * C)

            model.constrs.add(model.m1_Im[t] >= model.S1_Im[t])
            model.constrs.add(model.m1_Im[t] >= 0)
            model.constrs.add(model.m1_Im[t] <= model.S1_Im[t] + M_gen * (1 - model.n1_Im[t]))
            model.constrs.add(model.m1_Im[t] <= M_gen * model.n1_Im[t])

            model.constrs.add(model.m2_Im[t] >= self.P_rt[t] - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] >= - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] <= self.P_rt[t] - model.b_rt[t] + M_price * (1 - model.n2_Im[t]))
            model.constrs.add(model.m2_Im[t] <= - model.b_rt[t] + M_price * model.n2_Im[t])
        
        # General Constraints
        
        model.constrs.add(model.S[0] == self.init_SoC)
        model.constrs.add(model.S[24] == self.init_SoC)
        model.constrs.add(sum(model.q_da[t] for t in range(24)) == E_0_sum)
        model.constrs.add(sum(model.q_rt[t] for t in range(24)) == E_0_sum)  
              
        # Objective Function
            
        model.objective = pyo.Objective(expr=sum(self.P_da[t] * model.Q_da[t] + self.P_rt[t] * (model.u[t] - model.Q_da[t]) + model.f_max[t] + (- model.m1_Im[t] * model.m2_Im[t]) + model.u[t] * P_r for t in model.TIME), sense=pyo.maximize)

    def solve(self):
        self.build_model()
        SOLVER.solve(self)
        print(f"problem{n} solved")
        self.solved = True
        
    def report(self):
        if not self.solved:
            self.solve()
            self.solved = True            
        print(f"\noptimal value = {pyo.value(self.objective)}")

    def optimal_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        for t in range(T):
            self.b_da_values.append(pyo.value(self.b_da[t]))
            self.b_rt_values.append(pyo.value(self.b_rt[t]))
            self.q_da_values.append(pyo.value(self.q_da[t]))
            self.q_rt_values.append(pyo.value(self.q_rt[t]))
            self.u_values.append(pyo.value(self.u[t]))
            self.g_values.append(pyo.value(self.g[t]))
            self.c_values.append(pyo.value(self.c[t]))
            self.d_values.append(pyo.value(self.d[t]))
            self.S_values.append(pyo.value(self.S[t]))     
            self.Q_da_values.append(pyo.value(self.Q_da[t]))
            self.Q_c_values.append(pyo.value(self.Q_c[t]))
            
    def objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        return pyo.value(self.objective)
    
    def solve_with_fixed_vars(self, b_da_values, b_rt_values, q_da_values, q_rt_values, g_values, c_values, d_values, u_values):
        
        self.build_model()   
        model = self.model()
        
        for t in range(T):
            model.b_da[t].fix(b_da_values[t])
            model.b_rt[t].fix(b_rt_values[t])
            model.q_da[t].fix(q_da_values[t])
            model.q_rt[t].fix(q_rt_values[t])
            model.g[t].fix(g_values[t])
            model.c[t].fix(c_values[t])
            model.d[t].fix(d_values[t])
            model.u[t].fix(u_values[t])
                                
        return self.objective_value()

class deterministic_setting_5(pyo.ConcreteModel):
    def __init__ (self, n, init_SoC):
        super().__init__("Deterministic_Setting1")
        
        self.solved = False        
        self.n = n        
        self.init_SoC = init_SoC
        
        self.scenario = scenarios[n]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]
        
        self.b_da_values = []
        self.b_rt_values = []
        self.q_da_values = []
        self.q_rt_values = []
        self.u_values = []
        self.g_values = []
        self.c_values = []
        self.d_values = []
        self.S_values = []
        self.Q_da_values = []
        self.Q_c_values = []
        
    def build_model(self):
        model = self.model()
        
        model.TIME = pyo.RangeSet(0, T-1)
        
        model.ESSTIME = pyo.RangeSet(0, T)
        
        model.b_da = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.b_rt = pyo.Var(model.TIME, bounds=(-P_r, 0), domain=pyo.Reals)
        model.q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.g = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.c = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.d = pyo.Var(model.TIME, bounds=(0, B), domain=pyo.NonNegativeReals)
        model.u = pyo.Var(model.TIME, bounds=(0, S+C), domain=pyo.NonNegativeReals)
        
        model.S = pyo.Var(model.ESSTIME, bounds=(S_min, S_max), domain=pyo.Reals) 
        
        model.y_da = pyo.Var(model.TIME, domain=pyo.Binary)
        model.y_rt = pyo.Var(model.TIME, domain=pyo.Binary)
        model.Q_da = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_rt = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        model.Q_c = pyo.Var(model.TIME, domain=pyo.NonNegativeReals)
        
        # Linearization Real Variables
        model.m1_V = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m1_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m4_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m1_Im = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.m2_Im = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.S1_V = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.S1_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.S2_E = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)
        model.S1_Im = pyo.Var(model.TIME, bounds=(-M_set, M_set), domain=pyo.Reals)

        # Linearization Binary Variables
        model.n1_V = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n4_E = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_Im = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n1_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n2_F = pyo.Var(model.TIME, domain=pyo.Binary)
        model.n3_F = pyo.Var(model.TIME, domain=pyo.Binary)
        
        #부가정산금
        model.f_max = pyo.Var(model.TIME, domain=pyo.Reals)

        #Constraints

        if hasattr(model, 'constrs'):
            model.del_component('constrs')
            model.del_component('constrs_index')
        
        model.constrs = pyo.ConstraintList()
    
        for t in range(T):
            # q_da, q_rt constraint
            model.constrs.add(model.q_da[t] <= 1.1*self.E_0[t] + B)
            model.constrs.add(model.q_rt[t] <= 1.1*self.E_1[t] + B)
            
            # Demand response
            model.constrs.add(model.b_da[t] - self.P_da[t] <= M_price * (1-model.y_da[t]))
            model.constrs.add(self.P_da[t] - model.b_da[t] <= M_price * model.y_da[t])
            model.constrs.add(model.Q_da[t] == model.y_da[t] * model.q_da[t])
        
            model.constrs.add(model.b_rt[t] - self.P_rt[t] <= M_price * (1-model.y_rt[t]))
            model.constrs.add(self.P_rt[t] - model.b_rt[t] <= M_price * model.y_rt[t])
            model.constrs.add(model.Q_rt[t] == model.y_rt[t] * model.q_rt[t]) 
            
            model.constrs.add(model.Q_c[t] == model.Q_rt[t]*self.U[t])
 
            # b_rt <= b_da
            model.constrs.add(model.b_rt[t] <= model.b_da[t])
            
            # ESS operation
            model.constrs.add(model.S[t+1] == model.S[t] + v*model.c[t] - (model.d[t])/v)
            model.constrs.add(model.u[t] == model.g[t] + (model.d[t])/v - v*model.c[t])
            model.constrs.add(model.g[t] <= self.E_1[t])
            model.constrs.add(model.c[t] <= model.g[t])
            
            #f_V constraint
            model.constrs.add(model.S1_V[t] == model.b_rt[t] * model.u[t] - model.Q_da[t] * self.P_da[t] - model.u[t] * self.P_rt[t] + self.P_rt[t] * model.Q_da[t])
    

            # f_E linearization constraints
            model.constrs.add(model.S1_E[t] == self.P_rt[t] - model.b_da[t])

            model.constrs.add(model.m1_E[t] <= model.Q_da[t])
            model.constrs.add(model.m1_E[t] <= model.q_rt[t])
            model.constrs.add(model.m1_E[t] >= model.Q_da[t] - M_gen * (1 - model.n1_E[t]))
            model.constrs.add(model.m1_E[t] >= model.q_rt[t] - M_gen * model.n1_E[t])

            model.constrs.add(model.m4_E[t] <= model.u[t])
            model.constrs.add(model.m4_E[t] <= model.q_rt[t])
            model.constrs.add(model.m4_E[t] >= model.u[t] - M_gen * (1 - model.n4_E[t]))
            model.constrs.add(model.m4_E[t] >= model.q_rt[t] - M_gen * model.n4_E[t])
            
            model.constrs.add(model.S2_E[t] == model.m1_E[t]-model.m4_E[t])
                 
            #f_max linearization constraints
            model.constrs.add(model.f_max[t] >= model.S1_V[t])
            model.constrs.add(model.f_max[t] >= model.S1_E[t]*model.S2_E[t]) 
            model.constrs.add(model.f_max[t] >= 0)
            model.constrs.add(model.f_max[t] <= model.S1_V[t] + M_set*(1-model.n1_F[t]))
            model.constrs.add(model.f_max[t] <= model.S1_E[t]*model.S2_E[t] + M_set*(1-model.n2_F[t]))
            model.constrs.add(model.f_max[t] <= M_set*(1-model.n3_F[t]))
            model.constrs.add(model.n1_F[t]+model.n2_F[t]+model.n3_F[t]==1)        
        
            # f_Im linearization constraints
            model.constrs.add(model.S1_Im[t] == (model.u[t] - model.Q_c[t]) - 0.12 * C)

            model.constrs.add(model.m1_Im[t] >= model.S1_Im[t])
            model.constrs.add(model.m1_Im[t] >= 0)
            model.constrs.add(model.m1_Im[t] <= model.S1_Im[t] + M_gen * (1 - model.n1_Im[t]))
            model.constrs.add(model.m1_Im[t] <= M_gen * model.n1_Im[t])

            model.constrs.add(model.m2_Im[t] >= self.P_rt[t] - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] >= - model.b_rt[t])
            model.constrs.add(model.m2_Im[t] <= self.P_rt[t] - model.b_rt[t] + M_price * (1 - model.n2_Im[t]))
            model.constrs.add(model.m2_Im[t] <= - model.b_rt[t] + M_price * model.n2_Im[t])
        
        # General Constraints
        
        model.constrs.add(model.S[0] == self.init_SoC)
        model.constrs.add(model.S[24] == self.init_SoC)
        model.constrs.add(sum(model.q_da[t] for t in range(24)) == E_0_sum)
        model.constrs.add(sum(model.q_rt[t] for t in range(24)) == E_0_sum)  
              
        # Objective Function
            
        model.objective = pyo.Objective(expr=sum(self.P_da[t] * model.Q_da[t] + self.P_rt[t] * (model.u[t] - model.Q_da[t]) + model.f_max[t] + (- model.m1_Im[t] * model.m2_Im[t]) + model.u[t] * P_r for t in model.TIME), sense=pyo.maximize)

    def solve(self):
        self.build_model()
        SOLVER.solve(self)
        print(f"problem{n} solved")
        self.solved = True
        
    def report(self):
        if not self.solved:
            self.solve()
            self.solved = True            
        print(f"\noptimal value = {pyo.value(self.objective)}")

    def optimal_solutions(self):
        if not self.solved:
            self.solve()
            self.solved = True
        for t in range(T):
            self.b_da_values.append(pyo.value(self.b_da[t]))
            self.b_rt_values.append(pyo.value(self.b_rt[t]))
            self.q_da_values.append(pyo.value(self.q_da[t]))
            self.q_rt_values.append(pyo.value(self.q_rt[t]))
            self.u_values.append(pyo.value(self.u[t]))
            self.g_values.append(pyo.value(self.g[t]))
            self.c_values.append(pyo.value(self.c[t]))
            self.d_values.append(pyo.value(self.d[t]))
            self.S_values.append(pyo.value(self.S[t]))     
            self.Q_da_values.append(pyo.value(self.Q_da[t]))
            self.Q_c_values.append(pyo.value(self.Q_c[t]))
            
    def objective_value(self):
        if not self.solved:
            self.solve()
            self.solved = True
            
        return pyo.value(self.objective)
    
    def solve_with_fixed_vars(self, b_da_values, b_rt_values, q_da_values, q_rt_values, g_values, c_values, d_values, u_values):
        
        self.build_model()   
        model = self.model()
        
        for t in range(T):
            model.b_da[t].fix(b_da_values[t])
            model.b_rt[t].fix(b_rt_values[t])
            model.q_da[t].fix(q_da_values[t])
            model.q_rt[t].fix(q_rt_values[t])
            model.g[t].fix(g_values[t])
            model.c[t].fix(c_values[t])
            model.d[t].fix(d_values[t])
            model.u[t].fix(u_values[t])
                                
        return self.objective_value()
    
class original_obj_fcn():
    def __init__ (self, n):
        
        self.solved = False        
        self.n = n        
        self.scenario = scenarios[n]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]

        self.b_da_values = b_da_list[n-k]
        self.b_rt_values = b_rt_list[n-k]
        self.q_da_values = q_da_list[n-k]
        self.q_rt_values = q_rt_list[n-k]
        self.u_values = u_list[n-k]   
        self.Q_da_values = Q_da_list[n-k]
        self.Q_c_values = Q_c_list[n-k]
        
        self.f_P = 0
        self.f_V = []
        self.f_E = []
        self.f_max = 0
        self.f_Im = 0
        self.f_REC = 0
        
        self.obj_fcn = 0
    
    def original_obj(self):  
        for t in range(T):
            self.f_P += self.P_da[t]*self.Q_da_values[t]+(self.u_values[t]-self.Q_da_values[t])*self.P_rt[t]
            
            self.f_V.append(min(self.u_values[t], self.Q_c_values[t])*self.b_rt_values[t]-(self.Q_da_values[t]*self.P_da[t]+(min(self.u_values[t], self.Q_c_values[t])-self.Q_da_values[t])*self.P_rt[t])) 
            
            if self.Q_da_values[t] >= self.Q_c_values[t]:
                self.f_E.append((self.P_rt[t]-self.b_da_values[t])*(min(self.Q_da_values[t], self.q_rt_values[t])-min(max(min(self.u_values[t], self.Q_da_values[t]), self.Q_c_values[t]), self.q_rt_values[t])))
            else:
                self.f_E.append((self.P_rt[t]-self.b_da_values[t])*(min(self.Q_da_values[t], self.q_rt_values[t])-min(max(self.u_values[t], self.Q_da_values[t]), self.Q_c_values[t], self.q_rt_values[t])))
            
            self.f_max += max(0, self.f_V[t], self.f_E[t])
            
            if self.P_da[t] >= 0:
                self.f_Im += -(self.P_rt[t]-self.b_rt_values[t])*max(self.u_values[t]-self.Q_c_values[t]-0.12*C, 0)
            else:
                self.f_Im += -(-self.b_rt_values[t])*max(self.u_values[t]-self.Q_c_values[t]-0.12*C, 0)
            
            self.f_REC += P_r*self.u_values[t]
        
        self.obj_fcn += self.f_P + self.f_max + self.f_Im + self.f_REC        
        
        return self.obj_fcn
    
class original_obj_fcn_prime():
    def __init__ (self, n):
        
        self.solved = False        
        self.n = n        
        self.scenario = scenarios[n-k]        
        self.P_da = self.scenario[0]        
        self.P_rt = self.scenario[1]        
        self.E_0 = self.scenario[2]        
        self.E_1 = self.scenario[3]        
        self.U = self.scenario[4]

        self.b_da_values = b_da_list[n-k]
        self.b_rt_values = b_rt_list[n-k]
        self.q_da_values = q_da_list[n-k]
        self.q_rt_values = q_rt_list[n-k]
        self.u_values = u_list[n-k]   
        self.Q_da_values = Q_da_list[n-k]
        self.Q_c_values = Q_c_list[n-k]
        
        self.f_P = 0
        self.f_V = []
        self.f_E = []
        self.f_max = 0
        self.f_Im = 0
        self.f_REC = 0
        
        self.obj_fcn = 0
    
    def original_obj(self):  
        for t in range(T):
            self.f_P += self.P_da[t]*self.Q_da_values[t]+(self.u_values[t]-self.Q_da_values[t])*self.P_rt[t]
            
            self.f_V.append(min(self.u_values[t], self.Q_c_values[t])*self.b_rt_values[t]-(self.Q_da_values[t]*self.P_da[t]+(min(self.u_values[t], self.Q_c_values[t])-self.Q_da_values[t])*self.P_rt[t])) 
            
            if self.Q_da_values[t] >= self.Q_c_values[t]:
                self.f_E.append((self.P_rt[t]-self.b_da_values[t])*(self.Q_da_values[t]-max(min(self.u_values[t], self.Q_da_values[t]), self.Q_c_values[t])))
            else:
                self.f_E.append((self.P_rt[t]-self.b_da_values[t])*(self.Q_da_values[t]-min(max(self.u_values[t], self.Q_da_values[t]), self.Q_c_values[t])))
            
            self.f_max += max(0, self.f_V[t], self.f_E[t])
            
            if self.P_da[t] >= 0:
                self.f_Im += -(self.P_rt[t]-self.b_rt_values[t])*max(self.u_values[t]-self.Q_c_values[t]-0.12*C, 0)
            else:
                self.f_Im += -(-self.b_rt_values[t])*max(self.u_values[t]-self.Q_c_values[t]-0.12*C, 0)
            
            self.f_REC += P_r*self.u_values[t]
        
        self.obj_fcn += self.f_P + self.f_max + self.f_Im + self.f_REC        
        
        return self.obj_fcn

## Results

k=0
r=range(k, k+25)

# 모의시장 range(93)
# 실제시장 range(93, 185)
#ranged_r = range(len(scenarios)-123)

Tr = range(T)

## Optimal Solutions of Deterministic Setting 2

b_da_list = []
b_rt_list = []
q_da_list = []
q_rt_list = []
u_list = []
S_list = []
Q_da_list = []
Q_c_list = []
a = 0
b = 0
c = 0

## Det2' optimal solutions

for n in r:
    det = deterministic_setting_2(n, 0.5*S)
    det.solve()
    det.optimal_solutions()
    b_da_list.append(det.b_da_values)
    b_rt_list.append(det.b_rt_values)
    q_da_list.append(det.q_da_values)
    q_rt_list.append(det.q_rt_values)
    u_list.append(det.u_values)
    S_list.append(det.S_values)
    Q_da_list.append(det.Q_da_values)
    Q_c_list.append(det.Q_c_values)
    

for n in r:
    plt.plot(Tr, b_da_list[n-k])
    
plt.xlabel('Time')
plt.ylabel('b_da values')
plt.title('b_da')
plt.ylim(-110, 20)
plt.legend()
plt.show()

for n in r:
    plt.plot(Tr, b_rt_list[n-k])
    
plt.xlabel('Time')
plt.ylabel('b_rt values')
plt.title('b_rt')
plt.ylim(-110, 20)
plt.legend()
plt.show()

for n in r:
    plt.plot(Tr, q_da_list[n-k])
    
plt.xlabel('Time')
plt.ylabel('q_da values')
plt.title('q_da')
plt.ylim(0, 2000)
plt.legend()
plt.show()

for n in r:
    plt.plot(Tr, q_rt_list[n-k])
    
plt.xlabel('Time')
plt.ylabel('q_rt values')
plt.title('q_rt')
plt.ylim(0, 2000)
plt.legend()
plt.show()

for n in r:
    plt.plot(Tr, S_list[n-k])
    
plt.xlabel('Time')
plt.ylabel('S values')
plt.title('S')
plt.ylim(0, 2000)
plt.legend()
plt.show()

for n in r:
    plt.plot(Tr, Q_da_list[n-k])
    
plt.xlabel('Time')
plt.ylabel('Q_da values')
plt.title('Q_da')
plt.ylim(0, 2000)
plt.legend()
plt.show()

for n in r:
    plt.plot(Tr, Q_c_list[n-k])
    
plt.xlabel('Time')
plt.ylabel('Q_c values')
plt.title('Q_c')
plt.ylim(0, 2000)
plt.legend()
plt.show()

for n in r:
    plt.plot(Tr, u_list[n-k])
    
plt.xlabel('Time')
plt.ylabel('u values')
plt.title('u')
plt.ylim(0, 2000)
plt.legend()
plt.show()


## Price Analysis
"""
avg_day_ahead_prices=[]
avg_real_time_prices=[]

for t in Tr:
    sum_day_ahead_prices=0
    sum_real_time_prices=0
    for n in r:
        sum_day_ahead_prices+=day_ahead_prices[n][t]
        sum_real_time_prices+=real_time_prices[n][t]
    avg_day_ahead_prices.append(sum_day_ahead_prices/93)
    avg_real_time_prices.append(sum_real_time_prices/93)
    
for n in r:
    plt.plot(Tr, day_ahead_prices[n])
plt.xlabel('Time')
plt.ylabel('P_da values')
plt.title('P_da')
plt.ylim(-130, 300)
plt.legend()
plt.show()

for n in r:
    plt.plot(Tr, real_time_prices[n])
plt.xlabel('Time')
plt.ylabel('P_rt values')
plt.title('P_rt')
plt.ylim(-130, 300)
plt.legend()
plt.show()

plt.plot(Tr, avg_day_ahead_prices)
plt.xlabel('Time')
plt.ylabel('avg_P_da values')
plt.title('avg_P_da')
plt.ylim(-130, 300)
plt.legend()
plt.show()

plt.plot(Tr, avg_real_time_prices)
plt.xlabel('Time')
plt.ylabel('avg_P_rt values')
plt.title('avg_P_rt')
plt.ylim(-130, 300)
plt.legend()
plt.show()
"""

## Optimal Value Comparison 


d1_obj = []
d2_obj = []
difference = []

"""
for n in r:
    d1 = deterministic_setting_1(n, 0.5*S)
    d1_o = d1.objective_value()
    d2_o = original_obj_fcn(n).original_obj()
    #d2 = deterministic_setting_2(n, 0.5*S)
    d1_obj.append(d1_o)
    d2_obj.append(d2_o)
    difference.append(abs(d1_o-d2_o))

plt.plot(r, d1_obj, label='Original')
plt.plot(r, d2_obj, label='Approximation')
plt.plot(r, difference, label='Abs difference')

plt.xlabel('Scenario Index')
plt.ylabel('Values')
plt.title('Comparison of Deterministic Settings')

plt.ylim(0, 2000000)
plt.legend()
plt.show()
"""

# Optimal Initial SoC value

"""
d2_1_obj = []
d2_2_obj = []
d2_3_obj = []
d2_4_obj = []
d2_5_obj = []

for n in r:
    d2_1 = deterministic_setting_2_prime(n, 0.1*S)
    d2_2 = deterministic_setting_2_prime(n, 0.3*S)
    d2_3 = deterministic_setting_2_prime(n, 0.5*S)
    d2_4 = deterministic_setting_2_prime(n, 0.7*S)
    d2_5 = deterministic_setting_2_prime(n, 0.9*S)
    
    d2_1_obj.append(d2_1.objective_value())
    d2_2_obj.append(d2_2.objective_value())
    d2_3_obj.append(d2_3.objective_value())
    d2_4_obj.append(d2_4.objective_value())
    d2_5_obj.append(d2_5.objective_value())

plt.plot(r, d2_1_obj, label='0.1S')
plt.plot(r, d2_2_obj, label='0.3S')
plt.plot(r, d2_3_obj, label='0.5S')
plt.plot(r, d2_4_obj, label='0.7S')
plt.plot(r, d2_5_obj, label='0.9S')

def mean(a):
    return sum(a)/len(a)

print(mean(d2_1_obj), mean(d2_2_obj), mean(d2_3_obj), mean(d2_4_obj), mean(d2_5_obj))

plt.xlabel('Scenario Index')
plt.ylabel('Values')
plt.title('Comparison')
plt.ylim(0, 1400000)
plt.legend()
plt.show()


"""
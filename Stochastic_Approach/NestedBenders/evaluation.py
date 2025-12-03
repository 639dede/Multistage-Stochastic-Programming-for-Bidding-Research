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


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# 1. evaluation plots

## ---- Define K values ----
K = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]

## ---- Data ----
gap_1_SB = [12.66, 10.16, 11.77, 11.49, 11.68, 10.48, 11.67, 13.36, 12.94, 13.32]
iter_1_SB = [1493, 1395, 1138, 1012, 916, 884, 766, 755, 607, 535]
eval_1_SB = [8305, 8320, 8402, 8450, 8465, 8475, 8466, 8435, 8402, 8527]

gap_1_L = [2.12, 0.86, 2.12, 1.86, 4.85, 13.65, 9.63, 11.20, 14.70, 18.74]
iter_1_L = [656, 210, 154, 129, 109, 95, 87, 79, 67, 51]
eval_1_L = [8153, 8209, 8256, 8284, 8321, 7982, 8127, 8242, 7919, 8034]

gap_2_SB = [26.51, 30.44, 30.51, 29.18, 31.05, 27.37, 29.31, 28.89, 29.95, 32.79]
iter_2_SB = [1624, 1753, 1296, 1165, 1075, 1092, 906, 823, 793, 594]
eval_2_SB = [7634, 7937, 7916, 7921, 7994, 8007, 7958, 8020, 8074, 8173]

gap_2_L = [1.41, 0.97, 3.82, 3.63, 2.69, 3.30, 8.40, 6.33, 7.47, 16.00]
iter_2_L = [344, 254, 155, 137, 100, 88, 79, 73, 65, 47]
eval_2_L = [7740, 8204, 8188, 8100, 8037, 8150, 7941, 8080, 8091, 7801]


def _plot_series(index, K_vals, series_lists, title, ylab):
    # positions 1..n for even spacing
    x_pos = list(range(1, len(K_vals) + 1)) 
    plt.figure(figsize=(8, 4))
    plt.plot(x_pos, series_lists[0], color='blue',  linewidth=2, label='SB')
    plt.plot(x_pos, series_lists[1], color='gold',  linewidth=2, label='L')
    plt.title(f"{title}", fontsize=22)
    plt.xlabel("K", fontsize=22)
    plt.ylabel(ylab, fontsize=22)
    plt.xticks(x_pos, K_vals, fontsize=18)   # <-- X tick font size
    plt.yticks(fontsize=18) 
    plt.legend(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_gap(index, K_vals, gap_lists):
    _plot_series(index, K_vals, gap_lists, "UB-LB Gap", "Gap (%)")

def plot_iterations(index, K_vals, iter_lists):
    _plot_series(index, K_vals, iter_lists, "Iterations", "Iterations")

def plot_evaluations(index, K_vals, eval_lists):
    _plot_series(index, K_vals, eval_lists, "Evaluation", "Evaluation (kKRW)")



# 2. Solution plots

price_setting = 'sunny'    # 'cloudy', 'normal', 'sunny'

def load_csv(name, subdir):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sol_dir = os.path.join(current_dir, subdir)
    file_path = os.path.join(sol_dir, f"{name}.csv")
    return pd.read_csv(file_path)

def plot_DA_solutions():
    name_grid = [
        ["b_da_1", "q_da_1", "f_P_1"],
        ["b_da_2", "q_da_2", "f_P_2"],
        ["b_da_3", "q_da_3", "f_P_3"],
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)

    for i, row in enumerate(name_grid):
        for j, name in enumerate(row):
            ax = axes[i, j]
            df = load_csv(name, f"DA_solutions_{price_setting}")
            values = df.to_numpy().squeeze()
            hours = range(1, len(values) + 1)

            ax.plot(hours, values, marker="o", label=name)

            ax.set_title(name)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

            if name.startswith("b_da"):
                ax.set_ylim(-100, 20)
            elif name.startswith("q_da"):
                ax.set_ylim(0, 32000)

            ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Hour")
    axes[-1, 1].set_xlabel("Hour")
    axes[-1, 2].set_xlabel("Hour")

    fig.suptitle(f"Day-Ahead Solutions {price_setting}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_ID_solutions():
    name_grid = [
        ["f_P_1", "f_E_1", "f_Im_1"],
        ["f_P_2", "f_E_2", "f_Im_2"],
        ["f_P_3", "f_E_3", "f_Im_3"],
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)

    for i, row in enumerate(name_grid):
        for j, name in enumerate(row):
            ax = axes[i, j]
            df = load_csv(name, f"ID_solutions_{price_setting}")
            values = df.to_numpy().squeeze()
            hours = range(1, len(values) + 1)

            ax.plot(hours, values, marker="o", label=name)

            ax.set_title(name)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            if name.startswith("f_P") or name.startswith("f_E"):
                ax.set_ylim(-1e4, 3e4)
            elif name.startswith("f_Im"):
                ax.set_ylim(-1e4, 3e4)


    axes[-1, 0].set_xlabel("Hour")
    axes[-1, 1].set_xlabel("Hour")
    axes[-1, 2].set_xlabel("Hour")

    fig.suptitle(f"Intraday Solutions {price_setting}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
# ---- Main ----
if __name__ == "__main__":
    
    
    # 1. Evaluation plots 
    
    #plot_gap(1, K, [gap_1_SB, iter_1_L and gap_1_L])  # keep data order SB, L, Hyb
    #plot_iterations(1, K, [iter_1_SB, iter_1_L])
    #plot_evaluations(1, K, [eval_1_SB, eval_1_L])

    #plot_gap(2, K, [gap_2_SB, gap_2_L])
    #plot_iterations(2, K, [iter_2_SB, iter_2_L])
    #plot_evaluations(2, K, [eval_2_SB, eval_2_L])
    
    """
    # Trimmed K (example with K[:-1])
    K_trim = K[:-1]
    plot_gap(1, K_trim, [gap_1_SB[:-1], gap_1_L[:-1]])
    plot_iterations(1, K_trim, [iter_1_SB[:-1], iter_1_L[:-1]])
    plot_evaluations(1, K_trim, [eval_1_SB[:-1], eval_1_L[:-1]])

    plot_gap(2, K_trim, [gap_2_SB[:-1], gap_2_L[:-1]])
    plot_iterations(2, K_trim, [iter_2_SB[:-1], iter_2_L[:-1]])
    plot_evaluations(2, K_trim, [eval_2_SB[:-1], eval_2_L[:-1]])
    """
    
    # 2. Solution plots
    
    plot_DA_solutions()
    plot_ID_solutions()
    
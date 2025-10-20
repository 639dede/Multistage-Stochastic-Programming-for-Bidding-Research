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


# ---- Define K values ----
K = [1, 2, 3, 4, 5, 6, 8, 10, 15]

# ---- Data ----
gap_1_SB = [15.31, 14.88, 14.94, 13.30, 15.88, 16.52, 16.07, 14.84, 16.62]
iter_1_SB = [1947, 1356, 1100, 1075, 875, 770, 723, 601, 530]
eval_1_SB = [8284, 8223, 8206, 8390, 8393, 8393, 8332, 8486, 8385]

gap_1_L = [0.74, 2.97, 2.29, 7.05, 6.20, 1.84, 12.26, 10.34, 17.03]
iter_1_L = [586, 166, 164, 131, 110, 102, 81, 70, 51]
eval_1_L = [8257, 8165, 8336, 8134, 8235, 8242, 8144, 8273, 7995]

gap_2_SB = [35.69, 31.99, 35.5, 36.49, 35.72, 28.54, 37.44, 32.58, 31.76]
iter_2_SB = [1943, 1413, 1347, 1301, 1021, 944, 781, 786, 571]
eval_2_SB = [8167, 7900, 8174, 8094, 8066, 8035, 8174, 7956, 8100]

gap_2_L = [2.05, 1.54, 0.78, 3.00, 4.91, 5.00, 11.25, 7.37, 16.21]
iter_2_L = [686, 279, 150, 132, 111, 94, 74, 63, 46]
eval_2_L = [8254, 8180, 8352, 8356, 8410, 8196, 8087, 8057, 7725]


# ---- Helper: plot with even spacing but K labels ----
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


# ---- Main ----
if __name__ == "__main__":
    # Full K
    plot_gap(1, K, [gap_1_SB, iter_1_L and gap_1_L])  # keep data order SB, L, Hyb
    plot_iterations(1, K, [iter_1_SB, iter_1_L])
    plot_evaluations(1, K, [eval_1_SB, eval_1_L])

    plot_gap(2, K, [gap_2_SB, gap_2_L])
    plot_iterations(2, K, [iter_2_SB, iter_2_L])
    plot_evaluations(2, K, [eval_2_SB, eval_2_L])

    # Trimmed K (example with K[:-1])
    K_trim = K[:-1]
    plot_gap(1, K_trim, [gap_1_SB[:-1], gap_1_L[:-1]])
    plot_iterations(1, K_trim, [iter_1_SB[:-1], iter_1_L[:-1]])
    plot_evaluations(1, K_trim, [eval_1_SB[:-1], eval_1_L[:-1]])

    plot_gap(2, K_trim, [gap_2_SB[:-1], gap_2_L[:-1]])
    plot_iterations(2, K_trim, [iter_2_SB[:-1], iter_2_L[:-1]])
    plot_evaluations(2, K_trim, [eval_2_SB[:-1], eval_2_L[:-1]])

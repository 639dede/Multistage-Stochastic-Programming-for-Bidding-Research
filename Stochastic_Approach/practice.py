import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import bernoulli, truncnorm
from sklearn.cluster import KMeans

omit_dates = [pd.Timestamp("2024-12-19"), pd.Timestamp("2025-01-11")]


date_range = pd.date_range(start='2024-07-12', periods=5808, freq='H')
date_range = date_range[~date_range.normalize().isin(omit_dates)]

daily_dates = date_range.normalize().drop_duplicates()

target_date = list(daily_dates)[216]
print(target_date)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import bernoulli, truncnorm
from sklearn.cluster import KMeans


import pandas as pd

file_path_Energy = '.\Stochastic_Approach\Scenarios\Energy.csv'

df = pd.read_csv(file_path_Energy)

hourly_rows = []

# Loop through the DataFrame in chunks of 4 rows
for i in range(0, len(df), 4):
    chunk = df.iloc[i : i+4]
    
    # Optionally skip incomplete chunks
    # if len(chunk) < 4:
    #     continue

    new_row = {}
    new_row["timestamp"] = chunk["timestamp"].iloc[0]  # Use the first timestamp in the chunk
    
    # Sum the columns
    new_row["gen"] = chunk["gen"].sum()
    new_row["forecast_da"] = chunk["forecast_da"].sum()
    new_row["forecast_rt"] = chunk["forecast_rt"].sum()
    new_row["capacity"] = chunk["capacity"].sum()  # Now summing the capacity

    hourly_rows.append(new_row)

df_hourly = pd.DataFrame(hourly_rows)
df_hourly.to_csv("hourly_data.csv", index=False)

print("Done! Hourly data saved to hourly_data.csv")
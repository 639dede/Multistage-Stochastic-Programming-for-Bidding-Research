import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


data = pd.read_csv(".\Autoregressive\jeju_filtered_data.csv", index_col=0, parse_dates=True)

data['timestamp'] = pd.to_datetime(data['timestamp']).dt.date

data.set_index('timestamp')

m_t = data['forecast_rt']/data['forecast_da']

reshaped_array = m_t.values.reshape(230, 11)

# AR(1) Test

non_AR_1=[]

non_AR_num = 0

ar1_coefficients = []

all_residuals = []

for i in range(len(reshaped_array)):
    ts = reshaped_array[i, :]

    print(ts)
    
    # Fit an AR(1) model to the first time series
    model = sm.tsa.ARIMA(ts, order=(1, 0, 0)).fit()
    
    # Get the AR(1) coefficients
    ar1_coefficient = model.arparams[0]
    ar1_coefficients.append(ar1_coefficient)

    # Get the residuals
    residuals = model.resid
    all_residuals.extend(residuals) 
    
    # Perform the Ljung-Box test (lags=11 for 11 residual autocorrelations)
    ljung_box_result = acorr_ljungbox(residuals, lags=[10], return_df=True)

    p_value = ljung_box_result['lb_pvalue'].iloc[0]

    if p_value < 0.05:
        non_AR_1.append(i)
        non_AR_num+=1

average_ar1 = np.mean(ar1_coefficients)

print(all_residuals)

all_residuals = np.array(all_residuals)

variance_white_noise = np.var(all_residuals)

print(non_AR_num)
print(average_ar1)
print(variance_white_noise)
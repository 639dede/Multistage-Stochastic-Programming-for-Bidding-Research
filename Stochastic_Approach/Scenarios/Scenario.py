import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt
import os
import warnings
from scipy.stats import bernoulli, truncnorm
from sklearn.cluster import KMeans


# Given Parameters

P_r = 80
P_max = 300

T = 24

# CSV files & filtering

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)  # read first few KBs
    result = chardet.detect(rawdata)
    return result['encoding']

## Energy forecast file

file_path_Energy = './Stochastic_Approach/Scenarios/Energy_forecast.csv'

df = pd.read_csv(file_path_Energy)

invalid_rows = (df['forecast_da'] <= 0) | (df['forecast_rt'] <= 0)

df.loc[invalid_rows, 'forecast_da'] = 0.01
df.loc[invalid_rows, 'forecast_rt'] = 0

filtered_df_Energy = df

filtered_df_Energy_for_dist = df[(df['forecast_da'] > 0) & (df['forecast_rt'] > 0)]

timestamps_pre = pd.date_range(start="2024-03-01 00:00", end="2025-03-10 23:00", freq='H')

omit_dates = [pd.Timestamp("2024-12-19"), pd.Timestamp("2025-01-11"), pd.Timestamp("2025-02-15")]

mask = ~timestamps_pre.normalize().isin(omit_dates)

timestamps = timestamps_pre[mask]

# E_0 processing

E_0_values = df['forecast_da'].tolist()
forecast_data_0 = pd.DataFrame({'timestamp': timestamps, 'value': E_0_values})
forecast_data_0['hour'] = forecast_data_0['timestamp'].dt.hour
forecast_data_0['date'] = forecast_data_0['timestamp'].dt.date

E_0_daily = forecast_data_0.groupby('date')['value'].apply(list).tolist()
E_0 = forecast_data_0.groupby('hour')['value'].mean().tolist()

# E_1 processing

E_1_values = df['forecast_rt'].tolist()
forecast_data_1 = pd.DataFrame({'timestamp': timestamps, 'value': E_1_values})
forecast_data_1['hour'] = forecast_data_1['timestamp'].dt.hour
forecast_data_1['date'] = forecast_data_1['timestamp'].dt.date

E_1_daily = forecast_data_1.groupby('date')['value'].apply(list).tolist()
E_1 = forecast_data_1.groupby('hour')['value'].mean().tolist()

## Day-Ahead price

directory_path_da = './Stochastic_Approach/Scenarios/모의 실시간시장 가격/하루전'

files = os.listdir(directory_path_da)
csv_files = sorted([file for file in files if file.endswith('일간.csv')])

def process_da_file(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    data = df.loc[3:27, df.columns[2]]  
    return data.tolist()

day_ahead_prices_daily = []


for csv_file in csv_files:
    file_path = os.path.join(directory_path_da, csv_file)
    processed_data = process_da_file(file_path)
    day_ahead_prices_daily.append(processed_data)

day_ahead_prices = [item for sublist in day_ahead_prices_daily for item in sublist]


## Real-Time price

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)  # read first few KBs
    result = chardet.detect(rawdata)
    return result['encoding']

directory_path_rt = './Stochastic_Approach/Scenarios/모의 실시간시장 가격/실시간 임시'

files_rt = os.listdir(directory_path_rt)

csv_files_rt = sorted([file for file in files_rt if file.endswith('.csv')])

def process_rt_file(file_path):
    
    encoding = detect_encoding(file_path)

    df = pd.read_csv(file_path, encoding=encoding)
    data = df.iloc[3:99, 2]  
    reshaped_data = data.values.reshape(-1, 4).mean(axis=1)
    return reshaped_data

real_time_prices_daily = []

for xlsx_file in csv_files_rt:
    file_path = os.path.join(directory_path_rt, xlsx_file)
    processed_data = process_rt_file(file_path)
    real_time_prices_daily.append(processed_data)

real_time_prices = [item for sublist in real_time_prices_daily for item in sublist]

print(len(E_0_values), len(day_ahead_prices), len(real_time_prices))

E_0_values = np.array(E_0_values)
day_ahead_prices = day_ahead_prices
real_time_prices = real_time_prices


## Dataframe for TGMM

date_range_pre = pd.date_range(start='2024-03-01', periods=9000, freq='H')

omit_dates = [pd.Timestamp("2024-12-19"), pd.Timestamp("2025-01-11"), pd.Timestamp("2025-02-15")]

date_range = date_range_pre[~date_range_pre.normalize().isin(omit_dates)]

print(len(date_range))

df = pd.DataFrame({
    'date': date_range,
    'day_ahead_price': day_ahead_prices,
    'real_time_price': real_time_prices,
    'E_0_value': E_0_values
})

date_only = df["date"].dt.normalize()
neg_counts_by_day = df["day_ahead_price"].lt(0).groupby(date_only).sum()

eligible_days = neg_counts_by_day[neg_counts_by_day >= 2].index

data = df[date_only.isin(eligible_days)].reset_index(drop=True)

pd.set_option('display.max_rows', None)   # Show all rows
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.width', None)       # Don't wrap columns
pd.set_option('display.max_colwidth', None) # Don't truncate column text


# Delta_E distribution

filtered_df_Energy['delta'] = (
    filtered_df_Energy['forecast_rt'] 
    / filtered_df_Energy['forecast_da']
)

delta_values = filtered_df_Energy['delta'].tolist()

delta_E_daily = [
    delta_values[i : i + 24]
    for i in range(0, len(delta_values), 24)
]
assert all(len(day) == 24 for day in delta_E_daily), "Some daily_delta_E lists are not length 24."

filtered_df_Energy_for_dist['delta'] = filtered_df_Energy_for_dist['forecast_rt'] / filtered_df_Energy_for_dist['forecast_da']

delta_values_for_dist = filtered_df_Energy_for_dist['delta']

std_E = delta_values_for_dist.std()  

lower_E, upper_E = 0.8, 1.2

a_E, b_E = (lower_E - 1) / std_E, (upper_E - 1) / std_E

Energy_dist = truncnorm(a_E, b_E, loc=1, scale=std_E)


# Q_c distribution

std_c = 0.5

lower_c, upper_c = -1, 1

a_c, b_c = (lower_c) / std_c, (upper_c) / std_c

Q_c_truncnorm_dist = truncnorm(a_c, b_c, loc = 0, scale = std_c)

p = 0.05

def f_X(x):
    if x == 0:
        return 0.95  
    elif lower_c <= x <= upper_c:
        return p * Q_c_truncnorm_dist.pdf(x)
    else:
        return 0 

Q_c_x_values = np.linspace(-1.5, 1.5, 500)  
Q_c_f_X_values = np.array([f_X(x) for x in Q_c_x_values])


# Price Distributions

LB_price = -P_r
UB_price = P_max
LB_energy = -540
UB_energy = 18000
num_bins = 8

bin_edges_energy = np.linspace(LB_energy, UB_energy, num_bins + 1)
bin_edges_price = np.linspace(LB_price, UB_price, num_bins)

def merge_bins(data, conditioning_var, bin_edges, min_data_per_bin=10, tol=1e-9):

    df_tmp = data[[conditioning_var]].copy()

    edges = list(map(float, bin_edges))

    def _find_left_idx(edges_list, interval_left):
        arr = np.asarray(edges_list, dtype=float)
        hit = np.where(np.isclose(arr, float(interval_left), atol=tol, rtol=0))[0]
        if len(hit) == 0:
            return int(np.argmin(np.abs(arr - float(interval_left))))
        return int(hit[0])

    while True:
        binned = pd.cut(df_tmp[conditioning_var], bins=edges, include_lowest=True)
        counts = binned.value_counts().sort_index()

        too_small = counts[counts < min_data_per_bin]
        if too_small.empty:
            break

        interval = too_small.index[0]  
        i_left = _find_left_idx(edges, interval.left)    
        i_bin = i_left  

        left_neighbor = counts.iloc[i_bin - 1] if i_bin - 1 >= 0 else np.inf
        right_neighbor = counts.iloc[i_bin + 1] if i_bin + 1 < len(counts) else np.inf

        if left_neighbor <= right_neighbor and i_left > 0:
            edges.pop(i_left)
        else:
            if i_left + 1 < len(edges):
                edges.pop(i_left + 1)
            else:
                edges.pop(i_left)

    return np.array(edges, dtype=float)

bin_edges_energy = merge_bins(data, 'E_0_value', bin_edges_energy, min_data_per_bin=3)
bin_edges_price = merge_bins(data, 'day_ahead_price', bin_edges_price, min_data_per_bin=3)

def truncated_gaussian_pdf(x, mean, std, lower, upper):

    a, b = (lower - mean) / std, (upper - mean) / std
    pdf = truncnorm.pdf(x, a, b, loc=mean, scale=std)
    pdf = np.where(np.isfinite(pdf), pdf, 0)
    return pdf

def initialize_parameters(K, data):

    kmeans = KMeans(n_clusters=K, random_state=42).fit(data.reshape(-1, 1))
    means = kmeans.cluster_centers_.flatten()
    stds = np.std(data) * np.ones(K)
    weights = np.ones(K) / K
    return weights, means, stds

def e_step(data, weights, means, stds, lower, upper):

    K = len(weights)
    responsibilities = np.zeros((len(data), K))
    
    for k in range(K):
        responsibilities[:, k] = weights[k] * truncated_gaussian_pdf(data, means[k], stds[k], lower, upper)
    
    responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
    responsibilities_sum[responsibilities_sum == 0] = 1e-10
    
    responsibilities /= responsibilities_sum
    return responsibilities

def m_step(data, responsibilities, lower, upper, min_std=1e-3):

    K = responsibilities.shape[1]
    Nk = responsibilities.sum(axis=0)
    weights = Nk / len(data)
    means = np.zeros(K)
    stds = np.zeros(K)
    
    for k in range(K):
        if Nk[k] == 0:
            means[k] = np.random.choice(data)
            stds[k] = np.std(data)
            weights[k] = 1e-6  
            continue
        
        means[k] = np.sum(responsibilities[:, k] * data) / Nk[k]
        
        variance = np.sum(responsibilities[:, k] * (data - means[k])**2) / Nk[k]
        variance = max(variance, min_std**2)
        stds[k] = np.sqrt(variance)
    
    weights /= weights.sum()
    
    return weights, means, stds

def compute_log_likelihood(data, weights, means, stds, lower, upper, epsilon=1e-10):

    K = len(weights)
    likelihood = np.zeros((len(data), K))
    
    for k in range(K):
        likelihood[:, k] = weights[k] * truncated_gaussian_pdf(data, means[k], stds[k], lower, upper)
    
    total_likelihood = np.sum(likelihood, axis=1)
    total_likelihood = np.maximum(total_likelihood, epsilon)
    log_likelihood = np.sum(np.log(total_likelihood))
    return log_likelihood

def em_algorithm(data, K, lower, upper, max_iters=100, tol=1e-4, min_std=1e-3):

    weights, means, stds = initialize_parameters(K, data)
    log_likelihood_old = None
    
    for iteration in range(max_iters):
        try:
            # E-Step
            responsibilities = e_step(data, weights, means, stds, lower, upper)
            
            # M-Step
            weights, means, stds = m_step(data, responsibilities, lower, upper, min_std=min_std)
            
            log_likelihood = compute_log_likelihood(data, weights, means, stds, lower, upper)
            
            if log_likelihood_old is not None:
                if np.abs(log_likelihood - log_likelihood_old) < tol:
                    break
            log_likelihood_old = log_likelihood
            

        except Exception as e:
            print(f'Error at iteration {iteration}: {e}')
            break
    
    return weights, means, stds, responsibilities

def plot_tgmm(data, weights, means, stds, lower, upper, title='TGMM Fit'):

    x = np.linspace(lower, upper, 1000)
    pdf = np.zeros_like(x)
    
    for w, m, s in zip(weights, means, stds):
        pdf += w * truncated_gaussian_pdf(x, m, s, lower, upper)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Data Histogram')
    plt.plot(x, pdf, 'k', linewidth=2, label='TGMM Fit')
    plt.title(title)
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def sample_from_tgmm(weights, means, stds, lower, upper, num_samples=1):

    components = np.random.choice(len(weights), size=num_samples, p=weights)
    samples = []
    for comp in components:
        a, b = (lower - means[comp]) / stds[comp], (upper - means[comp]) / stds[comp]
        sample = truncnorm.rvs(a, b, loc=means[comp], scale=stds[comp])
        samples.append(sample)
    return np.array(samples)


def inspect_bins(data, conditioning_var, target_var, num_bins):

    data['bin'] = pd.qcut(data[conditioning_var], q=num_bins, duplicates='drop')
    for bin_interval in data['bin'].unique():
        subset = data[data['bin'] == bin_interval][target_var]
        count = len(subset)
        min_val = subset.min()
        max_val = subset.max()
    data.drop('bin', axis=1, inplace=True)

def plot_histogram(data, target_var, LB, UB):

    plt.figure(figsize=(10, 6))
    plt.hist(data[target_var], bins=100, range=(LB, UB), density=True, alpha=0.6, color='g')
    plt.title(f'Histogram of {target_var}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

def train_conditional_tgmm(data, conditioning_var, target_var, bin_edges, K=5, min_data_per_bin=30):

    data['bin'] = pd.cut(data[conditioning_var], bins=bin_edges, include_lowest=True)
    
    unique_bins = data['bin'].dropna().unique()
    
    tgmm_params = {}
    
    for bin_interval in unique_bins:
        subset = data[data['bin'] == bin_interval][target_var].values
        count = len(subset)
        if count < min_data_per_bin:
            continue
        try:
            weights, means, stds, responsibilities = em_algorithm(subset, K, LB_price, UB_price)
            tgmm_params[bin_interval] = {
                'weights': weights,
                'means': means,
                'stds': stds
            }
            #plot_tgmm(subset, weights, means, stds, LB_price, UB_price, title=f'TGMM Fit for {conditioning_var} in {bin_interval}')
        except Exception as e:
            print(f'Failed to train TGMM for bin {bin_interval}: {e}')
    
    data.drop('bin', axis=1, inplace=True)
    return tgmm_params, bin_edges


inspect_bins(data, 'E_0_value', 'day_ahead_price', num_bins)
plot_histogram(data, 'day_ahead_price', LB_price, UB_price)

inspect_bins(data, 'day_ahead_price', 'real_time_price', num_bins)
plot_histogram(data, 'real_time_price', LB_price, UB_price)

tgmm_model1_params, model1_bin_edges = train_conditional_tgmm(
    data=data,
    conditioning_var='E_0_value',
    target_var='day_ahead_price',
    bin_edges=bin_edges_energy,  
    K=5,
    min_data_per_bin=30
)

tgmm_model2_params, model2_bin_edges = train_conditional_tgmm(
    data=data,
    conditioning_var='day_ahead_price',
    target_var='real_time_price',
    bin_edges=bin_edges_price,  
    K=5,
    min_data_per_bin=30
)


def find_bin(value, bin_edges, tgmm_params):

    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= value <= bin_edges[i + 1]:
            bin_interval = pd.Interval(left=bin_edges[i], right=bin_edges[i + 1], closed='right')
            if bin_interval in tgmm_params:
                return bin_interval
            else:
                break  
    closest_bin = min(tgmm_params.keys(), key=lambda x: min(abs(value - x.left), abs(value - x.right)))
    return closest_bin


def sample_day_ahead_price(E0_value, tgmm_params, bin_edges, num_samples=1):

    bin_interval = find_bin(E0_value, bin_edges, tgmm_params)
    if bin_interval is None:
        raise ValueError("E0_value could not be assigned to any bin.")
    params = tgmm_params.get(bin_interval)
    if params is None:
        bin_interval = find_bin(E0_value, bin_edges, tgmm_params)
        params = tgmm_params.get(bin_interval)
        if params is None:
            raise ValueError(f"No TGMM parameters found for the bin: {bin_interval}.")
    samples = sample_from_tgmm(params['weights'], params['means'], params['stds'], LB_price, UB_price, num_samples)
    return samples
"""
new_E0_value = 2000 

try:
    sampled_day_ahead = sample_day_ahead_price(new_E0_value, tgmm_model1_params, model1_bin_edges, num_samples=100)
    print(f'\nSampled day_ahead_price values for E0_value={new_E0_value}:')
    print(sampled_day_ahead)
except ValueError as e:
    print(e)
"""

def sample_real_time_price(P_da_value, tgmm_params, bin_edges, num_samples=1):

    bin_interval = find_bin(P_da_value, bin_edges, tgmm_params)
    if bin_interval is None:
        raise ValueError("E0_value could not be assigned to any bin.")
    params = tgmm_params.get(bin_interval)
    if params is None:
        bin_interval = find_bin(P_da_value, bin_edges, tgmm_params)
        params = tgmm_params.get(bin_interval)
        if params is None:
            raise ValueError(f"No TGMM parameters found for the bin: {bin_interval}.")
    samples = sample_from_tgmm(params['weights'], params['means'], params['stds'], LB_price, UB_price, num_samples)
    return samples

"""

try:
    sampled_real_time = sample_real_time_price(new_E0_value, tgmm_model2_params, model2_bin_edges, num_samples=100)
    print(f'\nSampled real_time_price values for day_ahead_price={new_E0_value}:')
    print(sampled_real_time)
except ValueError as e:
    print(e)

"""


E_0_daily = E_0_daily[61:92]

P_da_daily = day_ahead_prices_daily[61:92]
P_rt_daily = real_time_prices_daily[61:92]


# Generate Scenario

class scenario():
    
    def __init__(self, N_t, E_0):
        
        self.E_0 = E_0
        self.T = 24
        self.N_t = N_t
        
        self.P_da = []
        self.delta_E = []
        self.Q_c = []
        self.P_rt = []
        self.delta = []

    def sample_multiple_P_da(self, n):
        
        P_da = [[] for i in range(n)]
        
        for i in range(n):
            for E_0_value in self.E_0:
                sampled_day_ahead = sample_day_ahead_price(E_0_value, tgmm_model1_params, model1_bin_edges, num_samples=1)
                P_da[i].append(sampled_day_ahead[0])
        
        return P_da
    
    def sample_delta_E(self):
        
        delta_samples = Energy_dist.rvs(self.T)
        return delta_samples.tolist()
    
    def sample_Q_c(self):
        
        Q_c_samples = []
        for _ in range(self.T):
            if np.random.rand() < 0.95:
                Q_c_samples.append(0)
            else:
                Q_c_sample = Q_c_truncnorm_dist.rvs()
                Q_c_samples.append(Q_c_sample)
        return Q_c_samples
    
    def sample_P_rt(self, P_da):
        
        P_rt_samples = []
        for P in P_da:
            sampled_rt = sample_real_time_price(
                P, tgmm_model2_params, model2_bin_edges, num_samples=1
            )
            P_rt_samples.append(sampled_rt[0])
        return P_rt_samples    
    
    def sample_single_delta(self, t, P_da):
        
        ## sample one delta_E
        delta = []
        
        delta_E = Energy_dist.rvs(1).tolist()[0]
        
        ## sample one delta_Q_c
        if np.random.rand() < 0.95:
                delta_Q_c = 0
        else:
                delta_Q_c = Q_c_truncnorm_dist.rvs()    
        
        ## sample one P_rt depending on P_da
        try:
            P_rt = sample_real_time_price(
                P_da, tgmm_model2_params, model2_bin_edges, num_samples=1
            )
        except ValueError as e:
            print(f"Error sampling P_rt for t={t}: {e}")        
        
        ## sample one delta depending on t = -1, ..., 23
        if t == 23:
            delta = [0, P_rt[0], delta_Q_c]

        else:
            delta = [delta_E, P_rt[0], delta_Q_c]
        
        return delta
    
    def sample_multiple_delta(self, t, N_t, P_da):
        
        ## sample one delta_E
        deltas = [] 
        
        for j in range(N_t):
            
            delta = []
            
            delta_E = Energy_dist.rvs(1).tolist()[0]
            
            ## sample one delta_Q_c
            if np.random.rand() < 0.95:
                    delta_Q_c = 0
            else:
                    delta_Q_c = Q_c_truncnorm_dist.rvs()    
            
            ## sample one P_rt depending on E_0
            try:
                P_rt = sample_real_time_price(
                    P_da, tgmm_model2_params, model2_bin_edges, num_samples=1
                )
            except ValueError as e:
                print(f"Error sampling P_rt for t={t}: {e}")        
            
            ## sample one delta depending on t = -1, ..., 23
            if t == 23:
                delta = [0, P_rt[0], delta_Q_c]

            else:
                delta = [delta_E, P_rt[0], delta_Q_c]
            
            deltas.append(delta)
        
        return deltas
    
    def generate_scenario_tree(self, P_da):
        
        scenario = []
            
        for t in range(self.T): 
            
            branches = np.array(self.sample_multiple_delta(t, 100, P_da[t]))
            kmeans = KMeans(n_clusters=self.N_t, random_state=0, n_init='auto')
            kmeans.fit(branches)
            
            reduced_branches = kmeans.cluster_centers_.tolist()
            scenario.append(reduced_branches)
            
        return scenario


if __name__ == '__main__':
    
    # Save Energy forecast csv file
    
    base = np.asarray(E_0, dtype=float)  # from your earlier calculation
    hours = np.arange(24)

    def remap_daylight_window(base, start_shift=0, end_shift=0, scale=1.0,
                            min_width=4, cut_frac=0.01):

        y = np.asarray(base, dtype=float)
        # detect original daylight window (ignore tiny twilight)
        cut = max(cut_frac * y.max(), 1e-9)
        nz = np.where(y > cut)[0]
        if nz.size == 0:
            
            # fallback if base is all zeros: assume 6..18 with a hump
            x = np.linspace(-1, 1, 24)
            y = (np.maximum(0, 1 - x**2)) * y.max()
            nz = np.where(y > cut)[0]

        s0, e0 = int(nz[0]), int(nz[-1])             # original sunrise/sunset
        s1 = int(np.clip(s0 + start_shift, 0, 23))   # new sunrise
        e1 = int(np.clip(e0 + end_shift,   0, 23))   # new sunset

        # keep a reasonable width
        if e1 <= s1:
            e1 = min(23, s1 + min_width)

        # original daylight segment and normalized positions
        seg0 = y[s0:e0+1]
        t0 = np.linspace(0, 1, len(seg0))

        # target positions in the new window
        seg_len = e1 - s1 + 1
        t1 = np.linspace(0, 1, seg_len)
        seg1 = np.interp(t1, t0, seg0) * scale

        out = np.zeros_like(y)
        out[s1:e1+1] = seg1
        return out

    # --- Make the three profiles ---
    E_0_cloudy = remap_daylight_window(base, start_shift=+2, end_shift=-2, scale=0.5)  # later start, earlier end
    E_0_normal = remap_daylight_window(base, start_shift= +1, end_shift= 0, scale=0.9)  # baseline
    E_0_sunny  = remap_daylight_window(base, start_shift=-1, end_shift=+1, scale=1.5)  # earlier start, later end

    # Save CSVs (optional)
    outdir = './Stochastic_Approach/Scenarios/Energy_forecast'
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(E_0_cloudy).to_csv(os.path.join(outdir, 'E_0_cloudy.csv'), index=False, header=False)
    pd.DataFrame(E_0_normal).to_csv(os.path.join(outdir, 'E_0_normal.csv'), index=False, header=False)
    pd.DataFrame(E_0_sunny ).to_csv(os.path.join(outdir, 'E_0_sunny.csv' ), index=False, header=False)

    # Plot (0, 20000) as requested
    def plot_E_0(ax, y, title):
        ax.plot(hours, y, linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel('Hour')
        ax.set_ylabel('E₀')
        ax.set_ylim(0, 20000)
        ax.grid(True)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    plot_E_0(axs[0], E_0_cloudy, "Cloudy")
    plot_E_0(axs[1], E_0_normal, "Normal")
    plot_E_0(axs[2], E_0_sunny,  "Sunny")
    plt.tight_layout()
    plt.savefig('./Stochastic_Approach/Scenarios/E_0_profiles_cloudy_normal_sunny.png', dpi=300)
    plt.close()
    
    
    E_0 = E_0_normal
    
    pd.DataFrame(E_0).to_csv("./Stochastic_Approach/Scenarios/Energy_forecast/E_0.csv", index=False)


    # Save Reduced Day Ahead price and Reduced Scenario Tree csv files
    
    scenario_generator = scenario(6, E_0)
    
    evaluation_num = 500
    
    sampled_P_da = np.array(scenario_generator.sample_multiple_P_da(evaluation_num))
    
    K_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, evaluation_num]
    
    Reduced_P_da = []
    Reduced_scenario_trees = []
    
    for k in K_list[:-1]:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(sampled_P_da)
        cluster_centers = kmeans.cluster_centers_  # shape: (k, 24)
        Reduced_P_da.append(cluster_centers)
    
    Reduced_P_da.append(sampled_P_da)
    
    for P_da_list in Reduced_P_da:
        scenario_trees = []
        for P_da in P_da_list:
            scenario_tree = scenario_generator.generate_scenario_tree(P_da)
            scenario_trees.append(scenario_tree)
        Reduced_scenario_trees.append(scenario_trees)
    
    
    base_dir = './Stochastic_Approach/Scenarios/Clustered_scenario_trees'
    os.makedirs(base_dir, exist_ok=True)

    for i, (k, scenario_trees) in enumerate(zip(K_list, Reduced_scenario_trees)):
        k_dir = os.path.join(base_dir, f'K{k}')
        os.makedirs(k_dir, exist_ok=True)

        for j, tree in enumerate(scenario_trees):
            rows = []
            for t in range(len(tree)):
                for b, branch in enumerate(tree[t]):
                    rows.append([t, b] + branch)
            
            csv_path = os.path.join(k_dir, f'scenario_{j}.csv')
            np.savetxt(csv_path, rows, delimiter=',')
    
    
    fig, axes = plt.subplots(len(K_list), 1, figsize=(10, 4 * len(K_list)), sharex=True)
    hours = np.arange(24)

    for i, (k, scenario_trees) in enumerate(zip(K_list, Reduced_scenario_trees)):
        ax = axes[i] if len(K_list) > 1 else axes
        
        for scenario in scenario_trees:
            N_t = len(scenario[0]) 
            for b in range(N_t):
                P_rt_values = [scenario[t][b][1] for t in range(24)]  
                ax.plot(hours, P_rt_values, color='black', alpha=0.3)
                
        ax.set_title(f"Scenario Tree Density for K = {k}")
        ax.set_ylabel("P_rt")
        ax.set_ylim(-120, 200)
        ax.grid(True)

    axes[-1].set_xlabel("Hour")
    plt.tight_layout()
    
    plt.savefig('./Stochastic_Approach/Scenarios/P_rt_density.png', dpi=300)
    
    #plt.show()
       
        
    T = sampled_P_da.shape[1]
    hours = np.arange(T)

    for i, P_da_list in enumerate(Reduced_P_da):
        fig, ax = plt.subplots(figsize=(10, 6))
        for profile in P_da_list:
            ax.plot(hours, profile, color='blue', alpha=0.6)
        ax.set_title(f"K = {K_list[i]}: Clustered Day-Ahead Price Profiles")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.tight_layout()
        #plt.show()
    
    
    save_dir = './Stochastic_Approach/Scenarios/Clustered_P_da'
    os.makedirs(save_dir, exist_ok=True)
    
    for i, P_da_list in enumerate(Reduced_P_da):
        k = K_list[i]
        csv_path = os.path.join(save_dir, f"K{k}.csv")
        np.savetxt(csv_path, P_da_list, delimiter=',')
    
    fig, axes = plt.subplots(len(K_list), 1, figsize=(12, 2.5 * len(K_list)), sharex=True, constrained_layout=True)
    hours = np.arange(24)

    for i, (k, (scenario_trees, P_da_list)) in enumerate(zip(K_list, zip(Reduced_scenario_trees, Reduced_P_da))):
        ax = axes[i] if len(K_list) > 1 else axes

        for P_da, scenario in zip(P_da_list, scenario_trees):
            ax.plot(hours, P_da, color='blue', linewidth=2.0, alpha=0.8)  # Day-ahead

            N_t = len(scenario[0])
            for b in range(N_t):
                P_rt_values = [scenario[t][b][1] for t in range(24)]
                ax.plot(hours, P_rt_values, color='black', linewidth=0.6, alpha=0.3)  # Real-time branches

        ax.set_title(f"K = {k}: P_da (blue) & P_rt branches (black)", fontsize=10)
        ax.set_ylabel("Price")
        ax.set_ylim(-120, 200)
        ax.grid(True)

    axes[-1].set_xlabel("Hour")
    
    plt.savefig('./Stochastic_Approach/Scenarios/combined_scenario_density.png', dpi=300)
    #plt.show()
        
    fig, axes = plt.subplots(len(K_list), 1, figsize=(12, 2.5 * len(K_list)), sharex=True, constrained_layout=True)
    for i, (k, P_da_list) in enumerate(zip(K_list, Reduced_P_da)):
        ax = axes[i] if len(K_list) > 1 else axes
        for profile in P_da_list:
            ax.plot(hours, profile, color='blue', linewidth=2.0, alpha=0.8)
        ax.set_title(f"K = {k}: Clustered Day-Ahead Price Profiles (P_da only)")
        ax.set_ylabel("P_da")
        ax.set_ylim(-120, 200)
        ax.grid(True)
    axes[-1].set_xlabel("Hour")
    plt.savefig('./Stochastic_Approach/Scenarios/P_da_only_byK.png', dpi=300)
    plt.close(fig)
        
 
    
    # For SDDiP
    """
    branch_num = 6
    
    scenario_generator = scenario(branch_num, E_0)
            
    P_da_list = [
        [
            120, 130, 120, 100, 
            90, 100, 100, 100, 
            90, 80, 90, 100, 
            90, 70, 90, 90,
            100, 120, 130, 120,
            100, 110, 140, 130
        ],
        [
            110, 100, 120, 110,
            90, 100, 110, 90,
            80, 40, -40, -79,
            -79, -70, -60, 50,
            50, 100, 90, 100,
            110, 120, 100, 100  
        ],
        [
            100, 90, 100, 90,
            80, 100, 0, -10,
            -50, -70, -79, -79,
            -79, -79, -60, -40,
            -50, -20, 0, 50,
            70, 80, 100, 90
        ]
    ]
    
    scenario_trees = []
    
    for P_da in P_da_list:
        
        scenario_tree = scenario_generator.generate_scenario_tree(P_da)
        scenario_trees.append(scenario_tree)
    

    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

    for idx, (P_da, tree) in enumerate(zip(P_da_list, scenario_trees)):
        # Plot P_da
        axs[0, idx].plot(range(24), P_da, color='tab:blue')
        axs[0, idx].set_title(f'Day-ahead Price {idx+1}')
        axs[0, idx].set_ylabel('P_da')
        axs[0, idx].set_ylim(-100, 300)
        axs[0, idx].set_xticks(range(0, 24, 4))
        axs[0, idx].grid(True)

        # Plot each P_rt branch (6 branches)
        
        P_rt_lists = [[] for _ in range(branch_num)]
        
        for branches in tree:
            
            for b_idx, branch in enumerate(branches):
                P_rt_lists[b_idx].append(branch[1])

        for P_rt_list in P_rt_lists:
            
            axs[1, idx].plot(range(24), P_rt_list, label=f's{idx}', alpha=0.3, color='black')

        axs[1, idx].set_title(f'Real-time Scenarios {idx+1}')
        axs[1, idx].set_xlabel('Hour')
        axs[1, idx].set_ylabel('P_rt')
        axs[1, idx].set_ylim(-100, 300)
        axs[1, idx].set_xticks(range(0, 24, 4))
        axs[1, idx].grid(True)

    plt.tight_layout()
    #plt.show()
    
    base_dir_P_da   = './Stochastic_Approach/Scenarios/P_da_settings'
    base_dir_trees  = './Stochastic_Approach/Scenarios/Trees_settings'
    
    os.makedirs(base_dir_P_da, exist_ok=True)
    os.makedirs(base_dir_trees, exist_ok=True)
    
    for idx, (P_da, tree) in enumerate(zip(P_da_list, scenario_trees)):

        # === Save P_da ===
        P_da_path = os.path.join(base_dir_P_da, f'P_da_{idx}.csv')
        pd.DataFrame(P_da).to_csv(P_da_path, index=False, header=False)

        # === Save Scenario Tree ===
        # Convert tree to 3D list → shape: (num_branches, 24, 3)
        # You can flatten each branch (24 rows, each with 3 columns)
        tree_path = os.path.join(base_dir_trees, f'Tree_{idx}.csv')

        # Flatten and save as DataFrame
        flat_tree = []
        for branch in tree:
            flat_tree.append(branch)  # branch: list of 24 [*, P_rt, *]

        # Resulting DataFrame: 3D flattened into 2D → long csv with index over branches
        # You may want to use MultiIndex later to identify branch #
        full_df = pd.concat([
            pd.DataFrame(b, columns=['x0', 'P_rt', 'x2']).assign(branch=i)
            for i, b in enumerate(flat_tree)
        ])

        # Optional: reorder columns
        full_df = full_df[['branch', 'x0', 'P_rt', 'x2']]

        full_df.to_csv(tree_path, index=False)
    """
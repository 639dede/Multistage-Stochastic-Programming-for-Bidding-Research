import pandas as pd

file_path = './Autoregressive/jeju_estim.csv'  

df = pd.read_csv(file_path)

print(df.columns)  

df['timestamp'] = pd.to_datetime(df[df.columns[0]])


start_time_1 = pd.to_datetime('00:00:00').time()
end_time_1 = pd.to_datetime('07:00:00').time()

start_time_2 = pd.to_datetime('18:00:00').time()
end_time_2 = pd.to_datetime('23:59:59').time()

df_filtered = df[~((df['timestamp'].dt.time >= start_time_1) & (df['timestamp'].dt.time < end_time_1)) &
                 ~((df['timestamp'].dt.time >= start_time_2) & (df['timestamp'].dt.time <= end_time_2))]

df_filtered.to_csv('./Autoregressive/jeju_filtered_data.csv', index=False)
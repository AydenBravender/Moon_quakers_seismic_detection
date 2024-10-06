import pandas as pd
import statistics

# Reading ground truth data
data = pd.read_csv("space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv")
total = []

# Converting column data to lists
name = data['filename'].tolist()
time_abs = data['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].tolist()
time_rel = data['time_rel(sec)'].tolist()
evid = data['evid'].tolist()
mq_type = data['mq_type'].tolist()

# Printing list data
print(f"filename: {name}")
print(f"time_abs: {time_abs}")
print(f"time_rel: {time_rel}")
print(f"evid: {evid}")
print(f"mq_type: {mq_type}")

# Reading predictions data
data_pred = pd.read_csv("detected_moonquakes_ayden.csv")

# Converting column data to lists
name_pred = data_pred['filename'].tolist()
time_rel_pred = data_pred['start_time'].tolist()

# Set NaN values in time_rel_pred to 0
time_rel_pred = [0 if pd.isna(t) else t for t in time_rel_pred]

# Printing list data
print(f"filename: {name_pred}")
print(f"time_rel: {time_rel_pred}")

# Calculate absolute differences
for i in range(len(name)):
    try:
        total.append(abs(time_rel[i] - time_rel_pred[i]))
    except IndexError:
        pass

# Calculate mean of total differences
mean = statistics.mean(total) if total else 0
print('---------------------------------------------------------------------------------')
print(mean)

import numpy as np
import os
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from obspy.signal.filter import bandpass
import re

# Define the directory containing MiniSEED files
data_directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'
output_csv = 'detected_moonquakes.csv'

# Prepare a list to store results
results = []

# Iterate over files in the specified directory
for filename in os.listdir(data_directory):
    if filename.endswith('.mseed'):
        mseed_file = os.path.join(data_directory, filename)
        
        # Read the MiniSEED file
        st = read(mseed_file)
        
        # Extract the first trace (adjust as needed if multiple traces exist)
        tr = st[0]
        
        # Get the sampling frequency
        sampling_rate = tr.stats.sampling_rate
        
        # Extract data and time array from the trace
        tr_times = tr.times()  # Time in seconds
        tr_data = tr.data  # Seismic velocity data
        
        # Bandpass filter to isolate relevant frequencies for moonquake detection
        minfreq = 0.5  # Lower bound frequency in Hz
        maxfreq = 1.0  # Upper bound frequency in Hz
        tr_filt_data = bandpass(tr_data, minfreq, maxfreq, df=sampling_rate, corners=4, zerophase=True)
        
        # Estimate arrival time of moonquake (use STA/LTA or similar detection method)
        sta_len = 120  # Short-term average window in seconds
        lta_len = 600  # Long-term average window in seconds
        
        # Run Obspy's STA/LTA trigger algorithm to detect events
        cft = classic_sta_lta(tr_filt_data, int(sta_len * sampling_rate), int(lta_len * sampling_rate))
        
        # Set thresholds for detecting triggers
        thr_on = 4.0  # Trigger on threshold
        thr_off = 1.5  # Trigger off threshold
        
        # Detect triggers
        on_off = np.array(trigger_onset(cft, thr_on, thr_off))
        
        # Identify the most confident trigger or note if none found
        if on_off.size > 0:
            trigger_indices = on_off[:, 0]
            confidences = cft[trigger_indices]
            max_conf_index = np.argmax(confidences)
            best_trigger = on_off[max_conf_index]
            trigger_time = tr_times[best_trigger[0]]
            results.append({'filename': filename, 'trigger_time': trigger_time})
        else:
            results.append({'filename': filename, 'trigger_time': None})  # Indicate no trigger found

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Extract event number from filename for sorting
df_results['event_number'] = df_results['filename'].apply(lambda x: int(re.search(r'evid(\d+)', x).group(1)))

# Sort the DataFrame by the event_number
df_results.sort_values(by='event_number', inplace=True)

# Reset index after sorting
df_results.reset_index(drop=True, inplace=True)

# Save to CSV, including the cases with no triggers
df_results[['filename', 'trigger_time']].to_csv(output_csv, index=False)

print(f"Detected moonquakes saved to {output_csv}")

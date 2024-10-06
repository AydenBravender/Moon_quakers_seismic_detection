import numpy as np
import os
import pandas as pd
from obspy import read
from obspy.signal.filter import bandpass
import re
import matplotlib.pyplot as plt

# Define the directory containing MiniSEED files
data_directory = '/home/ayden/nasa/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'
output_csv = 'detected_moonquakes_ayden.csv'

# Prepare a list to store results
results = []

# Ground truth data (replace with your actual timestamps)
# This is a dictionary where the filename is the key and the ground truth time is the value
ground_truth = {
    'example_file1.mseed': 10.5,  # Replace with actual filename and time
    'example_file2.mseed': 15.3,  # Replace with actual filename and time
    # Add more entries as needed
}

# Function to plot the data with moonquake vertical markers and ground truth
def plot_data(tr_times, tr_data, tr_filt_data, moonquake_start_time, ground_truth_time, filename):
    plt.figure(figsize=(12, 6))

    # Plot raw data
    plt.subplot(2, 1, 1)
    plt.plot(tr_times, tr_data, label="Raw Data")
    plt.title(f"Raw Seismic Data - {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Plot filtered data
    plt.subplot(2, 1, 2)
    plt.plot(tr_times, tr_filt_data, label="Filtered Data (0.2 - 1 Hz)", color='r')
    
    # Plot vertical line at the detected moonquake time (if any)
    if moonquake_start_time:
        plt.axvline(x=moonquake_start_time, color='g', linestyle='--', label=f'Moonquake Detected @ {moonquake_start_time:.2f}s')
    
    # Plot vertical line for ground truth time (if any)
    if ground_truth_time:
        plt.axvline(x=ground_truth_time, color='b', linestyle='--', label=f'Ground Truth @ {ground_truth_time:.2f}s')
    
    plt.title(f"Filtered Seismic Data - {filename}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Iterate over files in the specified directory
for filename in os.listdir(data_directory):
    if filename.endswith('.mseed'):
        mseed_file = os.path.join(data_directory, filename)
        print(f"Processing file: {filename}")
        
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
        minfreq = 0.1  # Adjusted lower bound frequency in Hz
        maxfreq = 1.0  # Upper bound frequency in Hz
        tr_filt_data = bandpass(tr_data, minfreq, maxfreq, df=sampling_rate, corners=4, zerophase=True)
        
        # Initialize variables to track the longest duration and highest magnitude
        max_duration = 0
        max_magnitude = 0
        best_start_time = None
        
        # Variables for tracking the current moonquake
        start_time = None
        duration = 0
        moonquake_detected = False
        amplitude_threshold = 0.00000000001 # Adjust as needed for detection sensitivity
        
        for i in range(1, len(tr_filt_data)):
            # Check if the amplitude exceeds the threshold
            if abs(tr_filt_data[i]) > amplitude_threshold:
                if not moonquake_detected:
                    # Moonquake started
                    start_time = tr_times[i]
                    moonquake_detected = True
                duration += 1 / sampling_rate  # Increment duration in seconds
            else:
                if moonquake_detected:
                    # Moonquake ended, calculate magnitude
                    magnitude = duration * np.max(tr_filt_data[i - int(duration * sampling_rate):i])
                    if duration > max_duration or (duration == max_duration and magnitude > max_magnitude):
                        max_duration = duration
                        max_magnitude = magnitude
                        best_start_time = start_time
                    moonquake_detected = False
                    duration = 0  # Reset duration for the next event
        
        # Handle the case where the moonquake continues until the end of the trace
        if moonquake_detected:
            magnitude = duration * np.max(tr_filt_data[-int(duration * sampling_rate):])
            if duration > max_duration or (duration == max_duration and magnitude > max_magnitude):
                max_duration = duration
                max_magnitude = magnitude
                best_start_time = start_time

        if best_start_time:
            # Save the result with the longest duration and highest magnitude
            results.append({'filename': filename, 'start_time': best_start_time, 'magnitude': max_magnitude})
            print(f"Moonquake detected in {filename} at {best_start_time} with magnitude {max_magnitude}")
        else:
            print(f"No significant moonquakes detected in {filename}")

        # Get ground truth time for this file if available
        ground_truth_time = ground_truth.get(filename, None)

        # Plot the raw and filtered data, marking the detected moonquake with a vertical line
        plot_data(tr_times, tr_data, tr_filt_data, best_start_time, ground_truth_time, filename)

# Check if any results were added
if len(results) > 0:
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Ensure 'filename' exists in the DataFrame and add event numbers for sorting
    if 'filename' in df_results.columns:
        # Extract event number from filename for sorting
        df_results['event_number'] = df_results['filename'].apply(lambda x: int(re.search(r'evid(\d+)', x).group(1)))

        # Sort the DataFrame by the event_number
        df_results.sort_values(by='event_number', inplace=True)

        # Find the moonquake with the greatest magnitude
        max_magnitude_row = df_results.loc[df_results['magnitude'].idxmax()]

        # Output the time the largest moonquake started
        print(f"Largest moonquake detected in {max_magnitude_row['filename']} started at {max_magnitude_row['start_time']} with magnitude {max_magnitude_row['magnitude']}.")
    
    # Reset index after sorting
    df_results.reset_index(drop=True, inplace=True)

    # Save to CSV, including the cases with no triggers
    df_results[['filename', 'start_time', 'magnitude']].to_csv(output_csv, index=False)

    print(f"Detected moonquakes saved to {output_csv}")
else:
    print("No moonquakes detected in any files.")

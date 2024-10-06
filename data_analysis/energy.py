import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read  # ObsPy for handling MiniSEED files
from obspy.signal.filter import bandpass  # Import bandpass filter from ObsPy
from scipy.signal import medfilt  # Import median filter

# Directory containing the MiniSEED files
data_directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'

# Get a sorted list of MiniSEED filenames
mseed_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.mseed')])

# Load event time data from CSV
event_time_data = pd.read_csv('/home/ayden/nasa/apollo12_catalog_GradeA_final.csv')

# Process each MiniSEED file in order
for filename in mseed_files:
    mseed_file = os.path.join(data_directory, filename)

    # Read the MiniSEED file using ObsPy
    st = read(mseed_file)
    tr = st[0]  # Assuming single trace per file
    time = np.arange(0, tr.stats.npts) * tr.stats.delta  # Create time array
    motion = tr.data  # Seismic data (velocity)

    # Bandpass filter to isolate relevant frequencies for moonquake detection
    minfreq = 0.7  # Lower bound frequency in Hz
    maxfreq = 1.0  # Upper bound frequency in Hz
    sampling_rate = 1.0 / tr.stats.delta  # Calculate the sampling rate
    tr_filt_data = bandpass(motion, minfreq, maxfreq, df=sampling_rate, corners=4, zerophase=True)

    # Apply median filter to remove short-term spikes (kernel size of 5)
    tr_filt_data = medfilt(tr_filt_data, kernel_size=5)

    # Extract the event ID from the filename
    event_id = filename.split('_')[-1].split('.')[0]  # Extract event ID
    event_id = event_id.replace("evid", "evid")  # Maintain 'evid' prefix

    # Find the row in the CSV that matches the event_id
    matching_event = event_time_data[event_time_data['evid'] == event_id]

    if not matching_event.empty:
        event_time = matching_event['time_rel(sec)'].values[0]  # Extract the event time
    else:
        print(f"No matching event found for {filename}. Skipping...")
        continue  # Skip this file if no matching event

    # Compute the squared velocity (motion)
    squared_motion = tr_filt_data ** 2

    # Compute the energy at each time step (cumulative sum of squared velocity * dt)
    energy = np.cumsum(squared_motion * tr.stats.delta)  # Using delta time directly

    # Compute the total time up to each step, avoiding division by zero
    time_intervals = np.cumsum(np.full(tr.stats.npts, tr.stats.delta))
    time_intervals[time_intervals == 0] = np.nan  # Replace zeros with NaN

    # Compute the power at each time step (energy divided by time)
    power = energy / time_intervals

    # Compute the rate of energy change
    rate_of_energy_change = np.gradient(energy, tr.stats.delta)

    # Compute the energy decay rate
    decay_rate = -np.gradient(energy, time)

    # Plotting
    plt.figure(figsize=(10, 10))

    # Plot Time vs Energy
    plt.subplot(5, 1, 1)
    plt.plot(time, energy, label="Energy", color="b")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('Time (sec)')
    plt.ylabel('Energy (J)')
    plt.title(f'{filename} - Time vs Energy')
    plt.grid(True)
    plt.legend()

    # Plot Rate of Energy Change
    plt.subplot(5, 1, 2)
    plt.plot(time, rate_of_energy_change, label="Rate of Energy Change", color="orange")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('Time (sec)')
    plt.ylabel('Rate of Energy Change (J/s)')
    plt.title('Rate of Energy Change')
    plt.grid(True)
    plt.legend()

    # Plot Time vs Filtered and Smoothed Motion
    plt.subplot(5, 1, 3)
    plt.plot(time, tr_filt_data, label="Filtered & Smoothed Motion (Velocity)", color="g")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Time vs Filtered Motion')
    plt.grid(True)
    plt.legend()

    # Plot Time vs Power
    plt.subplot(5, 1, 4)
    plt.plot(time, power, label="Power", color="r")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('Time (sec)')
    plt.ylabel('Power (W)')
    plt.title('Time vs Power')
    plt.grid(True)
    plt.legend()

    # Plot Energy Decay Rate
    plt.subplot(5, 1, 5)
    plt.plot(time, decay_rate, label="Energy Decay Rate", color="purple")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('Time (sec)')
    plt.ylabel('Decay Rate (J/s)')
    plt.title('Energy Decay Rate')
    plt.grid(True)
    plt.legend()

    # Adjust the layout
    plt.tight_layout()

    # Show the plots
    plt.show()

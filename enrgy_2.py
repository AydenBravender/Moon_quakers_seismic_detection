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

    # Compute the rate of energy change
    rate_of_energy_change = np.gradient(energy, tr.stats.delta)

    # Recalculate energy by integrating the rate of energy change over time
    new_energy = np.cumsum(rate_of_energy_change * tr.stats.delta)

    # Compute the power (energy divided by time)
    power = new_energy / time_intervals

    # Compute the energy decay rate
    decay_rate = -np.gradient(energy, time)

    # Compute the derivative of energy and power
    energy_derivative = np.gradient(new_energy, tr.stats.delta)
    power_derivative = np.gradient(power, tr.stats.delta)


    # Plotting
    plt.figure(figsize=(10, 16))  # Increased figure size to accommodate 8 plots

    # Plot Time vs Energy (from rate of energy change)S
    plt.subplot(7, 1, 1)
    plt.plot(time, new_energy, label="Energy (from Rate of Energy Change)", color="b")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('')
    plt.ylabel('Energy (J)')
    plt.title(f'{filename} - Time vs Energy')
    plt.grid(True)
    plt.legend()

    # Plot Rate of Energy Change
    plt.subplot(7, 1, 2)
    plt.plot(time, rate_of_energy_change, label="Rate of Energy Change", color="orange")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('')
    plt.ylabel('Rate of Energy Change (J/s)')
    plt.title('Rate of Energy Change')
    plt.grid(True)
    plt.legend()

    # Plot Time vs Filtered and Smoothed Motion
    plt.subplot(7, 1, 3)
    plt.plot(time, tr_filt_data, label="Filtered & Smoothed Motion (Velocity)", color="g")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('')
    plt.ylabel('Velocity (m/s)')
    plt.title('Time vs Filtered Motion')
    plt.grid(True)
    plt.legend()

    # Plot Time vs Power (from recalculated energy)
    plt.subplot(7, 1, 4)
    plt.plot(time, power, label="Power (from Rate of Energy Change)", color="r")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('')
    plt.ylabel('Power (W)')
    plt.title('Time vs Power')
    plt.grid(True)
    plt.legend()

    # Plot Energy Decay Rate
    plt.subplot(7, 1, 5)
    plt.plot(time, decay_rate, label="Energy Decay Rate", color="purple")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('')
    plt.ylabel('Decay Rate (J/s)')
    plt.title('Energy Decay Rate')
    plt.grid(True)
    plt.legend()

    # Plot Derivative of Energy
    plt.subplot(7, 1, 6)
    plt.plot(time, energy_derivative, label="Derivative of Energy", color="cyan")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('')
    plt.ylabel('d(Energy)/dt (J/s)')
    plt.title('Derivative of Energy')
    plt.grid(True)
    plt.legend()

    # Plot Derivative of Power
    plt.subplot(7, 1, 7)
    plt.plot(time, power_derivative, label="Derivative of Power", color="magenta")
    plt.axvline(x=event_time, color='k', linestyle='--', label="Event Time")
    plt.xlabel('')
    plt.ylabel('d(Power)/dt (W/s)')
    plt.title('Derivative of Power')
    plt.grid(True)
    plt.legend()

    # Adjust the layout
    plt.tight_layout()

    # Show the plots
    plt.show()

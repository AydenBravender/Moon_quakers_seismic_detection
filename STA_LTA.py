from SeismicPrediction import SeismicPrediction
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read  # ObsPy for handling MiniSEED files
from obspy.signal.filter import bandpass  # Import bandpass filter from ObsPy
from scipy.signal import medfilt  # Import median filter
import matplotlib.pyplot as plt


def main():
    filename = '//fs-059/studuser$/Gr11/a.bravender/nasa/space_apps_2024_seismic_detection/data/mars/training/data/XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed'
    st = read(filename)
    tr = st[0]  # Assuming single trace per file
    time = np.arange(0, tr.stats.npts) * tr.stats.delta  # Create time array

    pred1 = SeismicPrediction(tr, time)
    filtered_data = pred1.apply_bandpass_filter()
    decay_data = pred1.energy_decay(filtered_data)
    suppresed = pred1.staircase_data(decay_data, 1000)
    normalized = pred1.normalize(suppresed, 0, -1)
    print(pred1.create_high_freq(normalized))

    plt.ion()  # Enable interactive mode
    plt.figure(figsize=(10, 6))
    plt.plot(time, suppresed, color='blue', label='Energy Decay Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Decay Rate')
    plt.title('Energy Decay Over Time')
    plt.legend()
    plt.grid(True)  # Adds a grid for easier visualization

    # Show the plot
    plt.show()

    # Pause to keep the plot open
    plt.pause(100)

if __name__ == "__main__":
    main()
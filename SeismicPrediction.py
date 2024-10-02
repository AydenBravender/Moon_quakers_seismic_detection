import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read  # ObsPy for handling MiniSEED files
from obspy.signal.filter import bandpass  # Import bandpass filter from ObsPy
from scipy.signal import medfilt  # Import median filter
import matplotlib.pyplot as plt


class SeismicPrediction:
    """
    A class to represent the data

    Attributes
    ----------

    data : a list like object found in obspy.read
    time : array containg time of graph

    Methods
    -------

    apply_bandpass_filter():
        Applies a bandpass filter as well as a median filter on raw data
    """
    def __init__(self, data, time):
        self.data = data
        self.time = time


    def apply_bandpass_filter(self):
        """
        Applies a bandpass filter as well as a median filter on raw data

        returns: filtered data

        """
        # Bandpass filter to isolate relevant frequencies for moonquake detection
        minfreq = 0.7  # Lower bound frequency in Hz
        maxfreq = 1.0  # Upper bound frequency in Hz
        sampling_rate = 1.0 / self.data.stats.delta  # Calculate the sampling rate
        tr_filt_data = bandpass(self.data.data, minfreq, maxfreq, df=sampling_rate, corners=4, zerophase=True)

        # Apply median filter to remove short-term spikes (kernel size of 5)
        filtered_motion = medfilt(tr_filt_data, kernel_size=5)

        return filtered_motion


    def energy_decay(self, filtered_motion):
        """
        Calculates the energy decay rate from the filtered motion data. 
        The energy decay rate helps track the decrease in energy over time after seismic motion is detected.

        Attributes
        ----------

        filtered_motion : array of filtered data from apply_bandpass_filter()

        returns: decay_rate

        """
        squared_motion = filtered_motion ** 2 # Compute the squared velocity (motion)
        energy = np.cumsum(squared_motion * self.data.stats.delta)  # Compute the energy at each time step 
        decay_rate = -np.gradient(energy, self.time) # Compute the energy decay rate

        return decay_rate
    

    def staircase_data(self, data, window):
        """
        Suppresses fluctuations in the input data by keeping the maximum value
        within a sliding window.

        Parameters
        ----------
        data : numpy.ndarray
            The input data array to be suppressed.
        window : int
            The size of the window to consider for finding the maximum value.

        Returns
        -------
        numpy.ndarray
            An array where each value in the original data is replaced by the
            maximum value found within the specified window around that value.
        """
        suppressed_data = np.copy(data)  # Copy original data

        for i in range(len(suppressed_data)):
            suppressed_data[i] = np.mean(suppressed_data[i:i+window + 1])
        return suppressed_data
    
    
    def create_high_freq(self, decay_data):
        """
        Identifies potential seismic events based on the energy decay data.

        This method scans through the decay data and marks segments where
        the decay value falls below a specified threshold. It identifies
        the beginning of each potential quake segment and its length.

        Parameters
        ----------
        decay_data : numpy.ndarray
            The array of decay data from which potential seismic events are identified.

        Returns
        -------
        list
            A list of tuples, each containing the index of the start of a potential quake
            segment and its length.
        """
        potential_quakes = []
        curr_segment = []
        for i in range(len(decay_data)-1):
            if decay_data[i] <= -0.1e-17:
                curr_segment.append(decay_data[i])
                if decay_data[i+1] > -0.1e-17:
                    potential_quakes.append([i, len(curr_segment)])
                    curr_segment = []
        
        return potential_quakes
    


def main():
    filename = '/home/ayden/nasa/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-07-20HR00_evid00010.mseed'
    st = read(filename)
    tr = st[0]  # Assuming single trace per file
    time = np.arange(0, tr.stats.npts) * tr.stats.delta  # Create time array

    pred1 = SeismicPrediction(tr, time)
    filtered_data = pred1.apply_bandpass_filter()
    decay_data = pred1.energy_decay(filtered_data)
    suppresed = pred1.staircase_data(decay_data, 1000)
    print(pred1.create_high_freq(suppresed))

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
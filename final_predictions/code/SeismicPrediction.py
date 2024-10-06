import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read  # ObsPy for handling MiniSEED files
from obspy.signal.filter import bandpass  # Import bandpass filter from ObsPy
from scipy.signal import medfilt  # Import median filter
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report



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

    energy_decay():
        Calculates the energy decay rate from the filtered motion data

    staircase_data():
        Suppresses fluctuations in the input data by keeping the maximum value
        within a sliding window.

    normalize():
        Normalizes the array to a range between -1 and 0.

    create_high_freq():
        Identifies potential seismic events based on the energy decay data.

    predict():
        Process the results and predict significant seismic events based on specified criteria
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

    def normalize(self, arr):
        """
        Normalizes the array to a range between -1 and 0.

        Parameters
        ----------
        arr : numpy.ndarray
            The input data array to be normalized.

        Returns
        -------
        numpy.ndarray
            The normalized data.
        """
        avg_value = np.mean(arr)
        min_value = np.min(arr)
        max_value = np.max(arr)

        # # Normalize using the average and minimum
        norm_arr = (arr-avg_value) / (avg_value - min_value)

        return norm_arr
        
    
    def create_high_freq(self, decay_data, threshold):
        """
        Identifies potential seismic events based on the energy decay data.

        This method scans through the decay data and marks segments where
        the decay value falls below a specified threshold. It identifies
        the beginning of each potential quake segment and its length.

        Parameters
        ----------
        decay_data : numpy.ndarray
            The array of decay data from which potential seismic events are identified.

        threshold : int
            the number deciding which event are considered for a seismic event

        Returns
        -------
        list
            A list of tuples, each containing the index of the start of a potential quake
            segment and its length.
        """
        potential_quakes = []
        curr_segment = []
        for i in range(len(decay_data)):
            if decay_data[i] <= threshold:
                curr_segment.append(decay_data[i])
                try:
                    if decay_data[i+1] > threshold or i == len(decay_data)-1:
                        segment = decay_data[i-len(curr_segment)+1:i+1]
                        average = np.mean(decay_data)
                        # aydens_value = (min(segment)/average)
                        aydens_value = (min(segment)/max(segment))

                        start_index =  i-len(curr_segment)+1
                        end_index = i 

                        # Select the chunk of data within the specified range
                        chunk = self.data[start_index:end_index+1]

                        # Calculate the power (P = x^2) for the chunk
                        power = np.square(chunk)

                        # Calculate average power over the duration
                        average_power = np.mean(power)

                        # Calculate total energy (E = sum(x^2)) over the duration
                        total_energy = np.sum(power)

                        potential_quakes.append([int(self.time[[i-len(curr_segment)+1]]), len(curr_segment), aydens_value, average_power, total_energy])
                        curr_segment = []
                except IndexError:
                    if i == len(decay_data)-1:
                        segment = decay_data[i-len(curr_segment)+1:i+1]
                        average = sum(decay_data) / len(decay_data)
                        # aydens_value = (min(segment)/average)
                        aydens_value = (min(segment)/max(segment))

                        start_index =  i-len(curr_segment)+1
                        end_index = i 

                        # Select the chunk of data within the specified range
                        chunk = self.data[start_index:end_index+1]

                        # Calculate the power (P = x^2) for the chunk
                        power = np.square(chunk)

                        # Calculate average power over the duration
                        average_power = np.mean(power)

                        # Calculate total energy (E = sum(x^2)) over the duration
                        total_energy = np.sum(power)
                    

                        potential_quakes.append([int(self.time[[i-len(curr_segment)+1]]), len(curr_segment), aydens_value, average_power, total_energy])
                        curr_segment = []
        
        return potential_quakes


    def predict(self, results):
        """
        Process the results and predict significant seismic events based on specified criteria.

        This method processes a list of results, which contain information such as durations, times, 
        custom 'aydens_value', power, and energy of seismic events. It filters and sorts these events 
        to extract the most significant ones based on given thresholds, such as a minimum duration 
        and 'aydens_value'.

        Parameters:
        -----------
        results : list of lists
            A nested list where each sublist contains:
            - duration (int): The duration of the seismic event.
            - time (int): The time at which the seismic event occurred.
            - aydens_value (float): A custom calculated value for seismic event significance.
            - power (float): The power of the seismic event.
            - energy (float): The energy of the seismic event.

        Returns:
        --------
        final_results : list of int
            A list of times where significant seismic events occurred. Events are chosen based on 
            the criteria that:
            - The event duration is at least 3000 units.
            - The 'aydens_value' is at least 3.
            Events are sorted by duration in descending order. Only events separated by more than 
            5000 time units are considered as separate significant events.
        """
        final_results = []
        new_list = []
        # Extract durations and times
        durations = [inner_list[1] for outer_list in results for inner_list in outer_list]
        times = [inner_list[0] for outer_list in results for inner_list in outer_list]
        aydens_value = [inner_list[2] for outer_list in results for inner_list in outer_list]
        power = [inner_list[3] for outer_list in results for inner_list in outer_list]
        energy = [inner_list[4] for outer_list in results for inner_list in outer_list]
        
        for i in range(len(durations)):
            if durations[i] >= 3000 and aydens_value[i] >= 3:
                new_list.append([durations[i], times[i], power[i], energy[i]])
        new_list = sorted(new_list, key=lambda x: x[0], reverse=True)

        if new_list:
            final_results.append(new_list[0][1])
        for i in range(len(new_list)):
            n = 0
            for j in range(len(final_results)):
                if abs(new_list[i][1] - final_results[j]) > 5000:
                    n+=1
            if n == len(final_results):
                final_results.append(new_list[i][1]) 

        return final_results
    
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

    energy_decay():
        Calculates the energy decay rate from the filtered motion data

    staircase_data():
        Suppresses fluctuations in the input data by keeping the maximum value
        within a sliding window.

    normalize():
        Normalizes the array to a range between -1 and 0.

    create_high_freq():
        Identifies potential seismic events based on the energy decay data.
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
        min_value = np.min(arr)
        max_value = np.max(arr)

        norm_arr = (arr - max_value) / (max_value - min_value)

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
                        average = sum(decay_data) / len(decay_data)
                        aydens_value = (min(segment)/average)

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
                        aydens_value = (min(segment)/average)

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
        final_results = []
        new_list = []
        # Extract durations and times
        durations = [inner_list[1] for outer_list in results for inner_list in outer_list]
        times = [inner_list[0] for outer_list in results for inner_list in outer_list]
        aydens_value = [inner_list[2] for outer_list in results for inner_list in outer_list]
        power = [inner_list[3] for outer_list in results for inner_list in outer_list]
        energy = [inner_list[4] for outer_list in results for inner_list in outer_list]
        
        for i in range(len(durations)):
            if durations[i] >= 340 and aydens_value[i] >= 2.5:
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
    


# def main():
#     # Directory containing the MiniSEED files
#     data_directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'

#     # Get a sorted list of MiniSEED filenames
#     mseed_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.mseed')])

#     # Load event time data from CSV
#     event_time_data = pd.read_csv('apollo12_catalog_GradeA_final.csv')

#     # Process each MiniSEED file in order
#     for filename in mseed_files:
#         mseed_file = os.path.join(data_directory, filename)
#         # Read the MiniSEED file using ObsPy
#         st = read(mseed_file)
#         tr = st[0]  # Assuming single trace per file
#         time = np.arange(0, tr.stats.npts) * tr.stats.delta  # Create time array

#         event_id = filename.split('_')[-1].split('.')[0]  # Extract event ID
#         event_id = event_id.replace("evid", "evid")  # Maintain 'evid' prefix

#         # Find the row in the CSV that matches the event_id
#         matching_event = event_time_data[event_time_data['evid'] == event_id]

#         if not matching_event.empty:
#             event_time = matching_event['time_rel(sec)'].values[0]  # Extract the event time
#         else:
#             print(f"No matching event found for {filename}. Skipping...")
#             continue  # Skip this file if no matching event


#         pred1 = SeismicPrediction(tr, time)
#         filtered_data = pred1.apply_bandpass_filter()
#         decay_data = pred1.energy_decay(filtered_data)
#         suppresed = pred1.staircase_data(decay_data, 1000)
#         normalized = pred1.normalize(suppresed)
#         print(pred1.create_high_freq(normalized, -0.02))
#         print(event_time)

#         plt.figure(figsize=(10, 6))
#         plt.plot(time, normalized, color='blue', label='Energy Decay Rate')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Decay Rate')
#         plt.title('Energy Decay Over Time')
#         plt.legend()
#         plt.grid(True)  # Adds a grid for easier visualization

#         # Show the plot
#         plt.show()

# if __name__ == "__main__":
#     main()
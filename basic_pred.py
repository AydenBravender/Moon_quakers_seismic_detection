from SeismicPrediction import SeismicPrediction
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read  # ObsPy for handling MiniSEED files
from obspy.signal.filter import bandpass  # Import bandpass filter from ObsPy
from scipy.signal import medfilt  # Import median filter
import matplotlib.pyplot as plt


def predict(results):
    # final_results = []
    # for i in range(len(results)):
    #     duration = [inner_list[2] for outer_list in results for inner_list in outer_list]
    #     time = [inner_list[1] for outer_list in results for inner_list in outer_list]
    
    # sorted_duration = duration.sort(reverse=True)
    # sorted_time = time.sort(reverse=True)
    # for i in range(len(sorted_duration)):
    #     index = duration.index(sorted_duration[i])
    #     final_results.append(time[index])
    #     if abs(final_results[-1] - time[i]) >= 8000:
    #         final_results.append(time[i])
    
    # return final_results

    # --------------------------------------------------------------
    # final_results = []
    
    # # Extract durations and times
    # durations = [inner_list[2] for outer_list in results for inner_list in outer_list]
    # times = [inner_list[1] for outer_list in results for inner_list in outer_list]

    # # Create a list of (duration, time) pairs and sort by duration in descending order
    # duration_time_pairs = sorted(zip(durations, times), key=lambda x: x[0], reverse=True)

    # # Append the longest duration first
    # if duration_time_pairs:
    #     final_results.append(duration_time_pairs[0][1])  # Append time of the longest duration

    # # Now iterate through the sorted list and append valid times
    # last_appended_time = final_results[0] if final_results else None
    # for duration, time in duration_time_pairs[1:]:  # Skip the first since it's already added
    #     if last_appended_time is not None and abs(time - last_appended_time) >= 20000:
    #         final_results.append(time)
    #         last_appended_time = time  # Update last appended time

    # return final_results

    final_results = []
    new_list = []
    # Extract durations and times
    durations = [inner_list[2] for outer_list in results for inner_list in outer_list]
    times = [inner_list[1] for outer_list in results for inner_list in outer_list]
    
    for i in range(len(durations)):
        if durations[i] >= 2:
            new_list.append([durations[i], times[i]])
    new_list = sorted(new_list, key=lambda x: x[0], reverse=True)


    final_results.append(new_list[0][1])
    for i in range(len(new_list)):
        n = 0
        for j in range(len(final_results)):
            if abs(new_list[i][1] - final_results[j]) > 20000:
                n+=1
        if n == len(final_results):
            final_results.append(new_list[i][1]) 

    return final_results






    



def main():
    # Directory containing the MiniSEED files
    data_directory = 'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'

    # Get a sorted list of MiniSEED filenames
    mseed_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.mseed')])

    # Load event time data from CSV
    event_time_data = pd.read_csv('//fs-059/studuser$/Gr11/a.bravender/nasa/apollo12_catalog_GradeA_final.csv')
    output = []

    # Process each MiniSEED file in order
    for filename in mseed_files:
        mseed_file = os.path.join(data_directory, filename)
        # Read the MiniSEED file using ObsPy
        st = read(mseed_file)
        tr = st[0]  # Assuming single trace per file
        time = np.arange(0, tr.stats.npts) * tr.stats.delta  # Create time array

        event_id = filename.split('_')[-1].split('.')[0]  # Extract event ID
        event_id = event_id.replace("evid", "evid")  # Maintain 'evid' prefix

        # Find the row in the CSV that matches the event_id
        matching_event = event_time_data[event_time_data['evid'] == event_id]

        if not matching_event.empty:
            event_time = matching_event['time_rel(sec)'].values[0]  # Extract the event time
        else:
            print(f"No matching event found for {filename}. Skipping...")
            continue  # Skip this file if no matching event


        pred1 = SeismicPrediction(tr, time)
        filtered_data = pred1.apply_bandpass_filter()
        decay_data = pred1.energy_decay(filtered_data)
        suppresed = pred1.staircase_data(decay_data, 1000)
        normalized = pred1.normalize(suppresed)
        results = [pred1.create_high_freq(normalized, -0.08)]
        output.append(predict(results))
        

        # plt.figure(figsize=(10, 6))
        # plt.plot(time, normalized, color='blue', label='Energy Decay Rate')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Decay Rate')
        # plt.title('Energy Decay Over Time')
        # plt.legend()
        # plt.grid(True)  # Adds a grid for easier visualization

        # # Show the plot
        # plt.show()
    print(output)
    
if __name__ == "__main__":
    main()



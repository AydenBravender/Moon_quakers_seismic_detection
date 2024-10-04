from SeismicPrediction import SeismicPrediction
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read  # ObsPy for handling MiniSEED files
from obspy.signal.filter import bandpass  # Import bandpass filter from ObsPy
from scipy.signal import medfilt  # Import median filter
import csv


def predict(results):
    final_results = []
    new_list = []
    # Extract durations and times
    durations = [inner_list[2] for outer_list in results for inner_list in outer_list]
    times = [inner_list[1] for outer_list in results for inner_list in outer_list]
    aydens_value = [inner_list[3] for outer_list in results for inner_list in outer_list]
    
    for i in range(len(durations)):
        if durations[i] >= 340 and aydens_value[i] >= 5:
            new_list.append([durations[i], times[i]])
    new_list = sorted(new_list, key=lambda x: x[0], reverse=True)


    final_results.append(new_list[0][1])
    for i in range(len(new_list)):
        n = 0
        for j in range(len(final_results)):
            if abs(new_list[i][1] - final_results[j]) > 5000:
                n+=1
        if n == len(final_results):
            final_results.append(new_list[i][1]) 

    return final_results

def main():
    # Set the path to the specific MiniSEED file you want to process
    mseed_file = '/home/ayden/nasa/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1972-07-17HR00_evid00067.mseed'
    
    # Load event time data from CSV
    event_time_data = pd.read_csv('/home/ayden/nasa/space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv')
    output = []

    # Read the MiniSEED file using ObsPy
    st = read(mseed_file)
    tr = st[0]  # Assuming single trace per file
    time = np.arange(0, tr.stats.npts) * tr.stats.delta  # Create time array

    # Extract event ID (adjust if needed based on the file naming convention)
    event_id = os.path.basename(mseed_file).split('_')[-1].split('.')[0]
    event_id = event_id.replace("evid", "evid")  # Maintain 'evid' prefix

    # Find the row in the CSV that matches the event_id
    matching_event = event_time_data[event_time_data['evid'] == event_id]

    if not matching_event.empty:
        event_time = matching_event['time_rel(sec)'].values[0]  # Extract the event time
    else:
        print(f"No matching event found for {mseed_file}. Skipping...")
        return  # Exit if no matching event is found

    pred1 = SeismicPrediction(tr, time)
    filtered_data = pred1.apply_bandpass_filter()
    decay_data = pred1.energy_decay(filtered_data)
    suppresed = pred1.staircase_data(decay_data, 1000)
    normalized = pred1.normalize(suppresed)
    results = [pred1.create_high_freq(normalized, -0.08)]
    final = predict(results)
    output.append(final)

    print(results)

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(time, normalized, color='blue', label='Energy Decay Rate')
    plt.xlabel('Time (s)')
    plt.ylabel('Decay Rate')
    plt.title(f'Energy Decay Over Time - {os.path.basename(mseed_file)}')

    # Draw vertical lines for each event in 'final'
    for t in final:
        plt.axvline(x=t, color='k', linestyle='--', label=f"Event at {t:.2f}s")

    plt.axvline(x=event_time, color='r', linestyle='--', label="Catalog Event Time")  # Event time from the catalog
    plt.legend()
    plt.grid(True)

    plt.show()

    # Save the results to CSV
    with open('output_seismic_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in output:
            writer.writerow(row)

    print(f"Results saved to 'output_seismic_predictions.csv'.")

if __name__ == "__main__":
    main()

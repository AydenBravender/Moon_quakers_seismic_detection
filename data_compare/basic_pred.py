from Moon_quakers_seismic_detection.final_predictions.code.SeismicPrediction import SeismicPrediction
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read  # ObsPy for handling MiniSEED files
from obspy.signal.filter import bandpass  # Import bandpass filter from ObsPy
from scipy.signal import medfilt  # Import median filter
import csv


def process_mseed_file(mseed_file, event_time_data):
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
        return None, None  # Exit if no matching event is found

    pred1 = SeismicPrediction(tr, time)
    filtered_data = pred1.apply_bandpass_filter()
    decay_data = pred1.energy_decay(filtered_data)
    suppresed = pred1.staircase_data(decay_data, 1000)
    normalized = pred1.normalize(suppresed)
    results = [pred1.create_high_freq(normalized, -0.03)]
    final = pred1.predict(results)

    # Print prediction results for the current file
    print(f"Predictions for {os.path.basename(mseed_file)}: {final}")

    print(f"Processed {os.path.basename(mseed_file)}")
    return final, time, normalized, event_time


def main():
    # Set the directory containing MiniSEED files
    mseed_directory = '/home/ayden/nasa/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'
    
    # Load event time data from CSV
    event_time_data = pd.read_csv('space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv')
    
    output = []

    # Process each MiniSEED file in the directory
    for filename in os.listdir(mseed_directory):
        if filename.endswith(".mseed"):  # Ensure only MiniSEED files are processed
            mseed_file = os.path.join(mseed_directory, filename)

            # Process the file and get the results
            final, time, normalized, event_time = process_mseed_file(mseed_file, event_time_data)
            
            if final is None:
                continue  # Skip if no matching event found

            output.append([filename] + final)  # Save the filename and prediction results
            
            # Plotting the data
            plt.figure(figsize=(10, 6))
            plt.plot(time, normalized, color='blue', label='Energy Decay Rate')
            plt.xlabel('Time (s)')
            plt.ylabel('Decay Rate')
            plt.title(f'Energy Decay Over Time - {filename}')

            print(final)
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

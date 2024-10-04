from SeismicPrediction import SeismicPrediction
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read  # ObsPy for handling MiniSEED files
from obspy.signal.filter import bandpass  # Import bandpass filter from ObsPy
from scipy.signal import medfilt  # Import median filter
import matplotlib.pyplot as plt
import csv

    



def main():
    # Directory containing the MiniSEED files
    data_directory = '//fs-059/studuser$/Gr11/a.bravender/nasa/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA'

    # Get a sorted list of MiniSEED filenames
    mseed_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.mseed')])

    # Load event time data from CSV
    event_time_data = pd.read_csv('//fs-059/studuser$/Gr11/a.bravender/nasa/space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv')
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
        results = [pred1.create_high_freq(normalized, -0.024)]
        final = pred1.predict(results)
        output.append(final)
        
        print(results)
        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(time, normalized, color='blue', label='Energy Decay Rate')
        plt.xlabel(filename)
        plt.ylabel('Decay Rate')
        plt.title('Energy Decay Over Time')

        # Draw vertical lines for each event in 'final'
        for t in final:
            plt.axvline(x=t, color='k', linestyle='--', label=f"Event at {t:.2f}s")

        plt.axvline(x=event_time, color='r', linestyle='--', label="Catalog Event Time")  # If you still want the original event_time line
        plt.legend()
        plt.grid(True)  # Adds a grid for easier visualization

        plt.show()
    with open('output_seismic_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in output:
            writer.writerow(row)

    print(f"Results saved to 'output_seismic_predictions.csv'.")
    
if __name__ == "__main__":
    main()



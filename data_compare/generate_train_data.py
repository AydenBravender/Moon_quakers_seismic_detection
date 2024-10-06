from Moon_quakers_seismic_detection.final_predictions.code.SeismicPrediction import SeismicPrediction
from Moon_quakers_seismic_detection.final_predictions.code.RandomForestModel import RandomForestModel
import os
import numpy as np
import pandas as pd
from obspy import read  # ObsPy for handling MiniSEED files
import matplotlib.pyplot as plt
import csv

def main():
    # training_data = [[73374, 19516, 66.17392549039656, 2.4865298505464495e-18, 4.8527116563264506e-14], [12735, 19173, 39.14556222756553, 7.976772251135083e-19, 1.5293865437101295e-14], [73004, 19960, 44.04449368875055, 1.2337741438916345e-18, 2.4626131912077022e-14], 
    #                 [4525, 22606, 53.59326433219393, 1.672264464554961e-18, 3.780321048572945e-14], [52113, 29775, 29.21825849451307, 6.597095162820474e-19, 1.964285084729796e-14], [68342, 17719, 79.36209914844828, 8.6572173505819e-18, 1.5339723423496068e-13],
    #                 [72120, 20420, 105.46293977176782, 6.703629353398982e-17, 1.368881113964072e-12], [18380, 20507, 73.49346086608784, 1.214539665291525e-17, 2.4906564916133305e-13], [42189, 5497, 6.839583227007102, 2.7477787357006605e-18, 1.510453971014653e-14], 
    #                 [18380, 20507, 73.49346086608784, 1.214539665291525e-17, 2.4906564916133305e-13], [42189, 5497, 6.839583227007102, 2.7477787357006605e-18, 1.510453971014653e-14], [71860, 20672, 39.68131699586226, 8.3852386187417595e-19, 1.7333965272662965e-14],
    #                 [41422, 29794, 24.178898939574925, 4.77873179805441e-19, 1.423775351912331e-14], [62073, 2800, 25.62602731584809, 1.5059000468803565e-17, 4.2165201312649985e-14], [46291, 22610, 58.009107787533694, 1.3494611430908336e-18, 3.051131644528375e-14],
    #                 [26582, 20195, 45.058050723443756, 7.525183275254872e-19, 1.5197107624377213e-14], [73974, 17432, 41.54931788915631, 1.343001896545486e-18, 2.3411209060580912e-14], [56477, 15545, 38.995333338021545, 9.53058348795793e-19, 1.48152920320306e-14],
    #                 [45617, 23529, 53.917853706884124, 1.6210966651394173e-18, 3.814278343406535e-14], [53886, 10684, 21.695689421408353, 8.63279701131072e-19, 9.223280326884374e-15], [42937, 1314, 56.94184604234445, 7.092947791143017e-17, 9.320133397561925e-14], 
    #                 [65732, 25647, 25.948122061802174, 4.6032835781128535e-19, 1.1806041392786036e-14], [55838, 1438, 4.896261070952537, 5.424743715227665e-18, 7.800781462497382e-15], [13285, 26507, 41.05884165796437, 7.41663877544262e-19, 1.9659284402065753e-14], 
    #                 [54632, 29703, 23.43237081982058, 5.548347815611685e-19, 1.6480257516711388e-14], [36739, 23587, 54.186470228060024, 1.485351505641816e-17, 3.5034985963573515e-13], [25466, 14318, 137.26498028206862, 2.0546430966300598e-17, 2.94183798575492e-13], 
    #                 [35053, 25203, 45.92382102818788, 3.3119536382702797e-18, 8.347116754532586e-14], [29031, 9003, 23.529021480789037, 2.8108882186790652e-18, 2.5306426632767623e-14], [35053, 25203, 45.92382102818788, 3.3119536382702797e-18, 8.347116754532586e-14], 
    #                 [29031, 9003, 23.529021480789037, 2.8108882186790652e-18, 2.5306426632767623e-14], [8555, 51960, 18.37138325290266, 1.8622299427430846e-19, 9.676146782493068e-15], [80348, 15356, 91.81772690508954, 8.038523023367001e-16, 1.2343955954682366e-11], 
    #                 [39071, 20721, 89.47455497218877, 9.765767048054625e-16, 2.023564590027399e-11], [77626, 24190, 2.5733877052750516, 6.284533700950327e-20, 1.520228702259884e-15], [32102, 23081, 21.197617394046315, 2.4742375463039173e-19, 5.710787680624072e-15], 
    #                 [42047, 16239, 5.083808127259287, 9.574765434101282e-20, 1.5548461588437073e-15], [60141, 14420, 2.5600289623129413, 5.1011873002281005e-20, 7.355912086928921e-16], [12003, 24235, 34.53227023907869, 1.3746402991108257e-18, 3.331440764895086e-14], 
    #                 [55754, 1438, 51.272506022218586, 6.039285866322063e-17, 8.684493075771126e-14], [65278, 21192, 82.36876182861943, 7.387114977038297e-17, 1.5654774059339558e-12], [19777, 24604, 43.36335520976709, 1.7937341653060387e-18, 4.4133035403189775e-14], 
    #                 [15173, 24253, 27.44486578694629, 4.166819385176262e-19, 1.0105787054867988e-14], [23713, 20460, 52.82655300060642, 1.0266386386583283e-18, 2.1005026546949397e-14], [1587, 2064, 2.6294848638295254, 1.019153301502279e-19, 2.1035324143007038e-16], 
    #                 [64655, 17685, 32.97586737796324, 4.402704769437765e-19, 7.786183384750688e-15], [48830, 23971, 60.24342063799708, 3.1668149348705882e-18, 7.591172080378287e-14], [77396, 1171, 6.4131371180057695, 1.2376937190894045e-17, 1.4493393450536927e-14], 
    #                 [58244, 22375, 32.968024225954615, 6.406534194281304e-19, 1.433462025970442e-14], [79056, 23240, 71.79049901729995, 5.639178126019225e-17, 1.3105449964868677e-12], [28925, 1739, 2.7304950638915875, 1.936740723818587e-18, 3.367992118720523e-15], 
    #                 [79056, 23240, 71.79049901729995, 5.639178126019225e-17, 1.3105449964868677e-12], [28925, 1739, 2.7304950638915875, 1.936740723818587e-18, 3.367992118720523e-15], [8558, 21694, 39.47066311072109, 5.852173507538954e-19, 1.2695705207255008e-14],  
    #                 [65452, 21763, 55.85678869746913, 4.237428631638206e-18, 9.221915931034229e-14], [28352, 303069, 12.95570911011172, 8.909379957830716e-20, 2.7001568744397972e-14], [77409, 29572, 10.437523967905081, 2.4847699319812945e-19, 7.347961642855084e-15], 
    #                 [7439, 39734, 22.540539207253047, 2.149796647426919e-19, 8.54200199888612e-15], [16637, 38355, 2.833233896570098, 6.026933698703982e-20, 2.311630420137912e-15], [22577, 38007, 3.279655018076612, 5.630127346246764e-20, 2.1398425004880075e-15], 
    #                 [81255, 28569, 25.048422559418817, 1.6498448168359815e-19, 4.7134416572187155e-15], [75570, 19731, 2.5863963307082654, 5.2447089751131724e-20, 1.03483352787958e-15], [875, 14083, 2.6139557954791957, 4.097325953759903e-20, 5.770264140680071e-16], 
    #                 [65762, 13167, 3.787232164664552, 4.757486762489116e-20, 6.26418282016942e-16], [1892, 31420, 32.59482472466378, 1.7443527070317468e-19, 5.480756205493748e-15], [14975, 11599, 3.1332436569967945, 5.4191094435626813e-20, 6.285625043588354e-16], 
    #                 [78191, 8320, 4.965940254130458, 7.073476009940334e-20, 5.885132040270358e-16], [8865, 6451, 3.491758092208157, 4.963965758864754e-20, 3.2022543110436525e-16], [824, 49445, 3.0273626958082804, 6.898222213133678e-20, 3.4108259732839473e-15], 
    #                 [8329, 42064, 2.8017820615176805, 5.2675388770351466e-20, 2.215737553236064e-15], [25990, 21470, 16.05305789478061, 1.5243030207720754e-19, 3.2726785855976458e-15], [19545, 10672, 2.7959353729133074, 5.220679554805642e-20, 5.571509220888581e-16],
    #                 [40591, 10179, 2.7518489449435024, 4.239835431050937e-20, 4.3157284852667487e-16], [35372, 1780, 2.5488778625151225, 3.60418800959837e-20, 6.415454657085098e-17], [28995, 7336, 164.51546879898063, 2.971767309374168e-16, 2.1800884981568895e-12], 
    #                 [69738, 25098, 36.339932903932166, 9.194561986881199e-19, 2.3076511674674433e-14], [2580, 5434, 120.26298519978216, 1.1390372156152793e-18, 6.189528229653428e-15]] 
    
    # training_labels = [True, True, True, 
    #                 True, True, True, 
    #                 True, True, True, 
    #                 True, True, True,
    #                 True, False, True,
    #                 True, True, True,
    #                 True, True, False,
    #                 True, False, True,
    #                 True, True, True,
    #                 True, True, True, 
    #                 True, True, True,
    #                 True, False, True,
    #                 False, False, True,
    #                 False, True, True,
    #                 True, True, False,
    #                 True, True, False,
    #                 True, True, True,
    #                 True, True, True,
    #                 True, True, True,
    #                 True, False, False,
    #                 True, False, False,
    #                 False, True, False,
    #                 False, False, False,
    #                 False, True, False,
    #                 False, False, True,
    #                 True, True]
    

    # Directory containing the MiniSEED files
    data_directory = '/home/ayden/nasa/data/test'

    # Get a sorted list of MiniSEED filenames
    mseed_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.mseed')])

    # # Load event time data from CSV
    # event_time_data = pd.read_csv('space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv')

    # Prepare to store output results
    output = []
    output_to_csv = []

    # Process each MiniSEED file in order
    for filename in mseed_files:
        mseed_file = os.path.join(data_directory, filename)
        # Read the MiniSEED file using ObsPy
        st = read(mseed_file)
        tr = st[0]  # Assuming single trace per file
        time = np.arange(0, tr.stats.npts) * tr.stats.delta  # Create time array



        # model = RandomForestModel()
        # model.train_random_forest(training_data, training_labels)


        pred1 = SeismicPrediction(tr, time)
        filtered_data = pred1.apply_bandpass_filter()
        decay_data = pred1.energy_decay(filtered_data)
        suppressed = pred1.staircase_data(decay_data, 1000)
        normalized = pred1.normalize(suppressed)
        results = [pred1.create_high_freq(normalized, -0.008)]
        final = pred1.predict(results)

        # Initialize variable to hold found contents
        found_contents = []  
        

        # Loop through each value in final
        for val in final:
            for outer in results:  # First level of nesting
                for inner in outer:  # Second level of nesting
                    if inner[0] == val:  # Check if the element at index 0 matches val
                        found_contents.append(inner)  # Save the inner list that matches
                        break  # Exit inner loop if found
        
        # if len(found_contents) > 1:
        #     for i in range(len(found_contents)):
        #         result = model.predict(found_contents[i])
        #         if result == True:
        #             output_to_csv.append([filename, found_contents[i][0]])
        # else:
        #     output_to_csv.append([filename, found_contents[0][0]])
        print(found_contents)



        plt.figure(figsize=(10, 6))
        plt.plot(time, normalized, color='blue', label='Energy Decay Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('Decay Rate')
        plt.title(f'Energy Decay Over Time - {os.path.basename(mseed_file)}')

        # Draw vertical lines for each event in 'final'
        for t in found_contents:
            plt.axvline(x=t[0], color='k', linestyle='--', label=f"Event at {t[0]:.2f}s")

        # plt.axvline(x=event_time, color='r', linestyle='--', label="Catalog Event Time")  # Event time from the catalog
        plt.legend()
        plt.grid(True)

        plt.show()

        


if __name__ == "__main__":
    main()

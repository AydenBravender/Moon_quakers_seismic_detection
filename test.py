import numpy as np

decay_data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
threshold = 4

potential_quakes = []
curr_segment = []
for i in range(len(decay_data)):
    if decay_data[i] <= threshold:
        curr_segment.append(decay_data[i])
        try:
            if decay_data[i+1] > threshold or i == len(decay_data)-1:
                potential_quakes.append([i-len(curr_segment)+1, len(curr_segment)])
                curr_segment = []
        except IndexError:
            if i == len(decay_data)-1:
                potential_quakes.append([i-len(curr_segment)+1, len(curr_segment)])
                curr_segment = []

print(potential_quakes)

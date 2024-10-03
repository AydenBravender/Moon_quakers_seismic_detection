import numpy as np

# decay_data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
# threshold = 4

# potential_quakes = []
# curr_segment = []
# for i in range(len(decay_data)):
#     if decay_data[i] <= threshold:
#         curr_segment.append(decay_data[i])
#         try:
#             if decay_data[i+1] > threshold or i == len(decay_data)-1:
#                 potential_quakes.append([i-len(curr_segment)+1, len(curr_segment)])
#                 curr_segment = []
#         except IndexError:
#             if i == len(decay_data)-1:
#                 potential_quakes.append([i-len(curr_segment)+1, len(curr_segment)])
#                 curr_segment = []

# print(potential_quakes)
durations = [1, 2, 9, 4, 5, 6, 7, 8]
time = [1, 2, 3, 1, 6, 8, 1, 9]
new_list = []
final_result = []

for i in range(len(durations)):
    if durations[i] >= 2:
        new_list.append([durations[i], time[i]])
new_list = sorted(new_list, key=lambda x: x[0], reverse=True)
print(new_list)

final_result.append(new_list[0][1])
for i in range(len(new_list)):
    n = 0
    for j in range(len(final_result)):
        print(new_list[i][1], final_result[j])
        if abs(new_list[i][1] - final_result[j]) > 2:
            n+=1
    if n == len(final_result):
        final_result.append(new_list[i][1]) 

print(final_result)










sorted_durations = []

for i in range(len(durations)):
    if durations[i] >= 5:
        sorted_durations.append(durations[i])
sorted_durations.sort(reverse=True)



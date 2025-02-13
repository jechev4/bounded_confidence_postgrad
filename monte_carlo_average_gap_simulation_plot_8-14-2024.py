# **Monte Carlo Average Gap Simulation Plot - 8-14-2024**

import numpy as np
import matplotlib.pyplot as plt
import time

def Monte_Carlo_Gap_Probability(initial_vector_size, number_of_iterations):
    # Preallocate array to store maximum gaps
    maximum_gap_array = np.zeros(number_of_iterations)

    for each_iteration in range(number_of_iterations):
        # Generate and sort the opinion profile
        new_opinion_profile = np.sort(np.random.rand(initial_vector_size))

        # Calculate the gaps between consecutive elements
        gaps = np.diff(new_opinion_profile)

        # Find the maximum gap and store it
        maximum_gap_array[each_iteration] = np.max(gaps)

    return np.mean(maximum_gap_array)

def Run_Monte_Carlo_Simulations(start_size, end_size, step, number_of_iterations):
    # Preallocate array to store average maximum gaps for different opinion profile sizes
    final_gap_array = []

    for size in range(start_size, end_size + 1, step):
        average_max_gap = Monte_Carlo_Gap_Probability(size, number_of_iterations)
        final_gap_array.append((size, average_max_gap))
        print(f'Size: {size}, Average Max Gap: {average_max_gap:.3f}')

    return final_gap_array

number_of_simulations = 10000
start_size = 5
end_size = 1000
step = 5
start_time = time.time()

final_gap_array = Run_Monte_Carlo_Simulations(start_size, end_size, step, number_of_simulations)

# Extract sizes and average gaps for plotting
sizes = [size for size, avg_gap in final_gap_array]
average_gaps = [avg_gap for size, avg_gap in final_gap_array]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(sizes, average_gaps, marker='o', linestyle='-', color='b')
plt.title('Size of Opinion Profile vs Average Maximum Gap')
plt.xlabel('Size of Opinion Profile')
plt.ylabel('Average Maximum Gap')
plt.grid(True)
plt.show()

print('Time completed in:', "--- %s seconds ---" % (time.time() - start_time))

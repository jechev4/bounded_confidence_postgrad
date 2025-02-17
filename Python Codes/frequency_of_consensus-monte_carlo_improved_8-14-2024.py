# **Frequency Of Consensus - Monte Carlo Improved 8-14-2024**

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Efficiently calculates the bounded confidence opinion profile
def bounded_confidence(initial_vector, epsilon):
    new_opinion_profile = np.zeros_like(initial_vector)
    for i, opinion in enumerate(initial_vector):
        close_neighbors = np.abs(initial_vector - opinion) <= epsilon
        new_opinion_profile[i] = np.mean(initial_vector[close_neighbors])
    return new_opinion_profile

# Efficiently counts the number of fragmentations in the final array
def fragmentation_counter(final_array, tolerance):
    return np.sum(np.abs(final_array[0] - final_array) > tolerance)

# Monte Carlo simulation for bounded confidence model
def monte_carlo_simulation(number_of_agents, epsilon, tolerance):
    opinion_profile = np.random.rand(number_of_agents)
    norm_difference = tolerance + 1  # Start with a value higher than tolerance
    while norm_difference > tolerance:
        previous_profile = opinion_profile.copy()
        opinion_profile = bounded_confidence(opinion_profile, epsilon)
        norm_difference = np.linalg.norm(opinion_profile - previous_profile)

    final_array = bounded_confidence(opinion_profile, epsilon)
    number_of_fragmentations = fragmentation_counter(final_array, tolerance)

    # Return 0 for consensus and 1 for fragmentation
    return 0 if number_of_fragmentations == 0 else 1

# Main simulation loop over varying number of agents and epsilon values
start_time = time.time()
n = 1000  # Number of simulations per agent count and epsilon value
tol = 0.5 * 10**-5  # Tolerance

# Arrays to store the results for plotting
agent_sizes = []
epsilon_values = []
consensus_frequencies = []

# Testing different epsilon values from 0.02 to 0.4 in increments of 0.02
for e in np.arange(0.02, 0.42, 0.02):
    for num_of_agents in range(5, 105, 5):
        results = np.array([monte_carlo_simulation(num_of_agents, e, tol) for _ in range(n)])
        percent_of_consensus = np.mean(results == 0)

        # Store the data for plotting
        agent_sizes.append(num_of_agents)
        epsilon_values.append(e)
        consensus_frequencies.append(percent_of_consensus)

# Convert lists to arrays and reshape for 3D plotting
agent_sizes = np.array(agent_sizes).reshape(len(np.arange(0.02, 0.42, 0.02)), len(range(5, 105, 5)))
epsilon_values = np.array(epsilon_values).reshape(len(np.arange(0.02, 0.42, 0.02)), len(range(5, 105, 5)))
consensus_frequencies = np.array(consensus_frequencies).reshape(len(np.arange(0.02, 0.42, 0.02)), len(range(5, 105, 5)))

# Plotting the 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the 3D surface plot
ax.plot_surface(agent_sizes, epsilon_values, consensus_frequencies, cmap='viridis')

# Set labels
ax.set_xlabel('Number of Agents')
ax.set_ylabel('Epsilon')
ax.set_zlabel('Frequency of Consensus')

# Show plot
plt.title('3D Surface Plot of Consensus Frequency vs. Number of Agents and Epsilon')
plt.show()

print(f"\n--- Total Time: {time.time() - start_time} seconds ---")

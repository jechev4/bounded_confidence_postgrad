# **Monte Carlo Simulations - Normally Distributed Epsilon 9-22-2024**

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.stats as stats

# Efficiently calculates the bounded confidence opinion profile with individual epsilon values
def bounded_confidence(initial_vector, epsilon_array):
    new_opinion_profile = np.zeros_like(initial_vector)
    for i, opinion in enumerate(initial_vector):
        # Use individual epsilon for each agent
        epsilon = epsilon_array[i]
        close_neighbors = np.abs(initial_vector - opinion) <= epsilon
        if np.sum(close_neighbors) > 1:  # Ensure there are multiple close neighbors
            new_opinion_profile[i] = np.mean(initial_vector[close_neighbors])
        else:
            new_opinion_profile[i] = opinion  # No update if no close neighbors
    return new_opinion_profile

# Improved fragmentation detection logic with enhanced precision considerations
def is_fragmented(opinion_profile, tolerance=1e-9):
    for i in range(1, len(opinion_profile)):
        if not np.isclose(opinion_profile[i], opinion_profile[i - 1], atol=tolerance):
            return 1  # Fragmentation if any two elements are not within the tolerance
    # Additional check to ensure we haven't missed a group formation
    if np.max(opinion_profile) - np.min(opinion_profile) > tolerance:
        return 1  # Fragmentation if there's a significant spread in opinions
    return 0  # Consensus if all elements are close enough

# Updated Monte Carlo simulation function with individual epsilon values
def monte_carlo_simulation(number_of_agents, base_epsilon, epsilon_stddev, tolerance, rng_function):
    if rng_function == np.random.rand:
        opinion_profile = rng_function(number_of_agents)  # No 'size' keyword for np.random.rand
    else:
        opinion_profile = rng_function(size=number_of_agents)  # Use 'size' for scipy.stats functions

    # Ensure values are within [0, 1]
    opinion_profile = np.clip(opinion_profile, 0, 1)

    # Generate an array of epsilon values from a normal distribution
    epsilon_array = np.random.normal(loc=base_epsilon, scale=epsilon_stddev, size=number_of_agents)
    epsilon_array = np.clip(epsilon_array, 0, 1)  # Ensure epsilon values are within [0, 1]

    norm_difference = tolerance + 1  # Start with a value higher than tolerance
    while norm_difference > tolerance:
        previous_profile = opinion_profile.copy()
        opinion_profile = bounded_confidence(opinion_profile, epsilon_array)
        norm_difference = np.linalg.norm(opinion_profile - previous_profile)
        if norm_difference < tolerance:
            break

    # Use the improved fragmentation detection
    return is_fragmented(opinion_profile, tolerance)

# Main simulation loop over varying number of agents and epsilon values
start_time = time.time()
n = 100  # Number of repetitions for each simulation
tol = 0.5 * 10**-5  # Tolerance
min_agents = 5  # Minimum number of agents for the simulation  NOTE: CANNOT BE ZERO
max_agents = 500  # Maximum number of agents for the simulation (user can modify this)
agent_step = 5  # Incremental step for the number of agents (user can modify this)
min_epsilon = 0.02  # Minimum base value of epsilon (user can modify this)
max_epsilon = 0.4  # Maximum base value of epsilon (user cAAan modify this)
epsilon_step = 0.02  # Incremental step for epsilon (user can modify this)
epsilon_stddev = 0.05  # Standard deviation for the normal distribution of epsilon

# Parameters for the bimodal distribution (user can modify these)
mean1 = 0.3  # Mean of the first normal distribution
std1 = 0.1   # Standard deviation of the first normal distribution
mean2 = 0.7  # Mean of the second normal distribution
std2 = 0.1   # Standard deviation of the second normal distribution
mix_ratio = 0.5  # Mixing ratio between the two distributions (0.5 means equal mix)

# Arrays to store the results for plotting
agent_sizes = []
epsilon_values = []
consensus_frequencies = []

# Dictionary of random number generator functions
rng_functions = {
    'uniform': np.random.rand,
    'normal': lambda size: np.clip(stats.norm.rvs(loc=0.5, scale=0.15, size=size), 0, 1),  # Truncated normal
    'beta': lambda size: stats.beta.rvs(2, 5, size=size),  # Beta distribution
    'exponential': lambda size: np.clip(stats.expon.rvs(scale=0.5, size=size), 0, 1),  # Exponential distribution scaled
    'gamma': lambda size: np.clip(stats.gamma.rvs(2, size=size), 0, 1),  # Gamma distribution scaled
    'poisson': lambda size: np.clip(stats.poisson.rvs(mu=3, size=size) / 10, 0, 1),  # Poisson distribution scaled
    'bimodal': lambda size: np.clip(np.where(np.random.rand(size) < mix_ratio,
                                             stats.norm.rvs(loc=mean1, scale=std1, size=size),
                                             stats.norm.rvs(loc=mean2, scale=std2, size=size)), 0, 1)  # Bimodal distribution
}

# User selects the desired random number generator
selected_rng = 'bimodal'  # Change this to other keys like 'normal', 'beta', etc.

# Testing different epsilon values from min_epsilon to max_epsilon in increments of epsilon_step
for e in np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step):
    # Testing different agent sizes from min_agents to max_agents in increments of agent_step
    for num_of_agents in range(min_agents, max_agents + agent_step, agent_step):
        repetition_results = []
        for _ in range(n):  # Repeat the experiment n times
            result = monte_carlo_simulation(num_of_agents, e, epsilon_stddev, tol, rng_functions[selected_rng])
            repetition_results.append(result)

        # Calculate the average consensus frequency over n repetitions
        percent_of_consensus = np.sum(np.array(repetition_results) == 0) / len(repetition_results)

        # Store the data for plotting
        agent_sizes.append(num_of_agents)
        epsilon_values.append(e)
        consensus_frequencies.append(percent_of_consensus)

# Convert lists to arrays and reshape for 3D plotting
agent_sizes = np.array(agent_sizes).reshape(len(np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step)), len(range(min_agents, max_agents + agent_step, agent_step)))
epsilon_values = np.array(epsilon_values).reshape(len(np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step)), len(range(min_agents, max_agents + agent_step, agent_step)))
consensus_frequencies = np.array(consensus_frequencies).reshape(len(np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step)), len(range(min_agents, max_agents + agent_step, agent_step)))

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
plt.title('3D Surface Plot of Consensus Frequency vs. Number of Agents and Epsilon - Bimodal Distribution')
plt.show()

print(f"\n--- Total Time: {time.time() - start_time} seconds ---")

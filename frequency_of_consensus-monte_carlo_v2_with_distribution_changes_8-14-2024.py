# **Frequency Of Consensus - Monte Carlo V2 With Distribution Changes 8-14-2024**

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.stats as stats

# Efficiently calculates the bounded confidence opinion profile
def bounded_confidence(initial_vector, epsilon):
    new_opinion_profile = np.zeros_like(initial_vector)
    for i, opinion in enumerate(initial_vector):
        # Find close neighbors within epsilon, but also ensure the number is reasonable
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

# Updated Monte Carlo simulation function with floating-point precision considerations
def monte_carlo_simulation(number_of_agents, epsilon, tolerance, rng_function):
    if rng_function == np.random.rand:
        opinion_profile = rng_function(number_of_agents)  # No 'size' keyword for np.random.rand
    else:
        opinion_profile = rng_function(size=number_of_agents)  # Use 'size' for scipy.stats functions

    # Ensure values are within [0, 1]
    opinion_profile = np.clip(opinion_profile, 0, 1)

    norm_difference = tolerance + 1  # Start with a value higher than tolerance
    #iteration = 0
    while norm_difference > tolerance:
        #prints are debugging purposes and to check that the bounded confidence calculations are averaging correctly
        #print(f"\nIteration {iteration}:")
        #print(f"\nEpsilon {epsilon}:")
        #print(f"\nNumber of Agents {number_of_agents}:")
        #print(f"Original Opinion Profile: {opinion_profile}")
        previous_profile = opinion_profile.copy()
        opinion_profile = bounded_confidence(opinion_profile, epsilon)
        norm_difference = np.linalg.norm(opinion_profile - previous_profile)
        #print(f"Norm Difference: {norm_difference}")
        #print(f"Updated Opinion Profile: {opinion_profile}")
        #iteration += 1
        if norm_difference < tolerance:
            break

    # Use the improved fragmentation detection
    return is_fragmented(opinion_profile, tolerance)

# Main simulation loop over varying number of agents and epsilon values
start_time = time.time()
n = 10000  # Number of repetitions for each simulation
tol = 0.5 * 10**-5  # Tolerance
min_agents = 10  # Minimum number of agents for the simulation  NOTE: CANNOT BE ZERO
max_agents = 1000  # Maximum number of agents for the simulation (user can modify this)
agent_step = 10  # Incremental step for the number of agents (user can modify this)
min_epsilon = 0.02  # Minimum value of epsilon (user can modify this)
max_epsilon = 0.4  # Maximum value of epsilon (user can modify this)
epsilon_step = 0.02  # Incremental step for epsilon (user can modify this)

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
}

# User selects the desired random number generator
selected_rng = 'gamma'  # Change this to other keys like 'normal', 'beta', etc.

# Testing different epsilon values from min_epsilon to max_epsilon in increments of epsilon_step
for e in np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step):
    # Testing different agent sizes from min_agents to max_agents in increments of agent_step
    for num_of_agents in range(min_agents, max_agents + agent_step, agent_step):
        repetition_results = []
        for _ in range(n):  # Repeat the experiment n times
            result = monte_carlo_simulation(num_of_agents, e, tol, rng_functions[selected_rng])
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
plt.title('3D Surface Plot of Consensus Frequency vs. Number of Agents and Epsilon - Gamma Distribution')
plt.show()

print(f"\n--- Total Time: {time.time() - start_time} seconds ---")


# Notes on Modifications to Distributions for Bounded Confidence Model

# 1. Uniform Distribution:
# The uniform distribution generated by np.random.rand already produces values between 0 and 1.
# No modifications were necessary.

# 2. Normal Distribution:
# A normal distribution can generate values across the entire real number line, so to constrain it to [0, 1],
# we used a truncated normal distribution centered at 0.5 with a standard deviation of 0.15.
# The values were then clipped to ensure they stay within the [0, 1] range.

# Example modification:
# lambda size: np.clip(stats.norm.rvs(loc=0.5, scale=0.15, size=size), 0, 1)

# 3. Beta Distribution:
# The beta distribution inherently produces values in the range [0, 1]. It is defined by two shape parameters,
# which control the skewness of the distribution. For our purposes, no additional modifications were needed.

# Example:
# lambda size: stats.beta.rvs(2, 5, size=size)

# 4. Exponential Distribution:
# The exponential distribution typically generates values from 0 to infinity. To fit it within [0, 1],
# the distribution was scaled and then clipped. Scaling involves adjusting the rate parameter (1/scale),
# and clipping ensures that any values exceeding 1 are truncated.

# Example modification:
# lambda size: np.clip(stats.expon.rvs(scale=0.5, size=size), 0, 1)

# 5. Gamma Distribution:
# The gamma distribution, like the exponential, can produce values that exceed 1.
# To fit it within the [0, 1] range, we scaled the distribution and then applied clipping.

# Example modification:
# lambda size: np.clip(stats.gamma.rvs(2, size=size), 0, 1)

# General Note:
# The use of `np.clip` is essential in these modifications to enforce the boundary [0, 1].
# Without clipping, values could fall outside this range, which would violate the assumptions
# of the bounded confidence model that opinions are represented as numbers between 0 and 1.





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.stats as stats

# Efficiently calculates the bounded confidence opinion profile
def bounded_confidence(initial_vector, epsilon):
    new_opinion_profile = np.zeros_like(initial_vector)
    for i, opinion in enumerate(initial_vector):
        # Find close neighbors within epsilon, but also ensure the number is reasonable
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

# Updated Monte Carlo simulation function with floating-point precision considerations
def monte_carlo_simulation(number_of_agents, epsilon, tolerance, rng_function):
    if rng_function == np.random.rand:
        opinion_profile = rng_function(number_of_agents)  # No 'size' keyword for np.random.rand
    else:
        opinion_profile = rng_function(size=number_of_agents)  # Use 'size' for scipy.stats functions

    # Ensure values are within [0, 1]
    opinion_profile = np.clip(opinion_profile, 0, 1)

    norm_difference = tolerance + 1  # Start with a value higher than tolerance
    while norm_difference > tolerance:
        previous_profile = opinion_profile.copy()
        opinion_profile = bounded_confidence(opinion_profile, epsilon)
        norm_difference = np.linalg.norm(opinion_profile - previous_profile)
        if norm_difference < tolerance:
            break

    # Use the improved fragmentation detection
    return is_fragmented(opinion_profile, tolerance)

# Main simulation loop over varying number of agents and epsilon values
start_time = time.time()
n = 1000  # Number of repetitions for each simulation
tol = 0.5 * 10**-5  # Tolerance
min_agents = 5  # Minimum number of agents for the simulation  NOTE: CANNOT BE ZERO
max_agents = 1000  # Maximum number of agents for the simulation (user can modify this)
agent_step = 5  # Incremental step for the number of agents (user can modify this)
min_epsilon = 0.02  # Minimum value of epsilon (user can modify this)
max_epsilon = 0.4  # Maximum value of epsilon (user can modify this)
epsilon_step = 0.02  # Incremental step for epsilon (user can modify this)

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


""" TO CHECK THE UNDERLYING DISTRIBUTION, USE BELOW PLOT:
# Example parameters for the bimodal distribution
mean1 = 0.2
std1 = 0.1
mean2 = 0.8
std2 = 0.1
mix_ratio = 0.5

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

# Generate a sample of 10,000 points using the selected distribution
sample_size = 10000
sample = rng_functions[selected_rng](sample_size)

# Plot the histogram of the generated sample
plt.figure(figsize=(8, 6))
plt.hist(sample, bins=50, density=True, alpha=0.6, color='g')
plt.title(f'Histogram of {selected_rng.capitalize()} Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()
"""


# Testing different epsilon values from min_epsilon to max_epsilon in increments of epsilon_step
for e in np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step):
    # Testing different agent sizes from min_agents to max_agents in increments of agent_step
    for num_of_agents in range(min_agents, max_agents + agent_step, agent_step):
        repetition_results = []
        for _ in range(n):  # Repeat the experiment n times
            result = monte_carlo_simulation(num_of_agents, e, tol, rng_functions[selected_rng])
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


# Notes on Modifications to Distributions for Bounded Confidence Model

# 1. Uniform Distribution:
# The uniform distribution generated by np.random.rand already produces values between 0 and 1.
# No modifications were necessary.

# 2. Normal Distribution:
# A normal distribution can generate values across the entire real number line, so to constrain it to [0, 1],
# we used a truncated normal distribution centered at 0.5 with a standard deviation of 0.15.
# The values were then clipped to ensure they stay within the [0, 1] range.

# Example modification:
# lambda size: np.clip(stats.norm.rvs(loc=0.5, scale=0.15, size=size), 0, 1)

# 3. Beta Distribution:
# The beta distribution inherently produces values in the range [0, 1]. It is defined by two shape parameters,
# which control the skewness of the distribution. For our purposes, no additional modifications were needed.

# Example:
# lambda size: stats.beta.rvs(2, 5, size=size)

# 4. Exponential Distribution:
# The exponential distribution typically generates values from 0 to infinity. To fit it within [0, 1],
# the distribution was scaled and then clipped. Scaling involves adjusting the rate parameter (1/scale),
# and clipping ensures that any values exceeding 1 are truncated.

# Example modification:
# lambda size: np.clip(stats.expon.rvs(scale=0.5, size=size), 0, 1)

# 5. Gamma Distribution:
# The gamma distribution, like the exponential, can produce values that exceed 1.
# To fit it within the [0, 1] range, we scaled the distribution and then applied clipping.

# Example modification:
# lambda size: np.clip(stats.gamma.rvs(2, size=size), 0, 1)

# 6. Poisson Distribution:
# The Poisson distribution generates integer values, which are non-negative. However, these values
# can exceed 1 when scaled to the range [0, 1]. To fit within this range, we scaled the Poisson distribution
# by dividing by a factor (in this case, 10), and then clipped the values to ensure they remain within [0, 1].

# Example modification:
# lambda size: np.clip(stats.poisson.rvs(mu=3, size=size) / 10, 0, 1)

# 7. Bimodal Distribution:
# The bimodal distribution is a combination of two normal distributions. We allowed the user to define
# the mean and standard deviation for each of the two distributions, as well as the mixing ratio.
# The mixing ratio determines the proportion of the total samples that come from each normal distribution.
# After generating the values from the two distributions, the values were clipped to fit within [0, 1].

# Example modification:
# lambda size: np.clip(np.where(np.random.rand(size) < mix_ratio,
#                                stats.norm.rvs(loc=mean1, scale=std1, size=size),
#                                stats.norm.rvs(loc=mean2, scale=std2, size=size)), 0, 1)

# General Note:
# The use of `np.clip` is essential in these modifications to enforce the boundary [0, 1].
# Without clipping, values could fall outside this range, which would violate the assumptions
# of the bounded confidence model that opinions are represented as numbers between 0 and 1.

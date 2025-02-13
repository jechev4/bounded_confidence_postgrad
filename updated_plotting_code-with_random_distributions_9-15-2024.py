# **Updated Plotting Code - With Random Distributions 9-15-2024**

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define parameters for bimodal distribution
mean1, std1 = 0.3, 0.1
mean2, std2 = 0.7, 0.1
mix_ratio = 0.5  # 50% chance of selecting from either distribution

# Dictionary of random number generators for different distributions
rng_functions = {
    'uniform': np.random.rand,
    'normal': lambda size: np.clip(stats.norm.rvs(loc=0.5, scale=0.15, size=size), 0, 1),  # Truncated normal
    'beta': lambda size: stats.beta.rvs(2, 5, size=size),  # Beta distribution
    'exponential': lambda size: np.clip(stats.expon.rvs(scale=0.5, size=size), 0, 1),  # Exponential distribution
    'gamma': lambda size: np.clip(stats.gamma.rvs(2, scale=0.2, size=size), 0, 1),  # Adjusted Gamma distribution scaling
    'poisson': lambda size: np.clip(stats.poisson.rvs(mu=3, size=size) / 10, 0, 1),  # Adjusted Poisson distribution scaling
    'bimodal': lambda size: np.clip(np.where(np.random.rand(size) < mix_ratio,
                                             stats.norm.rvs(loc=mean1, scale=std1, size=size),
                                             stats.norm.rvs(loc=mean2, scale=std2, size=size)), 0, 1),  # Bimodal distribution
    'evenly_spaced': lambda size: np.linspace(0, 1, size)  # Evenly spaced values between 0 and 1
}

# Optimized bounded confidence function using vectorized operations
def bounded_confidence(initial_vector, epsilon):
    new_opinion_profile = np.zeros_like(initial_vector)
    size = initial_vector.size
    for i in range(size):
        distances = np.abs(initial_vector - initial_vector[i])
        close_neighbors = distances <= epsilon
        new_opinion_profile[i] = np.mean(initial_vector[close_neighbors])
    return new_opinion_profile

# Main simulation loop to reach consensus using different distributions
def run_simulation(opinion_profile_length, distribution_name='uniform', epsilon=0.24, tolerance=0.5*10**-4, max_iterations=1000):
    if distribution_name not in rng_functions:
        raise ValueError(f"Distribution '{distribution_name}' is not supported. Choose from {list(rng_functions.keys())}")

    # Generate the initial opinion profile using the selected distribution
    opinion_profile = rng_functions[distribution_name](opinion_profile_length)

    #print(f'Starting Opinion Profile ({distribution_name} distribution): \n', opinion_profile)

    # Preallocate a large enough array to hold the opinion profile at each step
    combined_array = np.zeros((max_iterations, opinion_profile.size))

    norm_difference = 1
    n = 0

    while tolerance < norm_difference and n < max_iterations:
        # Compute the new opinion profile
        previous_array = opinion_profile.copy()
        opinion_profile = bounded_confidence(opinion_profile, epsilon)

        # Compute the norm difference to check for consensus
        norm_difference = np.linalg.norm(opinion_profile - previous_array)

        # Store the opinion profile for this iteration
        combined_array[n, :] = opinion_profile
        n += 1

    # Slice combined_array to only include the iterations that actually occurred
    combined_array = combined_array[:n, :]

    print(f'\nTolerance: {tolerance}')
    print(f'\nNumber of Iterations needed for consensus (within tolerance): {n}')

    # Plot the results
    new_opinion_matrix = combined_array.T  # Transpose the matrix for plotting

    # Plot each agent's opinion over time
    for agent in range(new_opinion_matrix.shape[0]):
        plt.plot(np.arange(n), new_opinion_matrix[agent])

    plt.title(f'Opinions over Time ({distribution_name} distribution)')
    plt.xlabel('Time Step')
    plt.ylabel(f'Agent Opinion ({opinion_profile_length})')
    plt.show()


# User inputs for opinion profile length, epsilon, and tolerance
opinion_profile_length = 50
epsilon = 0.12
tolerance = 0.5 * 10 ** -10

# User input for the distribution: select from one of the random distributions or 'evenly_spaced'
distribution_name = 'beta'  # Can be 'uniform', 'normal', 'beta', 'exponential', 'gamma', 'poisson', 'bimodal', or 'evenly_spaced'

# Run the simulation with user-provided values
run_simulation(opinion_profile_length, distribution_name=distribution_name, epsilon=epsilon, tolerance=tolerance)

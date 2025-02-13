# **Highlight Bifurcation Region Monte Carlo Simulations Modification 9-29-2024**

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.stats as stats
from sklearn.cluster import KMeans

# Efficiently calculates the bounded confidence opinion profile with individual epsilon values
def bounded_confidence(initial_vector, epsilon_array):
    new_opinion_profile = np.zeros_like(initial_vector)
    for i, opinion in enumerate(initial_vector):
        epsilon = epsilon_array[i]
        close_neighbors = np.abs(initial_vector - opinion) <= epsilon
        if np.sum(close_neighbors) > 1:
            new_opinion_profile[i] = np.mean(initial_vector[close_neighbors])
        else:
            new_opinion_profile[i] = opinion
    return new_opinion_profile

# Detects the number of distinct groups in the final opinion profile.
def detect_groups(opinion_profile, tolerance=1e-9):
    """Detects the number of distinct groups in the final opinion profile."""
    num_agents = len(opinion_profile)
    max_clusters = min(10, num_agents)  # Ensure max_clusters is not greater than number of agents

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(opinion_profile.reshape(-1, 1))
        labels = kmeans.labels_
        unique_clusters = np.unique(labels)
        cluster_means = kmeans.cluster_centers_.flatten()

        # Check if the number of clusters and their means are distinct enough
        distinct_clusters = 0
        for i in range(len(cluster_means)):
            for j in range(i + 1, len(cluster_means)):
                if np.abs(cluster_means[i] - cluster_means[j]) > tolerance:
                    distinct_clusters += 1

        # If there are exactly two distinct clusters, it's a bifurcation
        if len(unique_clusters) == 2 and distinct_clusters == 1:
            return 2  # Two distinct groups (bifurcation)

        # If there are more than two distinct clusters, it's fragmentation
        if len(unique_clusters) > 2 and distinct_clusters > 1:
            return len(unique_clusters)  # More than two distinct groups (fragmentation)

    # If no distinct groups were found, it's consensus (all opinions converge)
    return 0  # Consensus

# Updated Monte Carlo simulation function to return number of groups
def monte_carlo_simulation(number_of_agents, base_epsilon, epsilon_stddev, tolerance, rng_function):
    if rng_function == np.random.rand:
        opinion_profile = rng_function(number_of_agents)
    else:
        opinion_profile = rng_function(size=number_of_agents)

    opinion_profile = np.clip(opinion_profile, 0, 1)
    epsilon_array = np.random.normal(loc=base_epsilon, scale=epsilon_stddev, size=number_of_agents)
    epsilon_array = np.clip(epsilon_array, 0, 1)

    norm_difference = tolerance + 1
    while norm_difference > tolerance:
        previous_profile = opinion_profile.copy()
        opinion_profile = bounded_confidence(opinion_profile, epsilon_array)
        norm_difference = np.linalg.norm(opinion_profile - previous_profile)
        if norm_difference < tolerance:
            break
    return detect_groups(opinion_profile, tolerance)

# Main simulation loop over varying number of agents and epsilon values
start_time = time.time()
n = 100  # Number of repetitions for each simulation
tol = 0.5 * 10**-5  # Tolerance
min_agents = 5
max_agents = 100
agent_step = 5
min_epsilon = 0.02
max_epsilon = 0.4
epsilon_step = 0.02
epsilon_stddev = 0.05

# Parameters for the bimodal distribution
mean1 = 0.3
std1 = 0.1
mean2 = 0.7
std2 = 0.1
mix_ratio = 0.5

agent_sizes = []
epsilon_values = []
consensus_frequencies = []
two_group_frequencies = []
fragmentation_frequencies = []

# Random number generator dictionary
rng_functions = {
    'uniform': np.random.rand,
    'normal': lambda size: np.clip(stats.norm.rvs(loc=0.5, scale=0.15, size=size), 0, 1),
    'beta': lambda size: stats.beta.rvs(2, 5, size=size),
    'exponential': lambda size: np.clip(stats.expon.rvs(scale=0.5, size=size), 0, 1),
    'gamma': lambda size: np.clip(stats.gamma.rvs(2, size=size), 0, 1),
    'poisson': lambda size: np.clip(stats.poisson.rvs(mu=3, size=size) / 10, 0, 1),
    'bimodal': lambda size: np.clip(np.where(np.random.rand(size) < mix_ratio,
                                             stats.norm.rvs(loc=mean1, scale=std1, size=size),
                                             stats.norm.rvs(loc=mean2, scale=std2, size=size)), 0, 1)
}

# User selects the desired random number generator
selected_rng = 'bimodal'

# Testing different epsilon values from min_epsilon to max_epsilon in increments of epsilon_step
robust_consensus_ranges = []
robust_bifurcation_ranges = []
robust_fragmentation_ranges = []

for e in np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step):
    consensus_sum = 0
    bifurcation_sum = 0
    fragmentation_sum = 0
    for num_of_agents in range(min_agents, max_agents + agent_step, agent_step):
        repetition_results = []
        for _ in range(n):
            result = monte_carlo_simulation(num_of_agents, e, epsilon_stddev, tol, rng_functions[selected_rng])
            repetition_results.append(result)

        # Calculate the frequency of consensus, two groups, and fragmentation
        percent_of_consensus = np.sum(np.array(repetition_results) == 0) / len(repetition_results)
        percent_of_two_groups = np.sum(np.array(repetition_results) == 2) / len(repetition_results)
        percent_of_fragmentation = np.sum(np.array(repetition_results) > 2) / len(repetition_results)

        consensus_sum += percent_of_consensus
        bifurcation_sum += percent_of_two_groups
        fragmentation_sum += percent_of_fragmentation

        # Store the data for plotting
        agent_sizes.append(num_of_agents)
        epsilon_values.append(e)
        consensus_frequencies.append(percent_of_consensus)
        two_group_frequencies.append(percent_of_two_groups)
        fragmentation_frequencies.append(percent_of_fragmentation)

    # Determine robust regions based on 90% rule
    if consensus_sum / len(range(min_agents, max_agents + agent_step, agent_step)) >= 0.9:
        robust_consensus_ranges.append(e)
    if bifurcation_sum / len(range(min_agents, max_agents + agent_step, agent_step)) >= 0.9:
        robust_bifurcation_ranges.append(e)
    if fragmentation_sum / len(range(min_agents, max_agents + agent_step, agent_step)) >= 0.9:
        robust_fragmentation_ranges.append(e)

# Convert lists to arrays and reshape for 3D plotting
agent_sizes = np.array(agent_sizes).reshape(len(np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step)),
                                            len(range(min_agents, max_agents + agent_step, agent_step)))
epsilon_values = np.array(epsilon_values).reshape(len(np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step)),
                                                  len(range(min_agents, max_agents + agent_step, agent_step)))
consensus_frequencies = np.array(consensus_frequencies).reshape(len(np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step)),
                                                                len(range(min_agents, max_agents + agent_step, agent_step)))
two_group_frequencies = np.array(two_group_frequencies).reshape(len(np.arange(min_epsilon, max_epsilon + epsilon_step, epsilon_step)),
                                                                len(range(min_agents, max_agents + agent_step, agent_step)))

# Plotting the 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the 3D surface plot for consensus
surface = ax.plot_surface(agent_sizes, epsilon_values, consensus_frequencies, cmap='viridis', alpha=0.7)

# Highlight regions where bifurcation happens 90% of the time
highlight_mask = two_group_frequencies >= 0.9  # Mask for bifurcation areas with 90% frequency

# Using a scatter plot to highlight bifurcation areas with red dots
ax.scatter(agent_sizes[highlight_mask], epsilon_values[highlight_mask],
           consensus_frequencies[highlight_mask], color='red', label="Bifurcation (>=90%)", s=50)

# Set labels
ax.set_xlabel('Number of Agents')
ax.set_ylabel('Epsilon')
ax.set_zlabel('Frequency of Consensus')

# Add a color bar for the consensus surface
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

# Show plot
plt.title('3D Surface Plot of Consensus with Bifurcation Areas Highlighted')
plt.legend()
plt.show()

# Display the ranges of robust consensus, bifurcation, and fragmentation
print(f"Robust consensus (>= 90%) occurs for epsilon > {robust_consensus_ranges[-1] if robust_consensus_ranges else 'N/A'}")
print(f"Robust bifurcation (>= 90%) occurs for epsilon in range {robust_bifurcation_ranges}")
print(f"Robust fragmentation (>= 90%) occurs for epsilon <= {robust_fragmentation_ranges[0] if robust_fragmentation_ranges else 'N/A'}")

print(f"\n--- Total Time: {time.time() - start_time} seconds ---")
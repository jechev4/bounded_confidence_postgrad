# **Histogram Distribution Check - 9-1-2024**

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Example parameters for the bimodal distribution
mean1 = 0.4
std1 = 0.1
mean2 = 0.60
std2 = 0.1
mix_ratio = 0.5

# Dictionary of random number generator functions
rng_functions = {
    'uniform': np.random.rand,
    'normal': lambda size: np.clip(stats.norm.rvs(loc=0.5, scale=0.15, size=size), 0, 1),  # Truncated normal
    'beta': lambda size: stats.beta.rvs(2, 5, size=size),  # Beta distribution
    'exponential': lambda size: np.clip(stats.expon.rvs(scale=0.5, size=size), 0, 1),  # Exponential distribution scaled
    'gamma': lambda size: np.clip(stats.gamma.rvs(2, scale=0.2, size=size), 0, 1),  # Adjusted Gamma distribution scaling
    'poisson': lambda size: np.clip(stats.poisson.rvs(mu=3, size=size) / 10, 0, 1),  # Adjusted Poisson distribution scaling   #current issue with poisson is that the scale makes the values 0.1, 0.2, 0.3, etc.
    'bimodal': lambda size: np.clip(np.where(np.random.rand(size) < mix_ratio,
                                             stats.norm.rvs(loc=mean1, scale=std1, size=size),
                                             stats.norm.rvs(loc=mean2, scale=std2, size=size)), 0, 1)  # Bimodal distribution
}

# User selects the desired random number generator
selected_rng = 'beta'  # Change this to other keys like 'normal', 'beta', etc.

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

# Note: To comment out this debugging code, simply add triple quotes around it:

#Poisson distribution
#Models the probability of a given number of events occurring in a fixed time interval. It's used to model rates of occurrence, such as the number of telephone calls per minute or the number of errors per page in a document.

#Beta distribution
#Describes the fraction of a time interval until the first event occurs in a Poisson process. The shape of the beta distribution can be specified using two positive values, alpha (α) and beta (β). The distribution is symmetrical if the parameters are equal, and positively or negatively skewed depending on whether alpha is less than or greater than beta.


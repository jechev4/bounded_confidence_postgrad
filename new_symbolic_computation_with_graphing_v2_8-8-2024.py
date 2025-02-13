# **Below Is New Symbolic Computation With Graphing? V2 8-8-2024**

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import Rational

def bounded_confidence_symbolic(initial_values, l, time_steps):
    n = sp.symbols('n')  # Define n as a symbolic variable
    epsilon = l / n
    history = [initial_values.copy()]

    for t in range(time_steps):
        new_values = initial_values.copy()
        for i in range(len(initial_values)):
            neighbors = []
            for j in range(len(initial_values)):
                diff = sp.Abs(initial_values[i] - initial_values[j])
                if sp.simplify(diff <= epsilon):
                    neighbors.append(initial_values[j])
            if neighbors:
                new_values[i] = sp.simplify(sp.Rational(sum(neighbors), len(neighbors)))
            else:
                new_values[i] = initial_values[i]  # Keep the original value if no neighbors are found
        history.append(new_values)
        initial_values = new_values

    return history

def bounded_confidence_numeric(initial_vector, epsilon):
    new_opinion_profile = np.empty(initial_vector.size)
    for element in range(0, initial_vector.size):
        new_array = abs(initial_vector - initial_vector[element])  # determine the "closeness" |xi - xj|
        counter = 0
        sum_elements = 0.0
        for each_element in range(0, initial_vector.size):
            if new_array[each_element] <= epsilon:
                sum_elements = sum_elements + initial_vector[each_element]
                counter += 1
        new_opinion_profile[element] = sum_elements / counter
    return new_opinion_profile

def plot_dynamics(vector, epsilon, tolerance=1e-8):
    combined_array = np.empty([0, vector.size], float)
    norm_difference = 1
    k = 0
    while tolerance < norm_difference:
        combined_array = np.vstack((combined_array, vector))
        bound_confidence_array = bounded_confidence_numeric(vector, epsilon)
        norm_difference = np.abs(np.linalg.norm(bound_confidence_array - vector))
        vector = bound_confidence_array
        k += 1

    print('\nNumber of Iterations needed for stationarity (within tolerance): ', k-1)
    print('\nStationary profile:\n', vector, '\n\n')

    new_opinion_matrix = combined_array.transpose()
    for plots in range(vector.size):
        plt.plot(np.arange(k), new_opinion_matrix[plots])

    plt.title(f'Opinion Profile Change Over Time (ε={round(epsilon, 2)})')
    plt.xlabel(f'Time Steps ({k-1})')
    plt.ylabel(f'Agent Opinions ({vector.size})')
    plt.show()

# Example usage
# Symbolic initial values
symbolic_initial_values = [Rational(1, 2), Rational(1, 3), Rational(3, 4), Rational(2, 3)]  # Arbitrary symbolic initial values
l = 2   # Change l to 1, 2, 3, ... l as needed
time_steps = 5  # Number of time steps to observe

# Get symbolic history
symbolic_history = bounded_confidence_symbolic(symbolic_initial_values, l, time_steps)

# Convert symbolic values to numeric for further simulation and plotting
#numeric_initial_values = np.array([float(v.evalf()) for v in symbolic_history[-1]])  # Use last symbolic state as initial for numeric simulation
#epsilon = 0.35 #+ 10**(-8)

# Run and plot numeric simulation
#plot_dynamics(numeric_initial_values, epsilon)

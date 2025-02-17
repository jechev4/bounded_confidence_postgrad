# **Below Is New Symbolic Computation 8-7-2024**

import sympy as sp

def bounded_confidence_model_symbolic(initial_values, l, time_steps):
    n = sp.symbols('n')  # Define n as a symbolic variable
    epsilon = l / n
    history = [initial_values.copy()]

    for t in range(time_steps):
        new_values = initial_values.copy()
        for i in range(len(initial_values)):
            neighbors = []
            for j in range(len(initial_values)):
                # Check if the symbolic inequality holds
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

def print_history(history):
    for t, step in enumerate(history):
        print(f"Time step {t}: {step}")

# Example usage
initial_values = [sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(3, 4), sp.Rational(2, 3)]  # Arbitrary symbolic initial values
l = 2   # Change l to 1, 2, 3, ... l as needed
time_steps = 5  # Number of time steps to observe

history = bounded_confidence_model_symbolic(initial_values, l, time_steps)
print_history(history)

#tolerance = 0.5*10**-8




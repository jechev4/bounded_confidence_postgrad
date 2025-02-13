# Bounded Confidence Model - Research & Experiments

## Overview
This repository contains my post-graduation research on **bounded confidence models** and their behavior under various conditions. The project explores how individual agents adjust their opinions over time based on proximity to others, using numerical and symbolic approaches.

The work includes implementations of **symbolic computations, Monte Carlo simulations, and statistical methods** to analyze opinion fragmentation, consensus, and stability.

## Key Features

### 1. Symbolic Computation for Opinion Evolution
- Implements **SymPy** for algebraic manipulations and formal proofs.
- Analyzes bounded confidence behavior through equations rather than pure numerical simulations.

### 2. Monte Carlo Simulations
- Uses **NumPy** and **SciPy** to run thousands of opinion evolution experiments.
- Plots the distribution of consensus outcomes and fragmentation probabilities.
- Improves prior methodologies for **frequency of consensus** estimation.

### 3. Statistical Analysis & Graphing
- Visualizes opinion dynamics through **Matplotlib**.
- Computes key statistical measures such as **gap size between clusters**, **rate of consensus**, and **emergence of multiple factions**.
- Applies empirical testing to validate theoretical expectations.

## Notable Sections in the Notebook
- **Symbolic Computation (8/7/2024 & 8/8/2024)**: Implements algebraic methods for bounded confidence analysis.
- **Monte Carlo Average Gap Simulation (8/14/2024)**: Estimates the gap between converging opinions using repeated trials.
- **Frequency of Consensus Simulation (8/14/2024)**: Improves the probability calculation of achieving full agreement.

## Requirements
This notebook requires the following Python libraries:
- `numpy`
- `scipy`
- `sympy`
- `matplotlib`

Install them using:
```sh
pip install numpy scipy sympy matplotlib
```

## How to Use
1. Open the notebook in Jupyter or VS Code.
2. Run the symbolic computation sections for algebraic analysis.
3. Execute Monte Carlo simulations to generate statistical insights.
4. Adjust parameters (e.g., tolerance values, agent counts) to test different scenarios.

## Future Work
- Exploring higher-dimensional opinion spaces.
- Integrating external datasets for real-world applicability.
- Optimizing the simulation efficiency with parallel processing.

---
This research is an ongoing project in opinion dynamics, bridging **mathematical theory, computational simulations, and real-world applicability**.


# Simulating the Dynamics of Programmed Cell Death: A Numerical Analysis

This repository contains the source code and documentation for the paper "Simulating the Dynamics of Programmed Cell Death," which conducts a comprehensive evaluation of numerical and machine learning methods for solving an Ordinary Differential Equation (ODE) model of apoptosis.

---

## Overview

Apoptosis, or programmed cell death, is a vital biological process. Mathematical models, particularly systems of ODEs, are crucial for understanding its complex cellular mechanisms. This project explores the critical role of the numerical solver in the accuracy and efficiency of these simulations.

We implement and compare four distinct solvers from scratch to analyze their performance on a six-dimensional ODE model of apoptosis. The goal is to provide a clear benchmark of their accuracy, stability, and computational trade-offs.

## Features

-   **ODE Model:** A six-variable nonlinear ODE system modeling the core dynamics of apoptosis.
-   **Numerical Solvers Implemented:**
    1.  **Explicit Euler Method:** A simple, first-order fixed-step solver.
    2.  **4th-Order Runge-Kutta (RK4):** A high-accuracy, fixed-step solver.
    3.  **Runge-Kutta-Fehlberg (RKF45):** An adaptive step-size solver with local error control.
-   **Machine Learning Solver:**
    4.  **Physics-Informed Neural Network (PINN):** A deep learning approach that learns the solution by enforcing the ODEs in its loss function.
-   **Comparative Analysis:** A framework for quantitatively comparing the performance of all four methods.

## The Apoptosis Model

The project is based on the six-variable ODE model proposed by Laise et al., which describes the concentration dynamics of:
-   **HIF-1** (Hypoxia-inducible factor)
-   **O2** (Oxygen)
-   **p300** (Coactivator)
-   **p53** (Tumor suppressor)
-   **Caspase** (Executioner proteins)
-   **K+** (Potassium ions)

The full system of equations can be found in the accompanying paper.

## How to Run the Code

### Prerequisites
- Python 3.7+
- The required Python libraries can be installed via `pip`:
  ```bash
  pip install -r requirements.txt
  ```

### Execution
The main analysis script can be run from the terminal:
```bash
python your_main_script_name.py 
```
This will execute all solvers, print the final comparison table to the console, and save all result graphs to a new directory named `final_report_graphs/`.

## Simulation Results

The four implemented methods were compared against each other. The adaptive RKF45 method, due to its high accuracy and error control, was used as the primary baseline.

### Performance Comparison

The following table summarizes the performance of each method when solving the system with the initial condition `y0 = [1,0,0,0,0,0]`.

| Method           | Mean Absolute Error (vs. RKF45) |
| ---------------- | ------------------------------- |
| **Euler Method** | %%% Fill in MAE from your run %%% |
| **RK4 Method**   | %%% Fill in MAE from your run %%% |
| **PINN (ML)**    | %%% Fill in MAE from your run %%% |

### Visual Comparison

The following graph shows a direct visual comparison of the solutions produced by each method for the `HIF-1` variable. The results highlight the high accuracy of the RK4 and PINN methods compared to the simpler Euler method.


## Repository Structure

```
.
├── final_report_graphs/   # Directory for output graphs
├── your_main_script.py    # Main script to run all analyses
├── report.pdf             # The final PDF report
├── report.tex             # The LaTeX source file for the report
└── README.md              # This file
```

## Authors

-   **Ziad Osama Ismaill**
-   **Anas Mohamed Abdelghany**
-   **Ahmed Mahmoud Adel**
-   **Hassan Badawy Mohamed**
-   **Mohamed Ehab Ahmed**
-   **Menna Atef Eid**
-   **Engy Mohamed Mahmoud**
-   **Nada Mostafa Kamel**
-   **Saga Sadek Zakaria**

All authors are affiliated with the Faculty of Engineering Biomedical Department, Cairo University.

## Acknowledgments

This work is based on the apoptosis model originally proposed by P. Laise, D. Fanelli, and A. Arcangeli in their 2012 paper, "A dynamical model of apoptosis and its role in tumor progression."

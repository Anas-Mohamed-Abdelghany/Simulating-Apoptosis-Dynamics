# solve_and_compare_ics.py

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. DEFINE THE BIOMEDICAL MODEL (APOPTOSIS)
# =============================================================================

# Define model parameters (from book Table 3.3)
params = {
    'ahif': 1.52, 'ao2': 1.8, 'ap53': 0.05, 'a3': 0.9, 'a4': 0.2,
    'a5': 0.001, 'a7': 0.7, 'a8': 0.06, 'a9': 0.1, 'a10': 0.7,
    'a11': 0.2, 'a12': 0.1, 'a13': 0.1, 'a14': 0.05
}

# Define the system of ODEs
def apoptosis_ode_system(t, y, p):
    """Defines the system of ODEs for apoptosis."""
    yhif, yo2, yp300, yp53, ycasp, ykp = y
    dyhif_dt = p['ahif'] - p['a3']*yo2*yhif - p['a4']*yhif*yp300 - p['a7']*yp53*yhif
    dyo2_dt = p['ao2'] - p['a3']*yo2*yhif + p['a4']*yhif*yp300 - p['a11']*yo2
    dyp300_dt = p['a8'] - p['a4']*yhif*yp300 - p['a5']*yp300*yp53
    dyp53_dt = p['ap53'] - p['a5']*yp300*yp53 - p['a9']*yp53
    dycasp_dt = p['a12'] + p['a9']*yp53 - p['a13']*ycasp
    dykp_dt = -p['a10']*ycasp*ykp + p['a11']*yo2 - p['a14']*ykp
    return np.array([dyhif_dt, dyo2_dt, dyp300_dt, dyp53_dt, dycasp_dt, dykp_dt])


# =============================================================================
# 2. IMPLEMENT THE EULER METHOD
# =============================================================================

def euler_solver(ode_func, y0, t_span, dt, p):
    """A from-scratch implementation of the Explicit Euler method."""
    t_start, t_end = t_span
    t_vals = np.arange(t_start, t_end + dt, dt)
    y_vals = np.zeros((len(t_vals), len(y0)))
    y_vals[0, :] = y0
    
    for i in range(len(t_vals) - 1):
        t_current = t_vals[i]
        y_current = y_vals[i, :]
        slope = ode_func(t_current, y_current, p)
        y_vals[i+1, :] = y_current + dt * slope
        
    return t_vals, y_vals


# =============================================================================
# 3. RUN SIMULATIONS AND CALCULATE ERROR
# =============================================================================

if __name__ == "__main__":
    # --- Simulation Setup ---
    time_span = (0, 100)
    time_step = 0.01  # A stable step size for Euler
    
    # Define the two initial conditions
    ic_1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ic_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print("Running Euler solver for both initial conditions...")
    
    # --- Solve for the first initial condition ---
    t_sol, y_sol1 = euler_solver(apoptosis_ode_system, ic_1, time_span, time_step, params)
    
    # --- Solve for the second initial condition ---
    _, y_sol2 = euler_solver(apoptosis_ode_system, ic_2, time_span, time_step, params)
    
    print("Solvers finished. Calculating relative approximate error...")

    # --- Calculate Relative Approximate Error ---
    # To avoid division by zero, we add a small epsilon where y_sol1 is close to zero
    epsilon = 1e-12
    # The error is |(solution1 - solution2) / solution1|
    relative_error = np.abs((y_sol1 - y_sol2) / (y_sol1 + epsilon))
    
    # --- Plotting the Results ---
    variable_labels = ['y_hif', 'y_o2', 'y_p300', 'y_p53', 'y_casp', 'y_kp']
    
    # Create a 3x2 grid for the solutions
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 8))
    fig1.suptitle('Comparison of Solutions for Different Initial Conditions', fontsize=16)

    for i in range(y_sol1.shape[1]):
        ax = axs1.flat[i]
        ax.plot(t_sol, y_sol1[:, i], color='black', label='y0 = [1,0,0,0,0,0]')
        ax.plot(t_sol, y_sol2[:, i], 'r--', label='y0 = [0,0,0,0,0,0]')
        ax.set_title(f"Solution for {variable_labels[i]}", fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.grid(True)
        ax.legend()
    
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # Create a separate 3x2 grid for the relative error
    fig2, axs2 = plt.subplots(3, 2, figsize=(15, 10))
    fig2.suptitle('Relative Approximate Error Between Solutions', fontsize=16)

    for i in range(relative_error.shape[1]):
        ax = axs2.flat[i]
        ax.plot(t_sol, relative_error[:, i], 'g-')
        ax.set_title(f"Relative Error for {variable_labels[i]}", fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Relative Error |(y1-y2)/y1|')
        ax.grid(True)
        # Use a logarithmic scale for the y-axis to better see the error dynamics
        ax.set_yscale('log')

    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()

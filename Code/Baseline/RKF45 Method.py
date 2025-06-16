import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# Parameters (from Table 3.3)
a = {
    'a_hif': 1.52, 'a_o2': 1.8, 'a_p53': 0.05, 'a3': 0.9, 'a4': 0.2,
    'a5': 0.001, 'a7': 0.7, 'a8': 0.06,
    'a9': 0.1, 'a10': 0.7, 'a11': 0.2, 'a12': 0.1,
    'a13': 0.1, 'a14': 0.05
}

# RHS of the apoptosis model
def apoptosis_rhs(t, y):
    y_hif, y_o2, y_p300, y_p53, y_casp, y_kp = y
    dydt = np.zeros_like(y)
    
    dydt[0] = a['a_hif'] - a['a3']*y_o2*y_hif - a['a4']*y_hif*y_p300 - a['a7']*y_p53*y_hif
    dydt[1] = a['a_o2'] - a['a3']*y_o2*y_hif + a['a4']* y_hif* y_p300 - a['a11'] * y_o2
    dydt[2] = - a['a4']*y_hif * y_p300 - a['a5']*y_p300 * y_p53 +  a['a8']
    dydt[3] = a['a_p53'] - a['a5']*y_p53*y_p300 - a['a9']*y_p53
    dydt[4] = a['a9']*y_p53 + a['a12'] - a['a13']*y_casp
    dydt[5] = -a['a10']*y_casp*y_kp +a['a11'] *y_o2 - a['a14']*y_kp
    
    return dydt

# Initial conditions and time grid
y0 = [1, 0, 0, 0, 0, 0]   # starting concentrations at t=0
t_span = (0, 100)
t_eval = np.linspace(0, 100, 101)

# 1) Solve with LSODA (via solve_ivp)
sol_lsoda = solve_ivp(apoptosis_rhs, t_span, y0, method='LSODA', t_eval=t_eval)

# 2) Hand-coded RKF45 integrator with truncation error storage
def rkf45_with_error(f, t0, y0, t_end, h0=0.1, tol=1e-6):
    t = t0
    y = np.array(y0, dtype=float)
    h = h0
    ts, ys, ees = [t], [y.copy()], [np.zeros_like(y)]  # Store truncation error ee = y5 - y4

    while t < t_end:
        if t + h > t_end:
            h = t_end - t
        
        # coefficients for Fehlberg RKF45
        c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
        a2 = np.array([1/4])
        a3 = np.array([3/32, 9/32])
        a4 = np.array([1932/2197, -7200/2197, 7296/2197])
        a5 = np.array([439/216, -8, 3680/513, -845/4104])
        a6 = np.array([-8/27, 2, -3544/2565, 1859/4104, -11/40])
        b4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
        b5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        
        k1 = f(t + c[0]*h, y) * h
        k2 = f(t + c[1]*h, y + a2[0]*k1) * h
        k3 = f(t + c[2]*h, y + a3[0]*k1 + a3[1]*k2) * h
        k4 = f(t + c[3]*h, y + a4[0]*k1 + a4[1]*k2 + a4[2]*k3) * h
        k5 = f(t + c[4]*h, y + a5[0]*k1 + a5[1]*k2 + a5[2]*k3 + a5[3]*k4) * h
        k6 = f(t + c[5]*h, y + a6[0]*k1 + a6[1]*k2 + a6[2]*k3 + a6[3]*k4 + a6[4]*k5) * h
        
        y4 = y + b4[0]*k1 + b4[1]*k2 + b4[2]*k3 + b4[3]*k4 + b4[4]*k5
        y5 = y + b5[0]*k1 + b5[1]*k2 + b5[2]*k3 + b5[3]*k4 + b5[4]*k5 + b5[5]*k6
        
        ee = y5 - y4  # Truncation error estimate at this step
        
        err = np.linalg.norm(ee, ord=np.inf)
        if err <= tol:
            t += h
            y = y4
            ts.append(t)
            ys.append(y.copy())
            ees.append(ee.copy())
        
        # adapt step size
        delta = 0.84 * (tol / (err + 1e-16))**0.25
        h = max(min(delta * h, 5*h), 0.1*h)
    
    return np.array(ts), np.array(ys), np.array(ees)

# Run RKF45 with error for both cases
ts_rkf, ys_rkf, ees_rkf = rkf45_with_error(apoptosis_rhs, 0, y0, 100)

# Now, add another plot for the case yini = [0,0,0,0,0,0]
y0_zero = [0, 0, 0, 0, 0, 0]
sol_lsoda_zero = solve_ivp(apoptosis_rhs, t_span, y0_zero, method='LSODA', t_eval=t_eval)
ts_rkf_zero, ys_rkf_zero, ees_rkf_zero = rkf45_with_error(apoptosis_rhs, 0, y0_zero, 100)

labels = ['HIF-1', 'O2', 'p300', 'p53', 'Caspase', 'K+']

# Print list(c(yt_hif, yt_o2, yt_p300, yt_p53, yt_casp, yt_kp)) for the two cases for t = 0, 25, 50, 100
print("Selected time points for both cases (LSODA):")
selected_times = [0, 25, 50, 100]
yt_dict = {}  # Store results for error calculation

for case_name, sol in [("y0 = [1,0,0,0,0,0]", sol_lsoda), ("y0 = [0,0,0,0,0,0]", sol_lsoda_zero)]:
    print(f"\nCase {case_name}:")
    yt_dict[case_name] = {}
    for t_sel in selected_times:
        idx = np.where(np.isclose(sol.t, t_sel))[0]
        if len(idx) > 0:
            yt = sol.y[:, idx[0]]
            yt_dict[case_name][t_sel] = yt
            # Print with variable names
            print(f"t = {t_sel}: ", end="")
            for i, name in enumerate(['yt_hif', 'yt_o2', 'yt_p300', 'yt_p53', 'yt_casp', 'yt_kp']):
                print(f"{name} = {yt[i]:.6f}", end=", " if i < 5 else "\n")
        else:
            print(f"t = {t_sel}: Not found in solution.")

# Print error between the two cases at selected times
print("\nError (absolute difference) between the two cases at selected times:")
for t_sel in selected_times:
    if t_sel in yt_dict["y0 = [1,0,0,0,0,0]"] and t_sel in yt_dict["y0 = [0,0,0,0,0,0]"]:
        yt1 = yt_dict["y0 = [1,0,0,0,0,0]"][t_sel]
        yt0 = yt_dict["y0 = [0,0,0,0,0,0]"][t_sel]
        abs_err = np.abs(yt1 - yt0)
        print(f"t = {t_sel}: ", end="")
        for i, name in enumerate(['yt_hif', 'yt_o2', 'yt_p300', 'yt_p53', 'yt_casp', 'yt_kp']):
            print(f"err_{name} = {abs_err[i]:.6f}", end=", " if i < 5 else "\n")
    else:
        print(f"t = {t_sel}: Not found in both cases.")

# Print truncation error estimate ee = y5 - y4 for both cases at selected times
print("\nTruncation error estimate (ee = y5 - y4) for RKF45 at selected times:")
def print_truncation_error(ts_rkf, ees_rkf, selected_times, case_label):
    print(f"\nCase {case_label}:")
    for t_sel in selected_times:
        idx = np.where(np.isclose(ts_rkf, t_sel))[0]
        if len(idx) > 0:
            ee = ees_rkf[idx[0]]
            print(f"t = {t_sel}: ", end="")
            for i, name in enumerate(['ee_hif', 'ee_o2', 'ee_p300', 'ee_p53', 'ee_casp', 'ee_kp']):
                print(f"{name} = {ee[i]:.2e}", end=", " if i < 5 else "\n")
        else:
            print(f"t = {t_sel}: Not found in RKF45 solution.")

print_truncation_error(ts_rkf, ees_rkf, selected_times, "y0 = [1,0,0,0,0,0]")
print_truncation_error(ts_rkf_zero, ees_rkf_zero, selected_times, "y0 = [0,0,0,0,0,0]")

# First plot: original initial conditions
fig1, axs1 = plt.subplots(3, 2, figsize=(12, 8))
for i, ax in enumerate(axs1.flat):
    ax.plot(ts_rkf, ys_rkf[:, i], label='RKF45', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel(labels[i])
    ax.set_title(f'({chr(97+i)}) {labels[i]}')
    ax.grid(True)
    ax.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Second plot: all-zero initial conditions
fig2, axs2 = plt.subplots(3, 2, figsize=(12, 8))
for i, ax in enumerate(axs2.flat):
    ax.plot(ts_rkf_zero, ys_rkf_zero[:, i], label='RKF45', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel(labels[i])
    ax.set_title(f'({chr(97+i)}) {labels[i]}')
    ax.grid(True)
    ax.legend()
fig2.suptitle('Apoptosis Model: RKF45 (y0 = [0,0,0,0,0,0])', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

# Explanation : Why is there a difference between the two cases?
#
# The difference between the two cases arises from the initial conditions provided to the model.
# In the first case, y0 = [1, 0, 0, 0, 0, 0], the HIF-1 concentration starts at 1, while all other species start at 0.
# In the second case, y0 = [0, 0, 0, 0, 0, 0], all species start at zero.
#
# The system of ODEs includes both production (source) and consumption (sink) terms for each species.
# Some species, such as HIF-1, O2, and p53, have nonzero constant production rates (see a['a_hif'], a['a_o2'], a['a_p53'], etc.).
# This means that even if the initial concentrations are zero, these species will begin to increase over time due to their source terms.
#
# However, the *dynamics* and *timing* of how each species rises and interacts depend on the initial conditions.
# For example, starting with HIF-1 = 1 (first case) means that the system immediately has HIF-1 available to participate in reactions,
# which can accelerate the production or consumption of other species through the nonlinear terms (e.g., y_o2*y_hif, y_hif*y_p300, etc.).
# In contrast, starting with all zeros (second case), the system must "build up" each species from their respective source terms,
# and the nonlinear interaction terms (which are products of concentrations) will initially be zero, so the system evolves more slowly at first.
#
# As a result, the time courses and possibly the steady-state values of the species can differ between the two cases,
# especially in the early phase of the simulation. The initial presence or absence of HIF-1 (or any other species) can
# significantly affect the transient dynamics due to the nonlinear couplings in the ODEs.
#
# In summary: The difference is due to the nonlinear and coupled nature of the ODE system, where initial concentrations
# influence how quickly and in what manner the system evolves, even though the source terms will eventually drive the
# system toward a steady state determined by the balance of production and consumption rates.

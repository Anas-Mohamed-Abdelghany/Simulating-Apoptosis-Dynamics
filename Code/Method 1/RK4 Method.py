import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


# Define parameters from Table 3.3
params = {
    "ahif": 1.52,
    "ao2": 1.8,
    "ap53": 0.05,
    "a3": 0.9,
    "a4": 0.2,
    "a5": 0.001,
    "a7": 0.7,
    "a8": 0.06,
    "a9": 0.1,
    "a10": 0.7,
    "a11": 0.2,
    "a12": 0.1,
    "a13": 0.1,
    "a14": 0.05
}

# System of ODEs (Eqs. 3.1 to 3.6)
def dydt(t, y, p):
    yhif, yo2, yp300, yp53, ycasp, ykp = y
    dy = np.zeros(6)

    dy[0] = p['ahif'] - p['a3']*yo2*yhif - p['a4']*yhif*yp300 - p['a7']*yp53*yhif
    dy[1] = p['ao2'] - p['a3']*yo2*yhif + p['a4']*yhif*yp300 - p['a11']*yo2
    dy[2] = -p['a4']*yhif*yp300 - p['a5']*yp300*yp53 + p['a8']
    dy[3] = p['ap53'] - p['a5']*yp53*yp300 - p['a9']*yp53
    dy[4] = p['a9']*yp53 + p['a12'] - p['a13']*ycasp
    dy[5] = -p['a10']*ycasp*ykp + p['a11']*yo2 - p['a14']*ykp

    return dy

# Runge-Kutta 4th order method
def runge_kutta_4(f, y0, t0, tf, dt, p):
    t_vals = np.arange(t0, tf + dt, dt)
    y_vals = np.zeros((len(t_vals), len(y0)))
    y = y0.copy()

    for i, t in enumerate(t_vals):
        y_vals[i] = y
        k1 = dt * f(t, y, p)
        k2 = dt * f(t + dt/2, y + k1/2, p)
        k3 = dt * f(t + dt/2, y + k2/2, p)
        k4 = dt * f(t + dt, y + k3, p)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t_vals, y_vals

# Time settings
t0 = 0
tf = 100  # Updated to match figure range
dt = 0.01

# Solve the system for both initial conditions
y0_1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y0_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

t_vals_1, y_vals_1 = runge_kutta_4(dydt, y0_1, t0, tf, dt, params)
t_vals_0, y_vals_0 = runge_kutta_4(dydt, y0_0, t0, tf, dt, params)

# Export to CSV for both cases
df1 = pd.DataFrame(y_vals_1, columns=["yhif", "yo2", "yp300", "yp53", "ycasp", "ykp"])
df1["time"] = t_vals_1
df1.to_csv("apoptosis_rk4_output_y0_1.csv", index=False)

df0 = pd.DataFrame(y_vals_0, columns=["yhif", "yo2", "yp300", "yp53", "ycasp", "ykp"])
df0["time"] = t_vals_0
df0.to_csv("apoptosis_rk4_output_y0_0.csv", index=False)

# Print error between the two initial conditions at t = (0, 25, 50, 75, 100)
selected_times = [0, 25, 50, 75, 100]
error_output = []
error_output.append("\nError (absolute difference) between y0 = [1,0,0,0,0,0] and y0 = [0,0,0,0,0,0] at selected times:")
for t_sel in selected_times:
    # Find the index closest to t_sel in t_vals_1 and t_vals_0 (should be the same)
    idx_1 = np.argmin(np.abs(t_vals_1 - t_sel))
    idx_0 = np.argmin(np.abs(t_vals_0 - t_sel))
    vals_1 = y_vals_1[idx_1]
    vals_0 = y_vals_0[idx_0]
    abs_err = np.abs(vals_1 - vals_0)
    line = f"t = {t_sel}: "
    for i, label in enumerate(["yhif", "yo2", "yp300", "yp53", "ycasp", "ykp"]):
        line += f"err_{label} = {abs_err[i]:.6f}"
        if i < 5:
            line += ", "
    error_output.append(line)

# Plot results for both initial conditions
labels = ["y_hif(t)", "y_o2(t)", "y_p300(t)", "y_p53(t)", "y_casp(t)", "y_kp(t)"]
fig, axs = plt.subplots(3, 2, figsize=(10, 8))

for i, ax in enumerate(axs.flat):
    ax.plot(t_vals_1, y_vals_1[:, i], color='black', label='y0 = [1,0,0,0,0,0]')
    ax.plot(t_vals_0, y_vals_0[:, i], color='red', linestyle='--', label='y0 = [0,0,0,0,0,0]')
    ax.set_title(f'({chr(97+i)}) {labels[i]}')
    ax.set_xlabel("t")
    ax.set_ylabel(labels[i])
    ax.grid(True)
    ax.legend()

plt.tight_layout()

# Print the error output before showing the plot, so both appear at the same time
for line in error_output:
    print(line)

plt.show()
print(f"err_{label} = {abs_err[i]:.6f}", end=", " if i < 5 else "\n")

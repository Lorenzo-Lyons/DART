from dart_dynamic_models import model_functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mf = model_functions()


# Define the range of throttle and steering inputs
th_vec = np.linspace(-1, 1, 30)  # Throttle (y-axis)
st_vec = np.linspace(-1, 1, 30) # Steering (x-axis)

# Define initial velocity conditions
vx, vy, w = 0, 0, 0

# Create meshgrid for inputs
ST, TH = np.meshgrid(st_vec, th_vec)  # Swap order to match (steering, throttle)

# Initialize output matrices
acc_x = np.zeros((len(th_vec), len(st_vec)))  # Rows: throttle, Columns: steering
acc_y = np.zeros((len(th_vec), len(st_vec)))
acc_w = np.zeros((len(th_vec), len(st_vec)))

# Evaluate the dynamic model over the input grid
for i, th in enumerate(th_vec):
    for j, st in enumerate(st_vec):
        acc_x[i, j], acc_y[i, j], acc_w[i, j] = mf.dynamic_bicycle(th, st, vx, vy, w)

# Set up figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Titles for the plots
titles = [r'Longitudinal Acceleration ($a_x$)', 
          r'Lateral Acceleration ($a_y$)', 
          r'Yaw Acceleration ($\dot{\omega}$)']

# Data to plot
data = [acc_x, acc_y, acc_w]

# Plot each heatmap
for i, ax in enumerate(axes):
    sns.heatmap(data[i],  
                xticklabels=np.round(st_vec, 2),  
                yticklabels=np.round(th_vec, 2),  
                cmap="inferno", ax=ax)

    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel("Steering (-1 to 1)")  # X-axis = Steering
    ax.set_ylabel("Throttle (0 to 1)")  # Y-axis = Throttle

    ax.invert_yaxis()  # ✅ Flip the y-axis so throttle = 0 is at the bottom

plt.tight_layout()
plt.show()

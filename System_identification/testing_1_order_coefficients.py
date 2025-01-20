# testing to see if we can replicate second order dynamics using some coefficients
import numpy as np
from matplotlib import pyplot as plt
from dart_dynamic_models import model_functions
mf = model_functions()



dt = 0.1
sim_time = 10 # seconds
transient_time = 2 # you need the input response to go to zero within this time otherwise you subtract energy from the signal

t = np.arange(0,sim_time,dt)

# forcing term will be a step input
forcing_term = np.ones(len(t))


# first order dynamics gain
time_constant = 0.1

n_past_inputs = int(transient_time/dt-1)
# produce matrix of past actions
past_actions_mat = np.ones((len(t),n_past_inputs+1))
past_actions_mat[:n_past_inputs+1,:] = np.tril(np.ones((n_past_inputs+1, n_past_inputs+1), dtype=int), -1) + np.eye(n_past_inputs+1) # replace with lower traingular ones matrix



# k_vals_step, k_vec_dev = produce_past_action_coefficients(z,w_Hz) 
k_vals_step = mf.produce_past_action_coefficients_1st_oder_step_response(time_constant,n_past_inputs+1,dt)
k_vals_impulse = mf.produce_past_action_coefficients_1st_oder(time_constant,n_past_inputs+1,dt)

# Forward integrate the state
# second order system initial conditions
x_vec = np.zeros(len(t))
x_vec_coefficients_step = np.zeros(len(t))
x_vec_coefficients_impulse = np.zeros(len(t))


ground_truth_refinement = 100
for i in range(1, len(t)):
    # integrate ground trough with a much higher dt to have better numerical accuracy
    x_ground_truth = x_vec[i-1]
    for k in range(ground_truth_refinement):
        x_dot = mf.continuous_time_1st_order_dynamics(x_ground_truth,forcing_term[i-1],time_constant)
        x_ground_truth += dt/ground_truth_refinement * x_dot
    x_vec[i] = x_ground_truth

    x_vec_coefficients_step[i] = past_actions_mat[i,:] @ k_vals_step
    x_vec_coefficients_impulse[i] = past_actions_mat[i,:] @ k_vals_impulse

    


plt.figure()
plt.plot(t,x_vec,label='real dynamics',color='dodgerblue',linewidth=3)
plt.plot(t,forcing_term,label='forcing term',color='orangered',zorder=20,linestyle='--')
plt.plot(t[:n_past_inputs+1],k_vals_step,label='k values (step response to unitary input acting for 1 dt)',color='gray')
plt.plot(t,x_vec_coefficients_step,label='x with coefficients step',color='black',linewidth=1)
#plt.plot(t,x_vec_coefficients_impulse,label='x with coefficients inpulse',color='darkgreen')
plt.xlabel('time [s]')
plt.ylabel('x')
plt.legend()

plt.show()


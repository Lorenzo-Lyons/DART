# testing to see if we can replicate second order dynamics using some coefficients
import numpy as np
from matplotlib import pyplot as plt


dt = 0.001
sim_time = 10 # seconds
t = np.arange(0,sim_time,dt)

# forcing term will be a step input
forcing_term = np.ones(len(t))


# second order system initial conditions
x_vec = np.zeros(len(t))
x_vec_coefficients = np.zeros(len(t))
x_dot = 0

z = 0.3
w_Hz = 1 # Hz
w = w_Hz * 2 * np.pi


n_past_inputs = 3999
# produce matrix of past actions
past_actions_mat = np.ones((len(t),n_past_inputs+1))
past_actions_mat[:n_past_inputs+1,:] = np.tril(np.ones((n_past_inputs+1, n_past_inputs+1), dtype=int), -1) + np.eye(n_past_inputs+1) # replace with lower traingular ones matrix


# define coefficients function
def produce_past_action_coefficients(z,w):
    # Generate the k coefficients for past actions
    #[d,c,b,z,w] = transform_parameters_norm_2_real()
    k_vec = np.zeros((n_past_inputs+1,1))
    for i in range(n_past_inputs+1):
        k_vec[i]=impulse_response(i*dt,z,w)
    return k_vec

def impulse_response(t,z,w):
    #second order impulse response
    #[d,c,b,z,w] = transform_parameters_norm_2_real()
    w = w * 2 * np.pi
    z = z

    if z >1:
        a = np.sqrt(z**2-1)
        f = w/(2*a) * (np.exp(-w*(z-a)*t) - np.exp(-w*(z+a)*t))
        
    elif z == 1:
        f = w**2 * t * np.exp(-w*t)

    elif z < 1:
        w_d = w * np.sqrt(1-z**2)
        f = w/(np.sqrt(1-z**2))*np.exp(-z*w*t)*np.sin(w_d*t)
    return f



k_vals = produce_past_action_coefficients(z,w_Hz) * dt




# Forward integrate the state
for i in range(1, len(t)):
    # integrate the steering angle
    x_ddot = w**2 * (forcing_term[i-1]-x_vec[i-1]) - 2*w*z * x_dot

    x_dot += x_ddot * dt
    x_vec[i] = x_vec[i-1] + dt * x_dot

    # using coefficients instead
    x_vec_coefficients[i] = past_actions_mat[i,:] @ k_vals






plt.figure()
plt.plot(t,x_vec,label='real dynamics',color='dodgerblue')
plt.plot(t,forcing_term,label='forcing term',color='orangered')
plt.plot(t[:n_past_inputs+1],k_vals,label='k values (impulse response)',color='gray')
plt.plot(t,x_vec_coefficients,label='x with coefficients',color='black')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('x')
plt.show()


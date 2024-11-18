# testing to see if we can replicate second order dynamics using some coefficients
import numpy as np
from matplotlib import pyplot as plt


dt = 0.001
sim_time = 10 # seconds
transient_time = 2 # you need the input response to go to zero within this time otherwise you subtract energy from the signal

t = np.arange(0,sim_time,dt)

# forcing term will be a step input
forcing_term = np.ones(len(t))



z = 1 #1.5369760990142822 # 0.880281388759613
w_Hz = 0.75 #6.818084716796875 # 11.37063980102539 #6.946717739105225 # Hz
w = w_Hz * 2 * np.pi


n_past_inputs = int(transient_time/dt-1)
# produce matrix of past actions
past_actions_mat = np.ones((len(t),n_past_inputs+1))
past_actions_mat[:n_past_inputs+1,:] = np.tril(np.ones((n_past_inputs+1, n_past_inputs+1), dtype=int), -1) + np.eye(n_past_inputs+1) # replace with lower traingular ones matrix


# define coefficients function
def produce_past_action_coefficients(z,w):
    # Generate the k coefficients for past actions
    #[d,c,b,z,w] = transform_parameters_norm_2_real()
    k_vec = np.zeros((n_past_inputs+1,1))
    k_vec_dev = np.zeros((n_past_inputs+1,1))
    for i in range(n_past_inputs+1):
        f,dev_f = impulse_response(i*dt,z,w)
        k_vec[i]=f * dt 
        k_vec_dev[i]=dev_f * dt
    return k_vec , k_vec_dev

def impulse_response(t,z,w):
    #second order impulse response
    #[d,c,b,z,w] = transform_parameters_norm_2_real()
    w = w * 2 * np.pi
    

    if z >1:
        a = np.sqrt(z**2-1)
        f = w/(2*a) * (np.exp(-w*(z-a)*t) - np.exp(-w*(z+a)*t))
        
    elif z == 1:
        f = w**2 * t * np.exp(-w*t)
        dev_f = w**2 * (np.exp(-w*t) - w*t*np.exp(-w*t))

    elif z < 1:
        w_d = w * np.sqrt(1-z**2)
        f = w/(np.sqrt(1-z**2))*np.exp(-z*w*t)*np.sin(w_d*t)
    return f, dev_f



k_vals, k_vec_dev = produce_past_action_coefficients(z,w_Hz) 




# Forward integrate the state
# second order system initial conditions
x_vec = np.zeros(len(t))
x_vec_coefficients = np.zeros(len(t))
x_dot_vec = np.zeros(len(t))
x_dot_vec_coefficients = np.zeros(len(t))
x_vec_coefficients_numerical = np.zeros(len(t))
x_dot = 0

max_x_ddot = 0.5
max_x_dot = 0.5

# numerically compute k_val in case you have actuator limitations
k_vals_numerical = np.zeros((n_past_inputs+1,1))

x_dot_numerical = 2  # initial condition is non-zero
Delta_st_int = 0
K_I = 100

for i in range(1, n_past_inputs+1):
    # integrate the steering angle
    #forcing_term_filtered = alpha_filter * forcing_term_filtered + (1-alpha_filter) * (forcing_term[i-1])
    

    #x_ddot = np.min([w**2 * (forcing_term[i-1] -x_vec[i-1]) - 2*w*z * x_dot,max_x_ddot])  # 
    x_ddot_numerical = w**2 * (0 -k_vals_numerical[i-1]) - 2*w*z * x_dot_numerical + K_I * Delta_st_int
    #x_ddot_numerical = np.min([x_ddot_numerical,max_x_ddot])
    #x_ddot_numerical = np.max([x_ddot_numerical,-max_x_ddot])

    x_dot_numerical += x_ddot_numerical * dt
    # x_dot_numerical = np.min([x_dot_numerical,max_x_dot])
    # x_dot_numerical = np.max([x_dot_numerical,-max_x_dot])

    k_vals_numerical[i] = k_vals_numerical[i-1] + dt * x_dot_numerical
    Delta_st_int = 0 - k_vals_numerical[i]







for i in range(1, len(t)):
    # integrate the steering angle
    #forcing_term_filtered = alpha_filter * forcing_term_filtered + (1-alpha_filter) * (forcing_term[i-1])
    
    x_ddot = w**2 * (forcing_term[i-1] -x_vec[i-1]) - 2*w*z * x_dot 
    #x_ddot = np.min([x_ddot,max_x_ddot])  
    #x_ddot = np.max([x_ddot,-max_x_ddot])

    #x_ddot = w**2 * (forcing_term[i-1] -x_vec[i-1]) - 2*w*z * x_dot 
    x_dot_vec[i] = x_dot
    x_dot += x_ddot * dt
    #x_dot = np.min([x_dot,max_x_dot])

    x_vec[i] = x_vec[i-1] + dt * x_dot

    # using coefficients instead
    x_vec_coefficients[i] = past_actions_mat[i,:] @ k_vals
    # using coefficients for derivative
    x_dot_vec_coefficients[i] = past_actions_mat[i,:] @ k_vec_dev


    x_vec_coefficients_numerical[i] = past_actions_mat[i,:] @ k_vals_numerical






plt.figure()
plt.plot(t,x_vec,label='real dynamics',color='dodgerblue')
plt.plot(t,forcing_term,label='forcing term',color='orangered')
plt.plot(t[:n_past_inputs+1],k_vals,label='k values (impulse response)',color='gray')
plt.plot(t[:n_past_inputs+1],k_vals_numerical,label='k values numerical(impulse response actuator sat)',color='blue')
plt.plot(t,x_vec_coefficients,label='x with coefficients',color='black')
#plt.plot(t,x_vec_coefficients_numerical/np.max(x_vec_coefficients_numerical),label='x with coefficients numerical',color='green')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('x')

plt.figure()
plt.title('x_dot')
plt.plot(t,x_dot_vec,label='x_dot true',color='dodgerblue')
plt.plot(t,x_dot_vec_coefficients,label='x_dot coefficients',color='k')


plt.show()


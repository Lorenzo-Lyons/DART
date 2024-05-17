#
#constant velocity vs throttle
from matplotlib import pyplot as plt
import numpy as np

data =np.array([[0.20,0.19],
                [0.21,0.38],
                [0.22,0.61],
                [0.23,0.85],
                [0.24,1.00],
                [0.25,1.19],
                [0.26,1.32],
                [0.27,1.64],
                [0.28,1.75],
                [0.29,1.91],
                [0.30,1.99],
                [0.31,2.11],
                [0.32,2.19],
                [0.35,2.6],
                [0.40,3.15],
        ])

plt.figure()
plt.plot(data[:,0],data[:,1],marker='.')
plt.xlabel('throttle')
plt.ylabel('v [m\s]')

# estimate the force based on the fitted throttle curve
a =  1.6837230920791626
b =  13.49715518951416
c =  0.3352389633655548

friction_force =    a * np.tanh(b  * data[:,1] ) + data[:,1] * c

# force measured from initial impulse (taken from graph)
force_impulse = np.array([1.81,
                          2.24,
                          2.8,
                          2.99,
                          3.25,
                          3.55,
                          3.56,
                          4.16,
                          4.70,
                          4.86,
                          5.12,
                          5.29,
                          5.71,
                          6.31,
                          7.81
                                ])

plt.figure()
plt.plot(data[:,0],friction_force, marker='.',color='blue',label='from constant velocity')
plt.plot(data[:,0],force_impulse, marker='.',color='orangered',label='initial impulse')
plt.legend()
plt.xlabel('throttle')
plt.ylabel('N')



# motor curve estimate based on 
a_estimate = np.divide(force_impulse,data[:,0])
b_estimate = - np.divide(friction_force,np.multiply(data[:,0],data[:,1])) + np.divide(a_estimate,data[:,1])
print('a_estimate =', a_estimate)
print('b_estimate =', b_estimate)



plt.show()
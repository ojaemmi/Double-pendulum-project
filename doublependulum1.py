import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
import scipy as sc
from scipy import integrate

import matplotlib.animation as animation

# Initiating all relevant constants: acc due to gravity in m/s^2, lengths of rods in m, masses of pendulums in kg
l1, l2 = 1, 1
l_max = l1+l2
m1, m2 = 1, 1
g = 9.81
t_max = 10              # simulation time in s
traj_points = 100       # amount of trajectory points shown


# Next we create a function that returns the an array with angular velocities and accelarations for of each state (shortened as s) of our double pendulum -system
# Each state is given as a matrix: s = [theta1, theta1dot, theta2, theta2dot]
# Theta1 and theta2 represent the angles of pendulum 1 and 2 respectively, theta1dot and theta2dot angular velocities respectively
# Each angular velocity is formulated using the Euler-Lagrange equations 
def derivatives(t, s):
    # Initiating 0-matrix, with the same structure
    dydx = np.zeros_like(s)

    dydx[0] = s[1]                                                 # Initiating the first item, which is the first derivate of theta1

    delta = s[2] - s[0]                                            # For the Euler-Lagrange equation we find the difference in angles

    den1 = (m1+m2) * l1 - m2 * l1 * cos(delta) * cos(delta)        # This is the denominator for the accelaration (2nd derivative) of theta1

    # Theta1 Doubledot:
    dydx[1] = ((m2 * l1 * s[1] * s[1] * sin(delta) * cos(delta) + m2 * g * sin(s[2]) * cos(delta) + m2 * l2 * s[3] * s[3] * sin(delta) - (m1+m2) * g * sin(s[0]))
               / den1)

    dydx[2] = s[3]                                                 # Initiating the third item, which is the first derivate of theta2

    den2 = (l2/l1) * den1                                          # Denominator for accelaration of theta2

    # Theta2 Doubledot
    dydx[3] = ((- m2 * l2* s[3] * s[3] * sin(delta) * cos(delta) + (m1+m2) * g * sin(s[0]) * cos(delta) - (m1+m2) * l1 * s[1] * s[1] * sin(delta) - (m1+m2) * g * sin(s[2]))
               / den2)

    return dydx    # Returns the derivative list

# Creating a time interval [0, t_max] with 0.01 intervals in between
dt = 0.01
t = np.arange(0, t_max, dt)

# For the animation we must give intial values for theta1, theta2, theta1dot and theta2dot
# Below t represents the angles theta, and tv the angular velocity: these are arbitrary and changeable for further examination
t1 = 40.0
tv1 = 0.0
t2 = 160.0
tv2 = 0.0

# Initiating state with the given variables
s = np.radians([t1, tv1, t2, tv2])

# Using SciPy to integrate ODEs
# Next creating NumPy array increasing rows and 4 colums (theta variables), then initiating at t=0 the intitial state above
y = np.empty((len(t), 4))
y[0] = s

y = sc.integrate.solve_ivp(derivatives, t[[0, -1]], s, t_eval=t).y.T


# Creating coordinates for both pendulums, using geometry
x1 = l1*sin(y[:, 0])
y1 = -l1*cos(y[:, 0])

x2 = l2*sin(y[:, 2]) + x1
y2 = -l2*cos(y[:, 2]) + y1

#Plotting figure, scaling the axis so the animation is readable
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-l_max, l_max), ylim=(-l_max, 1.))

# Drawing two rods, and showing the time spent
line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


# Lastly we create the animation function, which is responsible for updating the positions and visuals of the pendulum at each frame
def animate(b):
    # Arrays for pendulum 1 to restore x- and y-coordinates
    thisx = [0, x1[b], x2[b]]
    thisy = [0, y1[b], y2[b]]

    # Similar arrays for pendulum 2
    history_x = x2[:b]
    history_y = y2[:b]

    # Using .set_data()-method, we can draw the rods using our recovery arrays, also updating time in s
    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (b*dt))
    return line, trace, time_text

# Creating an ani-object using Matplotlib's FuncAnimation, lastly plotting the animation
ani = animation.FuncAnimation(fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()
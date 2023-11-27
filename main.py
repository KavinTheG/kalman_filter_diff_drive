from extended_kalman_filter import ExtendedKalmanFilter
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import time
import random

#### Initialization ####


# Assume Odometer data has std of 0.1m 
# R = np.ones((3,3)) * 0.01
R = np.eye(3) 

# Assume kinematic model / inputs has std of 0.1m
Q = np.eye(3)

# Initialize error covariance matrix
# P = np.array([0.01,0.01,0.003])
P = np.eye(3) * 0.1

# Transform matrix for odometer data (Y = CX + n) where n ~ N(0, R)
C = np.eye(3)

# Velocity of 1 m/s right
v = 3
omega = 0

# Time step 
dt = 0.1

# Duration = 10 seconds
total_time = 10

# Set initial position to origin
x_0 = 0
y_0 = 0
theta_0 = 0

# Track movements of robot using velocity linear and angular (w/o noise)
# I will add noise to this to represent odometry data
x_true = 0
y_true = 0
theta_true = 0

# Tracking of robot from kinematics
# 
x_kin = 0
y_kin = 0
theta_kin = 0

# Tracking of robot from sensor
x_sensor = 0
y_sensor = 0
theta_sensor = 0

# Initial Inputs
x_dot = v * np.cos(omega)
y_dot = v * np.sin(omega)
theta_dot = omega

x = [x_0, y_0, theta_0]
x_np = [[x_0], [y_0], [theta_0]]
kf = ExtendedKalmanFilter(x_np, Q, R, P, v, dt)


X_posteriori = [x_0, y_0, theta_0]



# Plotting variables
kinematic_plot_x = []
sensor_plot_x = []
kinematic_plot_y = []
sensor_plot_y = []

kalman_x = []
kalman_y = []


for _ in range (int(total_time / dt)):

    # Control input noise
    w = np.random.multivariate_normal(mean=np.zeros(3), cov=Q, size=1)

    theta_true += dt * theta_dot
    # calculate x and y velocity from theta angle
    x_dot = v * np.cos(theta_true)
    y_dot = v * np.sin(theta_true)
    theta_dot = omega

    x_true += dt * x_dot
    y_true += dt * y_dot
    X_dot_true = [x_dot, y_dot, theta_dot]

    # Control input
    U = [[v], [omega]]

    print("Velocity: ", v, " Angular Velocity: ", theta_dot, "\n")


    X_state_true = [x_true, y_true, theta_true]
    print("---- True X ", _ ," ----")
    pprint(X_state_true)


    print("\n----Kinematic Model ----")

    X_dot_kin = X_dot_true + w[0]
    x_kin += X_dot_kin[0] * dt
    y_kin += X_dot_kin[1] * dt
    theta_kin += X_dot_kin[2] * dt

    X_state_kin = [x_kin, y_kin, theta_kin]
    print("Kinematic State : ", X_state_kin)
    # print("Kinematic (velocity): ", X_dot_kin)

    
    print("---Sensor Model----")
    # Y value
    n = np.random.multivariate_normal(mean=np.zeros(3), cov=R, size=1)
    x_sensor = x_true + n[0][0]
    y_sensor = y_true + n[0][0]
    theta_sensor = theta_true + n[0][0]
    # Array to give  to kf to handle np array
    X_state_sensor = [[x_sensor], [y_sensor], [theta_sensor]]
    pprint(X_state_sensor)
    
    
    print("Sensor noise: ", n[0])
    print("Kinematic noise: ", w[0])
    print("\n---Kalman Filter----")
    kf.predict(U)
    kf.measure(X_state_sensor)
    kf_state = kf.get_posteriori()
    print("kf states: ", kf_state)
    

    ### Plot ###
    kinematic_plot_x.append(x_kin)
    sensor_plot_x.append(x_sensor)

    kinematic_plot_y.append(y_kin)
    sensor_plot_y.append(y_sensor)

    kalman_x.append(kf_state[0][0])
    kalman_y.append(kf_state[1][0])

    #time.sleep(0.5)
print(kf.get_posteriori())

plt.plot(kinematic_plot_x, kinematic_plot_y, color='green', label='Kinematic Model')
plt.plot(sensor_plot_x, sensor_plot_y, color='red', label='Odom Data')
plt.plot(kalman_x, kalman_y, color='blue', label='Kalman' )

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# Add x and y axes at origin
plt.axhline(0, color='black', linewidth=0.5)  # Horizontal line at y=0
plt.axvline(0, color='black', linewidth=0.5)  # Vertical line at x=0

plt.legend()
plt.show()
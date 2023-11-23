from extended_kalman_filter import ExtendedKalmanFilter
import numpy as np
from pprint import pprint
import matplotlib
import time
import random

#### Initialization ####


# Assume Odometer data has std of 0.1m 
R = np.eye(3) * 0.1

# Assume kinematic model / inputs has std of 0.1m
Q = np.eye(3) * 0.1

# Initialize error covariance matrix
P = np.eye(3)

# Transform matrix for odometer data (Y = CX + n) where n ~ N(0, R)
C = np.eye(3)

# Velocity of 1 m/s right
v = 1
omega = 0

# angular speed
omega = 0.5

# Time step 
dt = 0.1

# Duration = 10 seconds
total_time = 10 

# Set initial position to origin
x_0 = 0
y_0 = 0
theta_0 = 0

# Track true movements of robot
x_true = 0
y_true = 0
theta_true = 0

# Tracking of robot from kinematics
x_kin = 0
y_kin = 0
theta_kin = 0

# Tracking of robot from sensor
x_sensor = 0
y_sensor = 0
theta_sensor = 0

# Initial Inputs
x_dot = v * np.cos(0)
y_dot = v * np.sin(0)
theta_dot = omega

x = [x_0, y_0, theta_0]

kf = ExtendedKalmanFilter(x, Q, R, P, v)

for _ in range (int(total_time / dt)):

    # Control input noise
    w = np.random.multivariate_normal(mean=np.zeros(3), cov=Q, size=1).T
    theta_true += dt * theta_dot
    # calculate x and y velocity from theta angle
    x_dot = v * np.cos(theta_true)
    y_dot = v * np.sin(theta_true)

    x_true += dt * x_dot
    y_true += dt * y_dot

    theta_dot = omega


    # Control input
    U = [v, omega]

    factor = random.choice([-1, 1])
    offset = random.random()

    print("Velocity: ", v, " Angular Velocity: ", theta_dot, "\n")

    X_dot_true = [[x_dot], [y_dot], [theta_dot]]
    X_state_true = [[x_true], [y_true], [theta_dot]]
    print("---- True X ", _ ," ----")
    pprint(X_state_true)


    print("\n----Kinematic Model ----")
    
    X_dot_kin = X_dot_true + w
    x_kin += X_dot_kin[0] * dt
    y_kin += X_dot_kin[1] * dt
    theta_kin += X_dot_kin[2] * dt
    X_state_kin = [[x_kin], [y_kin], [theta_kin]]
    pprint(X_state_kin)

    
    print("---Sensor Model----")
    # Y value
    n = np.random.multivariate_normal(mean=np.zeros(3), cov=R, size=1).T
    X_dot_sensor = X_dot_true + n
    x_sensor += X_dot_sensor[0] * dt
    y_sensor += X_dot_sensor[1] * dt
    theta_sensor += X_dot_sensor[2] * dt
    X_state_sensor = [[x_sensor], [y_sensor], [theta_sensor]]
    pprint(X_state_sensor)
    

    print("\n")
    time.sleep(0.5)
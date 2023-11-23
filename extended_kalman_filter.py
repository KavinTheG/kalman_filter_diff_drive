import numpy as np

class ExtendedKalmanFilter:

    def __init__(self, X, Q, R, P, v, dt = 0.1) -> None:

        # x_k
        self.X_posteriori = X
        self.x = self.X_posteriori[0]
        self.y = self.X_posteriori[1]
        self.theta = self.X_posteriori[2]
        self.v = v
        self.X_previous = self.X_posteriori

        # x_k^-
        self.a_priori_state = X

        self.Q = Q
        self.R = R

        self.P_posteriori = P # a posteriori error
        self.P_priori = self.P_posteriori
        self.P_previous = self.P_posteriori
        
        # Set delta initially to 0.1 seconds
        self.deltaT = dt

        # Set up matrix A
        self.A = np.eye(3)
        self.A[0][2] = v * self.deltaT * np.sin(self.theta)

        # Set up matrix B
        self.B = np.zeros((3,2))
        # delta theta is equal self.theta - self.X_prev[2]
        self.B[0][0] = self.deltaT * np.cos(self.theta + (self.theta - self.X_previous[2]) / 2 )
        self.B[1][0] = self.deltaT * np.sin(self.theta + (self.theta - self.X_previous[2]) / 2 )
        self.B[2][1] = 1

        self.C = np.eye(3)






    def predict(self, U):
        # Set up Q1
        Q1 = [
              [self.Q[0][0] * self.deltaT + self.Q[2][2]*(self.deltaT**3 /3)* self.v**2 * np.sin(self.theta)**2],
              [-self.Q[2][2] * (self.deltaT**3 /3) * self.v**2 * np.sin(self.theta) * np.cos(self.theta)],
              [-self.Q[2][2] * (self.deltaT**2 /2) * self.v * np.sin(self.theta) ]
             ]

        # Set up Q2
        Q2 = [
              [-self.Q[2][2] * (self.deltaT**3 /3)* self.v**2 * np.sin(self.theta) * np.cos(self.theta)],
              [self.Q[1][1] * self.deltaT + self.Q[2][2]*(self.deltaT**3 /3)* self.v**2 * np.cos(self.theta)**2],
              [self.Q[2][2] * (self.deltaT**2 /2) * self.v * np.cos(self.theta) ]
             ]
        
        # Set up Q3
        Q3 = [
              [-self.Q[2][2] * (self.deltaT**2 /2)* self.v * np.sin(self.theta)],
              [self.Q[2][2] * (self.deltaT**2 /2)* self.v * np.cos(self.theta)],
              [self.Q[2][2] * self.deltaT ]
        ]

        # Set up Qk
        self.Qk = [Q1, Q2, Q3]

        # x_{k+1}^-
        self.X_priori = self.X_previous + self.B * U

        # P_{k+1}^-
        self.P_priori = self.A * self.P_previous * self.A.T + self.Qk


    
    def measure(self, Y):
        
        S = self.C * self.P_priori * self.C.T + self.R 

        # K_{k+1}
        self.kalman_gain = self.P_priori * self.C.T / S

        # x_{k+1}
        self.X_posteriori = self.X_priori + self.kalman_gain * (Y - self.C * self.X_priori)
        self.X_previous = self.X_posteriori

        # P_{k+1}
        self.P_posteriori = (np.eye(3) - self.kalman_gain * self.C) * self.P_priori
        self.P_previous = self.P_posteriori


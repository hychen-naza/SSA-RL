import numpy as np
import cvxopt
import sys

class ControlBarrierFunction():
    def __init__(self, max_speed, fake_env = None, dmin = 0.12, yita = 1, max_acc = 0.1, max_steering = np.pi/4):
        """
        Args:
            dmin: dmin for bx
            yita: yita for bx
        """
        self.dmin = dmin
        self.yita = yita
        self.max_acc = max_acc
        self.max_steering = max_steering
        self.max_speed = max_speed
        self.forecast_step = 3
        self.fake_env = fake_env

    def get_safe_control(self, full_robot_state, obs_states, f, g, u0):
        """
        Args:
            full_robot_state <x, y, vx, vy, theta>
            robot_state: np array current robot state <x, y, vx, vy>
            obs_state: np array closest static obstacle state <x, y, 0, 0>
            bx: barrier function -- dmin**2 - d**2
        """
        u0 = np.array(u0).reshape((2,1))
        robot_state = full_robot_state[:4]
        full_robot_state = full_robot_state.reshape((-1,1)) # shape (5, 1)
        L_gs = []
        reference_control_laws = []
        is_safe = True
        for obs_state in obs_states:
            d = np.array(robot_state - obs_state)
            d_pos = d[:2] # pos distance
            d_dot = d[2:] # vel 
            bx = self.dmin**2 - np.linalg.norm(d_pos) # barrier function
            # calculate Lie derivative
            p_d_p_robot_state = np.eye(5)
            p_d_pos_p_d = np.array([d_pos[0], d_pos[1], 0, 0, 0]).reshape((1,5)) / np.linalg.norm(d_pos) # shape (1, 5)
            p_d_pos_p_state = p_d_pos_p_d @ p_d_p_robot_state # shape (1, 5)

            p_bx_p_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_state
            
            L_f = p_bx_p_state @ (f @ full_robot_state) # shape (1, 1)
            L_g = p_bx_p_state @ g # shape (1, 2) g contains x information

            L_gs.append(-L_g)
            reference_control_laws.append(self.yita*bx + L_f + L_g @ u0)
            is_safe = False
        
        if (not is_safe):
            # Solve safe optimization problem
            # min_x (1/2 * x^T * Q * x) + (f^T * x)   s.t. Ax <= b
            # 
            u0 = u0.reshape(-1,1)
            Q = cvxopt.matrix(np.eye(2)) #
            p = cvxopt.matrix(np.zeros(2).reshape(-1,1))
            G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2)]))
            S_saturated = cvxopt.matrix(np.array([self.max_acc-u0[0][0], self.max_steering-u0[1][0], self.max_acc+u0[0][0], self.max_steering+u0[1][0]]).reshape(-1, 1))
            L_gs = np.array(L_gs).reshape(-1, 2)
            reference_control_laws = np.array(reference_control_laws).reshape(-1,1)
            #print(reference_control_laws)
            A = cvxopt.matrix([[cvxopt.matrix(L_gs), G]])
            #print(A.size)
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['maxiters'] = 300
            while True:
                try:
                    b = cvxopt.matrix([[cvxopt.matrix(reference_control_laws), S_saturated]])
                    sol = cvxopt.solvers.qp(Q, p, A, b)
                    u = sol["x"]
                    if (abs(u[0]) > 1e-2 or abs(u[1]) > 1e-2):
                        print(u)
                    break
                except ValueError:
                    # no solution, relax the constraint
                    # print(f"relax reference_control_law")
                    for i in range(len(reference_control_laws)):
                        reference_control_laws[i][0] += 0.2

            u = np.array([u[0]+u0[0][0], u[1]+u0[1][0]])
            u[0] = max(min(u[0], self.max_acc), -self.max_acc)
            u[1] = max(min(u[1], self.max_steering), -self.max_steering)
            return u, True
        
        u0 = u0.reshape(1,2)
        u = u0
        return u[0], False
        
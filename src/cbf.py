import numpy as np
import cvxopt
import sys

class ControlBarrierFunction():
    def __init__(self, max_speed, dmin = 0.12, k = 1, max_acc = 0.04):
        """
        Args:
            dmin: dmin for bx
            yita: yita for bx
        """
        self.dmin = dmin
        self.k = k
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.forecast_step = 3

    def get_safe_control(self, robot_state, obs_states, f, g, u0):
        """
        Args:
            robot_state <x, y, vx, vy>
            robot_state: np array current robot state <x, y, vx, vy>
            obs_state: np array closest static obstacle state <x, y, 0, 0>
            bx: barrier function -- dmin**2 - d**2
        """
        u0 = np.array(u0).reshape((2,1))
        L_gs = []
        L_fs = []
        obs_dots = []
        reference_control_laws = []
        is_safe = True

        for i, obs_state in enumerate(obs_states):
            d = np.array(robot_state - obs_state[:4])
            d_pos = d[:2] # pos distance
            d_vel = d[2:] # vel 
            d_abs = np.linalg.norm(d_pos)
            d_dot = self.k * (d_pos @ d_vel.T) / np.linalg.norm(d_pos)
            phi = np.power(self.dmin, 2) - np.power(np.linalg.norm(d_pos), 2) - d_dot

            # calculate Lie derivative
            # p d to p robot state and p obstacle state
            p_d_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_p_obs_state = np.hstack([-1*np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_pos_p_d = np.array([d_pos[0], d_pos[1]]).reshape((1,2)) / d_abs # shape (1, 2)
            p_d_pos_p_robot_state = p_d_pos_p_d @ p_d_p_robot_state # shape (1, 4)
            p_d_pos_p_obs_state = p_d_pos_p_d @ p_d_p_obs_state # shape (1, 4)

            # p d_dot to p robot state and p obstacle state
            p_vel_p_robot_state = np.hstack([np.zeros((2,2)), np.eye(2)]) # shape (2, 4)
            p_vel_p_obs_state = np.hstack([np.zeros((2,2)), -1*np.eye(2)]) # shape (2, 4)
            p_d_dot_p_vel = d_pos.reshape((1,2)) / d_abs # shape (1, 2)
            
            p_pos_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_pos_p_obs_state = np.hstack([-1*np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_dot_p_pos = d_vel / d_abs - 0.5 * (d_pos @ d_vel.T) * d_pos / np.power(d_abs, 3) 
            p_d_dot_p_pos = p_d_dot_p_pos.reshape((1,2)) # shape (1, 2)

            p_d_dot_p_robot_state = p_d_dot_p_pos @ p_pos_p_robot_state + p_d_dot_p_vel @ p_vel_p_robot_state # shape (1, 4)
            p_d_dot_p_obs_state = p_d_dot_p_pos @ p_pos_p_obs_state + p_d_dot_p_vel @ p_vel_p_obs_state # shape (1, 4)

            p_phi_p_robot_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_robot_state - \
                            self.k * p_d_dot_p_robot_state # shape (1, 4)
            p_phi_p_obs_state = -2 * np.linalg.norm(d_pos) * p_d_pos_p_obs_state - \
                            self.k * p_d_dot_p_obs_state # shape (1, 4)
        
            L_f = p_phi_p_robot_state @ (f @ robot_state.reshape((-1,1))) # shape (1, 1)
            L_g = p_phi_p_robot_state @ g # shape (1, 2) g contains x information
            obs_dot = p_phi_p_obs_state @ obs_state[-4:]
            L_fs.append(L_f)
            L_gs.append(L_g)
            obs_dots.append(obs_dot)
            reference_control_laws.append( -0.5*phi - L_f - obs_dot - np.dot(L_g, u0))
        

        # Solve safe optimization problem
        # min_x (1/2 * x^T * Q * x) + (f^T * x)   s.t. Ax <= b
        u0 = u0.reshape(-1,1)
        Q = cvxopt.matrix(np.eye(2)) #
        p = cvxopt.matrix(np.zeros(2).reshape(-1,1))
        G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2), np.array([[1,0],[-1,0]]), np.array([[0,1],[0,-1]])]))
        S_saturated = cvxopt.matrix(np.array([self.max_acc-u0[0][0], self.max_acc-u0[1][0], self.max_acc+u0[0][0], self.max_acc+u0[1][0], \
                                    self.max_speed-robot_state[2]-u0[0][0], self.max_speed+robot_state[2]+u0[0][0], \
                                    self.max_speed-robot_state[3]-u0[1][0], self.max_speed+robot_state[3]+u0[1][0]]).reshape(-1, 1))

        L_gs = np.array(L_gs).reshape(-1, 2)
        reference_control_laws = np.array(reference_control_laws).reshape(-1,1)
        A = cvxopt.matrix([[cvxopt.matrix(L_gs), G]])
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 600
        while True:
            try:

                b = cvxopt.matrix([[cvxopt.matrix(reference_control_laws), S_saturated]])
                #import pdb
                #pdb.set_trace()
                #print(f"Q {Q}, p {p}, A {A}, b {b}")
                sol = cvxopt.solvers.qp(Q, p, A, b)
                u = sol["x"]
                #if (abs(u[0]) > 1e-2 or abs(u[1]) > 1e-2):
                #    print(u)
                break
            except ValueError:
                # no solution, relax the constraint
                #print(f"relax reference_control_law\n\n\n")
                for i in range(len(reference_control_laws)):
                    reference_control_laws[i][0] += 0.01

        u = np.array([u[0]+u0[0][0], u[1]+u0[1][0]])
        u[0] = max(min(u[0], self.max_acc), -self.max_acc)
        u[1] = max(min(u[1], self.max_acc), -self.max_acc)
        return u, True
        
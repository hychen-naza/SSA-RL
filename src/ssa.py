import numpy as np
import math
import cvxopt
import sys
import collections

class SafeSetAlgorithm():
    def __init__(self, max_speed, is_qp = False, dmin = 0.12, k = 1, max_acc = 0.04):
        """
        Args:
            dmin: dmin for phi
            k: k for d_dot in phi
        """
        self.dmin = dmin
        self.k = k
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.forecast_step = 3
        self.records = collections.deque(maxlen = 10)
        self.acc_reward_normal_ssa = 0
        self.acc_reward_qp_ssa = 0
        self.acc_phi_dot_ssa = 0
        self.acc_phi_dot_qp = 0
        self.is_qp = is_qp

    def get_safe_control(self, robot_state, obs_states, f, g, u0):
        """
        Args:
            robot_state <x, y, vx, vy>
            obs_state: np array closest static obstacle state <x, y, vx, vy, ax, ay>
        """
        u0 = np.array(u0).reshape((2,1))
        robot_vel = np.linalg.norm(robot_state[-2:])
        
        L_gs = []
        L_fs = []
        reference_control_laws = []
        is_safe = True

        phis = []
        danger_indexs = []

        for i, obs_state in enumerate(obs_states):
            d = np.array(robot_state - obs_state[:4])
            d_pos = d[:2] # relative pos
            d_vel = d[2:] # relative vel 
            d_abs = np.linalg.norm(d_pos)
            d_dot = self.k * (d_pos @ d_vel.T) / d_abs
            phi = np.power(self.dmin, 2) - np.power(d_abs, 2) - d_dot
            
            # calculate Lie derivative
            # p d to p robot state and p obstacle state
            p_pos_p_robot_state = np.hstack([np.eye(2), np.zeros((2,2))]) # shape (2, 4)
            p_d_pos_p_pos = np.array([d_pos[0], d_pos[1]]).reshape((1,2)) / d_abs # shape (1, 2)
            p_d_pos_p_robot_state = p_d_pos_p_pos @ p_pos_p_robot_state # shape (1, 4)
            p_d_p_robot_state = -2 * d_abs * p_d_pos_p_robot_state

            # p d_dot to p robot state and p obstacle state
            p_vel_p_robot_state = np.hstack([np.zeros((2,2)), np.eye(2)]) # shape (2, 4)
            term1 = (d_pos.reshape((1,2)) / d_abs) @ p_vel_p_robot_state

            term2 = (p_pos_p_robot_state * d_abs - d_pos.reshape((2,1)) @ p_d_pos_p_robot_state) / np.power(d_abs, 2)
            term2 = d_vel.reshape((1,2)) @ term2
            p_d_dot_p_robot_state = term1 + term2  # shape (1, 4)
            p_phi_p_robot_state = p_d_p_robot_state - self.k * p_d_dot_p_robot_state # shape (1, 4)

            L_f = p_phi_p_robot_state @ (f @ robot_state.reshape((-1,1))) # shape (1, 1)
            L_g = p_phi_p_robot_state @ g # shape (1, 2) g contains x information
            L_fs.append(L_f) 

            if (phi > 0):
                L_gs.append(L_g)                                              
                reference_control_laws.append( -0.5*phi - L_f) 
                is_safe = False
                danger_indexs.append(i)

        if (not is_safe):
            # Solve safe optimization problem
            # min_x (1/2 * x^T * Q * x) + (f^T * x)   s.t. Ax <= b
            u0 = u0.reshape(-1,1)
            qp_parameter = self.find_qp(robot_state, obs_states, u0.flatten())
            u, reference_control_laws = self.solve_qp(robot_state, u0, L_gs, reference_control_laws, phis, qp_parameter, danger_indexs)
            return u, True                        
        u0 = u0.reshape(1,2)
        u = u0
        return u[0], False

    def check_same_direction(self, pcontrol, perpendicular_controls):
        if (len(perpendicular_controls) == 0):
            return True
        for control in perpendicular_controls:
            angle = self.calcu_angle(pcontrol, control)
            if (angle > np.pi/4):
                return False
        return True

    def calcu_angle(self, v1, v2):
        lv1 = np.sqrt(np.dot(v1, v1))
        lv2 = np.sqrt(np.dot(v2, v2))
        angle = np.dot(v1, v2) / (lv1*lv2)
        return np.arccos(angle)

    def solve_qp(self, robot_state, u0, L_gs, reference_control_laws, phis, qp_parameter, danger_indexs):
        q = qp_parameter
        Q = cvxopt.matrix(q)
        u_prime = -u0
        u_prime = qp_parameter @ u_prime
        p = cvxopt.matrix(u_prime) #-u0
        G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2), np.array([[1,0],[-1,0]]), np.array([[0,1],[0,-1]])]))
        S_saturated = cvxopt.matrix(np.array([self.max_acc, self.max_acc, self.max_acc, self.max_acc, \
                                    self.max_speed-robot_state[2], self.max_speed+robot_state[2], \
                                    self.max_speed-robot_state[3], self.max_speed+robot_state[3]]).reshape(-1, 1))
        #G = cvxopt.matrix(np.vstack([np.eye(2), -np.eye(2)]))
        #S_saturated = cvxopt.matrix(np.array([self.max_acc, self.max_acc, self.max_acc, self.max_acc]).reshape(-1, 1))
        L_gs = np.array(L_gs).reshape(-1, 2)
        reference_control_laws = np.array(reference_control_laws).reshape(-1,1)
        A = cvxopt.matrix([[cvxopt.matrix(L_gs), G]])
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 600
        while True:
            try:
                b = cvxopt.matrix([[cvxopt.matrix(reference_control_laws), S_saturated]])
                sol = cvxopt.solvers.qp(Q, p, A, b)
                u = sol["x"]
                break
            except ValueError:
                # no solution, relax the constraint   
                is_danger = False                 
                for i in range(len(reference_control_laws)):
                    if (self.is_qp and i in danger_indexs):
                        reference_control_laws[i][0] += 0.01
                        if (reference_control_laws[i][0] + phis[i] > 0):
                            is_danger = True
                    else:
                        reference_control_laws[i][0] += 0.01
        u = np.array([u[0], u[1]])
        return u, reference_control_laws

    def find_qp(self, robot_state, obs_states, u0, safest = False):
        if (not self.is_qp):
            return np.eye(2)
        # estimate obstacle positions in next few steps
        obs_poses = []
        for i in range(self.forecast_step):
            for obs in obs_states:
                obs_poses.append([obs[0]+i*obs[2]-robot_state[0], obs[1]+i*obs[3]-robot_state[1]])

        eigenvectors, max_dis_theta, min_dis_theta = self.find_eigenvector(robot_state, obs_poses)
        eigenvalues = self.find_eigenvalue(obs_poses, max_dis_theta, min_dis_theta)
        R = np.array([eigenvectors[0],eigenvectors[1]]).T
        R_inv = np.linalg.pinv(R)
        Omega = np.array([[eigenvalues[0], 0], [0, eigenvalues[1]]])
        qp = R @ Omega @ R_inv
        return qp

    def find_eigenvector(self, robot_state, obs_poses):
        xs = np.array([pos[0] for pos in obs_poses])
        ys = np.array([pos[1] for pos in obs_poses])

        theta1 = 0.5*np.arctan2(2*np.dot(xs,ys), np.sum(xs**2-ys**2))
        theta2 = theta1+np.pi/2

        first_order_theta1 = 0.5*np.sin(2*theta1)*np.sum(xs**2-ys**2) - np.cos(2*theta1)*np.dot(xs,ys)
        first_order_theta2 = 0.5*np.sin(2*theta2)*np.sum(xs**2-ys**2) - np.cos(2*theta2)*np.dot(xs,ys)

        second_order_theta1 = np.cos(2*theta1)*np.sum(xs**2-ys**2) + 2*np.sin(2*theta1)*np.dot(xs,ys)
        second_order_theta2 = np.cos(2*theta2)*np.sum(xs**2-ys**2) + 2*np.sin(2*theta2)*np.dot(xs,ys)
        
        if (second_order_theta1 < 0):            
            max_dis_theta = theta1
            min_dis_theta = theta2
        else:
            max_dis_theta = theta2
            min_dis_theta = theta1
        lambda1 = [np.cos(max_dis_theta), np.sin(max_dis_theta)]
        lambda2 = [np.cos(min_dis_theta), np.sin(min_dis_theta)]
        return [lambda1, lambda2], max_dis_theta, min_dis_theta

    def find_eigenvalue(self, obs_poses, max_dis_theta, min_dis_theta):
        max_dis = 0.
        min_dis = 0.

        xs = np.array([pos[0] for pos in obs_poses])
        ys = np.array([pos[1] for pos in obs_poses])

        for x, y in zip(xs, ys):
            max_dis += (-np.sin(max_dis_theta)*x + np.cos(max_dis_theta)*y)**2
            min_dis += (-np.sin(min_dis_theta)*x + np.cos(min_dis_theta)*y)**2
        return [min_dis*1e5, max_dis*1e5]

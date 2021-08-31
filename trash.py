# count the number of obstacles in vehicle surrounding regions
        counts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
        for pos in obs_poses:
            angle = math.atan2(pos[0]-robot_state[0], pos[1]-robot_state[1])*180/math.pi
            if (angle < 0):
                angle += 360
            region_num = angle // 45
            counts[int(region_num)] += 1

        # find the safest direction
        region = min(counts, key = counts.get)
        max_reward = float("-inf")
        best_region = None
        if (self.fake_env.model.reward_mse < 1e-3):
            min_count = counts[region]
            robot_state = np.array(robot_state)
            for key, value in counts.items():
                if (value == min_count):
                    angle = (key * 45 + (key + 1) * 45) / 2.
                    angle = angle * math.pi / 180
                    action = np.array([math.sin(angle), math.cos(angle)])
                    _, reward, _ = self.fake_env.step(robot_state, action)
                    if (reward[0] > max_reward):
                        max_reward = reward[0]
                        best_region = key                    
                    #print(f"robot_state {robot_state}, action {action}, reward {reward}")
            #print(f"counts {counts}, region {region}, best_region {best_region}")
            region = best_region

        #print(f"fake_env.model.reward_mse {}")
        safe_angle = (region * 45 + (region + 1) * 45) / 2.
        safe_angle = safe_angle * math.pi / 180
        safe_direction = [math.sin(safe_angle), math.cos(safe_angle)] # (x, y) direction
        vel = np.linalg.norm(robot_state[-2:])
        vel_direction = robot_state[-2:] / (vel + 1e-6)

        # calculate the safe control direction
        # safe_control + vel_direction = safe_direction
        safe_control_direction = [safe_direction[0]-vel_direction[0], safe_direction[1]-vel_direction[1]]
        ref_control = np.linalg.norm(u0)
        ref_control_direction = u0 / ref_control
        
        # calculate qp parameter
        x_parameter = safe_control_direction[0] * ref_control_direction[0]
        y_parameter = safe_control_direction[1] * ref_control_direction[1]
        if (x_parameter < 0 or y_parameter < 0):
            bias = 1 - min(x_parameter, y_parameter)
            x_parameter += bias
            y_parameter += bias
        return [x_parameter*100, y_parameter*100]



                    '''
            if (len(self.records) >= 3):
                past_records = []
                past_safe_control = True
                past_multi_obstacles = True
                for i in range(3):
                    tmp = self.records.pop()
                    past_safe_control = past_safe_control and tmp['is_safe_control']
                    past_multi_obstacles = past_multi_obstacles and tmp['is_multi_obstacles']
                    past_records.append(tmp)

                if (past_safe_control and past_multi_obstacles):
                    #print(f"past_records['control'] {past_records[0]['control']}, {past_records[1]['control']}, {past_records[2]['control']}")
                    angle1 = self.calcu_angle(past_records[0]['control'], past_records[2]['control'])
                    angle2 = self.calcu_angle(past_records[1]['control'], past_records[2]['control'])
                    if (angle1 < np.pi/4 and angle2 > (3*np.pi)/4):
                        perpendicular_controls = []
                        for i in range(3):
                            control = past_records[i]['control']
                            pcontrol = [-1*control[1], control[0]]
                            if (self.check_same_direction(pcontrol, perpendicular_controls)):
                                perpendicular_controls.append(pcontrol)
                            else:
                                pcontrol = [control[1], -1*control[0]]
                                if (self.check_same_direction(pcontrol, perpendicular_controls)):
                                    perpendicular_controls.append(pcontrol)

                        if (len(perpendicular_controls) == 3):
                            #print(f"perpendicular_controls {perpendicular_controls}, warning!!!!!!!!!\n\n\n")
                            #else:
                            u = np.mean(perpendicular_controls, axis = 0)
                            print(f"perpendicular_control is {u}")
                # add back
                past_records.reverse()
                for i in range(3):
                    self.records.append(past_records[i])
            '''
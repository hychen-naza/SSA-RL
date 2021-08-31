
######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

from builtins import object
import random
import numpy as np
import math

def l2( xy0, xy1 ):
    ox = xy1[0]
    oy = xy1[1]
    dx = xy0[0] - xy1[0]
    dy = xy0[1] - xy1[1]
    dist = math.sqrt( (dx * dx) + (dy * dy) )
    if (xy1[0] < -0.9):
        warp_dx = xy0[0] - (1 + (xy1[0] + 1))
        dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
        if (dist1 < dist):
            ox = (1 + (xy1[0] + 1))
            dist = dist1
            #print(f"case1")
    elif (xy1[0] > 0.9):
        warp_dx = xy0[0] - (-1 + (xy1[0] - 1))
        dist1 = math.sqrt( (warp_dx * warp_dx) + (dy * dy) )
        if (dist1 < dist):
            ox = (-1 + (xy1[0] - 1))
            dist = dist1
            #print(f"case2")
    return dist, ox, oy


class Asteroid(object):

    def __init__(self,
                 a_x, b_x, c_x,
                 a_y, b_y, c_y, t_start):
        self.a_x = a_x / 5
        self.b_x = b_x
        self.c_x = c_x
        self.a_y = a_y / 5
        self.b_y = b_y
        self.c_y = c_y
        self.t_start = t_start

    @property
    def params(self):
        return { 'a_x':     self.a_x,
                 'b_x':     self.b_x,
                 'c_x':     self.c_x,
                 'a_y':     self.a_y,
                 'b_y':     self.b_y,
                 'c_y':     self.c_y,
                 't_start': self.t_start }

    def x(self, t):

        t_shifted = t - self.t_start
        
        x = ((self.a_x * t_shifted * t_shifted)
             + (self.b_x * t_shifted)
             + self.c_x)

        return x

    def y(self, t):

        t_shifted = t - self.t_start
        
        y = ((self.a_y * t_shifted * t_shifted)
             + (self.b_y * t_shifted)
             + self.c_y)

        return y

    def v_x(self, t):

        t_shifted = t - self.t_start
        
        v_x = ((2 * self.a_x * t_shifted)
             + self.b_x)

        return v_x

    def v_y(self, t):

        t_shifted = t - self.t_start
        
        v_y = ((2 * self.a_y * t_shifted)
             + self.b_y)

        return v_y

FIELD_X_BOUNDS = (-0.95, 0.95)
FIELD_Y_BOUNDS = (-0.95, 1.0)

class AsteroidField(object):
    
    def __init__(self,
                 x_bounds = FIELD_X_BOUNDS,
                 y_bounds = FIELD_Y_BOUNDS):

        self.random_init()
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def random_init(self):
        asteroids = []
        for i in range(50):
            asteroids.append(self.random_init_obstacle(t = -100))
        self.asteroids = asteroids
        return 

    def random_init_obstacle(self, t, vehicle_x = 0, vehicle_y = -1, min_dist = 0.1):
        dist = -1
        x = y = -1
        while (dist < min_dist):
            x = random.uniform(FIELD_X_BOUNDS[0],FIELD_X_BOUNDS[1])
            y = random.uniform(FIELD_Y_BOUNDS[0],FIELD_Y_BOUNDS[1])
            dist = math.sqrt((vehicle_x-x)**2 + (vehicle_y-y)**2)
        b_x = random.uniform(1e-3, 1e-2) * random.choice([1,-1])
        b_y = random.uniform(1e-3, 1e-2) * random.choice([1,-1])
        a_x = random.uniform(5e-6, 1e-4) * random.choice([1,-1])
        a_y = random.uniform(5e-6, 1e-4) * random.choice([1,-1])
        return Asteroid(a_x, b_x, x, a_y, b_y, y, t)

    def unsafe_asteroid_locations(self, t, cx, cy, min_dist):

        """
        Returns (i, x, y) tuples indicating that the i-th asteroid is
        at location (x,y).
        """        
        locs =  [ (i, a.x(t), a.y(t), a.v_x(t), a.v_y(t), a.a_x, a.a_y)
                  for i,a in enumerate(self.asteroids)]
        unsafe_asteroids = []
        for i,x,y,x_v,y_v,x_a,y_a  in locs:
            if self.x_bounds[0] <= x <= self.x_bounds[1] and self.y_bounds[0] <= y <= self.y_bounds[1]:
                dist, ox, oy = l2([cx,cy], [x,y])
                if dist < min_dist:
                    unsafe_asteroids.append([i,(ox,oy,x_v,y_v,x_a,y_a)])
        return  unsafe_asteroids
        #return [ [i,(x,y,x_v,y_v,x_a,y_a)] for i,x,y,x_v,y_v,x_a,y_a  in locs
        #         if self.x_bounds[0] <= x <= self.x_bounds[1]
        #         and self.y_bounds[0] <= y <= self.y_bounds[1] and l2([x,y],[cx,cy]) < min_dist]

    def asteroid_locations(self, t, vehicle_x, vehicle_y, min_dist):

        """
        Returns (i, x, y) tuples indicating that the i-th asteroid is
        at location (x,y).
        """
        locs = []
        for i, a in enumerate(self.asteroids):
            if self.x_bounds[0] <= a.x(t) <= self.x_bounds[1] and self.y_bounds[0] <= a.y(t) <= self.y_bounds[1]:
                locs.append((i, a.x(t), a.y(t)))
            else:
                self.asteroids[i] = self.random_init_obstacle(t, vehicle_x, vehicle_y, min_dist)
                locs.append((i, a.x(t), a.y(t)))
        return locs

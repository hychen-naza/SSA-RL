
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

from builtins import str
from builtins import object
import math
import numpy as np

SPEED_CHANGE_ACCELERATE=1
SPEED_CHANGE_NONE=0
SPEED_CHANGE_DECELERATE=-1

SPEED_CHANGES = (SPEED_CHANGE_ACCELERATE,
                 SPEED_CHANGE_NONE,
                 SPEED_CHANGE_DECELERATE)

ANGLE_CHANGE_LEFT=1
ANGLE_CHANGE_NONE=0
ANGLE_CHANGE_RIGHT=-1

ANGLE_CHANGES = (ANGLE_CHANGE_LEFT,
                 ANGLE_CHANGE_NONE,
                 ANGLE_CHANGE_RIGHT)

class CraftState(object):

    def __init__(self, x, y, h, v, max_speed, speed_increment, angle_increment):

        self.x = x
        self.y = y
        self.h = h
        self.v = v
        self.v_x = v * math.cos(h)
        self.v_y = v * math.sin(h)
        self.max_speed = max_speed
        self.speed_increment = speed_increment
        self.angle_increment = angle_increment


    def __repr__(self):
        return "CraftState( x=%0.04f, y=%0.04f, h=%0.04f, v_x=%0.04f, v_y=%0.04f, max_speed=%0.04f, speed_increment=%0.04f, angle_increment=%0.04f)" % \
            (self.x, self.y, self.h, self.v_x, self.v_y, self.max_speed, self.speed_increment, self.angle_increment)

    @property
    def position(self):
        return (self.x, self.y, self.h)

    def steer(self, vx_speed_change, vy_speed_change):

        """
        Returns a new CraftState object.
        """
        
        #if angle_change not in ANGLE_CHANGES:
        #    raise RuntimeError('invalid angle change %s' % str(angle_change))

        #new_h = self.h + (angle_change)
        #new_h = new_h % (2.0 * math.pi)

        #if speed_change not in SPEED_CHANGES:
        #    raise RuntimeError('invalid speed change %s' % str(speed_change))
        #print(f"states ({self.x}, {self.y}, {self.v_x}, {self.v_y}), action {vx_speed_change}, {vy_speed_change}")
        self.v_x += vx_speed_change
        self.v_y += vy_speed_change
        self.v_x = max(min(self.v_x, self.max_speed), -self.max_speed)
        self.v_y = max(min(self.v_y, self.max_speed), -self.max_speed)
        
        new_v = np.linalg.norm([self.v_x, self.v_y])
        new_h = math.atan2(self.v_y, self.v_x)

        new_x = self.x + (new_v * math.cos( new_h ))
        new_y = self.y + (new_v * math.sin( new_h ))
        # warp around
        if (new_x > 1):
            new_x = -1 + (new_x - 1) 
        if (new_x < -1):
            new_x = 1 + (new_x + 1) 
        if (new_y < -1):
            new_y = -1
        #print(f"new states ({new_x}, {new_y})")
        return CraftState( x               = new_x,
                           y               = new_y,
                           h               = new_h,
                           v               = new_v,
                           max_speed       = self.max_speed,
                           speed_increment = self.speed_increment,
                           angle_increment = self.angle_increment )


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

from builtins import range
from builtins import object

SUCCESS = 'success'
FAILURE_TOO_MANY_STEPS = 'too_many_steps'

# Custom failure states for navigation.
NAV_FAILURE_COLLISION = 'collision'
NAV_FAILURE_OUT_OF_BOUNDS = 'out_of_bounds'

class BaseRunnerDisplay(object):

    def setup(self, x_bounds, y_bounds,
              in_bounds, goal_bounds,
              margin):
        pass
    
    def begin_time_step(self, t):
        pass

    def asteroid_at_loc(self, i, x, y):
        pass

    def asteroid_estimated_at_loc(self, i, x, y, is_match=False):
        pass

    def asteroid_estimates_compared(self, num_matched, num_total):
        pass

    def asteroid_set_color(self, id, color):
        pass

    def craft_at_loc(self, x, y, h, is_safe):
        pass

    def craft_steers(self, dh, dv):
        pass

    def collision(self):
        pass

    def out_of_bounds(self):
        pass

    def goal(self):
        pass

    def navigation_done(self, retcode, t):
        pass

    def estimation_done(self, retcode, t):
        pass
    
    def end_time_step(self, t):
        pass

    def teardown(self):
        pass

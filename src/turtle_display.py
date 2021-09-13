from __future__ import division
from __future__ import absolute_import

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

from past.utils import old_div
import math
import time
import turtle
import runner

class TurtleRunnerDisplay( runner.BaseRunnerDisplay ):

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x_bounds = (0.0,1.0)
        self.y_bounds = (0.0,1.0)
        self.asteroid_turtles = {}
        self.estimated_asteroid_turtles = {}
        self.craft_turtle = None

    def setup(self, x_bounds, y_bounds,
              in_bounds, goal_bounds,
              margin):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.margin = margin
        xmin,xmax = x_bounds
        ymin,ymax = y_bounds
        dx = xmax - xmin
        dy = ymax - ymin
        margin = 0.1
        turtle.setup(width=self.width, height=self.height)
        turtle.setworldcoordinates(xmin - (dx * margin),
                                   ymin - (dy * margin),
                                   xmax + (dx * margin),
                                   ymax + (dy * margin))
        turtle.tracer(0,1)
        turtle.hideturtle()
        turtle.penup()

        self._draw_goal(goal_bounds)
        self._draw_inbounds(in_bounds)

        self.craft_turtle = turtle.Turtle()
        self.craft_turtle.shape("triangle")
        self.craft_turtle.shapesize(0.3, 0.5)
        self.craft_turtle.penup()

    def _draw_inbounds(self, in_bounds):
        t = turtle.Turtle()
        t.hideturtle()
        t.pencolor("black")
        t.penup()
        t.setposition(in_bounds.x_bounds[0], in_bounds.y_bounds[0])
        t.pendown()
        t.setposition(in_bounds.x_bounds[1], in_bounds.y_bounds[0])
        t.setposition(in_bounds.x_bounds[1], in_bounds.y_bounds[1])
        t.setposition(in_bounds.x_bounds[0], in_bounds.y_bounds[1])
        t.setposition(in_bounds.x_bounds[0], in_bounds.y_bounds[0])

    def _draw_goal(self, goal_bounds):

        t = turtle.Turtle()
        t.hideturtle()
        t.color("green", "#aaffaa")
        t.penup()
        t.setposition(goal_bounds.x_bounds[0], goal_bounds.y_bounds[0])
        t.pendown()
        t.begin_fill()
        t.setposition(goal_bounds.x_bounds[1], goal_bounds.y_bounds[0])
        t.setposition(goal_bounds.x_bounds[1], goal_bounds.y_bounds[1])
        t.setposition(goal_bounds.x_bounds[0], goal_bounds.y_bounds[1])
        t.setposition(goal_bounds.x_bounds[0], goal_bounds.y_bounds[0])
        t.end_fill()
        
    def begin_time_step(self, t):
        for idx,trtl in list(self.asteroid_turtles.items()):
            trtl.clear()
            trtl.hideturtle()
        for idx,trtl in list(self.estimated_asteroid_turtles.items()):
            trtl.clear()
            trtl.hideturtle()
        self.craft_turtle.clear()
        self.craft_turtle.hideturtle()

    def asteroid_at_loc(self, i, x, y, nearest_obstacle = False, close_obstacle = False):
        if i not in self.asteroid_turtles:
            trtl = turtle.Turtle()
            trtl.shape("circle")
            trtl.color("grey")
            trtl.shapesize(self.margin * 20, self.margin * 20)
            trtl.penup()
            self.asteroid_turtles[i] = trtl
        self.asteroid_turtles[i].setposition(x,y)
        self.asteroid_turtles[i].color('grey')
        #Uncomment the following line to display asteroid IDs
        #self.asteroid_turtles[i]._write(str(i), 'center', 'arial')
        self.asteroid_turtles[i].showturtle()

    def asteroid_set_color(self, i, color = 'grey'):
        self.asteroid_turtles[i].color(color)

    def asteroid_estimated_at_loc(self, i, x, y, is_match=False):
        return
        if i not in self.estimated_asteroid_turtles:
            trtl = turtle.Turtle()
            trtl.shape("circle")
            trtl.color("#88ff88" if is_match else "#aa4444")
            trtl.shapesize(0.2,0.2)
            trtl.penup()
            self.estimated_asteroid_turtles[i] = trtl
        self.estimated_asteroid_turtles[i].color("#88ff88" if is_match else "#aa4444")
        self.estimated_asteroid_turtles[i].setposition(x,y)
        self.estimated_asteroid_turtles[i].showturtle()

    def craft_at_loc(self, x, y, h, is_ssa = False):
        self.craft_turtle.setposition(x,y)
        self.craft_turtle.settiltangle( old_div(h * 180, math.pi) )
        self.craft_turtle.color("red" if is_ssa else "black")
        self.craft_turtle.showturtle()

    def collision(self):
        self._explode_craft()

    def out_of_bounds(self):
        self._explode_craft()

    def navigation_done(self, retcode, t):
        if retcode in (runner.NAV_FAILURE_COLLISION,
                       runner.NAV_FAILURE_OUT_OF_BOUNDS,
                       runner.FAILURE_TOO_MANY_STEPS):
            self._explode_craft()

    def end_time_step(self, t):
        turtle.update()
        #time.sleep(0.1)

    def teardown(self):
        turtle.done()

    def _explode_craft(self):
        self.craft_turtle.shape("circle")
        self.craft_turtle.shapesize(1.0,1.0)
        self.craft_turtle.color("orange")


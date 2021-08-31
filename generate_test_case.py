
import argparse
import json
import math
import random

from asteroid import Asteroid, AsteroidField, \
    FIELD_X_BOUNDS, FIELD_Y_BOUNDS

INITIAL_CRAFT_STATE = { "x": 0.0,
                        "y": -1.1,
                        "h": math.pi * 0.5,
                        "v": 0.0,
                        "max_speed": 0.05,
                        "speed_increment": 0.01,
                        "angle_increment": 0.1 }

IN_BOUNDS = { "x_bounds": [-1.0, 1.0],
              "y_bounds": [-1.2, 1.2] }

GOAL_BOUNDS = { "x_bounds": [-1.0, 1.0],
                "y_bounds": [1.0, 1.2] }

class AsteroidFieldGenerator(object):

    def __init__(self, seed):
        self.random_state = random.Random(seed)

    def bounded_randval( self, low, high ):
        return ((high - low) * self.random_state.random()) + low

    def generate_asteroid_coeffs( self,
                                  ax_bounds, bx_bounds, cx_bounds,
                                  ay_bounds, by_bounds, cy_bounds ):

        return ( self.bounded_randval( *ax_bounds ),
                 self.bounded_randval( *bx_bounds ),
                 self.bounded_randval( *cx_bounds ),
                 self.bounded_randval( *ay_bounds ),
                 self.bounded_randval( *by_bounds ),
                 self.bounded_randval( *cy_bounds ) )

    def generate(self, t_past, t_future, t_step,
                 a_max = 0.0001,
                 b_max = 0.01,
                 asteroid_seed = 0):

        ax_left_bounds = (-a_max,  a_max)
        bx_left_bounds = (0.0,  b_max)
        cx_left_bounds = (-1.0, -1.0)

        ax_right_bounds = (-a_max, a_max)
        bx_right_bounds = (-b_max, 0.0)
        cx_right_bounds = (1.0,    1.0)

        ay_bounds = (-a_max, a_max)
        by_bounds = (-b_max, b_max)
        cy_bounds = (-1.0,   1.0)
        
        asteroids = []

        for t in range(t_past, t_future):
            
            if not (t%t_step):

                if len(asteroids) % 2:

                    coeffs = self.generate_asteroid_coeffs( ax_left_bounds,
                                                            bx_left_bounds,
                                                            cx_left_bounds,
                                                            ay_bounds,
                                                            by_bounds,
                                                            cy_bounds )
                else:

                    coeffs = self.generate_asteroid_coeffs( ax_right_bounds,
                                                            bx_right_bounds,
                                                            cx_right_bounds,
                                                            ay_bounds,
                                                            by_bounds,
                                                            cy_bounds )

                a_x,b_x,c_x,a_y,b_y,c_y = coeffs

                a = Asteroid( a_x = a_x,
                              b_x = b_x,
                              c_x = c_x,
                              a_y = a_y,
                              b_y = b_y,
                              c_y = c_y,
                              t_start = t )

                asteroids.append(a)

        return AsteroidField( asteroids = asteroids,
                              x_bounds = FIELD_X_BOUNDS,
                              y_bounds = FIELD_Y_BOUNDS )

def args_as_dict(args):
    if isinstance(args, dict):
        return args
    elif isinstance(args, argparse.Namespace):
        return vars(args)
    else:
        raise RuntimeError('cannot convert to dict: %s' % str(args))

def args_as_namespace(args):
    if isinstance(args, dict):
        return argparse.Namespace( **args )
    elif isinstance(args, argparse.Namespace):
        return args
    else:
        raise RuntimeError('cannot convert to namespace: %s' % str(args))
    
def params(args):

    my_args = args_as_namespace(args)

    field_generator = AsteroidFieldGenerator( seed = my_args.seed )
    field = field_generator.generate( t_past        = my_args.t_past,
                                      t_future      = my_args.t_future,
                                      t_step        = my_args.t_step,
                                      a_max         = my_args.asteroid_a_max,
                                      b_max         = my_args.asteroid_b_max,
                                      asteroid_seed = my_args.seed )

    craft_state = dict(INITIAL_CRAFT_STATE)
    craft_state['max_speed'] = my_args.craft_max_speed
    craft_state['angle_increment'] = my_args.craft_angle_increment
    
    return { "asteroids": [ dict(a.params)
                            for a in field.asteroids ],
             "initial_craft_state": craft_state,
             "in_bounds": dict(IN_BOUNDS),
             "goal_bounds": dict(GOAL_BOUNDS),
             "min_dist": my_args.min_dist,
             "noise_sigma": my_args.noise_sigma,
             "_args": vars(my_args) }

def main(args):

    p = params( args=args )

    f = open( args.outfile, 'w' )
    f.write( "params = " + json.dumps( p, indent=2 ) + "\n" )
    f.close()

    print("Wrote %s" % args.outfile)

def parser():
    prsr = argparse.ArgumentParser( "Generate a test case parameters and write to file." )
    prsr.add_argument("outfile",
                      help="name of file to write")
    prsr.add_argument("--t_past",
                      help    = "time in past (negative integer) from which to start generating asteroids",
                      type    = int,
                      default = -100)
    prsr.add_argument("--t_future",
                      help    = "time into future (postigive integer) at which to stop generating asteroids",
                      type    = int,
                      default = 1000)
    prsr.add_argument("--t_step",
                      help    = "add an asteroid every N-th time step",
                      type    = int,
                      default = 1)
    prsr.add_argument("--noise_sigma",
                      help    = "sigma of Gaussian noise applied to asteroid measurements",
                      type    = float,
                      default = 0.0)
    prsr.add_argument("--asteroid_a_max",
                      help    = "maximum magnitude for quadratic asteroid coefficient",
                      type    = float,
                      default = 0.0001)
    prsr.add_argument("--asteroid_b_max",
                      help    = "maximum magnitude for linear asteroid coefficient",
                      type    = float,
                      default = 0.01)
    prsr.add_argument("--craft_max_speed",
                      help    = "max speed for the student-piloted craft",
                      type    = float,
                      default = 0.05)
    prsr.add_argument("--craft_angle_increment",
                      help    = "heading change increment for student-piloted craft",
                      type    = float,
                      default = math.pi / 8.0)

    prsr.add_argument("--min_dist",
                      help    = "minimum distance craft must be from asteroid to avoid collision",
                      type    = float,
                      default = 0.02)
    prsr.add_argument("--seed",
                      help    = "random seed to use when generating asteroids",
                      type    = int,
                      default = 0)
    return prsr
    
if __name__ == '__main__':
    args = parser().parse_args()
    main(args)

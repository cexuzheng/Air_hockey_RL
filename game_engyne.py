from cmath import sqrt
from genericpath import exists
from hashlib import new
from tabnanny import check
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Colours
soft_pink = (255/255,192/255,203/255)

class air_hockey:
    def __init__(self, x_width = 4, y_height = 6, dt = 0.1, ball_radious = 0.1, hand_radious = 0.3,
            goal_fraction = 0.5, m_ball = 0.1, m_hand = 20, drag_coeff = 0.1, bounce_coeff = 0.96):
        self.x_width = x_width;
        self.y_height = y_height;
        self.dt = dt;

        points = np.linspace( 0, 2*np.pi, 100 )
        self.ball_radious = ball_radious;
        self.ball_pos = np.array(  [ ball_radious + np.random.rand()*( x_width - 2*ball_radious ), 
                    ball_radious + np.random.rand()*( y_height - 2*ball_radious ) ]  );         # [x,y]
        self.ball_contour = np.array( [  np.cos(points), np.sin(points) ] ).T + self.ball_pos
        ball_v_angle = np.random.rand() * 2*np.pi 
        self.ball_vel = np.random.rand()*np.array([ np.cos(ball_v_angle), np.sin(ball_v_angle) ])

        self.hand_radious = hand_radious;
        self.hand_pos = np.array(  [ hand_radious + np.random.rand()*( x_width - 2*hand_radious ), 
                    hand_radious + np.random.rand()*( y_height - 2*hand_radious ) ]  );         # [x,y]
        self.hand_contour = np.array( [  np.cos(points), np.sin(points) ] ).T + self.hand_pos
        self.hand_vel = np.zeros(2)

        self.self_goal = np.array([  [self.x_width/4, ball_radious], [self.x_width/4*3, ball_radious]  ])
        self.opponent_goal =  np.array([  [self.x_width/4, self.y_height - ball_radious], [self.x_width/4*3, self.y_height - ball_radious]  ])

        self.m_ball = m_ball
        self.m_hand = m_hand

        self.drag_coeff = drag_coeff
        self.bounce_coeff = bounce_coeff

        goal_corner = (1-goal_fraction)/2
        wall_segments = []  # [ [x1,y1], [x2,y2] ]
        wall_segments = wall_segments + [[ [ball_radious,     ball_radious],                  [self.x_width*goal_corner, ball_radious]                   ]]
        wall_segments = wall_segments + [[ [self.x_width*(1-goal_corner) ,ball_radious],      [x_width - ball_radious, ball_radious]                     ]]
        wall_segments = wall_segments + [[ [x_width - ball_radious, ball_radious],            [x_width - ball_radious, y_height - ball_radious]          ]]
        wall_segments = wall_segments + [[ [x_width - ball_radious, y_height - ball_radious], [self.x_width*(1-goal_corner) , y_height - ball_radious]   ]]
        wall_segments = wall_segments + [[ [self.x_width*goal_corner,y_height-ball_radious],  [ball_radious, y_height-ball_radious]                      ]]
        wall_segments = wall_segments + [[ [ball_radious, y_height-ball_radious],             [ball_radious, ball_radious]                               ]]
        self.ball_walls_segments = np.array( wall_segments )

        hand_limits = []
        hand_limits = hand_limits + [[  [hand_radious, hand_radious], [self.x_width-hand_radious, hand_radious]     ]]
        hand_limits = hand_limits + [[  [self.x_width-hand_radious, hand_radious], [self.x_width-hand_radious, self.y_height-hand_radious]     ]]
        hand_limits = hand_limits + [[  [self.x_width-hand_radious, self.y_height-hand_radious], [hand_radious, self.y_height-hand_radious]     ]]
        hand_limits = hand_limits + [[  [hand_radious, self.y_height-hand_radious], [hand_radious, hand_radious]     ]]
        self.hand_walls_segments = hand_limits

    def _draw(self):
        x_l2r = np.linspace(0, self.x_width, 11);
        y_d2u = np.linspace(0, self.y_height, 11);

        x_left = np.ones(11)*0;
        x_right = np.ones(11)*self.x_width;
        y_down = np.ones(11)*0;
        y_up = np.ones(11)*self.y_height;
        y_middle = np.ones(11)*self.y_height/2;

        x_goal = np.linspace(self.x_width/4, self.x_width*3/4, 11);

        fig = plt.figure(figsize=(6, 9))
        ax = fig.add_subplot(1,1,1) # ,projection="3d")

        ax.plot(x_l2r,  y_down, 'b', linewidth = 4.0)
        ax.plot(x_l2r,   y_up, 'b', linewidth = 4.0)
        ax.plot(x_left,  y_d2u, 'b', linewidth = 4.0)
        ax.plot(x_right, y_d2u, 'b', linewidth = 4.0)

        ax.plot(x_l2r, y_middle, c = soft_pink, linewidth = 2.0)
        ax.plot(x_goal, y_down, c = 'k', linewidth = 6.0)
        ax.plot(x_goal, y_up, c = 'k', linewidth = 6.0)
        # ax.fill_between(x_l2r, 0,1, facecolor='red')

    def vec_perp( self, vec ):
        v_perp = [vec[1], -vec[0]]/np.linalg.norm(vec)
        return v_perp

    def check_wall_collision( self, trajectory, walls ):
        walls = np.array( walls )
        trajectory = np.array( trajectory )
        n_walls = np.shape( walls )[0]
        v1 = trajectory[1,:] - trajectory[0,:]                          # particle movement
        v2 = walls[:,1,:] - walls[:,0,:]                                # wall vector
        v3 = walls[:,0,:] - trajectory[0,:]                             # initial points vector
        v1_perp = self.vec_perp(v1)
        triangle_height = np.dot(v3, v1_perp)
        v2_dot_v1_perp = np.dot(v2, v1_perp)
        v2_dot_v1_perp[ np.abs(v2_dot_v1_perp) < 1e-10 ] = 1e-10       # avoid zero division
        lambdas_2 = -triangle_height/v2_dot_v1_perp
        x_collision = walls[:,0,:] + lambdas_2.reshape(n_walls, 1) * v2
        lambdas = np.linalg.norm( x_collision - trajectory[1,:], axis= 1 ) / np.linalg.norm(v1)
        bool_collision = np.logical_and( lambdas_2 > -1e-10, lambdas_2 < 1.0+1e-10)
        bool_collision = np.logical_and( bool_collision, lambdas > -1e-10)
        bool_collision = np.logical_and( bool_collision, lambdas < 1.0+1e-10)
        return ( bool_collision, lambdas )
    
    def check_hand_collision(self, ball_trajectory, hand_trajectory):
        hand_trajectory = np.array(hand_trajectory); ball_trajectory = np.array(ball_trajectory)
        v1 = ball_trajectory[1,:] - ball_trajectory[0,:]
        v2 = hand_trajectory[1,:] - hand_trajectory[0,:]
        # d(t)^2 =  (x1-x2)^2 + 2t(v1(0) - v2(0))(x1-x2) + t^2( v1(0) - v2(0) )^2
        #         + (y1-y2)^2 + 2t(v1(1) - v2(1))(y1-y2) + t^2( v1(1) - v2(1) )^2
        
        x1_x2_aux = ball_trajectory[0,0] - hand_trajectory[0,0]
        y1_y2_aux = ball_trajectory[0,1] - hand_trajectory[0,1]
        v10_v20_aux = v1[0]-v2[0]
        v11_v21_aux = v1[1]-v2[1]
        min_allowed_dist = self.hand_radious + self.ball_radious
        
        c = (x1_x2_aux**2 + y1_y2_aux**2 - min_allowed_dist**2)
        b = 2*( v10_v20_aux*x1_x2_aux  +  v11_v21_aux*y1_y2_aux )
        a = ( v10_v20_aux**2 + v11_v21_aux**2 )
        sol1 = -1; sol2 = -1;
        if( b**2 - 4*a*c > 0):  # there exists a solution with dist = collision
            sol1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
            sol2 = (-b - np.sqrt(b**2 + 4*a*c))/(2*a)
        
        lambdas = -1; bool_collision = False
        if( sol1 > -1e-10 and sol1 < 1 + 1e-10):
            lambdas = sol1
            bool_collision = True
        elif( sol2 > -1e-10 and sol2 < 1 + 1e-10):
            lambdas = sol2
            bool_collision = True

        return( bool_collision, lambdas)

    def wall_bounce(self, v1, v2):
        v2_perp = self.vec_perp(v2)
        v1_after = v1 - 2 * np.dot( v1, v2_perp )*v2_perp
        return(v1_after)

    def hand_bounce(self, m1, m2, v1, v2, p_dir ):
        dp = 2*(np.dot(v2, p_dir) - np.dot(v1, p_dir))/(1/m1 + 1/m2)
        v1_after = v1 + dp/m1*p_dir
        v2_after = v2 - dp/m2*p_dir
        return(v1_after, v2_after)

    def check_any_coll(self, ball_trajectory, hand_trajectory):
        """ more specific function, orders the types of collision """
        

    def compute_physics(self):
        prev_ball_pos = self.ball_pos
        new_ball_pos = self.ball_pos + self.ball_vel*self.dt

        prev_hand_pos = self.hand_pos
        new_hand_pos = self.hand_pos + self.hand_vel*self.dt
        (wall_coll, wall_lam) = self.check_wall_collision( [prev_hand_pos, new_hand_pos], self.hand_walls_segments )
        if( np.any(wall_coll) ):        #there was a collision, hand must be stopped next to the edge for numerical reasons
            v2 = self.hand_vel*self.dt
            min_lambda = np.min(wall_lam)
            new_hand_pos = prev_hand_pos + v2*(min_lambda-1e-10)    # next to the edge
            self.hand_vel = np.zeros(2)

        (any_coll, coll_type, lambda_param ) = self.check_any_coll([prev_ball_pos, new_ball_pos], [prev_hand_pos, new_hand_pos])
        while any_coll:
            pass
        # (wall_coll, wall_lambdas) = self.check_wall_collision(  )



    #
    # ball and physics
    #


def update_line(i, Yhat, line):
    line.set_data(np.array([Yhat[i,:,0],Yhat[i,:,1]]))
    line.set_3d_properties(Yhat[i,:,2])
    return line

    


'''
ball_pos = np.random.rand(2)*x_width;
ball_pos[1] = ball_pos[1]*y_heigth/x_width;

ball_vel = np.random.rand(2)

'''







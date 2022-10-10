from cmath import sqrt
from genericpath import exists
from hashlib import new
import re
from shutil import which
from tabnanny import check
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Colours
soft_pink = (255/255,192/255,203/255)

GOAL_TO_ENEMY = 1; GOAL_TO_SELF = 2; WALL_COLLISION = 3; SELF_HAND_BALL_COLLISION = 4; ENEMY_HAND_BALL_COLLSION = 5; NONE_COLL = -1
SELF_HAND_WALL_COLLISION = 6; ENEMY_HAND_WALL_COLLISION = 7

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

        self.self_hand_pos = np.array(  [ hand_radious + np.random.rand()*( x_width - 2*hand_radious ), 
                    hand_radious + np.random.rand()*( y_height - 2*hand_radious ) ]  );         # [x,y]
        self.self_hand_contour = np.array( [  np.cos(points), np.sin(points) ] ).T + self.self_hand_pos
        self.self_hand_vel = np.zeros(2)

        self.enemy_hand_pos = np.array(  [ hand_radious + np.random.rand()*( x_width - 2*hand_radious ), 
                    hand_radious + np.random.rand()*( y_height - 2*hand_radious ) ]  );         # [x,y]
        self.enemy_hand_contour = np.array( [  np.cos(points), np.sin(points) ] ).T + self.enemy_hand_pos
        self.enemy_hand_vel = np.zeros(2)

        goal_corner = (1-goal_fraction)/2
        self.goal_corner = goal_corner
        self.self_goal = [  [self.x_width*goal_corner, ball_radious], [self.x_width*(1-goal_corner), ball_radious]  ]
        self.enemy_goal =[  [self.x_width*goal_corner, self.y_height - ball_radious], [self.x_width*(goal_corner), self.y_height - ball_radious]  ]

        self.m_ball = m_ball
        self.m_hand = m_hand

        self.drag_coeff = drag_coeff
        self.bounce_coeff = bounce_coeff

        
        wall_segments = []  # [ [x1,y1], [x2,y2] ]
        wall_segments = wall_segments + [[ [ball_radious,     ball_radious],                  [self.x_width*goal_corner, ball_radious]                   ]]
        wall_segments = wall_segments + [[ [self.x_width*(1-goal_corner) ,ball_radious],      [x_width - ball_radious, ball_radious]                     ]]
        wall_segments = wall_segments + [[ [x_width - ball_radious, ball_radious],            [x_width - ball_radious, y_height - ball_radious]          ]]
        wall_segments = wall_segments + [[ [x_width - ball_radious, y_height - ball_radious], [self.x_width*(1-goal_corner) , y_height - ball_radious]   ]]
        wall_segments = wall_segments + [[ [self.x_width*goal_corner,y_height-ball_radious],  [ball_radious, y_height-ball_radious]                      ]]
        wall_segments = wall_segments + [[ [ball_radious, y_height-ball_radious],             [ball_radious, ball_radious]                               ]]
        self.ball_walls_segments =  wall_segments 

        hand_limits = []
        hand_limits = hand_limits + [[  [hand_radious, hand_radious], [self.x_width-hand_radious, hand_radious]     ]]
        hand_limits = hand_limits + [[  [self.x_width-hand_radious, hand_radious], [self.x_width-hand_radious, self.y_height/2-hand_radious]     ]]
        hand_limits = hand_limits + [[  [self.x_width-hand_radious, self.y_height/2-hand_radious], [hand_radious, self.y_height/2-hand_radious]     ]]
        hand_limits = hand_limits + [[  [hand_radious, self.y_height/2-hand_radious], [hand_radious, hand_radious]     ]]
        self.self_hand_walls_segments = hand_limits

        hand_limits = []
        hand_limits = hand_limits + [[  [hand_radious, self.y_height/2+hand_radious], [self.x_width-hand_radious, self.y_height/2+hand_radious]     ]]
        hand_limits = hand_limits + [[  [self.x_width-hand_radious, self.y_height/2+hand_radious], [self.x_width-hand_radious, self.y_height-hand_radious]     ]]
        hand_limits = hand_limits + [[  [self.x_width-hand_radious, self.y_height-hand_radious], [hand_radious, self.y_height-hand_radious]     ]]
        hand_limits = hand_limits + [[  [hand_radious, self.y_height-hand_radious], [hand_radious, self.y_height/2+hand_radious]     ]]
        self.enemy_hand_walls_segments = hand_limits

    def _draw(self):
        x_l2r = np.linspace(0, self.x_width, 11);
        y_d2u = np.linspace(0, self.y_height, 11);

        x_left = np.ones(11)*0;
        x_right = np.ones(11)*self.x_width;
        y_down = np.ones(11)*0;
        y_up = np.ones(11)*self.y_height;
        y_middle = np.ones(11)*self.y_height/2;

        x_goal = np.linspace(self.x_width*self.goal_corner, self.x_width*(1-self.goal_corner), 11);

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

    def array_to_vel(self, trajectory):
        trajectory = np.array(trajectory)
        return( trajectory[1,:]-trajectory[0,:] )

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

    def check_any_coll(self, ball_trajectory, self_hand_trajectory, enemy_hand_trajectory):
        """ more specific function, orders the types of collision """
        (self_hand, l_self_hand) = self.check_hand_collision( ball_trajectory, self_hand_trajectory )
        (enemy_hand, l_enemy_hand) = self.check_hand_collision( ball_trajectory, enemy_hand_trajectory )
        (self_hand_wall, l_self_hand_wall) = self.check_wall_collision( self_hand_trajectory, self.self_hand_walls_segments )
        (enemy_hand_wall, l_enemy_hand_wall) = self.check_wall_collision( enemy_hand_trajectory, self.enemy_hand_walls_segments )
        (wall_coll, l_wall_coll) = self.check_wall_collision( ball_trajectory, [self.self_goal, self.enemy_goal] + self.ball_walls_segments )
        total_bool = np.concatenate( (self_hand, enemy_hand, self_hand_wall, enemy_hand_wall, wall_coll) )
        # any_coll = np.any(total_bool); coll_type = NONE_COLL; lambda_coll = -1
        if( np.any(total_bool) ):       # there is some collision
            total_lambda = np.concatenate( (l_self_hand, l_enemy_hand, l_self_hand_wall, l_enemy_hand_wall, l_wall_coll) )
            where_coll = np.where( total_bool )
            arg_min_i = np.argmin( total_lambda[total_bool] )
            which_coll = where_coll[arg_min_i]
            lambda_coll = total_lambda[which_coll]
            wall_i = which_coll - 6
            if( which_coll == 0 ):      # self_hand
                coll_type = SELF_HAND_BALL_COLLISION
            elif( which_coll == 1 ):    # enemy_hand
                coll_type = ENEMY_HAND_BALL_COLLSION
            elif( which_coll == 2 ):
                coll_type = SELF_HAND_WALL_COLLISION
            elif( which_coll == 3 ):
                coll_type = ENEMY_HAND_WALL_COLLISION
            elif( which_coll == 4 ):
                coll_type = GOAL_TO_SELF
            elif( which_coll == 5 ):
                coll_type = GOAL_TO_ENEMY
            else:
                coll_type = WALL_COLLISION
            return( True, coll_type, lambda_coll, wall_i )
        else:
            return( False, NONE_COLL, -1, -1)

    def hand_wall_coll(self, prev_hand_pos, new_hand_pos, lambda_coll ):
        coll_pos = (1-lambda_coll)*prev_hand_pos + lambda_coll*new_hand_pos
        return( coll_pos, coll_pos, np.zeros(2) )

    def hand_ball_coll(self, prev_ball_pos, new_ball_pos, prev_hand_pos, new_hand_pos, lambda_coll, time_left, v_ball, v_hand):
        prev_ball_pos = (1-lambda_coll)*prev_ball_pos + lambda_coll*new_ball_pos        # evolution
        prev_hand_pos = (1-lambda_coll)*prev_hand_pos + lambda_coll*new_hand_pos
        p_dir = prev_ball_pos - prev_hand_pos
        p_dir = p_dir / np.linalg.norm(p_dir)
        (v_ball_after, v_hand_after) = self.hand_bounce(self.m_ball, self.m_hand, 
                v_ball, v_hand, p_dir )

        v_ball_after = v_ball_after*self.bounce_coeff
        new_ball_pos = prev_ball_pos + v_ball_after*time_left

        v_hand_after =  v_hand_after*self.bounce_coeff
        new_hand_pos = prev_hand_pos + v_hand_after*time_left
        return(prev_ball_pos, new_ball_pos, prev_hand_pos, new_hand_pos, v_ball_after, v_hand_after)

    def ball_wall_coll(self, prev_ball_pos, new_ball_pos, v_wall, lambda_coll, time_left, v_ball):
        v_ball_after = self.wall_bounce(v_ball, v_wall)*self.bounce_coeff
        prev_ball_pos = (1-lambda_coll)*prev_ball_pos + lambda_coll*new_ball_pos
        new_ball_pos = prev_ball_pos + v_ball_after*time_left
        return( prev_ball_pos, new_ball_pos, v_ball_after )
    
    def compute_physics(self, update_contour = False):
        prev_ball_pos = self.ball_pos
        new_ball_pos = self.ball_pos + self.ball_vel*self.dt

        self_prev_hand_pos = self.self_hand_pos
        self_new_hand_pos = self.self_hand_pos + self.self_hand_vel*self.dt
        
        enemy_prev_hand_pos = self.enemy_hand_pos
        enemy_new_hand_pos = self.enemy_hand_pos + self.enemy_hand_vel*self.dt

        (any_coll, coll_type, lambda_coll, wall_i ) = self.check_any_coll([prev_ball_pos, new_ball_pos], 
                [self_prev_hand_pos, self_new_hand_pos], [enemy_prev_hand_pos, enemy_new_hand_pos])

        time_left = self.dt*(1-lambda_coll)
        while any_coll:
            if( coll_type == SELF_HAND_WALL_COLLISION ):
                ( self_prev_hand_pos, self_new_hand_pos, self.self_hand_vel) = self.hand_wall_coll(self_prev_hand_pos, self_new_hand_pos, lambda_coll)
                enemy_prev_hand_pos = (1-lambda_coll)*enemy_prev_hand_pos + lambda_coll*enemy_new_hand_pos
                prev_ball_pos = (1-lambda_coll)*prev_ball_pos + new_ball_pos
            elif( coll_type == ENEMY_HAND_WALL_COLLISION ):
                ( enemy_prev_hand_pos, enemy_new_hand_pos, self.enemy_hand_vel) = self.hand_wall_coll(enemy_prev_hand_pos, enemy_new_hand_pos, lambda_coll)
                self_prev_hand_pos = (1-lambda_coll)*self_prev_hand_pos + lambda_coll*self_new_hand_pos
                prev_ball_pos = (1-lambda_coll)*prev_ball_pos + new_ball_pos
            elif( coll_type == SELF_HAND_BALL_COLLISION ):
                ( prev_ball_pos, new_ball_pos, self_prev_hand_pos, self_new_hand_pos, self.ball_vel, 
                    self.self_hand_vel ) = self.hand_ball_coll(prev_ball_pos, new_ball_pos, self_prev_hand_pos,
                    self_new_hand_pos, lambda_coll, time_left, self.ball_vel, self.self_hand_vel)
                enemy_prev_hand_pos = (1-lambda_coll)*enemy_prev_hand_pos + lambda_coll*enemy_new_hand_pos
            elif( coll_type == ENEMY_HAND_BALL_COLLSION ):
                ( prev_ball_pos, new_ball_pos, enemy_prev_hand_pos, enemy_new_hand_pos, self.ball_vel, 
                    self.enemy_hand_vel ) = self.hand_ball_coll(prev_ball_pos, new_ball_pos, enemy_prev_hand_pos,
                    enemy_new_hand_pos, lambda_coll, time_left, self.ball_vel, self.enemy_hand_vel)
                self_prev_hand_pos = (1-lambda_coll)*self_prev_hand_pos + lambda_coll*self_new_hand_pos
            elif( coll_type == GOAL_TO_SELF):
                print( "OOOH!!, you have been scored a goal")
                (prev_ball_pos, new_ball_pos, self.ball_vel) = self.ball_wall_coll(prev_ball_pos, new_ball_pos,
                    self.array_to_vel(self.self_goal), lambda_coll, time_left, self.ball_vel)
                self_prev_hand_pos = (1-lambda_coll)*self_prev_hand_pos + lambda_coll*self_new_hand_pos
                enemy_prev_hand_pos = (1-lambda_coll)*enemy_prev_hand_pos + lambda_coll*enemy_new_hand_pos
            elif( coll_type == GOAL_TO_ENEMY):
                print( "YEAAAH!!, you have scored a goal")
                (prev_ball_pos, new_ball_pos, self.ball_vel) = self.ball_wall_coll(prev_ball_pos, new_ball_pos,
                    self.array_to_vel(self.enemy_goal), lambda_coll, time_left, self.ball_vel)
                self_prev_hand_pos = (1-lambda_coll)*self_prev_hand_pos + lambda_coll*self_new_hand_pos
                enemy_prev_hand_pos = (1-lambda_coll)*enemy_prev_hand_pos + lambda_coll*enemy_new_hand_pos
            elif( coll_type == WALL_COLLISION):
                (prev_ball_pos, new_ball_pos, self.ball_vel) = self.ball_wall_coll(prev_ball_pos, new_ball_pos,
                    self.array_to_vel(self.ball_walls_segments[wall_i]), lambda_coll, time_left, self.ball_vel)
                self_prev_hand_pos = (1-lambda_coll)*self_prev_hand_pos + lambda_coll*self_new_hand_pos
                enemy_prev_hand_pos = (1-lambda_coll)*enemy_prev_hand_pos + lambda_coll*enemy_new_hand_pos
            # once updated compute again
            (any_coll, coll_type, lambda_coll, wall_i ) = self.check_any_coll([prev_ball_pos, new_ball_pos], 
                [self_prev_hand_pos, self_new_hand_pos], [enemy_prev_hand_pos, enemy_new_hand_pos])
            time_left = self.dt*(1-lambda_coll)
        
        if(update_contour):
            self.ball_contour = self.ball_contour-self.ball_pos + new_ball_pos
            self.self_hand_contour = self.self_hand_contour - self.self_hand_pos + self_new_hand_pos
            self.enemy_hand_contour = self.enemy_hand_contour - self.enemy_hand_pos + enemy_new_hand_pos
        
        self.ball_pos = new_ball_pos; self.self_hand_pos = self_new_hand_pos; self.enemy_hand_pos = enemy_new_hand_pos
                




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







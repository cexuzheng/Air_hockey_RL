import numpy as np
from game_engyne import air_hockey
from skimage.draw import disk, line


class env_v1():
    def __init__(self, x_width = 4, y_heigth = 6, n_actions = 17, hand_v = 1, dt = 0.1):
        self.air_hockey = air_hockey(x_width=x_width, y_height=y_heigth, dt=dt)
        self.x_width = x_width
        self.y_heigth = y_heigth
        self.init_pos = np.array( [x_width/2, y_heigth/5] )
        self.dt = self.air_hockey.dt
        thetas = np.linspace( 0, np.pi/(n_actions-1)*(n_actions-2), n_actions )
        self.actions = hand_v*np.array([np.cos(thetas), np.sin(thetas)])
        self.actions = np.concatenate( ([[0],[0]], self.actions), axis=1)
        self.draw_canvas()

    def reset(self, ball_pos = None, ball_vel=None, self_pos = None, enemy_pos = None):
        if(ball_pos is None):
            self.air_hockey.ball_pos = np.array( [self.x_width/2, self.y_heigth/2] )
        else:
            self.air_hockey.ball_pos = np.array(ball_pos)

        if( ball_vel is None):
            theta = (np.random.rand()-0.5)*2*np.pi
            v = np.random.rand()*10
            self.air_hockey.ball_vel = v*np.array( [np.cos(theta), np.sin(theta)] )
        else:
            self.air_hockey.ball_vel = np.array(ball_vel)
        
        if(self_pos is None):
            self.air_hockey.self_hand_pos = self.init_pos
        else:
            self.air_hockey.self_hand_pos = np.array(self_pos)

        if(enemy_pos is None):
            self.air_hockey.enemy_hand_pos = [self.x_width, self.y_heigth] - self.init_pos
        else:
            self.air_hockey.enemy_hand_pos = np.array(enemy_pos)

        self.air_hockey.self_hand_vel = np.zeros( 2 )
        self.air_hockey.enemy_hand_vel = np.zeros( 2 )


    def step(self, n_action):
        reward = 0; done = False; info = ()
        self.air_hockey.self_hand_vel = self.actions[:,n_action]
        self_goal, enemy_goal = self.air_hockey.compute_physics()
        new_state = np.concatenate( (self.air_hockey.ball_pos, self.air_hockey.ball_vel, self.air_hockey.self_hand_pos)  )
        if( self.air_hockey.ball_pos[1] < self.y_heigth/2 ):
            reward -= 1
        else:
            reward += 1
        
        reward += enemy_goal*1000 - self_goal*1000

        return(new_state, reward, done, info)
    
    def draw_canvas(self, ppm = 300):
        self.ppm = ppm
        self.x_pixels = self.x_width*ppm
        self.y_pixels = self.y_heigth*ppm
        self.canvas = np.ones( (self.x_pixels, self.y_pixels, 3), 'uint8')*255
        rr,cc = line( 0,0, self.x_pixels-1, 0 )
        self.canvas[rr,cc,:] = (0,0,0)
        rr,cc = line( 0,0, 0,self.y_pixels-1)
        self.canvas[rr,cc,:] = (0,0,0)
        rr,cc = line( self.x_pixels-1,0, self.x_pixels-1,self.y_pixels-1)
        self.canvas[rr,cc,:] = (0,0,0)
        rr,cc = line( 0,self.y_pixels-1, self.x_pixels-1,self.y_pixels-1)
        self.canvas[rr,cc,:] = (0,0,0)
        rr,cc = line(0, int(self.y_pixels/2), self.x_pixels-1, int(self.y_pixels/2) )
        self.ball_px_rad = self.air_hockey.ball_radious*ppm
        self.hand_px_rad = self.air_hockey.hand_radious*ppm

    
    def draw(self):
        drawing = np.copy( self.canvas )
        rr,cc = disk( self.air_hockey.ball_pos*self.ppm, self.ball_px_rad )
        indices = np.logical_and( np.logical_and(rr >= 0, rr < self.x_pixels), np.logical_and(cc >= 0, cc < self.y_pixels ) )
        rr = rr[ indices ]; cc = cc[indices]
        drawing[rr,cc,:] = (255,0,0)

        rr,cc = disk( self.air_hockey.self_hand_pos*self.ppm, self.hand_px_rad )
        indices = np.logical_and( np.logical_and(rr >= 0, rr < self.x_pixels), np.logical_and(cc >= 0, cc < self.y_pixels ) )
        rr = rr[ indices ]; cc = cc[indices]
        drawing[rr,cc,:] = (0,0,255)

        rr,cc = disk( self.air_hockey.enemy_hand_pos*self.ppm, self.hand_px_rad )
        indices = np.logical_and( np.logical_and(rr >= 0, rr < self.x_pixels), np.logical_and(cc >= 0, cc < self.y_pixels ) )
        rr = rr[ indices ]; cc = cc[indices]
        drawing[rr,cc,:] = (0,255,0)

        return drawing




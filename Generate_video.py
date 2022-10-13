import numpy as np
from game_engyne import air_hockey
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(i, lin_ball, ball_contours, lin_self_hand, self_hand_contours, lin_enemy_hand, enemy_hand_contours):
    lin_ball.set_data( [ball_contours[i,:,0], ball_contours[i,:,1]] )
    lin_self_hand.set_data( [self_hand_contours[i,:,0], self_hand_contours[i,:,1]] )
    lin_enemy_hand.set_data( [enemy_hand_contours[i,:,0], enemy_hand_contours[i,:,1]] )
    return lin_ball

game = air_hockey( x_width = 90, y_height=120, ball_radious=2.5, hand_radious=3, hand_fric_coeff = 0.1, ball_fric_coeff = 0.01 , bounce_coeff=0.97, dt = 1/60);

self = game

game._draw()
theta = np.random.rand()*np.pi
theta = 1.087182779223424
theta = np.pi*3/2
v_vel = (7+np.random.rand()*5)*50
# theta = np.pi/3
game.set_ball_pos_vel([game.x_width/2, game.y_height*0.9],  v_vel*np.array( [np.cos(theta), np.sin(theta)] ) )
# game.set_ball_pos_vel([1,1],  [-1,-1] )


lin_ball = game.ax.plot( game.ball_contour[:,0], game.ball_contour[:,1], '--r' )[0]
lin_self_hand = game.ax.plot( game.self_hand_contour[:,0], game.self_hand_contour[:,1], '--b' )[0]
lin_enemy_hand = game.ax.plot( game.enemy_hand_contour[:,0], game.enemy_hand_contour[:,1], '--b' )[0]

T = 4; n_loop = int(T/game.dt)
ball_contour_shape = game.ball_contour.shape
ball_contours = np.zeros( (n_loop, ball_contour_shape[0], ball_contour_shape[1]) )
self_hand_contour_shape = game.self_hand_contour.shape
self_hand_contours = np.zeros( (n_loop, self_hand_contour_shape[0], self_hand_contour_shape[1]) )
enemy_hand_contour_shape = game.enemy_hand_contour.shape
enemy_hand_contours = np.zeros( (n_loop, enemy_hand_contour_shape[0], enemy_hand_contour_shape[1]) )

print(game.ball_vel)
for i in range(n_loop):
    print(i)
    print(game.ball_pos, game.ball_vel)
    print(game.enemy_hand_vel)
    game.compute_physics(True)
    ball_contours[i,:,:] = game.ball_contour
    self_hand_contours[i,:,:] = game.self_hand_contour
    enemy_hand_contours[i,:,:] = game.enemy_hand_contour
    
# print( len( game.ball_walls_segments)  )

print(theta)

line_ani = animation.FuncAnimation( game.fig, update_line, n_loop,
     fargs=(lin_ball, ball_contours, lin_self_hand, self_hand_contours, lin_enemy_hand, enemy_hand_contours), interval=int(1000*game.dt),save_count = 1000)



"""
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

line_ani.save('corner.mp4', writer=writer)

"""

plt.show()

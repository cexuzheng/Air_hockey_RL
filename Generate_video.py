import numpy as np
from game_engyne import air_hockey
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(i, lin_ball, ball_contours, lin_self_hand, self_hand_contours, lin_enemy_hand, enemy_hand_contours):
    lin_ball.set_data( [ball_contours[i,:,0], ball_contours[i,:,1]] )
    lin_self_hand.set_data( [self_hand_contours[i,:,0], self_hand_contours[i,:,1]] )
    lin_enemy_hand.set_data( [enemy_hand_contours[i,:,0], enemy_hand_contours[i,:,1]] )
    return lin_ball

game = air_hockey(drag_coeff = 0.0, bounce_coeff=1.0);

self = game
game.enemy_hand_contour -= game.enemy_hand_pos
game.enemy_hand_pos = np.array([0.5, 3.5])
game.enemy_hand_contour += game.enemy_hand_pos

game._draw()
theta = np.random.rand()*np.pi
game.set_ball_pos_vel([game.x_width/2, game.y_height/2],  np.random.rand()*6*np.array( np.cos(theta), np.sin(theta) ) )

lin_ball = game.ax.plot( game.ball_contour[:,0], game.ball_contour[:,1], '--r' )[0]
lin_self_hand = game.ax.plot( game.self_hand_contour[:,0], game.self_hand_contour[:,1], '--b' )[0]
lin_enemy_hand = game.ax.plot( game.enemy_hand_contour[:,0], game.enemy_hand_contour[:,1], '--b' )[0]

T = 9; n_loop = int(T/game.dt)
ball_contour_shape = game.ball_contour.shape
ball_contours = np.zeros( (n_loop, ball_contour_shape[0], ball_contour_shape[1]) )
self_hand_contour_shape = game.self_hand_contour.shape
self_hand_contours = np.zeros( (n_loop, self_hand_contour_shape[0], self_hand_contour_shape[1]) )
enemy_hand_contour_shape = game.enemy_hand_contour.shape
enemy_hand_contours = np.zeros( (n_loop, enemy_hand_contour_shape[0], enemy_hand_contour_shape[1]) )

for i in range(n_loop):
    print(i)
    print(game.ball_pos)
    game.compute_physics(True)
    ball_contours[i,:,:] = game.ball_contour
    self_hand_contours[i,:,:] = game.self_hand_contour
    enemy_hand_contours[i,:,:] = game.enemy_hand_contour
    
# print( len( game.ball_walls_segments)  )

print(game.self_hand_pos)
print(game.enemy_hand_pos)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
line_ani = animation.FuncAnimation( game.fig, update_line, n_loop,
     fargs=(lin_ball, ball_contours, lin_self_hand, self_hand_contours, lin_enemy_hand, enemy_hand_contours), interval=150,save_count = 1000)


line_ani.save('random.mp4', writer=writer)


plt.show()

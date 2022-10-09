import numpy as np
from game_engyne import air_hockey
import matplotlib.pyplot as plt

print("hello")

game = air_hockey();

game._draw()
plt.show()

x_prev = [1, 1 ]
x_next = [1, 1 ]

print( game.ball_walls_segments )


(bools, lambdas) = game.check_wall_collision([x_prev, x_next], game.ball_walls_segments )

print (bools)
print(lambdas)
i = np.argmin( lambdas )

print( i )

ball_tr = [ [0.1, 1],[0.9, 1]  ]
hand_tr = [ [0.9, 1],[0.1, 1]  ]

ball_tr = [ [0.1, 1],[0.8, 1]  ]
hand_tr = [ [0.9, 1],[0.9, 1]  ]

ball_tr = [ [1, 2],[1, 1.5]  ]
hand_tr = [ [0.1, 1],[0.9, 1]  ]

print(game.hand_radious + game.ball_radious)

hand_tr = np.array(hand_tr); ball_tr = np.array(ball_tr)
v1 = ball_tr[1,:] - ball_tr[0,:]
v2 = hand_tr[1,:] - hand_tr[0,:]
(coll_bool, coll_lambda) = game.check_hand_collision( ball_tr, hand_tr)

print(coll_bool)
print(coll_lambda)

pos_ball = ball_tr[0,:] + coll_lambda*v1
pos_hand = hand_tr[0,:] + coll_lambda*v2

print("dist = ", np.linalg.norm(pos_ball-pos_hand) )
print( pos_ball )
print( pos_hand )


import numpy as np
from agent import RL_Agent_v1
from env_versions import env_v4
import matplotlib.pyplot as plt
import torch
import time
import sys
from tqdm.notebook import tqdm


def train_v1(env, agent, max_episode_num = 5000, max_steps = 99, save_file = "RL_agent", mean_n = 10,
             e_greedy = -1, e_max = 1, e_min = 0.5, e_decay_rate = 0.01, clip_update = 10, show_every = 10, save_every = 10):
    start_time = time.time()
    all_rewards = []
    mean_rewards = []
    # optimizer = optim.RMSprop(agent.parameters())
    for episode in range(max_episode_num):
        env.reset()
        rewards = []
        state = np.concatenate( (env.air_hockey.ball_pos, env.air_hockey.ball_vel, env.air_hockey.self_hand_pos)  )
        if(e_greedy != -1):
            e_greedy = e_min + (e_max - e_min)*np.exp(-e_decay_rate*episode)

        for steps in range(max_steps):
            action = agent.get_action(state, e_greedy)
            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            agent.step( state, action, done, new_state, reward )
            if done:
                break
            state = new_state
        all_rewards.append(np.sum(rewards))
        mean_rewards.append( np.mean(all_rewards[-mean_n:]) )
        if (episode+1) % show_every == 0:
            sys.stdout.write("episode: {}, total reward: {}, last_average_reward: {}, time: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  
                    np.round(np.mean(all_rewards[-mean_n:]), decimals = 3), time.time()-start_time))
            
        if ( episode+1 ) % save_every == 0:        # update the clipped
            agent.save_model(save_file)


    plt.figure()
    plt.subplot(111)
    plt.plot(all_rewards,'b')
    plt.plot(mean_rewards,'r')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.show()
    return (all_rewards, mean_rewards)

def compute_colision(v_ball, p_ball, p_hand):
    v_perp = np.array([1,0])
    Lamb = np.dot(v_perp, v_ball)
    pc = p_ball + v_ball*Lamb
    return pc

def clip_function(curr_states, rewards, next_states, actions):
    curr_states = np.array(curr_states)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    actions = np.array( actions )
    batch_size = rewards.shape[0]
    expected_value = np.zeros( batch_size )
    p_hand = curr_states[:,4:6]
    pc = compute_colision( curr_states[:,2:4], curr_states[:,0:2], p_hand )
    ac_1 = pc[:,0] < p_hand[:,0]; ac_0 = pc[:,0] > p_hand[:,0]
    expected_value[ (ac_1 and actions(ac_1) == 1) or (ac_0 and actions(ac_0) == 0)   ] = 50
    expected_value[ (ac_1 and actions(ac_1) == 0) or (ac_0 and actions(ac_0) == 1)   ] = -50
    return expected_value
        

env = env_v4()
agent = RL_Agent_v1(memory_capacity=1000, learning_rate=3e-2, batch_size=4, n_actions = 3, nlayer1=64, nlayer2=64, gamma=0.7)
print( agent.agent.forward( [2,3,0,0,2,1] ) )
agent.function = clip_function
agent.learn_mode = 'function'
(all_rewards, mean_rewards) = train_v1(env, agent, max_episode_num = 10, max_steps = 40, save_file = "post_training_DQN_5", mean_n=50,
            e_greedy = 1, e_max = 1, e_min = 0.01, e_decay_rate = 0.001, show_every = 250, save_every = 500)
print( agent.agent.forward( [2,3,0,0,2,1] ) )

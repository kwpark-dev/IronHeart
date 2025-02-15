import torch
import gymnasium as gym
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

from network.actor import Actor
from network.critic import Critic
from player.ddpg import AgentDDPG, ReplayBuffer


EPISODE = 4096

if __name__ == "__main__":
    # name = "InvertedPendulum-v5"
    # name = "Reacher-v5"
    name = "HalfCheetah-v5"
    
    env = gym.make(name)
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.shape[0]
    
    max_action = env.action_space.high
    min_action = env.action_space.low
    
    print(max_action, in_dim, out_dim, env.action_space.low)
    
    config = {'in_dim':in_dim,
              'out_dim': out_dim,
              'actor_lr':2e-4,
              'critic_lr':1e-4,
              'tau':4e-3,
              'noise':0.1*max_action[0],
              'max': max_action[0],
              'min': min_action[0],
              'start': 256,
              'gamma':0.99,
              'batch':128,
              'capacity':int(1e5)}
    
    agent = AgentDDPG(Actor, Critic, Adam, ReplayBuffer, config)
    reward_list = []
    actor_list = []
    critic_list = []
    q_est_list = []
    q_target_list = []
    for _ in range(EPISODE):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        while True:
            action = agent.action(state)
            next_state, reward, done, trun, _ = env.step(action.numpy())    
            
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([done], dtype=torch.float32)
            agent.storage.push(state, action, reward, done, next_state)
            
            state = next_state
            
            if done or trun:
                break
            
        agent.learn()
        
        # now test target actor
        R = 0
        data_est = []
        data_target = []
        
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        while True:
            action = agent.action(state, mode='test')
            q_est, q_target = agent.get_q_value(state, action)
            next_state, reward, done, trun, _ = env.step(action.numpy())    
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            state = next_state
            R += reward
            data_est.append(q_est)
            data_target.append(q_target)
            
            if done or trun:
                print("Cumulative reward: {}".format(R))
                
                break
            
        reward_list.append(R)
        data_est = np.array(data_est)
        data_target = np.array(data_target)
        
        q_est_list.append(np.array([data_est.mean(),
                                    data_est.var(),
                                    data_est.min()]))
        
        q_target_list.append(np.array([data_target.mean(),
                                       data_target.var(),
                                       data_target.min()]))
                
    actor_list = agent.actor_list
    critic_list = agent.critic_list
    actor_grad = agent.actor_grad
    critic_grad = agent.critic_grad
        
    q_est_list = np.array(q_est_list)
    q_target_list = np.array(q_target_list)    
        
    a_est, v_est, m_est = q_est_list.T
    a_target, v_target, m_target = q_target_list.T    
        
    fig, ax = plt.subplots(2, 4, figsize=(17, 9))
    
    ax[0][0].plot(reward_list, color='black')
    ax[0][0].set_title('Cumulative Reward')
    ax[1][0].plot(a_est, color='orange', label='estimated')
    ax[1][0].plot(a_target, color='cyan', label='target')
    ax[1][0].set_title('Mean Q value')
    ax[1][0].legend()
    
    ax[0][1].plot(v_est, color='orange', ls='--', label='estimated')
    ax[0][1].plot(v_target, color='cyan', ls='--', label='target')
    ax[0][1].set_title('Var Q value')
    ax[0][1].legend()
    ax[1][1].plot(m_est, color='orange', ls=':', label='estimated')
    ax[1][1].plot(m_target, color='cyan', ls=':', label='target')
    ax[1][1].set_title('Min Q value')
    ax[1][1].legend()    
    
    ax[0][2].plot(actor_list, color='blue')
    ax[0][2].set_title('Actor Loss')
    ax[1][2].plot(actor_grad, color='blue')
    ax[1][2].set_title('Gradient Norm')
    
    ax[0][3].plot(critic_list, color='red')
    ax[0][3].set_title('Critic Loss')
    ax[1][3].plot(critic_grad, color='red')
    ax[1][3].set_title('Gradient Norm')
    
    fig.suptitle(name)
    plt.savefig('./images/ddpg/{}_gam_0{}.jpg'.format(name, 
                                                     int(config['gamma']*100)))
    # plt.show()
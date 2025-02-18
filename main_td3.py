import torch
import gymnasium as gym
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from network.actor import Actor
from network.critic import TwinCritic
from player.td3 import AgentTD3, ReplayBuffer
from utils.estimator import monte_carlo_return


EPISODE = 7815

if __name__ == "__main__":
    # name = "InvertedPendulum-v5"
    # name = "Reacher-v5"
    name = "Hopper-v5"
    
    max_step = 1000
    if name == "Reacher-v5":
        max_step = 200
    
    env = gym.make(name, max_episode_steps=max_step)
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.shape[0]
    
    max_action = env.action_space.high
    min_action = env.action_space.low
    
    print(max_action, in_dim, out_dim, env.action_space.low)
    
    config = {'in_dim':in_dim,
              'out_dim': out_dim,
              'actor_lr':1e-5,
              'critic_lr':1e-4,
              'tau':2e-3,
              'noise':0.1*max_action[0],
              'max': max_action[0],
              'min': min_action[0],
              'start': 192,
              'gamma':0.99,
              'smooth':0.1,
              'smooth_clip': 0.5,
              'batch':128,
              'capacity':int(5e5),
              'period':2}
    
    agent = AgentTD3(Actor, TwinCritic, Adam, ReplayBuffer, config)
    reward_list = []
    actor_list = []
    critic_list = []
    q_est_list = []
    q_target_list = []
    bias_list = []
    for _ in tqdm(range(EPISODE)):
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
        reward_col = []
        done_col = []
        
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
            reward_col.append(reward)
            done_col.append(done)
            
            if done or trun:
                print("Cumulative reward: {}".format(R))
                
                break
            
        reward_list.append(R)
        data_est = np.array(data_est)
        data_target = np.array(data_target)
        mc_return = monte_carlo_return(reward_col, gamma=config['gamma'])
        # mc_mean = mc_return.mean()
        
        bias_est = data_est/mc_return - 1.
        bias_target = data_target/mc_return - 1.
        
        q_est_list.append(np.array([data_est.mean(),
                                    data_est.var(),
                                    data_est.min()]))
        
        q_target_list.append(np.array([data_target.mean(),
                                       data_target.var(),
                                       data_target.min()]))
        
        bias_list.append(np.array([bias_est.mean().clip(-1, 1),
                                   bias_target.mean().clip(-1, 1)]))
        
        # print("bias: ", (data_est - mc_return).mean())
                
    actor_list = agent.actor_list
    critic_list = agent.critic_list
    actor_grad = agent.actor_grad
    critic_grad = agent.critic_grad
        
    q_est_list = np.array(q_est_list)
    q_target_list = np.array(q_target_list)    
    bias_list = np.array(bias_list)
        
    a_est, v_est, m_est = q_est_list.T
    a_target, v_target, m_target = q_target_list.T    
    
    bias_est, bias_target = bias_list.T
    # step = np.array(range(EPISODE))*config['batch']
    
    fig, ax = plt.subplots(3, 3, figsize=(13, 13))
    
    ax[0][0].plot(reward_list, color='black')
    ax[0][0].set_title('Cumulative Reward')
    ax[1][0].plot(a_est, color='orange', label='estimated')
    ax[1][0].plot(a_target, color='cyan', label='target')
    ax[1][0].set_title('Mean Q value')
    ax[1][0].legend()
    
    ax[0][1].plot(v_est, color='orange', ls='--', label='estimated')
    ax[0][1].plot(v_target, color='cyan', ls='--', label='target')
    ax[0][1].set_title('Variance Q value')
    ax[0][1].legend()
    ax[1][1].plot(m_est, color='orange', ls=':', label='estimated')
    ax[1][1].plot(m_target, color='cyan', ls=':', label='target')
    ax[1][1].set_title('Min Q value')
    ax[1][1].legend()    
    
    ax[0][2].plot(actor_list, color='blue')
    ax[0][2].set_title('Actor Loss')
    ax[1][2].plot(actor_grad, color='blue')
    ax[1][2].set_title('Gradient Norm')
    
    ax[2][0].plot(critic_list, color='red')
    ax[2][0].set_title('Critic Loss')
    ax[2][1].plot(critic_grad, color='red')
    ax[2][1].set_title('Gradient Norm')
    
    ax[2][2].plot(bias_est, color='orange', ls='-.', label='estimated')
    ax[2][2].plot(bias_target, color='cyan', ls='-.', label='target')
    ax[2][2].set_title('Bias Q value')
    ax[2][2].legend()
    
    fig.suptitle(name)
    plt.savefig('./images/td3/{}_gam_0{}.jpg'.format(name, 
                                                     int(config['gamma']*100)))
    # plt.show()
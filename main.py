import torch
import gymnasium as gym
from torch.optim import Adam
import matplotlib.pyplot as plt

from network.actor import Actor
from network.critic import Critic
from player.ddpg import AgentDDPG, ReplayBuffer

# # Create the MuJoCo environment (Hopper-v4 as an example)
# env = gym.make("Hopper-v5", render_mode="human")  # Use "human" for visualization

# # Reset the environment
# obs, info = env.reset()

# for _ in range(1000):  # Run for 1000 timesteps
#     action = env.action_space.sample()  # Sample a random action
#     obs, reward, terminated, truncated, info = env.step(action)

#     env.render()  # Render the environment

#     if terminated or truncated:
#         obs, info = env.reset()

# env.close()


EPISODE = 128

if __name__ == "__main__":
    env = gym.make("Hopper-v5")
    # env = gym.make("HalfCheetah-v5")
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.shape[0]
    
    max_action = env.action_space.high
    min_action = env.action_space.low
    
    print(max_action, in_dim, out_dim, env.action_space.low)
    
    config = {'in_dim':in_dim,
              'out_dim': out_dim,
              'actor_lr':1e-4,
              'critic_lr':8e-4,
              'tau':4e-4,
              'noise':0.1,
              'gamma':0.99,
              'batch':64,
              'capacity':10000}
    
    agent = AgentDDPG(Actor, Critic, Adam, ReplayBuffer, config)
    reward_list = []
    actor_list = []
    critic_list = []
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
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        
        while True:
            action = agent.action(state, mode='test')
            next_state, reward, done, trun, _ = env.step(action.numpy())    
            next_state = torch.tensor(next_state, dtype=torch.float32)
            
            state = next_state
            R += reward
            
            if done or trun:
                print("Cumulative reward: {}".format(R))
                
                break
            
        reward_list.append(R)
        
    actor_list = agent.actor_list
    critic_list = agent.critic_list
        
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    
    ax[0].plot(reward_list, color='black')
    ax[0].set_title('Cumulative Reward')
    
    ax[1].plot(actor_list, color='blue')
    ax[1].set_title('Actor Loss')
    
    ax[2].plot(critic_list, color='red')
    ax[2].set_title('Critic Loss')
    
    plt.show()
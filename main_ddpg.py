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


EPISODE = 2048

if __name__ == "__main__":
    env = gym.make("Hopper-v5")
    # env = gym.make("HalfCheetah-v5")
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.shape[0]
    
    max_action = env.action_space.high
    min_action = env.action_space.low
    
    print(max_action, in_dim, out_dim, env.action_space.low)
    # save actor lr: 1e-6, critic lr: 6e-4
    config = {'in_dim':in_dim,
              'out_dim': out_dim,
              'actor_lr':4e-6,
              'critic_lr':4e-5,
              'tau':1e-3,
              'noise':0.1,
              'max': max_action[0],
              'min': min_action[0],
              'start': 256,
              'gamma':0.99,
              'batch':128,
              'capacity':4096}
    
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
    actor_grad = agent.actor_grad
    critic_grad = agent.critic_grad
        
    fig, ax = plt.subplots(1, 5, figsize=(21, 4))
    
    ax[0].plot(reward_list, color='black')
    ax[0].set_title('Cumulative Reward')
    
    ax[1].plot(actor_list, color='blue')
    ax[1].set_title('Actor Loss')
    
    ax[2].plot(actor_grad, color='blue')
    ax[2].set_title('Gradient Norm')
    
    ax[3].plot(critic_list, color='red')
    ax[3].set_title('Critic Loss')
    
    ax[4].plot(critic_grad, color='red')
    ax[4].set_title('Gradient Norm')
    
    plt.show()
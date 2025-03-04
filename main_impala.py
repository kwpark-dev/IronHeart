import torch.optim as optim
import torch.multiprocessing as mp

from network.actor import ActorCritic
from player.impala import explorer, learner
import gymnasium as gym




if __name__ == "__main__":
    mp.set_start_method("spawn")
    
    lr = 1e-3
    batch_size = 128
    rollout_size = 64
    epoch = 256
    
    # name = "InvertedPendulum-v5"
    # name = "Reacher-v5"
    name = "Hopper-v5"
    
    max_step = 1000
    if name == "Reacher-v5":
        max_step = 200
    
    env = gym.make(name, max_episode_steps=max_step)
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    # min_action = env.action_space.low
            
    trendy = ActorCritic(in_dim, out_dim, max_action)
    trendy.share_memory()
    
    optimizer = optim.Adam(trendy.parameters(), lr=lr)
    pipe = mp.Queue()
    stop_event = mp.Event()
    num = 4
    
    process = []
    for idx in range(num):
        p = mp.Process(target=explorer, args=(env, idx, trendy, pipe, stop_event, rollout_size))
        p.start()
        process.append(p)
        
    lp = mp.Process(target=learner, args=(trendy, optimizer, pipe, batch_size,
                                          epoch, stop_event))
    lp.start()
    
    lp.join()
    
    for p in process:
        p.join()
    
    pipe.close()
    
    print("Queue is removed.")
    print("All done")
        
        
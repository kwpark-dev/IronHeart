import torch.optim as optim
import torch.multiprocessing as mp

from network.actor import ActorCritic
from player.impala import explorer, learner
import gymnasium as gym




if __name__ == "__main__":
    mp.set_start_method("spawn")
    
    in_dim = 4
    out_dim = 2
    lr = 4e-5
    batch_size = 256
    rollout_size = 128
    epoch = 256
    
    trendy = ActorCritic(in_dim, out_dim)
    trendy.share_memory()
    
    optimizer = optim.Adam(trendy.parameters(), lr=lr)
    pipe = mp.Queue()
    stop_event = mp.Event()
    num = 4
    env = gym.make("Hopper-v5")
    
    process = []
    for idx in range(num):
        p = mp.Process(target=explorer, args=(env, idx, trendy, pipe, stop_event, rollout_size))
        p.start()
        process.append(p)
        
    lp = mp.Process(target=learner, args=(trendy, optimizer, pipe, batch_size, rollout_size,
                                          epoch, stop_event))
    lp.start()
    
    lp.join()
    
    for p in process:
        p.join()
    
    pipe.close()
    print("Queue is removed.")
        
    print("All done")
        
        
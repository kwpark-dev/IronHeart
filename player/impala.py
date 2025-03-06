import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque




def v_trace(behavior, target, reward, value, done, gamma=0.99, lamb=1.0):
    c_on = 1.0 
    rho_on = 1.0 
    
    behavior = behavior.squeeze()
    target = target.squeeze()
    reward = reward.squeeze()
    value = value.squeeze()
    done = done.squeeze()
    
    ratio = target/(behavior + 1e-4)
    c = torch.clamp(ratio, 0, c_on) * lamb
    rho = torch.clamp(ratio, 0, rho_on)
    
    T = len(reward)
    _v_trace = torch.zeros_like(value)

    _delta = rho[:T-1] * (reward[:T-1] + gamma * value[1:] * (1 - done[1:]) - value[:-1])
    _delta_last = rho[-1] * (reward[-1] + gamma * value[-1] * (1 - done[-1]))
    
    delta = torch.cat((_delta, _delta_last.unsqueeze(dim=0)))

    _v_trace_list = [torch.tensor(0, dtype=torch.float32)]
    for i in reversed(range(T - 1)):
        _v_trace_list.append(delta[i] + gamma * c[i] * (_v_trace[i+1] - value[i+1]) * (1 - done[i+1]))
    
    _v_trace = torch.stack(_v_trace_list[::-1])
    v_trace = value + _v_trace
    
    _q_s = reward[:T-1] + gamma * v_trace[1:]
    _q_s_last = reward[-1]
    q_s = torch.cat((_q_s, _q_s_last.unsqueeze(dim=0)))
    
    return v_trace, ratio, q_s



def explorer(env, idx, trendy, pipe, stop_event, rollout_size): # actor 
    # env = gym.make("CartPole-v1")
    
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    
    while not stop_event.is_set():
        rollout = []
        for _ in range(rollout_size):
            
            trendy.eval()
            with torch.no_grad():
                action, b_logit, _ = trendy(state)
            
            b_prob = torch.sigmoid(b_logit.sum(dim=0))
            next_state, reward, done, trun, _ = env.step(action.numpy())
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor([reward], dtype=torch.float32)
            done = torch.tensor([done], dtype=torch.float32)
            
            rollout.append((state, action, reward, b_prob, done))
            state = next_state
            
            if done or trun:
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32)
    
        pipe.put(rollout)
    
    print("Explorer_{}_ wants to stop exploring".format(idx))
    
    while not pipe.empty():
        pipe.get()
    
    print("Message cleared")
    
            
    
def learner(trendy, optimizer, pipe, batch_size, epoch, stop_event): # learner
    data_table = []
    measured = 0
    
    trendy.train()
    
    for j in range(epoch):
        mini_batch = []
        
        while len(mini_batch) < batch_size:
            mini_batch.extend(pipe.get())
        
        T = len(mini_batch)
        measured += T
        state, action, reward, b_prob, done = zip(*mini_batch)
        
        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.stack(reward)
        b_prob = torch.stack(b_prob)
        done = torch.stack(done)
        
        _, t_logit, t_value = trendy(state)
        t_prob = torch.sigmoid(t_logit.sum(dim=1))
        # t_prob = torch.gather(_t_prob, 1, action.unsqueeze(1))
        # t_prob = t_prob.squeeze(1)
        
        v_s, imp_ratio, q_s = v_trace(b_prob, t_prob, reward, t_value, done)
        bias = v_s - t_value
        critic_loss = 0.5*(bias**2).mean()
        
        base = q_s - t_value
        log_probs = torch.log(t_prob.squeeze() + 1e-4)
        actor_loss = - (imp_ratio * (log_probs * base)).mean()
        
        entropy = (0.05 * t_prob * t_prob.log()).mean()
        
        loss = actor_loss + critic_loss + entropy
        print(loss.item(), actor_loss.item(), critic_loss.item(), entropy.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # torch.save(trendy.state_dict(), './model/model_epoch_{}.pth'.format(j))
        
        ratio_mean = imp_ratio.mean().item()
        # ratio_max = imp_ratio.max().item()
        # ratio_min = imp_ratio.min().item()
        
        data_table.append([ratio_mean,
                           actor_loss.item(),
                           critic_loss.item(),
                           entropy.item()])
    
    rlist, alist, clist, elist = np.array(data_table).T
    
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    
    ax[0][0].plot(rlist, color='black')
    ax[0][0].set_title('Mean Importance Sampling')
    
    ax[0][1].plot(alist, color='orange')
    ax[0][1].set_title('Actor Loss')
    
    ax[1][0].plot(clist, color='blue')
    ax[1][0].set_title('Critic Loss')
    
    ax[1][1].plot(elist, color='red')
    ax[1][1].set_title('Entropy')
    
    plt.savefig('./images/impala/test.jpg')
    # plt.show()
    
    
    
    stop_event.set()

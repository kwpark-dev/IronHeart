import torch
import torch.nn as nn
import random
from collections import deque
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR




class AgentTD3:
    def __init__(self, actor_net, critic_net, optimizer, storage, config):
        self._parse(config)
    
        self.storage = storage(self.capacity)
        
        self.actor = actor_net(self.in_dim, self.out_dim, self.max)
        self.actor_target = actor_net(self.in_dim, self.out_dim, self.max)
        self._sync(self.actor, self.actor_target)
        
        self.actor_opt = optimizer(self.actor.parameters(), lr=self.actor_lr)
        
        self.critic = critic_net(self.in_dim, self.out_dim)
        self.critic_target = critic_net(self.in_dim, self.out_dim)
        self._sync(self.critic, self.critic_target)
        
        self.critic_opt = optimizer(self.critic.parameters(), lr=self.critic_lr)

        if self.schedule:
            self.scheduler = StepLR(self.critic_opt, 
                                    step_size=self.decay_step, 
                                    gamma=self.decay)
        
        
        self.actor_list = []
        self.actor_grad = []
        self.critic_list = []
        self.critic_grad = []
        
        self.i = 0
        
        
    def _sync(self, net, target_net):
        weights = target_net.state_dict()
        net.load_state_dict(weights)
    
    
    def _parse(self, config):
        self.in_dim = config['in_dim']
        self.out_dim = config['out_dim']
        
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.tau = config['tau']
        self.noise = config['noise']
        self.max = config['max']
        self.min = config['min']
        self.gamma = config['gamma']
        self.batch = config['batch']
        self.capacity = config['capacity']
        self.period = config['period']
        self.smooth = config['smooth']
        self.clip = config['smooth_clip']
        self.schedule = config['schedule']
        self.decay = config['decay']
        self.decay_step = config['decay_step']
        
    
    def learn(self): # training mode
        state, action, reward, done, next_state = self.storage.fetch(self.batch)
        
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            smooth = (torch.randn_like(next_action) * self.smooth).clamp(-self.clip, self.clip)
            q_one_target, q_two_target = self.critic_target(next_state, next_action + smooth)
            backup = reward + self.gamma * (1 - done) * torch.min(q_one_target, q_two_target)
        
        q_one_est, q_two_est = self.critic(state, action.squeeze(1))
        critic_loss = ((backup - q_one_est)**2).mean() + ((backup - q_two_est)**2).mean()
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        # critic gradient clipping
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        # trace grad norm
        critic_grad = self._trace_grad(self.critic)
        self.critic_grad.append(critic_grad)
        self.critic_opt.step()

        # print(critic_loss.item())

        self.critic_list.append(critic_loss.item())
            
        if (self.i+1) % self.period == 0:
            actor_loss = - self.critic.q_one(torch.cat([state, self.actor(state)], dim=-1)).mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            # actor gradient clipping
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            # trace grad norm
            actor_grad = self._trace_grad(self.actor)
            self.actor_grad.append(actor_grad)
            self.actor_opt.step()

            # print(actor_loss.item(), critic_loss.item())

            self.actor_list.append(actor_loss.item())
            # self.critic_list.append(critic_loss.item())
            
            with torch.no_grad(): # if not, the targets are added to grad. graph
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.i += 1
        
        
    def _trace_grad(self, model):
        grad_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        
        return grad_norm ** 0.5        
        
            
    def action(self, state, mode='train'):
        self.actor.eval() # anyhow sample actions after turning off util layers
        with torch.no_grad():
            action = self.actor(state)
        
        if mode == 'train':
            noise = self.noise * torch.randn(self.out_dim)
            
            return torch.clamp(action + noise, self.min, self.max)
    
        else: 
            return action
        
        
    def get_q_value(self, state, action):
        self.critic.eval()
        self.critic_target.eval()
        
        with torch.no_grad():
            q_one_est, q_two_est = self.critic(state, action)
            q_one_target, q_two_target = self.critic_target(state, action)
            
        q_est = torch.min(q_one_est, q_two_est)
        q_target = torch.min(q_one_target, q_two_target)
        
        return q_est.item(), q_target.item()



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    
    def __len__(self):
        return len(self.buffer)
    
    
    def push(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))
    
    
    def fetch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, done, next_state = zip(*batch)
        
        return (torch.stack(state),
                torch.stack(action),
                torch.stack(reward),
                torch.stack(done),
                torch.stack(next_state))
        
import torch
import torch.nn as nn
from torch.distributions import Normal



def orthogonal_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(Actor, self).__init__()
        
        self.scale = scale
        self.backbone = nn.Sequential(nn.Linear(state_dim, 512),
                                      nn.LayerNorm(512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.LayerNorm(256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.LayerNorm(256),
                                      nn.ReLU())
        
        # note that a range of actions is -1 ~ 1
        self.head = nn.Sequential(nn.Linear(256, action_dim),
                                  nn.Tanh())
        
        # self.apply(orthogonal_init)
    
    
    def forward(self, x):
        feature = self.backbone(x)
        action = self.head(feature) * self.scale
        
        return action 
    
    
    
class StochasticActor(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(StochasticActor, self).__init__()
        
        self.scale = scale
        self.dist_net = nn.Sequential(nn.Linear(state_dim, 512),
                                      nn.LayerNorm(512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.LayerNorm(256),
                                      nn.ReLU(),
                                      nn.Linear(256, action_dim*2),
                                      nn.Tanh())
        
        
    def forward(self, x):
        mean, log_std = self.dist_net(x).chunk(2, dim=1)
        
        return mean, log_std
        
        
    def sample(self, x, mode='train'):
        mean, log_std = self.forward(x)
        
        if mode == 'train':
            normal = Normal(mean, log_std.exp())
            y = normal.rsample()
            pi = torch.tanh(y)
            # once the action is squashed by non-linear function, 
            # the probability should be corrected by Jacobian det
            # f(y) = f(x) |dx/dy| 
            logp = (normal.log_prob(y) - torch.log(1 - pi**2 + 1e-8)).sum(dim=-1)
                
            return pi * self.scale, logp
        
        else: 
            return torch.tanh(mean) * self.scale
             
    

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(ActorCritic, self).__init__()
        
        self.scale = scale
        self.backbone = nn.Sequential(nn.Linear(state_dim, 512),
                                      nn.LayerNorm(512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.LayerNorm(256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.LayerNorm(256),
                                      nn.ReLU())
        
        # note that a range of actions is -1 ~ 1
        self.actor_head = nn.Sequential(nn.Linear(256, action_dim),
                                        nn.Tanh())
       
        self.critic_head = nn.Linear(256, 1)
        self.prob_head = nn.Linear(action_dim, action_dim)
               
        self.apply(orthogonal_init)
    
    
    def forward(self, x):
        feature = self.backbone(x)
        action = self.actor_head(feature)
        log_p = self.prob_head(action).sum()
        value = self.critic_head(feature)
        
        return action*self.scale, log_p, value




if __name__ == "__main__":
    state = torch.randn(4, 3)
    actor = Actor(3, 9, 3)
    action = actor(state)
    
    print(action)
import torch
import torch.nn as nn



def orthogonal_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 512),
                                 nn.LayerNorm(512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.LayerNorm(256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.LayerNorm(256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))
    
        self.apply(orthogonal_init)
    
    
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        value = self.net(xy)
        
        return value
    
    

class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TwinCritic, self).__init__()
        
        self.q_one = nn.Sequential(nn.Linear(state_dim + action_dim, 512),
                                   nn.LayerNorm(512),
                                   nn.ReLU(),
                                   nn.Linear(512, 256),
                                   nn.LayerNorm(256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.LayerNorm(256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
        
        self.q_two = nn.Sequential(nn.Linear(state_dim + action_dim, 512),
                                   nn.LayerNorm(512),
                                   nn.ReLU(),
                                   nn.Linear(512, 256),
                                   nn.LayerNorm(256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.LayerNorm(256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))
                
        self.apply(orthogonal_init)

        
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        q1 = self.q_one(xy)
        q2 = self.q_two(xy)
        
        return q1, q2
    
    
    
    
    
if __name__ == "__main__":
    action = torch.randn(3, 4)
    state = torch.randn(3, 5)
    
    critic = TwinCritic(state.shape[-1], action.shape[-1])
    q1, q2 = critic(state, action)
    print(q1, q2)
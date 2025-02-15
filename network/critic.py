import torch
import torch.nn as nn



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.backbone = nn.Sequential(nn.Linear(state_dim + action_dim, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.ReLU())
        
        self.head = nn.Linear(256, 1)
                
    
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        feature = self.backbone(xy)
        value = self.head(feature)
        
        return value 
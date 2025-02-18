import torch.nn as nn



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, scale):
        super(Actor, self).__init__()
        
        self.scale = scale
        self.backbone = nn.Sequential(nn.Linear(state_dim, 512),
                                      nn.ReLU(),
                                      nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.ReLU())
        
        # note that a range of actions is -1 ~ 1
        self.head = nn.Sequential(nn.Linear(256, action_dim),
                                  nn.Tanh())
        
    
    def forward(self, x):
        feature = self.backbone(x)
        action = self.head(feature) * self.scale
        
        return action 
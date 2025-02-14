import torch.nn as nn



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        self.backbone = nn.Sequential(nn.Linear(state_dim, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 128),
                                      nn.ReLU(),
                                      nn.Linear(128, 128),
                                      nn.ReLU())
        
        # note that a range of actions is -1 ~ 1
        self.head = nn.Sequential(nn.Linear(128, action_dim),
                                  nn.Tanh())
        
    
    def forward(self, x):
        feature = self.backbone(x)
        action = self.head(feature)
        
        return action 
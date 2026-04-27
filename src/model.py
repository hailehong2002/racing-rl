import torch
import torch.nn as nn
from torch.distributions import Categorical
from config import STATE_LAYER, ACTION_LAYER, HIDDEN_LAYER

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RacingAgent(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(self.STATE_LAYER, self.HIDDEN_LAYER)),
            nn.Tanh(),
            layer_init(nn.Linear(self.HIDDEN_LAYER, self.HIDDEN_LAYER)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(self.HIDDEN_LAYER, self.ACTION_LAYER), std=0.01)
        self.critic = layer_init(nn.Linear(self.HIDDEN_LAYER, 1), std=1.0)
        
    def forward(self, state: torch.Tensor):
        feature = self.backbone(state)
        return self.actor(feature), self.critic(feature)
        
    def get_action_and_value(self, state: torch.Tensor):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits) #Converts raw logits into a probability distribution (Applied softmax internally)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
    
    @torch.no_grad()  #Use for rollouts to get value estimates without tracking gradients
    def get_value(self, state: torch.Tensor):
        _, value = self.forward(state)
        return value
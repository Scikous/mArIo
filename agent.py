import torch.nn as nn
from model import Network
import torch.optim as optim




#Deep-Q-Learning handler
class Agent(nn.Module):
    def __init__(self) -> None:
        super(Agent, self).__init__()

        #learning parameters
        self.gamma = 1.0
        self.min_gamma = 0.005
        self.gamma_decay = 5e-5
        self.learning_rate = 5e-4
        #initialize networks
        self.online_network = Network()
        self.target_network = Network()
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        #backprop
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        loss = nn.MSELoss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

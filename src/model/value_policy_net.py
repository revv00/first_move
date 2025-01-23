import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from logging import getLogger
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, ActionLabelsBlack

logger = getLogger(__name__)

class ValuePolicyNet(nn.Module):
    def __init__(self, config: Config):
        super(ValuePolicyNet, self).__init__()
        self.config = config
        self.n_labels = len(ActionLabelsRed)
        mc = self.config.model
        
        # Input layer
        self.input_conv = nn.Conv2d(9, mc.cnn_filter_num, mc.cnn_first_filter_size, 
                                   padding='same', bias=False)
        self.input_bn = nn.BatchNorm2d(mc.cnn_filter_num)
        
        # Residual blocks using torchvision's BasicBlock
        self.residual_blocks = nn.Sequential(*[
            BasicBlock(mc.cnn_filter_num, mc.cnn_filter_num)
            for _ in range(mc.res_layer_num)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(mc.cnn_filter_num, 4, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_dense = nn.Linear(4 * 14 * 10, self.n_labels)
        
        # Value head
        self.value_conv = nn.Conv2d(mc.cnn_filter_num, 2, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_dense1 = nn.Linear(2 * 14 * 10, mc.value_fc_size)
        self.value_dense2 = nn.Linear(mc.value_fc_size, 1)
        
    def forward(self, x):
        # Input layer
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual blocks
        x = self.residual_blocks(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = torch.flatten(policy, start_dim=1)
        policy = F.softmax(self.policy_dense(policy), dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = torch.flatten(value, start_dim=1)
        value = F.relu(self.value_dense1(value))
        value = torch.tanh(self.value_dense2(value))
        
        return policy, value
        
    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
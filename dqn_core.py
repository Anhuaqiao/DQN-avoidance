import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import os

# replay buffer size
REPLAY_SIZE = 10000
# size of minibatch
small_BATCH_SIZE = 16
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

# hyper parameters
# discount factor γ
GAMMA = 0.9
# the start value of epsilon E
INITIAL_EPSILON = 0.1
# the final value of epsilon
FINAL_EPSILON = 0.01

class DQN:
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        # init experience replay
        self.replay_buffer = deque()
        # you can create the network by the two parameters
        self.create_Q_network()
        # after create the network, we can define the training methods
        self.create_updating_method()
        # set the value in choose_action
        self.epsilon = INITIAL_EPSILON
        self.model_path = model_file + "/save_model.pth"
        self.model_file = model_file
        self.log_file = log_file
        # 因为保存的模型名字不太一样，只能检查路径是否存在
        # Init session
        self.session = tf.InteractiveSession()
        if os.path.exists(self.model_file):
            print("model exists , load model\n")
            self.model = torch.load(self.model_path)
        else:
            print("model don't exists , create new one\n")
            model = nn.Module



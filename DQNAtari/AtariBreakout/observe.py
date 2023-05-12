import os, random
import numpy as np
import torch
from torch import nn
import itertools
from baselines_wrappers import DummyVecEnv
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
import time

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()


def compute_shape(net, env):
    with torch.no_grad():
        n_shape = net(torch.as_tensor(env.observation_space.sample()[None]).float()).shape[1]

    return n_shape

class Network(nn.Module):
    def __init__(self, env, device):
        super().__init__()

        self.num_actions = env.action_space.n
        self.device = device

        super().__init__()
        self.device = device
        self.num_actions = env.action_space.n

        depths = (32, 64, 64)
        n_input_channels = env.observation_space.shape[0]
        final_layer = 512

        self.fc1 = nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2)
        self.fc4 = nn.ReLU()
        self.fc5 = nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1)
        self.fc6 = nn.ReLU()
        self.fc7 = nn.Flatten()

        shape = compute_shape(nn.Sequential(self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7), env)

        self.fc8 = nn.Linear(shape, final_layer)
        self.fc9 = nn.ReLU()
        self.fc10 = nn.Linear(final_layer, self.num_actions)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        out = self.fc8(out)
        out = self.fc9(out)
        return self.fc10(out)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)

        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions
    
    def load_network(self, path):
        self.load_state_dict(torch.load(path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

make_env = lambda: make_atari_deepmind('ALE/Breakout-v5', scale_values=True) #add, render_mode=human to gym.make

vec_env = DummyVecEnv([make_env for _ in range(1)])

env = BatchedPytorchFrameStack(vec_env, k=4)
env.action_space.n = 4

net = Network(env, device)
net = net.to(device)

net.load_network('./atari_breakout_new_network_run_1_normal.pack')

obs = env.reset()
beginning_episode = True
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net.act(obs, 0.0)

    if beginning_episode:
        action = [1]
        beginning_episode = False

    obs, rew, done, _ = env.step(action)
    #env.render()
    time.sleep(0.02)

    if done[0]:
        obs = env.reset()
        beginning_episode = True
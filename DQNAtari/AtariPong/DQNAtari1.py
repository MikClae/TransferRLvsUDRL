import os
from torch import nn
import torch
from collections import deque
import itertools
import numpy as np
import random
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from baselines_wrappers import DummyVecEnv, Monitor
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 1000000
MIN_REPLAY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000
NUM_ENVS = 4
TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
LR = 5e-5
SAVE_PATH = 'atari_breakout_network.pack'
SAVE_INTERVAL = 10000
LOG_DIR = 'logs/atari_breakout'
LOG_INTERVAL = 1000


def conv_net(observation_space, depths=(32, 64, 64), final_layer=512):
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten()
    )

    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out


class Network(nn.Module):
    def __init__(self, env, device):
        super().__init__()
        self.device = device
        self.num_actions = env.action_space.n

        conv_network = conv_net(env.observation_space)

        self.net = nn.Sequential(
            conv_network,
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, observations, eps):
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        q_values = self(observations_tensor)
        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            if eps >= random.random():
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions

    def calculate_loss(self, batch, target_net):
        observations = [t[0] for t in batch]
        actions = np.asarray([t[1] for t in batch])
        rewards = np.asarray([t[2] for t in batch])
        finishes = np.asarray([t[3] for t in batch])
        new_observations = [t[4] for t in batch]

        if isinstance(observations[0], PytorchLazyFrames):
            observations = np.stack([obs.get_frames() for obs in observations])
            new_observations = np.stack([obs.get_frames() for obs in new_observations])
        else:
            observations = np.asarray(observations)
            new_observations = np.asarray(new_observations)

        observations_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        finishes_tensor = torch.as_tensor(finishes, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_observations_tensor = torch.as_tensor(new_observations, dtype=torch.float32, device=self.device)

        # Compute Targets
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_observations_tensor)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rewards_tensor + GAMMA * (1 - finishes_tensor) * max_target_q_values

        # Compute Loss
        q_values = online_net(observations_tensor)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_tensor)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss

    def save_network(self, path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as file:
            file.write(params_data)

    def load_network(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        with open(path, 'rb') as file:
            params_numpy = msgpack.loads(file.read())

        params = {k: torch.as_tensor(v, device=self.device) for k, v in params_numpy.items()}

        self.load_state_dict(params)


make_env = lambda: Monitor(make_atari_deepmind("ALE/Breakout-v5", scale_values=True), allow_early_resets=True)
vector_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
# env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)]])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

env = BatchedPytorchFrameStack(vector_env, k=4)

replay_buffer = deque(maxlen=BUFFER_SIZE)
episode_info_buffer = deque([], maxlen=100)

episode_count = 0

summary_writer = SummaryWriter(LOG_DIR)

online_net = Network(env, device=device)
target_net = Network(env, device=device)

online_net = online_net.to(device)
target_net = target_net.to(device)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

# Initialize replay buffer
observations = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    actions = [env.action_space.sample() for i in range(NUM_ENVS)]

    new_observations, rewards, finishes, _ = env.step(actions)

    for observation, action, reward, finish, new_observation in zip(observations, actions, rewards, finishes,
                                                                    new_observations):
        transition = (observation, action, reward, finish, new_observation)
        replay_buffer.append(transition)

    observations = new_observations

# Main Training Loop
observations = env.reset()
iterations = 1500000
for iteration in tqdm(range(iterations)):
    epsilon = np.interp(iteration * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    if isinstance(observations[0], PytorchLazyFrames):
        observations_ = np.stack([obs.get_frames() for obs in observations])
        actions = online_net.act(observations_, epsilon)
    else:
        actions = online_net.act(observations, epsilon)

    new_observations, rewards, finishes, infos = env.step(actions)

    for observation, action, reward, finish, new_observation, info in zip(observations, actions, rewards, finishes,
                                                                          new_observations, infos):
        transition = (observation, action, reward, finish, new_observation)
        replay_buffer.append(transition)

        if finish:
            episode_info_buffer.append(info['episode'])
            episode_count += 1

    observations = new_observations

    transitions = random.sample(replay_buffer, BATCH_SIZE)
    loss = online_net.calculate_loss(transitions, target_net)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update Target Net
    if iteration % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if iteration % LOG_INTERVAL == 0 and iteration != 0:
        reward_mean = np.mean([e['r'] for e in episode_info_buffer]) or 0
        length_mean = np.mean([e['l'] for e in episode_info_buffer]) or 0

        # print()
        # print('Step:', iteration)
        # print('Avg Rew:', reward_mean)
        # print('Avg Length:', length_mean)
        # print('Episodes:', episode_count)

        summary_writer.add_scalar("Avg Reward", reward_mean, global_step=iteration)
        summary_writer.add_scalar("Avg Length", length_mean, global_step=iteration)
        summary_writer.add_scalar("Episodes", episode_count, global_step=iteration)

    # Saving

    if iteration % SAVE_INTERVAL == 0 and iteration != 0:
        online_net.save_network(SAVE_PATH)
        # print("Model saved")

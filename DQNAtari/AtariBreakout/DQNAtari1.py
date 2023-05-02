from torch import nn
import torch
from collections import deque
import itertools
import numpy as np
import random
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from baselines_wrappers import DummyVecEnv, Monitor, SubprocVecEnv
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 1000000
MIN_REPLAY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 1300000
NUM_ENVS = 4
TARGET_UPDATE_FREQ = 10000 // NUM_ENVS
LR = 5e-5
SAVE_PATH = './atari_breakout_new_network_run_1.pack'
SAVE_INTERVAL = 10000
LOG_DIR = 'logs/atari_breakout'
LOG_INTERVAL = 1000


def compute_shape(net, env):
    with torch.no_grad():
        n_shape = net(torch.as_tensor(env.observation_space.sample()[None]).float()).shape[1]

    return n_shape


class Network(nn.Module):
    def __init__(self, env, device):
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

    def act(self, observations, eps):
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        q_values = self.forward(observations_tensor)
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

        loss = nn.functional.mse_loss(action_q_values, targets)

        return loss

    def save_network(self, path):
        torch.save(self.state_dict(), path)

    def load_network(self, path):
        self.load_state_dict(torch.load(path))

    def load_part(self, path):
        state_dict_ = torch.load(path)
        with torch.no_grad():
            self.fc5.weight.copy_(state_dict_['fc5.weight'])
            self.fc5.bias.copy_(state_dict_['fc5.bias'])
#            self.fc6.weight.copy_(state_dict_['fc6.weight'])
#            self.fc6.bias.copy_(state_dict_['fc6.bias'])
#             self.fc7.weight.copy_(state_dict_['fc7.weight'])
#             self.fc7.bias.copy_(state_dict_['fc7.bias'])
            self.fc8.weight.copy_(state_dict_['fc8.weight'])
            self.fc8.bias.copy_(state_dict_['fc8.bias'])
#            self.fc9.weight.copy_(state_dict_['fc9.weight'])
#            self.fc9.bias.copy_(state_dict_['fc9.bias'])
            self.fc10.weight.copy_(state_dict_['fc10.weight'])
            self.fc10.bias.copy_(state_dict_['fc10.bias'])


make_env = lambda: Monitor(make_atari_deepmind("ALE/Pong-v5", scale_values=True), allow_early_resets=True)
vector_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
# vector_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

env = BatchedPytorchFrameStack(vector_env, k=4)
env.action_space.n = 4

replay_buffer = deque(maxlen=BUFFER_SIZE)
episode_info_buffer = deque([], maxlen=100)

episode_count = 0

summary_writer = SummaryWriter(LOG_DIR)

online_net = Network(env, device=device)
target_net = Network(env, device=device)

online_net = online_net.to(device)
target_net = target_net.to(device)

# online_net.load_part('./atari_pong_new_network_run_1.pack')

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
lives = np.zeros(NUM_ENVS)
for iteration in tqdm(range(iterations)):
    epsilon = np.interp(iteration, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    if isinstance(observations[0], PytorchLazyFrames):
        observations_ = np.stack([obs.get_frames() for obs in observations])
        actions = online_net.act(observations_, epsilon)
    else:
        actions = online_net.act(observations, epsilon)

    new_observations, rewards, finishes, infos = env.step(actions)
    for info in range(len(infos)):
        if infos[info]['lives'] == lives[info] - 1:
            rewards[info] = -1
        lives[info] = infos[info]['lives']

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
        print(' Step:', iteration)
        print('Avg Rew:', reward_mean)
        print('Avg Length:', length_mean)
        print('Epsilon:', epsilon)

        summary_writer.add_scalar("Avg Reward", reward_mean, global_step=iteration)
        summary_writer.add_scalar("Avg Length", length_mean, global_step=iteration)

    # Saving

    if iteration % SAVE_INTERVAL == 0 and iteration != 0:
        online_net.save_network(SAVE_PATH)
        print("Model saved")

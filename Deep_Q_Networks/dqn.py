import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from torch.nn import Sequential

from pureastroidgame import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

plt.ion()

print(torch.xpu.is_available())

device = torch.device(
    "xpu" if torch.xpu.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.sequence = Sequential(
            nn.Linear(n_observations, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, n_actions),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sequence.forward(x)



BATCH_SIZE = 128
GAMMA = 0.99
#GAMMA = 0.1
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 2500
TAU = 0.005
#TAU = 0.1
#LR = 3e-4
LR = 0.001

# n_actions = env.action_space.n
n_actions = len(bot.action_space)

# state, info = env.reset()
state, _ = reset()
n_observations = len(state)

#print(state.shape)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.SGD(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():

            return policy_net(state).max(1).indices.view(1, 1)
    else:
        #return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return torch.tensor([[bot.action_space[random.randint(0, n_actions-1)]]], device=device, dtype=torch.long)


episode_durations = []
episode_scores = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    score_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Duration')
    #plt.ylabel('Score')
    #plt.scatter(durations_t.numpy(), score_t.numpy())
    plt.xlabel('Episode')
    plt.plot(durations_t.numpy())
    # plt.ylabel('Score')
    # plt.plot(score_t.numpy())

    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)

    #print(state_batch.shape)

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 500

highest_duration = 0

for i_episode in range(num_episodes):

    #state, info = env.reset()
    state, _ = reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    #state = torch.moveaxis(state, 3, 1)

    score = 0
    for t in count():
        action = select_action(state)

        observation, reward, terminated, truncated, _ = step(action.item())

        score += reward

        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            #next_state = torch.moveaxis(next_state, 3, 1)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)


        if done:
            if t+1 > highest_duration:
                highest_duration = t + 1
            episode_durations.append(t + 1)
            episode_scores.append(score)
            #if len(episode_durations) % 40 == 0:
            plot_durations()
            print(f"lasted {t+1} ticks with a score of {score}")
            break


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

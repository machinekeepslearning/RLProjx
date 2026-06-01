import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy
from torch.nn import Sequential

from Environments.pureastroidgame import *

import torch
import torch.nn as nn
import torch.optim as optim
import time
import keyboard

policy_path = "asteroid_policy.pt"
target_path = "asteroid_target.pt"

plt.ion()

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
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.sequence.forward(x)

episode_durations = []
episode_rewards = []
avg_q_values = []

BATCH_SIZE = 128
GAMMA = 0.99
#GAMMA = 0.1
EPS_START = 0.95
EPS_END = 0.1
EPS_DECAY = 500000
TAU = 0.005
#TAU = 0.1
#LR = 3e-4
LR = 1e-4

frame_skip = 1

preload = False

# n_actions = env.action_space.n
n_actions = len(bot.action_space)

# state, info = env.reset()
state, _ = reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
if preload:
    policy_net.load_state_dict(torch.load(policy_path, weights_only=True))
    target_net.load_state_dict(torch.load(target_path, weights_only=True))
else:
    target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.SGD(policy_net.parameters(), lr=LR)
memory = ReplayMemory(1000000)

if preload:
    steps_done = 5 * EPS_DECAY
else:
    steps_done = 0

terminate = False


def termTraining():
    global terminate

    terminate = True


keyboard.add_hotkey('q', termTraining)

notified = False

global_start = time.time()


def select_action(state):
    global steps_done, notified
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if eps_threshold < 0.5 and not notified:
        print(f"Policy Mode at {time.time() - global_start}")
        print(f"Episode: {len(episode_durations)}")
        notified = True

    with torch.no_grad():
        decision = policy_net(state).max(1)

    max_q = decision[0][0].cpu().numpy()

    q_values.append(max_q)

    if sample > eps_threshold:
        return decision.indices.view(1, 1)
    else:
        #return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return torch.tensor([[bot.action_space[random.randint(0, n_actions - 1)]]], device=device, dtype=torch.long)


fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)
fig.set_size_inches(8, 6.4, True)

def plot_durations(show_result=False):
    global ax1, ax2

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    score_t = torch.tensor(episode_rewards, dtype=torch.float)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Duration (ticks)')
    ax1.plot(score_t.numpy())
    ax2.plot(durations_t.numpy())

    if len(score_t) >= 10:
        means = score_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        ax1.plot(means.numpy())

        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        ax2.plot(means.numpy())
    plt.pause(0.001)


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

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 700

start = time.time()

for i_episode in range(num_episodes):
    state, _ = reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    step_start = time.time()

    total_reward = 0
    q_values = []
    for t in count():
        global observation, d_reward, terminated, truncated

        action = select_action(state)

        reward = 0
        for i in range(frame_skip):
            observation, d_reward, terminated, truncated, _ = step(action.item())
            reward += d_reward

        total_reward += reward

        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if steps_done % 316000 == 0:
            torch.save(policy_net.state_dict(), policy_path)
            torch.save(target_net.state_dict(), target_path)
            print(f"Saved Data after {time.time()-start} seconds")

        if done or terminate:
            end = time.time() - step_start
            episode_durations.append((t + 1))
            episode_rewards.append(total_reward)
            avg_q_values.append(numpy.mean(q_values))
            plot_durations()
            break
    if terminate:
        break

torch.save(policy_net.state_dict(), policy_path)
torch.save(target_net.state_dict(), target_path)

print('Complete')
print(f"Training took {time.time() - start} seconds")



plot_durations(show_result=True)

qfig, qaxis = plt.subplots()
qaxis.plot(avg_q_values)

print(f"Average Max Q Vals: {avg_q_values}")

fig.savefig("dqn_fire.png")
qfig.savefig("q_vals.png")

plt.ioff()
plt.show()

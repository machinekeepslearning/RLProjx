import torch
import os, sys
import gymnasium_env
from Deep_Q_Networks.helpers.helper_modules import *
import matplotlib.pyplot as plt

os.add_dll_directory(os.environ["ffmpeg_dir"])

env_id = "AsteroidEnv-v0"

torch.manual_seed(0)

import time
import keyboard
from torchrl.envs import GymEnv, StepCounter, TransformedEnv, Compose

transforms = [StepCounter()]

env = TransformedEnv(GymEnv(env_id, obs_type="grayscale", bounds=(84, 84)), Compose(transforms))
env.auto_register_info_dict()
env.set_seed(0)

from tensordict.nn import TensorDictSequential as Seq, TensorDictModule
from tensordict import TensorDict
from torchrl.modules import EGreedyModule, MLP, ConvNet, QValueActor
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, PrioritizedReplayBuffer, ReplayBuffer
from torch.optim import Adam

from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl._utils import logger as torchrl_logger
from torchrl.record import CSVLogger

ITFDict = TensorDictModule(IntToFloat(), in_keys=["observation"], out_keys=["observation_float"])

CNNDict = TensorDictModule(ConvNet(in_features=1, num_cells=[8, 16, 32], kernel_sizes=[3, 2, 2], strides=[3, 2, 2]),
                           in_keys=["observation_float"], out_keys=["observation_float"])

MLPDict = TensorDictModule(MLP(out_features=env.action_spec.shape[-1], num_cells=[512, 512]),
                           in_keys=["observation_float"], out_keys=["action_value"])

value_net = Seq(ITFDict, CNNDict, MLPDict)

policy = QValueActor(value_net, spec=env.action_spec)
exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=50_000, eps_init=0.5
)
complete_policy = Seq(policy, exploration_module)
init_rand_steps = 80000
frames_per_batch = 32
optim_steps = 5

rb = PrioritizedReplayBuffer(storage=LazyTensorStorage(500_000), alpha=1, beta=0.5)
collector = Collector(
    env,
    complete_policy,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
    auto_register_policy_transforms=True,
)
#rb = ReplayBuffer(storage=LazyTensorStorage(500_000))
loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True, double_dqn=True)
optim = Adam(loss.parameters(), lr=0.001)
updater = SoftUpdate(loss, eps=0.99)

path = "./training_loop"
logger = CSVLogger(exp_name="dqn",
                   log_dir=path)

total_count = 0
total_episodes = 0
t0 = time.time()
for i, data in enumerate(collector):
    logger.log_scalar("chosen_action_value", torch.mean(data["chosen_action_value"]))
    logger.log_scalar("step_count", torch.mean(data["step_count"].float()))
    logger.log_scalar("reward", torch.mean(data["reward"].float()))
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        for _ in range(optim_steps):
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            exploration_module.step(data.numel())
            updater.step()
            if i % 10:
                torchrl_logger.info(f"Max num steps: {max_length}, rb length {len(rb)}")
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()
    if max_length >= 100000 or keyboard.is_pressed("`"):
        break

torch.save(complete_policy.state_dict(), "saved_weights/MLP_RGB.pt")

t1 = time.time()

torchrl_logger.info(
    f"solved after {total_count} steps, {total_episodes} episodes and in {t1 - t0}s"
)

logger.flush()

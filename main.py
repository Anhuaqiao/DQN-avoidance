import torch
import torch.nn as nn
from torch import optim
from dqn import DQN
import random
import math
from env import Env
from utils import ReplayMemory, Transition
from itertools import count
from collections import deque
import keyboard
import os
import time
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename='step.log', level=logging.INFO, format='%(message)s')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

lr = 2.5E-4  # learning rate 0.00025
epoch = 10000  # training epoch
batch_size = 500

# hyper parameters
GAMMA = 0.99
EPS_START = 0.1
EPS_END = 0.01
EPS_DECAY = 5000
WARMUP = 1000

eps_threshold = EPS_START
steps_done = 0

paused = False
model_file = os.path.join('model', 'Q_net.pth')


def select_action(state: torch.Tensor, surv_dur: torch.Tensor) -> int:
    """
    epsilon greedy
    - epsilon: choose random action
    - 1-epsilon: argmax Q(a,s)

    Input: state shape (1,4,84,84)

    Output: action shape (1,1)
    """
    global eps_threshold
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return Q_net(state, surv_dur).max(1)[1]
    else:
        return random.sample(env.action_space, 1)[0]


def pause_training():
    global paused
    paused = not paused


def save_model():
    print("saving model...")
    torch.save(Q_net.state_dict(), model_file)
    print("done.")


keyboard.add_hotkey('p', pause_training)

print("press s to start training")
keyboard.wait('s')

env = Env()
n_action = len(env.action_space)

log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, "log.txt")

# create q value network and target network
gpu = torch.device("cuda:0")
cpu = torch.device("cpu")
Q_net = DQN(in_channels=4, n_actions=len(env.action_space)).to(gpu)
target_net = DQN(in_channels=4, n_actions=len(env.action_space)).to(gpu)
if os.path.isfile(model_file):
    Q_net.load_state_dict(torch.load(model_file))
target_net.load_state_dict(Q_net.state_dict())
keyboard.add_hotkey('m', save_model)
target_net.eval()

memory = ReplayMemory(2500)

# optimizer
optimizer = optim.AdamW(Q_net.parameters(), lr=lr, amsgrad=True)

# warming up
print("Warming up...")
warm_up_step = 0
for episode in count():
    episode_start_time = time.time()
    surv_dur = 0.0  # 存活时长
    frame = env.restart()
    frame = torch.from_numpy(frame).to(gpu)
    frame = torch.stack((frame, frame, frame, frame)).unsqueeze(0)

    for step in count():
        warm_up_step += 1
        step_start_time = time.time()
        # take one step
        action = random.sample(env.action_space, 1)[0]
        next_frame, done, reward, surv_dur = env.step(action)

        # convert to tensor
        cur_survive_duration = step_start_time-episode_start_time
        surv_dur = torch.tensor([surv_dur]).unsqueeze(0).to(gpu)
        reward = torch.tensor([reward])  # (1)
        done = torch.tensor([done])  # (1)
        next_frame = torch.from_numpy(next_frame).to(gpu)  # (84,84)
        next_frame = torch.stack((next_frame, frame[0][0], frame[0][1], frame[0][2])).unsqueeze(0)

        memory.push(frame, action, next_frame, surv_dur, reward, done)

        frame = next_frame

        if done:
            break

    if warm_up_step > WARMUP:
        break

rewardList = []
lossList = []
rewarddeq = deque([], maxlen=100)
lossdeq = deque([], maxlen=100)
avgrewardlist = []
avglosslist = []

# start training
print("Training ...")
pb = 0
for episode in count():
    episode_start_time = time.time()
    surv_dur = 0.0  # 存活时长
    surv_dur = torch.tensor([surv_dur], device=gpu).unsqueeze(0).to(gpu)

    frame = env.restart()
    frame = torch.from_numpy(frame).to(gpu)
    frame = torch.stack((frame, frame, frame, frame)).unsqueeze(0)

    total_loss = 0.0
    total_reward = 0

    for step in count():
        # if paused:
        #    print("training paused. Press p to continue")
        #    keyboard.wait('p')
        step_start_time = time.time()
        action = select_action(frame, surv_dur)

        next_frame, done, reward, surv_dur = env.step(action)
        total_reward += reward

        # convert to tensor
        surv_dur = torch.tensor([surv_dur]).unsqueeze(0).to(gpu)
        reward = torch.tensor([reward])  # (1)
        done = torch.tensor([done])  # (1)
        next_frame = torch.from_numpy(next_frame).to(gpu)  # (84,84)
        next_frame = torch.stack((next_frame, frame[0][0], frame[0][1], frame[0][2])).unsqueeze(0)

        memory.push(frame, action, next_frame, surv_dur, reward, done)

        frame = next_frame

        # train
        Q_net.train()

        # 将transitions列表转换为一个包含个属性列表的Transition，
        # [(transition1),(transition2),...] -> ([state],[action],[next_state],[reward])
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(gpu)  # (bs,4,84,84)
        next_state_batch = torch.cat(batch.next_state).to(gpu)  # (bs,4,84,84)
        surv_dur_batch = torch.cat(batch.surv_dur).to(gpu) # (bs,1)
        action_batch = torch.tensor(batch.action).unsqueeze(1).to(gpu)  # (bs,1)
        reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(gpu)  # (bs,1)
        done_batch = torch.tensor(batch.done).unsqueeze(1).to(gpu)  # (bs,1)

        try:
            state_Q_values = Q_net(state_batch, surv_dur_batch)  # Q(st,a), (bs,n_actions)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception

        selected_state_Q_value = state_Q_values.gather(1, action_batch)  # 实际选择的a: Q(st,a), (bs, 1)

        with torch.no_grad():
            try:
                next_state_Q_values = target_net(next_state_batch,surv_dur_batch)  # Q'(st+1, a), (s(bs,n_actions)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    else:
                        raise exception
            selected_next_state_Q_value = next_state_Q_values.max(1, keepdim=True)[0]  # max_a Q'(st+1,a), (bs,1)

        # TD target
        TD_target = selected_next_state_Q_value * GAMMA * ~done_batch + reward_batch  # (bs,1)
        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_state_Q_value, TD_target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_stop_time = time.time()
        duration = (step_stop_time - step_start_time) * 1000
        logging.info(f"step {steps_done} cost {duration} ms")
        #print(f"step {steps_done} cost {duration} ms")

        # update target_net every 1500 steps
        if steps_done % 1000 == 0:
            target_net.load_state_dict(Q_net.state_dict())

        if done:
            break
    episode_stop_time = time.time()
    pb = max(pb, int((episode_stop_time - episode_start_time) * 50))
    rewardList.append(total_reward)
    lossList.append(total_loss)
    rewarddeq.append(total_reward)
    lossdeq.append(total_loss)
    avgreward = sum(rewarddeq) / len(rewarddeq)
    avgloss = sum(lossdeq) / len(lossdeq)
    avglosslist.append(avgloss)
    avgrewardlist.append(avgreward)
    output = f"Episode {episode}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, Epsilon {eps_threshold:.2f}, TotalStep {steps_done}"
    print(output)
    with open(log_path, "a") as f:
        f.write(f"{output}\n")

# plot loss-epoch and reward-epoch
plt.figure(1)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(range(len(lossList)), lossList, label="loss")
plt.plot(range(len(lossList)), avglosslist, label="avg")
plt.legend()
plt.savefig(os.path.join(log_dir, "loss.png"))

plt.figure(2)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.plot(range(len(rewardList)), rewardList, label="reward")
plt.plot(range(len(rewardList)), avgrewardlist, label="avg")
plt.legend()
plt.savefig(os.path.join(log_dir, "reward.png"))

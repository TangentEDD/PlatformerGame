import random
import torch
import torch.nn as nn

def train_step(memory, batch_size, qnet, target_qnet, optimizer, gamma):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = qnet(states).gather(1, actions)
    next_q_values = target_qnet(next_states).max(1)[0].unsqueeze(1)
    target = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
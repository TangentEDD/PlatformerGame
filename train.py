from platform_env import PlatformEnv
from qnet import QNet
from utils import train_step
import torch
import torch.optim as optim
import random
import collections
import pygame
env = PlatformEnv()
state_dim = 4
action_dim = 3
qnet = QNet(state_dim, action_dim)
target_qnet = QNet(state_dim, action_dim)
target_qnet.load_state_dict(torch.load("qnet.pth"))

def demo_episode(env, qnet, max_steps=200):
    state = env.reset()
    done = False
    total_reward = 0
    # 初始化 pygame 窗口
    pygame.init()
    screen = pygame.display.set_mode((int(env.width*50), int(env.height*50)))
    clock = pygame.time.Clock()

    for t in range(max_steps):
        #如果游戏中按下任意键，直接跳过演示，进入下一轮训练
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:  # 任意键跳过演示
                pygame.quit()
                return total_reward
        # 用当前模型选择动作（不探索）
        with torch.no_grad():
            action = qnet(torch.tensor(state)).argmax().item()

        state, reward, done = env.step(action)
        total_reward += reward

        # 渲染画面
        env.render(screen)
        clock.tick(30)  # 控制帧率

        if done:
            break

    print(f"[DEMO] Total reward: {total_reward}")
    pygame.quit()

optimizer = optim.Adam(qnet.parameters(), lr=0.001)
memory = collections.deque(maxlen=20000)

gamma = 0.99
epsilon = 0.4
batch_size = 64
for episode in range(7000):
    state = env.reset()
    total_reward = 0
    for t in range(env.max_steps):
        if random.random() < epsilon:
            action = random.randint(0, action_dim-1)
        else:
            with torch.no_grad():
                action = qnet(torch.tensor(state)).argmax().item()

        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_step(memory, batch_size, qnet, target_qnet, optimizer, gamma)
        if done:
            break

    if episode % 100 == 0:
        target_qnet.load_state_dict(qnet.state_dict())
    if episode % 500 == 0:
        print(f"Episode {episode}, total reward: {total_reward}, epsilon: {epsilon:.3f}")
    
        demo_episode(env, qnet)
    if total_reward > 2000:
        demo_episode(env, qnet)
        # 如果模型成功，提前保存模型并结束训练
        torch.save(qnet.state_dict(), "qnetBest.pth")
        print("Model saved as qnetBest.pth")
        break
    epsilon = max(0.05, epsilon * 0.995)
    print(f"Episode {episode}, total reward: {total_reward}")

torch.save(qnet.state_dict(), "qnet2.pth")


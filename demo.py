import pygame
import torch
from platform_env import PlatformEnv
from qnet import QNet

env = PlatformEnv()
state_dim = 4
action_dim = 3
qnet = QNet(state_dim, action_dim)

gametype = "demo"  # "demo" or "play"
pygame.init()
screen = pygame.display.set_mode((int(env.width*50), int(env.height*50)))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 18)

def wait_for_start():
    global gametype
    waiting = True
    while waiting:
        screen.fill((0, 0, 0))
        tutorial_text = "Arrow keys to move, Space to jump. Reach the green dot!"
        text_tutorial = font.render(tutorial_text, True, (255, 255, 255))
        rect_tutorial = text_tutorial.get_rect(center=(screen.get_width()//2, screen.get_height()//2 - 40))
        screen.blit(text_tutorial, rect_tutorial)
        text1 = font.render("Press X to start model demonstration", True, (255, 255, 255))
        rect1 = text1.get_rect(center=(screen.get_width()//2, screen.get_height()//2 - 20)) 
        screen.blit(text1, rect1)
        text2 = font.render("Press C to start playing", True, (255, 255, 255))
        rect2 = text2.get_rect(center=(screen.get_width()//2, screen.get_height()//2 + 20))
        screen.blit(text2, rect2)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:  # x键开始
                waiting = False
                gametype = "demo"
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:  # c键开始玩
                waiting = False
                gametype = "play"
def play_once():  #模型演示进程
    state = env.reset()
    done = False
    running = True
    total_reward = 0
    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                done = True

        with torch.no_grad():
            action = qnet(torch.tensor(state)).argmax().item()

        next_state, reward, done = env.step(action)
        state = next_state

        env.render(screen)
        total_reward += reward
        text = font.render(f"Reward: {total_reward:.2f}", True, (0, 0, 0))
        screen.blit(text, (0, 10))
        pygame.display.flip()
        clock.tick(30)

    return "end"

def wait_for_restart_or_exit():
    waiting = True
    while waiting:
        
        text3 = font.render("Press X to replay, press Z to exit", True, (0, 0, 0))
        rect3 = text3.get_rect(center=(screen.get_width()//2, screen.get_height()//2))
        screen.blit(text3, rect3)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    return "restart"
                elif event.key == pygame.K_z:
                    return "exit"


while True:
    wait_for_start()
    if gametype == "play":
        env.play()
    elif gametype == "demo":
        qnet.load_state_dict(torch.load("qnet.pth"))
        qnet.eval()
        play_once() 
    choice = wait_for_restart_or_exit()
    if choice == "restart":
        continue
    elif choice == "exit":
        break

pygame.quit()

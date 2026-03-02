import numpy as np
import pygame
# -------------------------
# 1. 平台环境定义
# -------------------------
class PlatformEnv:
    def __init__(self):
        
        self.width = 20.0
        self.height = 10.0
        self.goal = (19.0, 0.7) #目标点
        self.max_steps = 200
        self.steps = 0
        self.platforms = [ # 平台列表，每个平台由左边界x1、右边界x2和高度y定义
            {"x1": 2.0, "x2": 4.0, "y": 1.0}, 
            {"x1": 5.0, "x2": 6.0, "y": 1.5}, 
            {"x1": 7.0, "x2": 8.0, "y": 2.0}, 
            {"x1": 9.0, "x2": 12.0, "y": 1.5},
            {"x1": 13.0, "x2": 16.0, "y": 1.0},
            {"x1": 17.0, "x2": 19.0, "y": 0.7},
        ]
        
        self.just_landed = False
        self.is_grounded = False
        self.reset()
    
    def reset(self):
        self.x = 2.1
        self.y = 1.2
        self.vx = 0.0
        self.vy = 0.0
        self.steps = 0
        self.stagnant_steps = 0
        self.last_x = self.x
        self.last_y = self.y

        self.just_landed = False
        self.is_grounded = True
        self.checkpointsets = [
            (5.5, 1.5), (7.5, 2.0), (9.5, 1.5),(13.5, 1.0),(17.5, 0.7)
        ]
        self.checkpoint = self.checkpointsets[0] # 初始检查点为第一个平台的检查点
        return self._get_state()

    def _get_state(self):
        
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)


    def step(self, action):
        # 动作空间: 0=左移, 1=右移, 2=跳跃
        if action == 0:
            self.vx = -0.2
        elif action == 1:
            self.vx = 0.2
        elif action == 2 and self.is_grounded:  # 跳跃
            self.vy = 0.6
            self.is_grounded = False
        
        # 更新位置
        self.x += self.vx
        self.y += self.vy
        self.vy -= 0.09  # 重力
        self.steps += 1
        friction = 0.4 if self.is_grounded else 0.0 #摩擦力
        self.vx *= friction

        # 边界处理
        self.x = max(0.0, min(self.width, self.x))
        if self.y < 0.0:
            self.y = 0.0
            self.vy = 0.0
            self.is_grounded = True

        # 奖励
        reward = -0.01
        done = False
        # 距离惩罚 越远离目标点越高
        reward += -0.01 * abs(self.goal[0] - self.x)
        reward += -0.008 * abs(self.goal[1] - self.y)
        
        if abs(self.x - self.goal[0]) < 0.5 and abs(self.y - self.goal[1]) < 0.5:
            reward = 3000.0
            done = True
        elif self.y > self.height or self.y <= 0:  # 掉出地图
            reward -= 20.0
            done = True
        if abs(self.x - self.checkpoint[0]) < 0.5 and abs(self.y - self.checkpoint[1]) < 0.5:
            reward += 200.0 # 到达检查点奖励
            # 更新检查点为下一个平台的检查点，直到最后一个平台
            if self.checkpoint in self.checkpointsets:
                idx = self.checkpointsets.index(self.checkpoint)
                if idx < len(self.checkpointsets) - 1:
                    self.checkpoint = self.checkpointsets[idx + 1]
            # 如果已经到达最后一个检查点，把它移到地图外，表示不再有检查点
            else:
                self.checkpoint = (self.width + 1.0, self.height + 1.0)
        for p in self.platforms:#检测是否落到平台上
            # 条件1：x 在平台范围内
            if p["x1"] <= self.x <= p["x2"]:
                #条件2：球在平台下方，并且下一步会穿过平台高度（说明是从下往上跳）
                if self.y < p["y"] and self.y + self.vy > p["y"]:
                    self.y = p["y"] - 0.2 # 保持在平台下方
                    self.vy = 0.0
                # 条件3：球在平台上方，并且下一步会穿过平台高度（说明是从上往下落）


                elif self.y > p["y"] and self.y + self.vy < p["y"]:
                    self.y = p["y"] + 0.2 # 保持在平台上方
                    self.vy = 0.0
                    self.is_grounded = True
                    if not self.just_landed:
                        reward += 30
                        self.just_landed = True
        # 停滞惩罚逻辑
        if abs(self.x - self.last_x) < 0.01 and abs(self.y - self.last_y) < 0.01:
            self.stagnant_steps += 1
        else:
            self.stagnant_steps = 0

        self.last_x = self.x
        self.last_y = self.y

        if self.stagnant_steps >= 10:  # 连续10步几乎没动
            reward -= 0.2 * self.stagnant_steps
                    
        if self.steps >= self.max_steps:
            reward -= 100.0
            done = True
        return self._get_state(), reward, done

    def render(self, screen):
        screen.fill((135, 206, 235))  # 天空背景

        # 画平台
        for p in self.platforms:
            pygame.draw.rect(
                screen,
                (139, 69, 19),
                pygame.Rect(p["x1"]*50, self.height*50 - p["y"]*50, (p["x2"]-p["x1"])*50, 10)
            )

        # 画角色
        pygame.draw.circle(
            screen,
            (255, 0, 0),
            (int(self.x*50), int(self.height*50 - self.y*50)),
            10
        )

        # 画目标点
        pygame.draw.circle(
            screen,
            (0, 255, 0),
            (int(self.goal[0]*50), int(self.height*50 - self.goal[1]*50)),
            10
        )
        # 画检查点
        if self.checkpoint[0] > 0 and self.checkpoint[1] > 0: # 检查点有效时才画
            pygame.draw.circle(
                screen,
                #黄色的检查点
                (255, 255, 0),
                (int(self.checkpoint[0]*50), int(self.height*50 - self.checkpoint[1]*50)),
                10
            )
        pygame.display.flip()


#玩家操作
    def play(self):
        done = False
        pygame.init()
        screen = pygame.display.set_mode((int(self.width*50), int(self.height*50)))
        clock = pygame.time.Clock()
        running = True
        state = self.reset()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            horizontal_action = -1
            jump_action = -1

            if keys[pygame.K_LEFT]:
                horizontal_action = 0
            elif keys[pygame.K_RIGHT]:
                horizontal_action = 1

            if keys[pygame.K_SPACE]:
                jump_action = 2

            # 先执行水平动作
            if horizontal_action != -1:
                state, reward, done = self.step(horizontal_action)

            # 再执行跳跃动作
            if jump_action != -1:
                state, reward, done = self.step(jump_action)
            # 每一帧都调用 step()
            action = 3  # 无动作
            state, reward, done = self.step(action)
            print(f"x={self.x:.2f}, y={self.y:.2f}, vx={self.vx:.2f}, vy={self.vy:.2f}, done={done}")

            self.render(screen)
            clock.tick(30)

            if done:
                print("Episode finished!")
                running = False

        
        return "end"




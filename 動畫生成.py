'''import pygame
import sys
import tkinter as tk
from tkinter import ttk
import math

class CircleAnimator:
    def __init__(self):
        # 初始化Pygame
        pygame.init()
        
        # 設置顯示器
        self.WIDTH = 1920
        self.HEIGHT = 1080
        
        # 創建兩個視窗：一個用於編輯，一個用於投影
        self.edit_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.projection_screen = None  # 稍後初始化
        
        # 初始化時鐘
        self.clock = pygame.time.Clock()
        
        # 儲存圓形參數
        self.circles = []
        self.is_animating = False
        
        # 初始化Tkinter視窗
        self.root = tk.Tk()
        self.root.title("參數設定")
        self.setup_gui()
        
    def setup_gui(self):
        # 創建輸入框和標籤
        ttk.Label(self.root, text="內徑:").grid(row=0, column=0, padx=5, pady=5)
        self.inner_radius = ttk.Entry(self.root)
        self.inner_radius.grid(row=0, column=1, padx=5, pady=5)
        self.inner_radius.insert(0, "50")
        
        ttk.Label(self.root, text="外徑:").grid(row=1, column=0, padx=5, pady=5)
        self.outer_radius = ttk.Entry(self.root)
        self.outer_radius.grid(row=1, column=1, padx=5, pady=5)
        self.outer_radius.insert(0, "100")
        
        ttk.Label(self.root, text="移動距離 (像素):").grid(row=2, column=0, padx=5, pady=5)
        self.move_distance = ttk.Entry(self.root)
        self.move_distance.grid(row=2, column=1, padx=5, pady=5)
        self.move_distance.insert(0, "500")
        
        ttk.Label(self.root, text="移動速度 (像素/秒):").grid(row=3, column=0, padx=5, pady=5)
        self.move_speed = ttk.Entry(self.root)
        self.move_speed.grid(row=3, column=1, padx=5, pady=5)
        self.move_speed.insert(0, "200")
        
        # 創建按鈕
        ttk.Button(self.root, text="開始動畫", command=self.start_animation).grid(row=4, column=0, columnspan=2, pady=10)
        
    def create_circle(self, pos):
        try:
            inner_r = float(self.inner_radius.get())
            outer_r = float(self.outer_radius.get())
            distance = float(self.move_distance.get())
            speed = float(self.move_speed.get())
            
            self.circles.append({
                'pos': list(pos),
                'inner_radius': inner_r,
                'outer_radius': outer_r,
                'start_pos': list(pos),
                'distance': distance,
                'speed': speed,
                'progress': 0  # 動畫進度 (0-1)
            })
            
        except ValueError as e:
            print(f"請輸入有效的數值: {e}")
    
    def start_animation(self):
        self.is_animating = True
        # 初始化投影螢幕
        self.projection_screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT),
            pygame.FULLSCREEN | pygame.HWSURFACE,
            display=2  # 第三個顯示器
        )
        self.root.iconify()  # 最小化Tkinter視窗
        
    def draw_circle(self, screen, circle):
        # 繪製圓環
        for r in range(int(circle['outer_radius']), int(circle['inner_radius']-1), -1):
            alpha = int(255 * (1 - (r - circle['inner_radius']) / 
                              (circle['outer_radius'] - circle['inner_radius'])))
            surface = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (255, 255, 255, alpha), (r, r), r)
            screen.blit(surface, (circle['pos'][0]-r, circle['pos'][1]-r))
    
    def update_circles(self):
        dt = self.clock.get_time() / 1000.0  # 轉換為秒
        
        for circle in self.circles:
            if circle['progress'] < 1:
                # 更新進度
                movement = circle['speed'] * dt / circle['distance']
                circle['progress'] = min(circle['progress'] + movement, 1)
                
                # 更新位置
                circle['pos'][0] = circle['start_pos'][0] + circle['distance'] * circle['progress']
    
    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.is_animating:
                    self.create_circle(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 清空畫面
            self.edit_screen.fill((0, 0, 0))
            if self.is_animating and self.projection_screen:
                self.projection_screen.fill((0, 0, 0))
            
            # 更新和繪製所有圓形
            if self.is_animating:
                self.update_circles()
            
            for circle in self.circles:
                self.draw_circle(self.edit_screen, circle)
                if self.is_animating and self.projection_screen:
                    self.draw_circle(self.projection_screen, circle)
            
            # 更新顯示
            pygame.display.flip()
            if self.is_animating and self.projection_screen:
                pygame.display.flip()
            
            # 更新Tkinter
            self.root.update()
        
        pygame.quit()
        self.root.destroy()

if __name__ == '__main__':
    animator = CircleAnimator()
    animator.run()'''
    

import pygame
import sys
import tkinter as tk
from tkinter import ttk

class CircleAnimator:
    def __init__(self):
        pygame.init()
        
        # 獲取顯示器信息
        display_info = pygame.display.Info()
        self.WIDTH = 1920  # 可以根據實際螢幕調整
        self.HEIGHT = 1080
        
        # 設置在第二個顯示器上顯示
        self.screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT),
            pygame.FULLSCREEN | pygame.HWSURFACE,
            display=1  # 使用第二個顯示器 (索引從0開始)
        )
        
        self.clock = pygame.time.Clock()
        self.circles = []
        self.is_animating = False
        
        # 創建參數輸入窗口
        self.setup_parameter_window()
        
    def setup_parameter_window(self):
        self.root = tk.Tk()
        self.root.title("圓圈參數設定")
        
        # 設置參數輸入
        ttk.Label(self.root, text="內徑 (像素):").grid(row=0, column=0, padx=5, pady=5)
        self.inner_radius = ttk.Entry(self.root)
        self.inner_radius.grid(row=0, column=1, padx=5, pady=5)
        self.inner_radius.insert(0, "30")
        
        ttk.Label(self.root, text="外徑 (像素):").grid(row=1, column=0, padx=5, pady=5)
        self.outer_radius = ttk.Entry(self.root)
        self.outer_radius.grid(row=1, column=1, padx=5, pady=5)
        self.outer_radius.insert(0, "50")
        
        ttk.Label(self.root, text="移動速度 (像素/秒):").grid(row=2, column=0, padx=5, pady=5)
        self.speed = ttk.Entry(self.root)
        self.speed.grid(row=2, column=1, padx=5, pady=5)
        self.speed.insert(0, "200")
        
        # 添加說明文字
        self.instruction_label = ttk.Label(
            self.root, 
            text="請在黑色畫布上點擊要生成圓圈的位置\n按空白鍵開始動畫\n按ESC退出",
            justify=tk.CENTER
        )
        self.instruction_label.grid(row=3, column=0, columnspan=2, pady=10)
        
    def create_circle(self, pos):
        try:
            inner_r = float(self.inner_radius.get())
            outer_r = float(self.outer_radius.get())
            speed = float(self.speed.get())
            
            self.circles.append({
                'pos': list(pos),
                'start_x': pos[0],
                'y': pos[1],
                'inner_radius': inner_r,
                'outer_radius': outer_r,
                'speed': speed
            })
            
        except ValueError as e:
            print(f"請輸入有效的數值: {e}")
    
    def draw_circles(self):
        for circle in self.circles:
            # 繪製外圈（白色）
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (int(circle['pos'][0]), int(circle['y'])),
                int(circle['outer_radius'])
            )
            # 繪製內圈（黑色）
            pygame.draw.circle(
                self.screen,
                (0, 0, 0),
                (int(circle['pos'][0]), int(circle['y'])),
                int(circle['inner_radius'])
            )
    
    def update_circles(self):
        if self.is_animating:
            dt = self.clock.get_time() / 1000.0  # 轉換為秒
            for circle in self.circles:
                circle['pos'][0] += circle['speed'] * dt
    
    def run(self):
        running = True
        while running:
            self.clock.tick(30)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.is_animating:
                    # 當點擊時創建新圓圈
                    self.create_circle(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and not self.is_animating:
                        # 按空白鍵開始動畫
                        self.is_animating = True
            
            # 更新畫面
            self.screen.fill((0, 0, 0))  # 清空畫面
            self.update_circles()
            self.draw_circles()
            pygame.display.flip()
            
            # 更新Tkinter視窗
            self.root.update()
            
            # 顯示FPS
            fps = self.clock.get_fps()
            pygame.display.set_caption(f'FPS: {fps:.1f}')
        
        pygame.quit()
        self.root.destroy()

if __name__ == '__main__':
    animator = CircleAnimator()
    animator.run()
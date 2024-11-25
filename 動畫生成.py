'''import pygame
import sys
import tkinter as tk
from tkinter import ttk

class CircleAnimator:
    def __init__(self):
        pygame.init()
        
        # 獲取所有顯示器信息
        self.displays = pygame.display.get_num_displays()
        if self.displays < 3:
            print("需要至少三個顯示器！")
            sys.exit(1)
            
        # 獲取顯示器信息
        display_info = pygame.display.Info()
        self.WIDTH = 1800  # 可以根據實際螢幕調整
        self.HEIGHT = 520
        
        # 創建兩個視窗：一個用於選取位置（第二螢幕），一個用於投影（第三螢幕）
        # 選取位置的視窗（第二螢幕）
        self.selection_screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT),
            pygame.FULLSCREEN | pygame.HWSURFACE,
            display=1  # 第二個顯示器
        )
        
        # 投影視窗（第三螢幕）
        # 創建第二個視窗需要使用額外的display surface
        self.projection_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.projection_screen = pygame.display.set_mode(
            (self.WIDTH, self.HEIGHT),
            pygame.FULLSCREEN | pygame.HWSURFACE,
            display=1  # 第三個顯示器
        )
        
        self.clock = pygame.time.Clock()
        self.circles = []
        self.is_animating = False
        
        # 創建參數輸入窗口（在主螢幕）
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
            text="請在第二螢幕的黑色畫布上點擊要生成圓圈的位置\n圓圈將在第三螢幕上投影\n按空白鍵開始動畫\n按ESC退出",
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
    
    def draw_circles(self, screen):
        for circle in self.circles:
            # 繪製外圈（白色）
            pygame.draw.circle(
                screen,
                (255, 255, 255),
                (int(circle['pos'][0]), int(circle['y'])),
                int(circle['outer_radius'])
            )
            # 繪製內圈（黑色）
            pygame.draw.circle(
                screen,
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
            self.clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.is_animating:
                    # 當點擊時在選取視窗創建新圓圈
                    self.create_circle(pygame.mouse.get_pos())
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_KP_ENTER and not self.is_animating:
                        # 按空白鍵開始動畫
                        self.is_animating = True
            
            # 更新選取視窗
            self.selection_screen.fill((0, 0, 0))  # 清空畫面
            if not self.is_animating:
                self.draw_circles(self.selection_screen)  # 在選取視窗顯示靜態圓圈
            pygame.display.update()
            
            # 更新投影視窗
            self.projection_screen.fill((0, 0, 0))  # 清空畫面
            self.update_circles()
            self.draw_circles(self.projection_screen)  # 在投影視窗顯示動畫
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
'''

import pygame
import sys
import os
import tkinter as tk
from tkinter import ttk

class CircleAnimator:
    def __init__(self):
        pygame.init()
        
        # 獲取所有顯示器數量
        self.displays = pygame.display.get_num_displays()
        if self.displays < 3:
            print("需要至少三個顯示器！")
            sys.exit(1)
        
        # 計算總視窗大小（假設所有螢幕的高度相同）
        display_info = pygame.display.Info()
        self.WIDTH = display_info.current_w * 2  # 第二和第三顯示器寬度之和
        self.HEIGHT = display_info.current_h     # 單個螢幕的高度
        
        # 創建單一視窗覆蓋兩個顯示器
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{display_info.current_w},0"  # 從第二顯示器開始
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.NOFRAME)
        
        self.clock = pygame.time.Clock()
        self.circles = []
        self.is_animating = False
        
        # 創建 Tkinter 參數輸入窗口
        self.setup_parameter_window()
        
    def setup_parameter_window(self):
        self.root = tk.Tk()
        self.root.title("圓圈參數設定")
        
        ttk.Label(self.root, text="內徑 (像素):").grid(row=0, column=0, padx=5, pady=5)
        self.inner_radius = ttk.Entry(self.root)
        self.inner_radius.grid(row=0, column=1, padx=5, pady=5)
        self.inner_radius.insert(0, "10")
        
        ttk.Label(self.root, text="外徑 (像素):").grid(row=1, column=0, padx=5, pady=5)
        self.outer_radius = ttk.Entry(self.root)
        self.outer_radius.grid(row=1, column=1, padx=5, pady=5)
        self.outer_radius.insert(0, "15")
        
        ttk.Label(self.root, text="移動速度 (像素/秒):").grid(row=2, column=0, padx=5, pady=5)
        self.speed = ttk.Entry(self.root)
        self.speed.grid(row=2, column=1, padx=5, pady=5)
        self.speed.insert(0, "60")
        
        self.instruction_label = ttk.Label(
            self.root, 
            text="請在第二螢幕的黑色畫布上點擊要生成圓圈的位置\n圓圈將在第三螢幕上投影\n按空白鍵開始動畫\n按 ESC 退出",
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
    
    def draw_circles(self, surface):
        for circle in self.circles:
            pygame.draw.circle(surface, (255, 255, 255), (int(circle['pos'][0]), int(circle['y'])), int(circle['outer_radius']))
            pygame.draw.circle(surface, (0, 0, 0), (int(circle['pos'][0]), int(circle['y'])), int(circle['inner_radius']))
    
    def update_circles(self):
        if self.is_animating:
            dt = self.clock.get_time() / 1000.0  # 轉換為秒
            for circle in self.circles:
                circle['pos'][0] += circle['speed'] * dt
    
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
                    elif event.key == pygame.K_SPACE:
                        self.is_animating = True
            
            # 分割畫面為兩個區域
            selection_surface = self.screen.subsurface((0, 0, self.WIDTH // 2, self.HEIGHT))
            projection_surface = self.screen.subsurface((self.WIDTH // 2, 0, self.WIDTH // 2, self.HEIGHT))
            
            # 清空背景並繪製選取視窗
            selection_surface.fill((0, 0, 0))
            self.draw_circles(selection_surface)
            
            # 清空背景並繪製投影視窗
            projection_surface.fill((0, 0, 0))
            self.update_circles()
            self.draw_circles(projection_surface)
            
            pygame.display.flip()
            self.root.update()
        
        pygame.quit()
        self.root.destroy()

if __name__ == '__main__':
    animator = CircleAnimator()
    animator.run()

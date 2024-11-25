
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
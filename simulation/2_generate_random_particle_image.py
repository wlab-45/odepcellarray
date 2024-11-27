import numpy as np
import cv2
import os
import random
from tkinter import Tk
from tkinter.filedialog import askdirectory

def create_canvas_and_draw_circles(output_folder, radius, file_i, size=150):
    # 創建一個黑色畫布
    canvas_width = 1814
    canvas_height = 1220
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    # 設定輸出文本檔案的資料夾路徑
    output_txt_folder = 'C:/Users/Vivo/CGU/odep_cellarray/Cell_Manipulation_Simulation/virtual_particle_points_txt'
    
    # 隨機生成 20 個點座標
    points = []
    for _ in range(30):
        x = random.randint(0, canvas_width - 1)
        y = random.randint(0, canvas_height - 1)
        points.append((x, y))
    
    # 打印生成的點座標
    print("生成的點座標:")
    for point in points:
        print(point)

    # 在畫布上畫紅色圓形
    for point in points:
        cv2.circle(canvas, point, radius, (0, 0, 255), -1)  # 使用紅色 (BGR)

    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 輸出結果至指定資料夾
    output_path = os.path.join(output_folder, f'{file_i}.png')
    cv2.imwrite(output_path, canvas)
    print(f"結果已輸出至: {output_path}")


    
    # 將座標寫入文本檔案
    output_txt_path = os.path.join(output_txt_folder, f'{file_i}.txt')
    
    try:
        with open(output_txt_path, 'w') as f:
            for point in points:
                f.write(f"{point}\n")
        print(f"座標已輸出至: {output_txt_path}")
    except Exception as e:
        print(f"寫入文件時發生錯誤: {e}")

# 使用範例
Tk().withdraw()
    
# 打開文件對話框以選擇資料夾
output_folder = askdirectory(title="選擇放置虛擬資料圖片的資料夾")
rad = input("enter the radius of virtual particles: ")
filenumber = input("enter the number of virtual image that you want to generate: ")
radius = float(rad)
radius = round(radius)
filenumber = int(filenumber)

# 創建指定數量的虛擬圖片
for file_i in range(filenumber):
    create_canvas_and_draw_circles(output_folder, radius, file_i, size=150)

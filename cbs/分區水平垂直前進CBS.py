import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog,simpledialog
import os
import math, random
import heapq
import copy
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import math
import time
import heapq
import math
from collections import namedtuple, defaultdict
import multiprocessing as mp
import time
from functools import partial
#from cbs import cbs_planning
from ORCA_RVO2 import orca_planning


def create_canvas_and_draw_circles(output_folder, radius, file_i, size=150):
    # 創建一個黑色畫布
    canvas_width = 1814
    canvas_height = 1220
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    # 設定輸出文本檔案的資料夾路徑
    output_txt_folder = 'C:/Users/Vivo/CGU/odep_cellarray/Cell_Manipulation_Simulation/virtual_particle_points_txt'
    
    # 隨機生成 20 個點座標
    points = []
    while len(points) < 65:
        x = random.randint(0, canvas_width - 5)
        y = random.randint(0, canvas_height - 5)
        if all(np.sqrt((x - px)**2 + (y - py)**2) >= 30 for px, py in points):
            points.append((x, y))

    # 在畫布上畫紅色圓形
    for point in points:
        cv2.circle(canvas, point, radius, (0, 0, 255), -1)  # 使用紅色 (BGR)

    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 輸出結果至指定資料夾
    output_path = os.path.join(output_folder, f'{file_i}.png')
    cv2.imwrite(output_path, canvas)
    #print(f"結果已輸出至: {output_path}")


    
    # 將座標寫入文本檔案
    output_txt_path = os.path.join(output_txt_folder, f'{file_i}.txt')
    
    try:
        with open(output_txt_path, 'w') as f:
            for point in points:
                f.write(f"{point}\n")
        #print(f"座標已輸出至: {output_txt_path}")
    except Exception as e:
        print(f"寫入文件時發生錯誤: {e}")

#step 3
def select_png_file():
    # 隱藏主視窗
    root = tk.Tk()
    root.withdraw()

    # 使用 askopenfilename 選擇 PNG 檔案
    file_path = filedialog.askopenfilename(
        title='選擇 原始PNG 檔案 (rawimage)',
        filetypes=[('PNG Files', '*.png')]
    )
    
    if not file_path:
        raise ValueError("路徑無效")

    # 打印選擇的檔案路徑
    if file_path:
        print(f'選擇的檔案: {file_path}')
        return file_path
    else:
        print('未選擇任何檔案')
        
def generate_array(image, size, length, width):
    for i in range(int(length)):
        for j in range(int(width)):
            top_left = (i * size, j * size)  # 左上角座標
            bottom_right = (top_left[0] + size, top_left[1] + size)  # 右下角座標
            # 繪製虛線正方形
            cv2.rectangle(image, top_left, bottom_right, (255, 191, 0), 2)
    return image 

def wholestep3_draw_array_picture():
    file_path=select_png_file()
    image=cv2.imread(file_path)
    file_name = os.path.basename(file_path)
    #set size of each square arraysize
    root = tk.Tk()
    root.withdraw()
    size = simpledialog.askinteger("Input", "Enter the length of each square in the array:")
    size=int(size)
    length = 9 #simpledialog.askinteger("Input", "Enter the number of columns in the array:")
    width = 5 #simpledialog.askinteger("Input", "Enter the number of rows in the array:")
    length = int(length)
    width = int(width)
    target_numbers = length * width
    arrayimage =generate_array(image , size, length, width)

    cv2.imshow('Array of Squares', arrayimage )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_arrayimage_directory="C:/Users/Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\array_image"
    array_image_path = os.path.join(save_arrayimage_directory, f'arrayimage_{file_name}')
    cv2.imwrite(array_image_path, arrayimage)
    print("array success")
    return size, target_numbers, arrayimage, file_name, length, width

#step 5
def read_coordinates_from_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                x, y = map(int, line.strip('()').split(','))
                coordinates.append((x, y))
    return coordinates

def form_sort_list(arrayimage, coordinates_file_path, size, target_numbers, file_name):
    all_coordinates =  read_coordinates_from_file(coordinates_file_path) # 初始化空列表以存儲座標
    
    all_sorted_coordinates = sorted(all_coordinates, key=lambda point: math.sqrt(point[0]**2 + point[1]**2))
    
    obsticle_coordinates = []
    # 繪製障礙物，確保不與紅色圓點重疊
    obsticle_image = arrayimage.copy()
    radius = 15
    for i in range(5):
        while True:
            x, y = np.random.randint(size*int(math.sqrt(target_numbers)), 1814), np.random.randint(size*int(math.sqrt(target_numbers)), 1220)
            if all(math.sqrt((x - cx)**2 + (y - cy)**2) > 4 * radius for cx, cy in all_coordinates):
                cv2.circle(obsticle_image, (x, y), radius, (255, 0, 0), -1)  # 繪製圓形
                obsticle_coordinates.append((x, y))
                break
    save_arrayimage_directory="C:/Users/Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\array_image"
    array_image_path = os.path.join(save_arrayimage_directory, f'arrayimage_{file_name}')
    cv2.imwrite(array_image_path, obsticle_image)

    return all_sorted_coordinates, obsticle_image, obsticle_coordinates

def calculate_distance(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

# 無優先版本
def picking_target(all_coordinate, size, columns, rows):
    target_coordinate = []
    target_numbers = columns * rows
    # 過濾不符合條件的點 (陣列正右側區)
    
    sorted_available_coordinate = sorted(all_coordinate, key=lambda point: math.sqrt(point[0]**2 + point[1]**2))  # 依距離排序

    # 確保目標數量足夠
    if len(sorted_available_coordinate) >= target_numbers:
        target_coordinate = sorted_available_coordinate[:target_numbers]
    else:
        print(f"可移動粒子數量 {len(sorted_available_coordinate)} 不足，需要至少 {target_numbers} 個")

    return target_coordinate

def draw_light_image(arrayimage, target_coordinate):
    light_image=arrayimage.copy()
    good=len(target_coordinate)
    for i in range(good):
        cv2.circle(light_image , target_coordinate[i] , int(2*7+5),(250,250,255), 10) 
    return light_image    
        
def wholestep5_draw_light_image(arrayimage, target_numbers,size, file_name, columns, rows):
    #generate 4 list
    all_coordinate=[]
    target_coordinate=[]
    obstacle_coordinate=[]
    ##set coordinates file path
    # 隱藏主視窗
    txt_folder = 'C:/Users/Vivo/CGU/odep_cellarray/Cell_Manipulation_Simulation/virtual_particle_points_txt'
    file_name_without_ext = os.path.splitext(file_name)[0]
    coordinates_file_path = os.path.join(txt_folder, f'{file_name_without_ext}.txt')
    #generating all_cooridinate
    all_coordinate, obsticle_image, obstacle_coordinate = form_sort_list(arrayimage, coordinates_file_path, size, target_numbers, file_name)
    print(f'length of all_coordinate list={len(all_coordinate)}')
    
    #picking target
    target_coordinate =picking_target(all_coordinate, size, columns, rows)
    print(f'length of target_coordinate list={len(target_coordinate)}')
    
    #驗證用
    light_image=draw_light_image(obsticle_image,target_coordinate)
    scale_percent = 70  # 縮放比例
    width = int(light_image.shape[1] * scale_percent / 100)
    height = int(light_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_light_image= cv2.resize(light_image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('step5: processed image', resized_light_image )
    light_image_directory="C:/Users/Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\lightimg_with_particle"
    light_image_path = os.path.join(light_image_directory, f'light_image_{file_name}')
    cv2.imwrite(light_image_path, light_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return target_coordinate, obstacle_coordinate, light_image, all_coordinate

##step 6 
# 以rows為子列表的方式組織array
def structure_and_center_of_array (columns, rows, size):
    array_layers = [[] for _ in range(rows)]
    for i in range(rows):  #假設每層3格(3*3矩陣)則標號為layer(row) 0,1,2 每層格編號(左至右)
        for j in range(1, columns*2, 2):
            k=i+1
            array_layers[i].append((size *j// 2, size *(2*k-1)//2))
    return array_layers

# 分配座標

# version2
def assignment(target_coordinate, size, columns, rows):
    matched_target_and_array= [[] for _ in range (rows)] 
    #儲存左上、右上、左下、右下陣列中心點
    array = structure_and_center_of_array(columns, rows, size)
    target_coordinate_batches = [[] for _ in range (rows)] 
    
    for start in range(rows):
        if len(target_coordinate) < columns:
            print(f"剩餘 target_coordinate ({len(target_coordinate)}) 少於預期的 {columns} 個")
            break  
        
        #先依y座標排序，再分batch
        target_coordinate = sorted(target_coordinate, key= lambda p: p[1])# 排序y座標
        target_coordinate_batch = target_coordinate[:columns]
        target_coordinate = target_coordinate[columns:]  # 扣除已經排序的點
        target_coordinate_batches[start]= sorted(target_coordinate_batch, key= lambda p: p[0])# 排序x座標

        #print(f" 點區域  第 {start + 1} 批的點數: {len(target_coordinate_batches[start])}")
        for j in range(columns):
            matched_target_and_array[start].append((target_coordinate_batches[start][j], array[start][j]))
    print(f'length of matched_target_and_array list={len(matched_target_and_array)}')
    return matched_target_and_array

 

def straight_path(start_point, target_center, step_size):
    path = []
    dx = target_center[0] - start_point[0]
    dy = target_center[1] - start_point[1]
    all_length = math.sqrt(dx**2 + dy**2)
    step_count = int(all_length / step_size)
    
    if all_length == 0:  # 如果起點終點一樣，直接連到終點
        path.append(start_point)  
    elif step_count == 0:  # 如果距離太短，直接連到終點
        path.extend([start_point, target_center])
    else:
        step_x = dx / step_count 
        step_y = dy / step_count  
        path.append(start_point)
        
        for step_idx in range(1, step_count + 1): 
            next_point = (
                round(start_point[0] + step_x * step_idx),
                round(start_point[1] + step_y * step_idx)
            )
            path.append(next_point)
        
        if path[-1] != target_center:
            path.append(target_center)
    
    return path


def path_for_batch(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, size, image_width, image_height, step_size, Rl, obstacle_radius=15):
    start_time = time.time()
    grid_size = size
    final_paths = orca_planning(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, grid_size, image_width, image_height, step_size, Rl, obstacle_radius)
    if final_paths is None:
        print("無法找到路徑，請檢查參數或障礙物配置。")
        raise ValueError("無法找到路徑")  
    end_time = time.time()
    print(f"\n⏱️ 總耗時: {end_time - start_time:.2f} 秒")
    return final_paths

# for a*
def convert_to_grid_coordinates(image_width, image_height, obstacle_coordinates, grid_size, obstacle_radius=15):
    # 計算網格的行數和列數
    num_rows = (image_height + grid_size - 1) // grid_size
    num_cols = (image_width + grid_size - 1) // grid_size

    # 建立網格，可通行的默認為 True
    walkable_grid = [[True for _ in range(num_cols)] for _ in range(num_rows)]

    # 檢查每個障礙物
    for ox, oy in obstacle_coordinates:
        # 計算障礙物所影響的網格範圍
        start_grid_x = max(0, (ox - obstacle_radius) // grid_size)
        end_grid_x = min(num_cols - 1, (ox + obstacle_radius) // grid_size)
        start_grid_y = max(0, (oy - obstacle_radius) // grid_size)
        end_grid_y = min(num_rows - 1, (oy + obstacle_radius) // grid_size)

        # 設定受影響的網格為不可通行
        for grid_y in range(start_grid_y, end_grid_y + 1):
            for grid_x in range(start_grid_x, end_grid_x + 1):
            
                # 計算網格的中心坐標
                grid_center_x = grid_x * grid_size + grid_size // 2
                grid_center_y = grid_y * grid_size + grid_size // 2

                # 計算障礙物中心到網格中心的曼哈頓距離
                dx = abs(grid_center_x - ox)
                dy = abs(grid_center_y - oy)

                # 若障礙物到網格中心的距離小於網格一半加上障礙物半徑，設置為不可通行
                if dx <= grid_size // 2 + obstacle_radius or dy <= grid_size // 2 + obstacle_radius:
                    walkable_grid[grid_y][grid_x] = False
    return walkable_grid


# 網格轉step_size
def interpolate_path(start, end, step_size):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    all_length = math.sqrt(dx**2 + dy**2)
    
    if all_length == 0:  # 如果起點終點一樣
        return [start]
        
    step_count = int(all_length / step_size)
    if step_count == 0:  # 如果距離太短
        return [start, end]
    
    interpolated_points = [start]
    step_x = dx / step_count
    step_y = dy / step_count
    
    for step_idx in range(1, step_count + 1):
        next_point = (
            round(start[0] + step_x * step_idx),
            round(start[1] + step_y * step_idx)
        )
        interpolated_points.append(next_point)
        
    if interpolated_points[-1] != end:
        interpolated_points.append(end)
        
    return interpolated_points

def convert_to_pixel_coordinates(particle, goal, path, grid_size, step_size):
    if path:
        start_point = particle
        actual_path = []
        trimmed_path = path[:-1]  # 不包含最後一點
        
        # 處理起點到第一個網格點的路徑
        if len(trimmed_path) > 1:
            # **直接從起點插值到第二個網格的中心點**(避免路徑返回問題)
            first_grid_point = (trimmed_path[1][0] * grid_size + grid_size//2, trimmed_path[1][1] * grid_size + grid_size//2)
            initial_segment = interpolate_path(start_point, first_grid_point, step_size)[:-1]  # 不包含终点，避免重复
            actual_path.extend(initial_segment)
            start_index = 1  # 確保後續從 path[1] 開始
                        
            # **處理中間路徑點**
            for grid_index in range(start_index, len(trimmed_path)-1):  # **從 start_index 開始，避免重複 trimmed_path[0]**
                current = (trimmed_path[grid_index][0] * grid_size + grid_size//2, trimmed_path[grid_index][1] * grid_size + grid_size//2)
                next_point = (trimmed_path[grid_index+1][0] * grid_size + grid_size//2, trimmed_path[grid_index+1][1] * grid_size + grid_size//2)
                interpolated = interpolate_path(current, next_point, step_size)[:-1]  
                actual_path.extend(interpolated)
            
            # 處理最後一段到目標點的路徑
            last_grid_point = (trimmed_path[-1][0] * grid_size + grid_size//2, trimmed_path[-1][1] * grid_size + grid_size//2)
            final_segment = interpolate_path(last_grid_point, goal, step_size)
            actual_path.extend(final_segment)
            
        else:
            # **如果 trimmed_path 只有一個點，則只能插值到這個點** 
            actual_path = straight_path(start_point, goal, step_size)
        return actual_path
    else:
        print(f"無法從粒子 {particle} 到達目標 {goal}")
        return []
    
def get_particle_occupied_grids(image_width, image_height,center_pos, grid_size, Rl = 15): 
    occupied = []
    cx, cy = center_pos
    save_radius = Rl +10 +2 # 10 為光圈寬度 2 為保險距離
    # 計算網格的行數和列數
    num_rows = (image_height + grid_size - 1) // grid_size
    num_cols = (image_width + grid_size - 1) // grid_size
    
    start_grid_x = max(0, (cx - save_radius) // grid_size)
    end_grid_x = min(num_cols, (cx + save_radius) // grid_size)
    start_grid_y = max(0, (cy - save_radius) // grid_size)
    end_grid_y = min(num_rows, (cy + save_radius) // grid_size)
    
    for grid_x in range( start_grid_x, end_grid_x + 1):
        for grid_y in range(start_grid_y, end_grid_y + 1):
            grid_center_x = grid_x * grid_size + grid_size // 2
            grid_center_y = grid_y * grid_size + grid_size // 2
            dx = abs(grid_center_x - cx)
            dy = abs(grid_center_y - cy)

            # 檢查是否在圓內
            if dx <= grid_size // 2 + save_radius or dy <= grid_size // 2 + save_radius:
                occupied.append((grid_x, grid_y))
    return set(occupied) #返回該粒子所佔的網格 (數)

# 路徑尋找
def draw_and_get_paths(image, whole_path_batch_astar, obstacle_coordinate_changed_bybatch, batch_size, size):
    whole_paths_batch = [[] for _ in range(batch_size)]
    no_found = 0
    
    # 取得影像尺寸
    image_height, image_width = image.shape[:2]
    
    # 產生網格
    grid_size = size
    walkable_grid = convert_to_grid_coordinates(image_width, image_height, obstacle_coordinate_changed_bybatch, grid_size, obstacle_radius=15)
    
    # 繪製網格
    for y in range(0, image_height, grid_size):
        cv2.line(image, (0, y), (image_width, y), (200, 200, 200), 1)  # 橫線
    for x in range(0, image_width, grid_size):
        cv2.line(image, (x, 0), (x, image_height), (200, 200, 200), 1)  # 縱線

    # 繪製障礙物（不可通行區域）
    for grid_y in range(len(walkable_grid)):
        for grid_x in range(len(walkable_grid[0])):
            if not walkable_grid[grid_y][grid_x]:  # 若為障礙物
                top_left = (grid_x * grid_size, grid_y * grid_size)
                bottom_right = ((grid_x + 1) * grid_size, (grid_y + 1) * grid_size)
                cv2.rectangle(image, top_left, bottom_right, (255, 255,0), -1)  # 紅色填充不可通行區域

    # 繪製所有路徑
    for path in whole_path_batch_astar:
        if path:
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            for j in range(len(path) - 1):
                cv2.line(image, path[j], path[j + 1], color, 2)

    return image

# 主函數
def whole_step_6_draw_path(batch_size, copyimage, size, obstacle_coordinate_changed_btbatch, file_name, whole_path_batch_astar, step_size):

    ###設定光圖形最段距離移動步距， 與移動rate有關
    #root = tk.Tk()
    #root.withdraw()
    step_size = 3 #simpledialog.askinteger("Input", "設定光圖形一幀移動的距離， 與光圖形移動rate有關 (integer):")
    step_size =int(step_size)
    
    # 繪製移動路徑                                                           
    image_with_paths = draw_and_get_paths(light_image, whole_path_batch_astar, obstacle_coordinate_changed_btbatch, batch_size, size)
    scale_percent = 50  # 缩放比例
    width = int(image_with_paths.shape[1] * scale_percent / 100)
    height = int(image_with_paths.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_image_with_path= cv2.resize(image_with_paths, dim, interpolation=cv2.INTER_AREA)
    # 顯示影像
    cv2.imshow("Result of path", resized_image_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    path_save_directory="C:/Users/Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\path_image"
    path_image_path = os.path.join(path_save_directory, f'path_{file_name}')
    cv2.imwrite(path_image_path, image_with_paths)   



#避免碰撞發生
#檢查現有路徑是否會有碰撞發生
## step 7s
# 找到光圈圓心到圓點的單位向量
# 這個函數可以保留，用於計算拉動方向
def find_unit_vector(light_coor, particle_coor):
    # 計算分量差值
    dx = light_coor[0] - particle_coor[0]
    dy = light_coor[1] - particle_coor[1]
    # 檢查dx和dy是否都為0
    if dx == 0 and dy == 0:
        return (0, 0)
    # 計算單位向量
    distance_two_points = math.sqrt(dx**2 + dy**2)
    unit_vector = (dx/distance_two_points, dy/distance_two_points)
    return unit_vector

# 模擬移動 (修改版 - 光圈引導粒子拉動效果)
def simulate_movement(canvas, step_size, whole_paths, all_particle_coor, target_numbers, Rl, Rp, obstacle_coordinate, file_name, matched_target_and_array):  

    # 靜止的粒子座標從 all_particle_coor 的 target_numbers 索引開始
    static_particles_coords = all_particle_coor[target_numbers:]

    max_path_length = 0
    if whole_paths: # 確保 whole_paths 不是空的列表
        max_path_length = max(len(path) for path in whole_paths)
    else:
        print("[警告] whole_paths 為空，無法進行模擬")
        return # 如果沒有路徑，提前結束

    # 這一部分邏輯保留，因為影片時長由最長路徑決定
    for k in range(len(whole_paths)):
        while len(whole_paths[k]) < max_path_length:
            if whole_paths[k]:  # 確保原始路徑非空，可以複製最後一點
                whole_paths[k].append(whole_paths[k][-1])
            else:
                print(f"[警告] 第{k}個 whole_paths 在填充時為空。")
                pass # 不填充空路徑，依賴後續繪製檢查 path 是否非空

    # 設置影片輸出區
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 請確認或修改這裡的輸出路徑
    outputpath = f"C:/Users/Vivo/odepcellarray_fromlab/cbs/movement_simulation_{file_name}_rvo版.mp4"
    out = cv2.VideoWriter(outputpath, fourcc, 30.0, (canvas.shape[1], canvas.shape[0]))

    if not out.isOpened():
        print(f"Error: Could not open video writer for path {outputpath}")
        return

    # 初始化移動粒子的當前位置列表
    # all_particle_coor 包含了所有粒子的初始位置。
    # 移動粒子是前 target_numbers 個。
    # 複製這些位置，以便在模擬中更新。
    moving_particles_current_pos = [list(particle) for batch in matched_target_and_array for (particle, array) in batch]


    print(f"Starting simulation visualization for {len(whole_paths)} moving light circles over {max_path_length} frames.")

    # 模擬循環，遍歷每個影格
    for step in range(max_path_length):
        display_img = canvas.copy() #空白畫布

        # 繪製靜態障礙物 (保持不變)
        for obstacle in obstacle_coordinate:
            cv2.circle(display_img, obstacle, 15, (255, 0, 0), -1) # 藍色障礙物

        # 繪製靜止的粒子 (非目標粒子 保持不變)
        for circle in static_particles_coords:
            cv2.circle(display_img, circle, radius=Rp, color=(0, 0, 250), thickness=-1) # 靜止粒子 (紅)

        # 繪製移動的光圈和他們拉動的粒子
        # 遍歷 whole_paths (每個 path 對應一個移動的光圈)
        for i, path in enumerate(whole_paths):
            # 檢查路徑是否為空，以及當前時間步長是否在路徑範圍內
            if not path or step >= len(path):
                
                if path: 
                    light_coor = path[-1]
                else:
                    continue # 跳過這個粒子/光圈的繪製和更新
                particle_coor = moving_particles_current_pos[i] # 取出粒子當前位置（應該是最後位置）

            else: # 如果在路徑範圍內，光圈移動到路徑上的下一個點
                light_coor = path[step] # 光圈的目標位置就是路徑上的當前點
                particle_coor = moving_particles_current_pos[i] # 取出粒子當前位置

                # --- 粒子被光圈拉動的邏輯 ---
                dx = light_coor[0] - particle_coor[0]
                dy = light_coor[1] - particle_coor[1]
                dist_to_light = math.sqrt(dx**2 + dy**2)

                if dist_to_light > 0:
                    # 計算單位向量 (拉動方向)
                    unit_vector = (dx / dist_to_light, dy / dist_to_light)

                    # 計算這個影格粒子應該移動的距離
                    # 粒子朝光圈移動，但最大移動距離限制在 step_size
                    move_distance = min(dist_to_light, step_size) # 拉力強度/粒子最大速度限制

                    # 更新粒子位置 (使用浮點數計算，最後繪圖時取整)
                    particle_coor[0] += unit_vector[0] * move_distance
                    particle_coor[1] += unit_vector[1] * move_distance

                # --- 確保粒子在光圈影響範圍內 (Rl-Rp) ---
                # 在粒子移動 *之後*，檢查它是否超出了光圈的有效拉動範圍
                dx_after_move = particle_coor[0] - light_coor[0]
                dy_after_move = particle_coor[1] - light_coor[1]
                dist_after_move = math.sqrt(dx_after_move**2 + dy_after_move**2)

                # Rl-Rp 是粒子中心到光圈中心的最大距離，再遠就拉不動了
                max_pull_distance_from_light_center = Rl - Rp # 使用 Rl 和 Rp

                if dist_after_move > max_pull_distance_from_light_center:
                    # 如果粒子超出了有效範圍，將它拉回到邊界上
                    if dist_after_move > 1e-6: # 避免除以零
                         scale = max_pull_distance_from_light_center / dist_after_move
                         # 將粒子位置拉回到以 light_coor 為中心，max_pull_distance_from_light_center 為半徑的圓周上
                         particle_coor[0] = light_coor[0] + (particle_coor[0] - light_coor[0]) * scale
                         particle_coor[1] = light_coor[1] + (particle_coor[1] - light_coor[1]) * scale
                    # else: # 如果距離非常小但仍然 > 0 且 > max_pull_distance，也拉回
                        # 這部分邏輯通常涵蓋在上面的 if dist_after_move > 1e-6 中

                # 更新粒子在列表中的位置 (儲存浮點數位置)
                moving_particles_current_pos[i] = particle_coor.copy() # 確保是複製，不是引用

            # 繪製光圈 (在路徑點上)
            cv2.circle(display_img, (int(round(light_coor[0])), int(round(light_coor[1]))), radius=Rl, color=(250, 250, 250), thickness=10) # 白色光圈

            # 繪製粒子 (在更新後的位置上)
            cv2.circle(display_img, (int(round(particle_coor[0])), int(round(particle_coor[1]))), Rp, (0, 0, 250), -1) # 移動粒子 (紅色)


        # 縮放並顯示影格 (保持不變)
        scale_percent = 50
        width = int(display_img.shape[1] * scale_percent / 100)
        height = int(display_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(display_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('Movement Simulation', resized_image)
        out.write(display_img)

        # 檢查用戶中斷 (保持不變)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Simulation interrupted by user.")
            break

    # 釋放影片寫入器並關閉視窗 (保持不變)
    out.release()
    cv2.destroyAllWindows()
    print(f"Simulation video saved to {outputpath}")


# step7 主函數    
def whole_step7_simulate_moving(size, target_numbers, whole_paths, all_sorted_coordinate, step_size, Rl, Rp, obstacle_coordinate, file_name, columns, rows, matched_target_and_array):
    canvas = np.zeros((1220, 1814, 3), dtype=np.uint8)
    generate_array(canvas, size, columns, rows)
    simulate_movement(canvas, step_size, whole_paths, all_sorted_coordinate, target_numbers, Rl, Rp, obstacle_coordinate, file_name, matched_target_and_array)
    return

# version2
if __name__ == '__main__':
    obstacle_coordinate = []
    target_coordinate = []
    Rl, Rp = 15, 9  # 光圈半徑和粒子半徑
    output_folder = 'C:\\Users\\Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\raw_image'
    folder = 'C:\\Users\\Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\raw_image'
    image_width = 1814
    image_height = 1220
    step_size = 3
    
    size, target_numbers, arrayimage, file_name, columns, rows = wholestep3_draw_array_picture()
    target_coordinate, obstacle_coordinate, light_image, all_sorted_coordinate = wholestep5_draw_light_image(arrayimage, target_numbers, size, file_name, columns, rows)

    # 分區(L型)分配座標
    whole_paths = [[] for _ in range(len(target_coordinate))]
    sum_path_length = 0
    sum_path_counts = 0
    batch_size = 0
    matched_target_and_array = assignment(target_coordinate, size, columns, rows)
    for start in range(rows): # rows數= batch數
        obstacle_coordinate_changed_btbatch = obstacle_coordinate.copy()
        copyimage = light_image.copy()
        matched_target_and_array_batch = matched_target_and_array[start]  # 取出對應的批次
        batch_size = columns
        
        # #更新障礙物列表(非批次的粒子)
        before_start = matched_target_and_array[:start]  # start 之前的部分
        after_start = matched_target_and_array[start + 1:]  # start 之後的部分
        obstacle_coordinate_changed_btbatch.extend([destination for batch in before_start for _, destination in batch])
        obstacle_coordinate_changed_btbatch.extend([start_point for batch in after_start for start_point, _ in batch])

        whole_path_batch_astar = path_for_batch(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, size, image_width, image_height, step_size, Rl, obstacle_radius= 15)
        whole_step_6_draw_path(batch_size, copyimage, size, obstacle_coordinate_changed_btbatch, file_name, whole_path_batch_astar, step_size)
        

        max_path_length = max(len(path) for path in whole_path_batch_astar)  # 找最長的路徑

        for k in range(len(whole_path_batch_astar)):
            while len(whole_path_batch_astar[k]) < max_path_length:
                whole_path_batch_astar[k].append(whole_path_batch_astar[k][-1])  # 最後一點填充(表一直在陣列終點)
            for m in range( int(1 * sum_path_length)):   #可以設定每批開始移動的間隔時間
                whole_path_batch_astar[k].insert(0, whole_path_batch_astar[k][0])  # 填充第一個點來分批移動
            
        sum_path_length += int( max_path_length)

        # 將批次的路徑添加到 whole_paths
        for l, path in enumerate(whole_path_batch_astar):
            index =  sum_path_counts + l  
            if index < len(whole_paths):
                whole_paths[index].extend(path)
            else:
                print(f"Index out of range: {index}")

        sum_path_counts += batch_size

    for i in range(len(whole_paths)):
        if len(whole_paths[i]) == 0:
            print(f"第 {i} 批次路徑為空，無法進行模擬")
            continue
    whole_step7_simulate_moving(size, target_numbers, whole_paths, all_sorted_coordinate, step_size, Rl, Rp, obstacle_coordinate, file_name, columns, rows, matched_target_and_array)

    
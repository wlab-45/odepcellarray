import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog,simpledialog
import os
import math, random
import heapq
import csv
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random


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
    
    # 打印生成的點座標
    #print("生成的點座標:")
    #for point in points:
    #    print(point)

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
    width = 6 #simpledialog.askinteger("Input", "Enter the number of rows in the array:")
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

def picking_target(all_coordinate, size, columns, rows):
    target_coordinate=[]
    circle_in_array=[]
    circle_in_array_not_target=[]
    if len(all_coordinate) >= target_numbers:
        #偵測array中的點數目分為兩種情況
        for (x, y) in all_coordinate:
            if x**2 < int(columns * size) and y**2 < int(rows * size):
                circle_in_array.append((x, y))

        if len(circle_in_array) <= target_numbers:  
            target_coordinate = all_coordinate[:target_numbers]
        else:
            print("circle_in_array > target, we'll move the rest circle out")
            target_coordinate = all_coordinate[:target_numbers]
            circle_in_array_not_target = circle_in_array[target_numbers:]
        update_all_coordinate = all_coordinate[target_numbers:] 
    else:
        print("all_coordinate list length error")
    return target_coordinate, circle_in_array_not_target, update_all_coordinate

def draw_light_image(arrayimage, target_coordinate):
    light_image=arrayimage.copy()
    good=len(target_coordinate)
    for i in range(good):
        cv2.circle(light_image , target_coordinate[i] , int(2*7+5),(250,250,255), 10) 
    return light_image    
        
def wholestep5_draw_light_image(arrayimage, target_numbers,size, file_name, columns, rows):
    #generate 4 list
    all_coordinate=[]
    upadate_all_coordinate=[]
    target_coordinate=[]
    obstacle_coordinate=[]
    circle_in_array_not_traget=[]
    ##set coordinates file path
    # 隱藏主視窗
    txt_folder = 'C:/Users/Vivo/CGU/odep_cellarray/Cell_Manipulation_Simulation/virtual_particle_points_txt'
    file_name_without_ext = os.path.splitext(file_name)[0]
    coordinates_file_path = os.path.join(txt_folder, f'{file_name_without_ext}.txt')
    #generating all_cooridinate
    all_coordinate, obsticle_image, obstacle_coordinate = form_sort_list(arrayimage, coordinates_file_path, size, target_numbers, file_name)
    print(f'length of all_coordinate list={len(all_coordinate)}')
    
    #picking target
    target_coordinate , circle_in_array_not_traget, upadate_all_coordinate=picking_target(all_coordinate, size, columns, rows)
    print(f'length of target_coordinate list={len(target_coordinate)}')
    
    #驗證用
    light_image=draw_light_image(obsticle_image,target_coordinate)
    scale_percent = 70  # 缩放比例
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
    return target_coordinate, obstacle_coordinate, circle_in_array_not_traget, light_image ,upadate_all_coordinate, all_coordinate

##step 6 
def structure_and_center_of_array(columns, rows, size):
    array_arrangeby_batches = [[] for _ in range(min(rows, columns))]
    
    for start in range(min(rows, columns)):
        k = start * 2 + 1  # 讓每個 start 產生不同的 k
        # 提取 L 型區域的行
        for j in range(k, columns * 2, 2):
            array_arrangeby_batches[start].append((size * j // 2, size * k // 2))

        # 提取 L 型區域的列
        for i in range(k + 2, rows * 2, 2):
            array_arrangeby_batches[start].append((size * k // 2, size * i // 2))
        #print(f"陣列區第{start + 1}批長度:{len(array_arrangeby_batches[start])}")
    #print(f"陣列區總長度(批次數):{len(array_arrangeby_batches)}")
    return array_arrangeby_batches

# 分配座標
# version2
def assignment(target_coordinate, size, columns, rows):
    matched_target_and_array=[[] for _ in range (min(rows, columns))]
    target_numbers= len(target_coordinate)
    #儲存左上、右上、左下、右下陣列中心點
    array = structure_and_center_of_array(columns, rows, size)
    points_arrangeby_batches = [[] for _ in range (min(rows, columns))]

    for start in range(min(rows, columns)):
        batch_size = (columns - start) + (rows - start) - 1
        if len(target_coordinate) < batch_size:
            print(f"剩餘 target_coordinate ({len(target_coordinate)}) 少於預期的 {batch_size}")
            break  
        target_coordinate_batches = target_coordinate[:batch_size]
        target_coordinate = target_coordinate[batch_size:]  # 扣除已經排序的點
        points_sortedby_y = sorted(target_coordinate_batches, key=lambda p: p[1])# 排序y座標
        # 依據分段，先取得橫列的點，再取得直行的點
        points_in_row_of_batch = points_sortedby_y[:columns - start]
        points_in_column_of_batch = points_sortedby_y[columns - start:]

        # 重新排序，確保橫列按 x 排序，直行按 y 排序
        points_in_row_of_batch.sort(key=lambda p: p[0])
        points_in_column_of_batch.sort(key=lambda p: p[1])
        # 合併 L 型區域的點
        points_arrangeby_batches[start] = points_in_row_of_batch + points_in_column_of_batch
        #print(f" 點區域  第 {start + 1} 批的點數: {len(points_arrangeby_batches[start])}")
        for j in range(len(points_arrangeby_batches[start])):
            matched_target_and_array[start].append((points_arrangeby_batches[start][j], array[start][j]))
    #print(f'length of matched_target_and_array list={len(matched_target_and_array)}')
    return matched_target_and_array   
    
def straight_path(matched_target_and_array, batch_counts, step_size):
    whole_paths = [[] for _ in range(batch_counts)]
    
    for i, (current_point, target_center) in enumerate(matched_target_and_array):
        dx = target_center[0] - current_point[0]
        dy = target_center[1] - current_point[1]
        all_length = math.sqrt(dx**2 + dy**2)
        
        if all_length == 0:  # 如果起點終點一樣，直接連到終點
            whole_paths[i].append(current_point)
            continue
        
        step_count = int(all_length / step_size)  
        if step_count == 0:  # 如果距離太短，直接連到終點
            whole_paths[i].extend([current_point, target_center])
            continue
        
        step_x = dx / step_count 
        step_y = dy / step_count  
        whole_paths[i].append(current_point)
        
        for step_idx in range(1, step_count + 1): 
            next_point = (
                round(current_point[0] + step_x * step_idx),
                round(current_point[1] + step_y * step_idx)
            )
            whole_paths[i].append(next_point)
        
        if whole_paths[i][-1] != target_center:
            whole_paths[i].append(target_center)
    #for i, path in enumerate(whole_paths):
        #print(f"Path {i}: {path}")
    
    return whole_paths

def check_paths_for_obstacles(whole_paths, obstacle_coordinates, size ): #size可根據光大小設定
    overlapping_rows_in_wholepaths = []
    #print("我在檢查路徑是否與障礙物重疊了!!!!")
    for path_index, path in enumerate(whole_paths):
        for point in path:
            for obstacle in obstacle_coordinates:
                if (obstacle[0] - size <= point[0] <= obstacle[0] + size and
                    obstacle[1] - size <= point[1] <= obstacle[1] + size):
                    overlapping_rows_in_wholepaths.append(path_index)
                    #print(f"路徑 {path_index} 與障礙物重疊了")
                    break  
            if path_index in overlapping_rows_in_wholepaths:
                break  # 若已確定重疊，則不需再檢查
        #print(f"我檢查完{path_index}了!!!!")
    return overlapping_rows_in_wholepaths

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
                grid_center_x = grid_x * grid_size + grid_size / 2
                grid_center_y = grid_y * grid_size + grid_size / 2

                # 計算障礙物中心到網格中心的曼哈頓距離
                dx = abs(grid_center_x - ox)
                dy = abs(grid_center_y - oy)

                # 若障礙物到網格中心的距離小於網格一半加上障礙物半徑，設置為不可通行
                if dx <= grid_size / 2 + obstacle_radius or dy <= grid_size / 2 + obstacle_radius:
                    walkable_grid[grid_y][grid_x] = False
    return walkable_grid

def a_star(start_grid, target_grid, obstacle_coordinates, walkable_grid, grid_size):
    #A* 路徑規劃 
    def g_scores(current, neighbor):
        # 計算實際的 x 和 y 差距
        dx = abs(current[0] - neighbor[0])
        dy = abs(current[1] - neighbor[1])
        if dx == dy:  # 對角線移動
            return 14 * dx  # 每個對角移動成本為 14
        else:
            return 10 * (dx + dy)  # 直線移動成本為 10

    def heuristic_with_obstacles(current, target, obstacles_coordinates, grid_size):
        # h(n) 優化版本
        dx = abs(target[0] - current[0])
        dy = abs(target[1] - current[1])
        base_distance = 10 * math.sqrt(dx**2 + dy**2)  # 使用曼哈頓距離，與 g(n) 保持相似比例
    
        penalty = 0
        # 使用曼哈頓距離檢查障礙物
        for ox, oy in obstacles_coordinates:
            # 檢查是否在路徑上：用矩形框快速檢查
            if (min(current[0], target[0]) <= ox <= max(current[0], target[0]) and 
                min(current[1], target[1]) <= oy <= max(current[1], target[1])):
                # 使用曼哈頓距離估算
                dist = abs(ox - current[0]) + abs(oy - current[1])
                if dist < grid_size:  # 障礙物在 500 像素範圍內
                    penalty += 80 * (1 - dist / grid_size)
        return base_distance + penalty

    def get_neighbors(current_grid, walkable_grid):
        # 取得鄰居節點
        x, y = current_grid
        # 定義相鄰節點的相對位置
        directions_first = [
            (-1, 0),   # 左
            (-1, -1),  # 左斜上
            (0, -1),   # 正上
            (-1, 1),   # 左斜下
            (0, 1),    # 正下
                (1, 0),    # 右
            (1, -1),   # 右斜上
            (1, 1)     # 右斜下
        ]
        '''ection_second = [   
            (1, 0),    # 右
            (1, -1),   # 右斜上
            (1, 1)     # 右斜下
        ]'''
        neighbors = []
        for dx, dy in directions_first:
            # 移除 grid_size 相乘
            neighbor_x = x + dx
            neighbor_y = y + dy
            # 確保鄰居在邊界內且是可通行的
            if (0 <= neighbor_x < len(walkable_grid[0]) and 
                0 <= neighbor_y < len(walkable_grid) and 
                walkable_grid[neighbor_y][neighbor_x]):  # 檢查是否可通行
                neighbors.append((neighbor_x, neighbor_y))
        '''if not neighbors:
            for dx, dy in direction_second:
                # 移除 grid_size 相乘
                neighbor_x = x + dx
                neighbor_y = y + dy
                # 確保鄰居在邊界內且是可通行的
                if (0 <= neighbor_x < len(walkable_grid[0]) and 
                    0 <= neighbor_y < len(walkable_grid) and 
                    walkable_grid[neighbor_y][neighbor_x]):  # 檢查是否可通行
                    neighbors.append((neighbor_x, neighbor_y))'''
        return neighbors

    open_set = []
    closed_set = set()
    came_from = {}
    g_score = {start_grid: 0}
    h_score = heuristic_with_obstacles(start_grid, target_grid, obstacle_coordinates, grid_size)
    f_score = {start_grid: g_score[start_grid] + h_score}
    heapq.heappush(open_set, (f_score[start_grid], start_grid))

    while open_set:
        current = heapq.heappop(open_set)[1]

        # 檢查是否到達目標
        if current == target_grid:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed_set.add(current)

        for neighbor in get_neighbors(current, walkable_grid):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + g_scores(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic_with_obstacles(neighbor, target_grid, obstacle_coordinates, grid_size)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
    return []  # 若沒有找到路徑，返回空列表

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

# 路徑尋找
def draw_and_get_paths(image, matched_target_and_array, obstacle_coordinates, step_size, batch_counts, size):
    start_time = time.time()
    whole_paths = straight_path(matched_target_and_array, batch_counts, step_size)
    no_found = 0
    # 檢查路徑是否與障礙物重疊
    overlapping_rows_in_wholepaths = check_paths_for_obstacles(whole_paths, obstacle_coordinates, size)
    if len(overlapping_rows_in_wholepaths) == 0:
        pass
    else:    
        for row in overlapping_rows_in_wholepaths:
            start_point = whole_paths[row][0]
            target_center = whole_paths[row][-1]
            image_height, image_width = image.shape[:2]
            walkable_grid = convert_to_grid_coordinates(image_width, image_height, obstacle_coordinates, size, obstacle_radius=15)
            grid_size = size
            start_grid = (start_point[0] // grid_size, start_point[1] // grid_size)
            target_grid = (target_center[0] // grid_size, target_center[1] // grid_size)
            if start_grid[0] < 0 or start_grid[1] < 0 or target_grid[0] >= len(walkable_grid[0]) or target_grid[1] >= len(walkable_grid):
                raise ValueError("Start or target grid out of bounds")
            
            path = a_star(start_grid, target_grid, obstacle_coordinates, walkable_grid, grid_size)
        
            if path:
                actual_path = []
                path.remove(path[-1])               
                
                # 處理起點到第一個網格點的路徑
                first_grid_point = (path[0][0] * grid_size + grid_size//2, path[0][1] * grid_size + grid_size//2 )
                initial_segment = interpolate_path(start_point, first_grid_point, step_size)[:-1]  # 不包含終點，避免重複
                actual_path.extend(initial_segment)
                
                # 處理中間路徑點
                for i in range(len(path)-1):
                    current = (path[i][0] * grid_size + grid_size//2, path[i][1] * grid_size + grid_size//2)
                    next_point = (path[i+1][0] * grid_size + grid_size//2, path[i+1][1] * grid_size + grid_size//2)
                    # 在兩個網格點之間插入 step_size 的點
                    interpolated = interpolate_path(current, next_point, step_size)[:-1]  # 不包含終點，避免重複
                    actual_path.extend(interpolated)
                
                # 處理最後一段到目標點的路徑
                last_grid_point = (path[-1][0] * grid_size + grid_size//2, path[-1][1] * grid_size + grid_size//2)
                final_segment = interpolate_path(last_grid_point, target_center, step_size)
                actual_path.extend(final_segment)
                whole_paths[row] = actual_path
            else:
                print(f"無法找到從 {start_point} 到 {target_center} 的路徑")
                no_found+=1
    end_time = time.time()            
    total_time = end_time - start_time
    
    # 繪製所有路徑
    for path in whole_paths:
        if path:
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            for j in range(len(path) - 1):
                cv2.line(image, path[j], path[j + 1], color, 2)
                
    print(f"單圖陣列 路徑規劃(先直線後用a*) 處理時間={total_time}")
    return image, whole_paths, total_time, no_found

# 主函數
def whole_step_6_draw_path(batch_counts, light_image, size, obsticle_coordinates, file_name, columns, rows, matched_target_and_array):

    ###設定光圖形最段距離移動步距， 與移動rate有關
    #root = tk.Tk()
    #root.withdraw()
    step_size = 3 #simpledialog.askinteger("Input", "設定光圖形一幀移動的距離， 與光圖形移動rate有關 (integer):")
    step_size =int(step_size)
    
    # 繪製移動路徑
    image_with_paths, whole_paths, total_time, no_found = draw_and_get_paths(light_image, matched_target_and_array, obsticle_coordinates, step_size, batch_counts, size)
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
    return  whole_paths, step_size, total_time, no_found

## step 7
#檢測碰撞
def collision(Rl, Rp, start_point, particle_coor):
    distance_squared = (particle_coor[0] - start_point[0])**2 + (particle_coor[1] - start_point[1])**2
    if  distance_squared <= (Rl + Rp)**2 +2:                         #distance_squared >= (Rl - Rp-4)**2:  # 碰撞发生（在光圈内或刚好接触）
        return 1  # 返回1表示发生了碰撞
    else:
        return 0
    
#找到光圈圓心到圓點的單位向量   
def find_unit_vector(light_coor, particle_coor):
    # 计算u_vector的x和y分量的差值
    dx = light_coor[0] - particle_coor[0]
    dy = light_coor[1] - particle_coor[1]
    # 检查dx和dy是否都为零
    if dx == 0 and dy == 0:
        return (0, 0)
    # 计算圆点中心到光圈圆心的单位向量
    distance_two_points = math.sqrt(dx**2 + dy**2)
    unit_vector = (dx/distance_two_points, dy/distance_two_points)
    #vector=(dx,dy)
    return  unit_vector

# 模擬移動
def simulate_movement(canvas, step_size, whole_paths, all_particle_coor, target_numbers, Rl, Rp, obstacle_coordinate, file_name):
    circles = all_particle_coor[target_numbers:]  # 靜止的小圓座標
    max_path_length = max(len(path) for path in whole_paths)  # 找最長的路徑
    
    # 填充路徑到相同長度
    for k in range(len(whole_paths)):
        while len(whole_paths[k]) < max_path_length:
            whole_paths[k].append(whole_paths[k][-1])  # 最後一點填充(表一直在陣列終點)
                
    # 設置影片輸出區
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outputpath = f"C:/Users/Vivo/CGU/odep_cellarray/Cell_Manipulation_Simulation/movement_simulation_{file_name}_分批校正版.mp4"
    out = cv2.VideoWriter(outputpath, fourcc, 20.0, (canvas.shape[1], canvas.shape[0]))

    all_particle_coor_list = [list(coord) for coord in all_particle_coor] 
    number = 0
    for step in range(max_path_length):
        display_img = canvas.copy() 
        
        '''模擬粒子掉落而已'''
        # **隨機選擇一個粒子的索引**
        random_index = random.randint(0, len(whole_paths) - 1)
        selected_particle_before_update = all_particle_coor_list[random_index].copy()  # 複製避免影響原數據

        for obstacle in obstacle_coordinate:
            cv2.circle(display_img, obstacle, 15, (255, 0, 0), -1)
        
        # 畫靜止的圓
        for circle in circles:
            cv2.circle(display_img, circle, radius=Rp, color=(0, 0, 250), thickness=-1)

        # 繪製光圈和他的粒子
        for i, path in enumerate(whole_paths):
            if step < len(path):  # 確保路徑不會超出範圍
                number += 1
                point = path[step]  # 當前目標位置
                particle_coor = all_particle_coor_list[i]  # 當前粒子位置

                # 繪製光圈與粒子
                cv2.circle(display_img, tuple(particle_coor), Rp, (0, 0, 250), -1)
                cv2.circle(display_img, point, radius=Rl, color=(250, 250, 250), thickness=10)
                
                # 更新粒子位置
                unit_vector = find_unit_vector(point, particle_coor)
                particle_coor[0] += int(unit_vector[0] * step_size)
                particle_coor[1] += int(unit_vector[1] * step_size)

                # 確保粒子在光圈內
                dx = particle_coor[0] - point[0]
                dy = particle_coor[1] - point[1]
                distance_squared = dx**2 + dy**2
                if distance_squared > (Rl - Rp)**2:  # 掉出光圈外
                    distance = math.sqrt(distance_squared)
                    scale = (Rl - Rp) / distance
                    particle_coor[0] = point[0] + int(dx * scale)
                    particle_coor[1] = point[1] + int(dy * scale)
        
        if random_signal() == 1:
            print("有粒子掉落")
        
        scale_percent = 50
        width = int(display_img.shape[1] * scale_percent / 100)
        height = int(display_img.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_image = cv2.resize(display_img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Movement Simulation', resized_image)
        out.write(display_img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()

# step7 主函數    
def whole_step7_simulate_moving(size, target_numbers, whole_paths, all_sorted_coordinate, step_size, Rl, Rp, obstacle_coordinate, file_name, columns, rows):
    canvas = np.zeros((1220, 1814, 3), dtype=np.uint8)
    generate_array(canvas, size, columns, rows)
    simulate_movement(canvas, step_size, whole_paths, all_sorted_coordinate, target_numbers, Rl, Rp, obstacle_coordinate, file_name)
    return

#避免碰撞發生
#檢查現有路徑是否會有碰撞發生
def distance_calculation(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance
    
def collision_in_path(whole_path, Rl):
    
    max_path_length = max(len(path) for path in whole_path) 
    collision_list = []  
    
    for step in range(max_path_length): 
        active_points = []  # 當前時間步所有粒子的位置
        path_indices = []  # 記錄對應的 path 編號
        
        for idx, path in enumerate(whole_path_batch):
            if step < len(path):  
                active_points.append(path[step])
                path_indices.append(idx)  # 記錄這個點對應的路徑編號
        
        # 檢查當前時間步的所有點，是否有兩點距離過近
        for i in range(len(active_points)):
            for j in range(i + 1, len(active_points)):  
                dx = abs(active_points[i][0] - active_points[j][0])
                dy = abs(active_points[i][1] - active_points[j][1])
                
                if dx**2 + dy**2 < (2*(Rl+10))**2:  # 10是來自於光圈thickness
                    collision_list.append((path_indices[i], path_indices[j], step))  
    
    return collision_list  

#修正路徑直到沒有碰撞
def resolve_collisions(whole_path, Rl, step_size):
    # whole_path格式更改為一維
    '''whole_path = []
    for batch in whole_path:
        for path in batch:
            whole_path.append(path)'''

    collision_info = collision_in_path(whole_path, Rl)
    n = 0
    collision_info = [collision for collision in collision_info if collision[2] > 2* (Rl+10) // step_size +1]
    #print(f"過濾後碰撞信息: {collision_info}")
    while collision_info:
        n += 1
        #print(f"第{n}次碰撞數量: {len(collision_info)}")
        # 1. 建立碰撞關係
        added_path = []
        point_1, point_2, step = collision_info[0]
        inactive_path = []
        if point_1 < point_2:
            path_of_point2_before = whole_path[point_2][:step-2*(Rl+10)//step_size-3 ]
            path_of_point2_after = whole_path[point_2][step-2*(Rl+10)//step_size-3:] 
            for i in range(2*(Rl+10)// step_size + 3):
                if len(path_of_point2_before) == 0:
                    #path_of_point2_before.insert(0, whole_path[point_2][0])
                    inactive_path.append((point_1, point_2, step))
                    break
                else:
                    added_path.append(path_of_point2_before[-1])
            new_path_of_point2 = path_of_point2_before + added_path + path_of_point2_after  
            whole_path[point_2] = new_path_of_point2
            #print(f"{point_1}路徑為:{whole_path[point_1]}")
            #print(f"第{point_2}路徑改善為:{whole_path[point_2]}")
        else:
            path_of_point1_before = whole_path[point_1][:step-2*(Rl+10)//step_size-3]
            path_of_point1_after = whole_path[point_1][step- 2*(Rl+10)//step_size-3:] 
            for i in range(2* (Rl+10)//step_size+3):
                if len(path_of_point2_before) == 0:
                    #path_of_point2_before.insert(0, whole_path[point_2][0])
                    inactive_path.append((point_1, point_2, step))
                    break
                else:
                    added_path.append(path_of_point1_before[-1])
            new_path_of_point1 = path_of_point1_before + added_path + path_of_point1_after  
            whole_path[point_1] = new_path_of_point1
        
        collision_info = collision_in_path(whole_path, Rl)
        collision_info = [collision for collision in collision_info if collision[2] > 2*(Rl+10) // step_size+1]
        if len(inactive_path) > 0:
            collision_info.remove(inactive_path[0])
        print(f"第{n}次改善結果: {collision_info}")
        if len(collision_info) == 0:
            return whole_path, len(collision_info)
    return whole_path, len(collision_info)

#檢查外部先被填滿
def check_external_filling(whole_path_batch):
    #檢查是否有外部先被填滿的狀況
    paths_length = [len(path) for path in whole_path_batch]
    path_error = []
    # 檢查當前時間步的所有點，是否有兩點距離過近
    for i in range(len(whole_path_batch)):
        for j in range(i + 1, len(whole_path_batch)):
            if paths_length[i] > paths_length[j]:
                path_error.append((i, paths_length[i], j, paths_length[j]))
                #print(f"路徑 {i} 長度 {paths_length[i]} 比路徑 {j} 長度 {paths_length[j]}長")
    return path_error

#校正外部先被填滿
def correction_of_path(whole_path_batch, step_size, size, obsticle_coordinates, path_error,  light_image):

    while path_error:
        new_assigned_match = [[]for _ in range(2)]
        idx_first_p, path_length_first, idx_second_p, path_length_second = path_error[0]
        path_length_diff = abs(path_length_first - path_length_second)
        light_image_copy= light_image.copy()
        # 外部顆粒延遲移動
        if  idx_first_p < idx_second_p:
            for i in range (path_length_diff):
                whole_path_batch[idx_second_p].insert(0, whole_path_batch[idx_second_p][0])
            path_error = check_external_filling(whole_path_batch)
        else: 
            for i in range (path_length_diff):
                whole_path_batch[idx_first_p].insert(0, whole_path_batch[idx_first_p][0])
            path_error = check_external_filling(whole_path_batch)
        if not path_error:
            return whole_path_batch
    return whole_path_batch



def random_signal(): #模擬有粒子拖不動時
    return 1 if random.randint(1, 100) == 1 else 0  # 以 1% 概率返回 1
    

# 主函數
if __name__ == '__main__':
    Rl, Rp = 15, 9  # 光圈半徑和粒子半徑    
    obstacle_coordinate = []
    target_coordinate = []
    circle_in_array_not_traget = []
    upadate_all_coordinate = []
    one_exp_path_times = []
    whole_path_time = []
    one_exp_path_no_found = []
    # whole_time= []
    output_folder = 'C:\\Users\\Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\raw_image'
    folder = 'C:\\Users\\Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\raw_image'

    size, target_numbers, arrayimage, file_name, columns, rows = wholestep3_draw_array_picture()
    target_coordinate, obstacle_coordinate, circle_in_array_not_traget, light_image, upadate_all_coordinate, all_sorted_coordinate = wholestep5_draw_light_image(arrayimage, target_numbers, size, file_name, columns, rows)

    # 分區(L型)分配座標
    whole_paths = [[] for _ in range(len(target_coordinate))]
    sum_path_length = 0
    sum_path_counts = 0
    batch_counts = 0
    matched_target_and_array = assignment(target_coordinate, size, columns, rows)
    for start in range(min(rows, columns)):
        obstacle_coordinate_changed = obstacle_coordinate.copy()
        matched_target_and_array_batch = []
        copyimage = light_image.copy()
        matched_target_and_array_batch = matched_target_and_array[start]  # 取出對應的批次
        batch_counts = len(matched_target_and_array[start])

        #更新障礙物列表(非批次的粒子)
        before_start = matched_target_and_array[:start]  # start 之前的部分
        after_start = matched_target_and_array[start + 1:]  # start 之後的部分
        obstacle_coordinate_changed.extend([destination for batch in before_start for _, destination in batch])
        obstacle_coordinate_changed.extend([start_point for batch in after_start for start_point, _ in batch])

        whole_path_batch, step_size, path_time, path_no_found = whole_step_6_draw_path(batch_counts, copyimage, size, obstacle_coordinate_changed, file_name, columns, rows, matched_target_and_array_batch)
        # 打印 whole_paths 以確認結果
        #for i, path in enumerate(whole_path_batch):
        #    print(f"Path_batch {start} number{i}: {len(path)}")

        #檢查是否有外部先被填滿的狀況
        path_error = check_external_filling(whole_path_batch)
        
        if len(path_error) > 0:
            #print(f"!!!!!第 {start} 批次的路徑存在外部先被填滿的狀況")
            correction_of_path(whole_path_batch, step_size, size, obstacle_coordinate, path_error, light_image)
            
        #檢查碰撞
        '''whole_path_batch, length_of_path_collision = resolve_collisions(whole_path_batch, Rl, step_size)
        if length_of_path_collision > 0:
            print(f"!!!!!第 {start} 批次的路徑仍然存在碰撞")'''
        
        # 填充 path 至一樣長度
        max_path_length = max(len(path) for path in whole_path_batch)  # 找最長的路徑

        for k in range(len(whole_path_batch)):
            while len(whole_path_batch[k]) < max_path_length:
                whole_path_batch[k].append(whole_path_batch[k][-1])  # 最後一點填充(表一直在陣列終點)
            for m in range( int(1/5 * sum_path_length)):   #可以設定每批開始移動的間隔時間
                whole_path_batch[k].insert(0, whole_path_batch[k][0])  # 填充第一個點來分批移動
            
        #for i, path in enumerate(whole_path_batch):
            #print(f"second check Path_batch {start} number{i}: {len(path)}")
        sum_path_length += int( max_path_length)

        # 將批次的路徑添加到 whole_paths
        for l, path in enumerate(whole_path_batch):
            index =  sum_path_counts + l  
            if index < len(whole_paths):
                whole_paths[index].extend(path)
            else:
                print(f"Index out of range: {index}")

        sum_path_counts += batch_counts
        
    #最後檢查可能的碰撞
    whole_paths, length_of_path_collision = resolve_collisions(whole_paths, Rl, step_size)
    if length_of_path_collision > 0:
        print(f"!!!!!第 {start} 批次的路徑已超過迴圈數限制，仍然存在碰撞")
    
    # 打印 whole_paths 以確認結果
    #for i, path in enumerate(whole_paths):
    #    print(f"Path {i}: {len(path)}")
    # 打印 whole_paths 的行數
    #print(f"Number of rows in whole_paths: {len(whole_paths)}")
    
    whole_step7_simulate_moving(size, target_numbers, whole_paths, all_sorted_coordinate, step_size, Rl, Rp, obstacle_coordinate, file_name, columns, rows)

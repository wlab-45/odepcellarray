import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog,simpledialog
import os
import math, random
from typing import List, Tuple, Set
import heapq
#import matplotlib.pyplot as plt
#import multiprocessing

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
        
def generate_array(image,size,target_numbers):
    # 繪製四個正方形
    for i in range(int(math.sqrt(target_numbers))):
        for j in range(int(math.sqrt(target_numbers))):
            top_left = (i * size, j * size)  # 左上角座標
            bottom_right = (top_left[0] + size, top_left[1] + size)  # 右下角座標
            # 繪製虛線正方形
            cv2.rectangle(image, top_left, bottom_right, (255, 191, 0), 2)
    return image 

def wholestep3_draw_array_picture():
    file_path=select_png_file()
    image=cv2.imread(file_path)
    file_name = os.path.basename(file_path)
    name, extension = os.path.splitext(file_name)
    #set size of each square arraysize
    root = tk.Tk()
    root.withdraw()
    size = simpledialog.askinteger("Input", "Enter the size of each square in the array (integer):")
    size=int(size)
    target_numbers = simpledialog.askinteger("Input", "Enter the number of squares in the array (you should input a square of integer or there may be an error):")
    target_numbers = int(target_numbers)
    arrayimage =generate_array(image , size, target_numbers)

    cv2.imshow('Array of Squares', arrayimage )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    save_arrayimage_directory="C:/Users/Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\array_image"
    array_image_path = os.path.join(save_arrayimage_directory, f'arrayimage_{file_name}')
    cv2.imwrite(array_image_path, arrayimage)
    print("array success")
    return size, target_numbers, arrayimage

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

def form_sort_list(arrayimage, coordinates_file_path, size, target_numbers):
    all_coordinates =  read_coordinates_from_file(coordinates_file_path) # 初始化空列表以存儲座標
    
    all_sorted_coordinates = sorted(all_coordinates, key=lambda point: math.sqrt(point[0]**2 + point[1]**2))
    
    obsticle_coordinates = []
    # 繪製障礙物，確保不與紅色圓點重疊
    obsticle_image = arrayimage.copy()
    radius = 15
    for i in range(30):
        while True:
            x, y = np.random.randint(size*int(math.sqrt(target_numbers)), 1814), np.random.randint(size*int(math.sqrt(target_numbers)), 1220)
            if all(math.sqrt((x - cx)**2 + (y - cy)**2) > 4 * radius for cx, cy in all_coordinates):
                cv2.circle(obsticle_image, (x, y), radius, (255, 0, 0), -1)  # 繪製圓形
                obsticle_coordinates.append((x, y))
                break
    return all_sorted_coordinates, obsticle_image, obsticle_coordinates

def calculate_distance(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

def picking_target(all_coordinate, target_numbers, size, obstacle_image):
    target_coordinate=[]
    circle_in_array=[]
    circle_in_array_not_target=[]
    if len(all_coordinate) >= target_numbers:
        #偵測array中的點數目分為兩種情況
        for (x, y) in all_coordinate:
            if x < int(math.sqrt(target_numbers) * size) and y < int(math.sqrt(target_numbers) * size):
                circle_in_array.append((x, y))

        if len(circle_in_array) <= target_numbers:  
            target_coordinate = all_coordinate[:target_numbers]
        else:
            print("circle_in_array > target, we'll move the rest circle out")
            target_coordinate = all_coordinate[:target_numbers]
            circle_in_array_not_target = circle_in_array[target_numbers:]
        update_all_coordinate = all_coordinate[target_numbers:] #只有第一次確定擷target_numbers個，all更新，去除前四個
    else:
        print("all_coordinate list length error")
    return target_coordinate, circle_in_array_not_target, update_all_coordinate

def draw_light_image(arrayimage, target_coordinate):
    light_image=arrayimage.copy()
    good=len(target_coordinate)
    for i in range(good):
        cv2.circle(light_image , target_coordinate[i] , int(2*10),(250,250,255),10)  # 繪製圓形
    return light_image    
        
def wholestep5_draw_light_image(arrayimage, target_numbers,size):
    #generate 4 list
    all_coordinate=[]
    upadate_all_coordinate=[]
    target_coordinate=[]
    obstacle_coordinate=[]
    circle_in_array_not_traget=[]
    ##set coordinates file path
    # 隱藏主視窗
    root = tk.Tk()
    root.withdraw()
    coordinates_file_path = filedialog.askopenfilename(
        title='選擇 txt 檔案',
        filetypes=[('txt Files', '*.txt')]
    )
    #generating all_cooridinate
    all_coordinate, obsticle_image, obstacle_coordinate = form_sort_list(arrayimage, coordinates_file_path, size, target_numbers)
    print(f'length of all_coordinate list={len(all_coordinate)}')
    
    #picking target
    target_coordinate , circle_in_array_not_traget, upadate_all_coordinate=picking_target(all_coordinate, target_numbers, size, obsticle_image)
    print(f'length of target_coordinate list={len(target_coordinate)}')
    
    #驗證用
    light_image=draw_light_image(obsticle_image,target_coordinate)
    scale_percent = 70  # 缩放比例
    width = int(light_image.shape[1] * scale_percent / 100)
    height = int(light_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_light_image= cv2.resize(light_image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('step5: processed image', resized_light_image )
    cv2.imwrite("C:/Users/Vivo/CGU/odep_cellarray/Cell_Manipulation_Simulation/virtual_test_image/lightimg_with_particle/0000.png", light_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return target_coordinate, obstacle_coordinate, circle_in_array_not_traget, light_image ,upadate_all_coordinate, all_coordinate


##step 6 (結構樣要重寫)
def structure_and_center_of_array (target_numbers, size):
    num_layers = int(math.sqrt(target_numbers))
    array_layers = [[] for _ in range(num_layers)]
    for i in range(num_layers):  #假設每層3格(3*3矩陣)則標號為layer(row) 0,1,2 每層格編號(左至右)
        for j in range(1, num_layers*2, 2):
            k=i+1
            array_layers[i].append((size *j// 2, size *(2*k-1)//2))
    return array_layers

# 分配座標
def assignment(target_coordinate, size):
    matched_target_and_array=[]
    target_numbers= len(target_coordinate)
    #儲存左上、右上、左下、右下陣列中心點
    array_layers = structure_and_center_of_array(target_numbers,size)
    points = sorted(target_coordinate, key=lambda p: p[1])# 排序y座標
    points_layers = [[] for _ in range(int(math.sqrt(target_numbers)))]
    for i in range(int(math.sqrt(target_numbers))):
        points_layers[i] = sorted(points[ :int(math.sqrt(target_numbers))], key=lambda p: p[0])  # 左下、右下(x座標排序)
        points = points[int(math.sqrt(target_numbers)):]  # 扣除已經排序的點
    
    for i in range(int(math.sqrt(target_numbers))):
        for j in range(int(math.sqrt(target_numbers))):
            matched_target_and_array.append(points_layers[i][j])
            matched_target_and_array.append(array_layers[i][j])
    print (f'match:{matched_target_and_array}')
    return matched_target_and_array

# a*演算法 
def draw_and_get_paths(image, matched_target_and_array, obstacle_coordinates, step_size):

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """基礎的啟發式函數：計算歐幾里得距離"""
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def heuristic_with_obstacles(a: Tuple[int, int], b: Tuple[int, int], obstacles: List[Tuple[int, int]], radius: int) -> float:
        """改進版啟發式函數，考慮障礙物的影響"""
        base_heuristic = heuristic(a, b)
        penalty = 0
        for ox, oy in obstacles:
            distance = math.sqrt((a[0] - ox) ** 2 + (a[1] - oy) ** 2)
            if distance < radius * 2:  # 若靠近障礙物，添加懲罰
                penalty += (radius * 2 - distance)
        return base_heuristic + penalty

    def is_collision(point: Tuple[int, int], obstacle_coordinates: List[Tuple[int, int]], obstacle_radius: int) -> bool:
        """是否碰撞"""
        for ox, oy in obstacle_coordinates:
            if math.sqrt((point[0] - ox) ** 2 + (point[1] - oy) ** 2) < obstacle_radius:
                return True
        return False

    def is_valid_direction(direction: Tuple[int, int], obstacle_coordinates: List[Tuple[int, int]], obstacle_radius: int) -> bool:
        """檢查某個方向是否有效，避免與障礙物碰撞"""
        return not is_collision(direction, obstacle_coordinates, obstacle_radius)

    def get_neighbors(point: Tuple[int, int], step_size: int, target_point: Tuple[int, int], obstacle_coordinates: List[Tuple[int, int]], obstacle_radius: int) -> List[Tuple[int, int]]:
        """取得鄰居節點，首先考慮正交線的上半部方向，若有障礙才考慮下半部方向"""
        x, y = point
        # 從圖像獲取尺寸
        image_height, image_width = image.shape[:2]

        # 上半部的方向（正上、左右斜上、左、右）
        directions_up = [
            (x, y - step_size),  # 正上
            (x - step_size, y - step_size),  # 左斜上
            (x + step_size, y - step_size),  # 右斜上
            (x - step_size, y),  # 左
            (x + step_size, y)   # 右
        ]

        # 下半部的方向（左右斜下、正下）
        directions_down = [
            (x - step_size, y + step_size),  # 左斜下
            (x + step_size, y + step_size),  # 右斜下
            (x, y + step_size)  # 正下
        ]

        neighbors = []

        # 先檢查上半部的五個方向
        for direction in directions_up:
            if 0 <= direction[0] < image_width and 0 <= direction[1] < image_height and is_valid_direction(direction, obstacle_coordinates, obstacle_radius):
                neighbors.append(direction)

        # 若上半部無法找到有效方向，再檢查下半部的三個方向
        if not neighbors:
            for direction in directions_down:
                if 0 <= direction[0] < image_width and 0 <= direction[1] < image_height and is_valid_direction(direction, obstacle_coordinates, obstacle_radius):
                    neighbors.append(direction)
        return neighbors


    def a_star(start: Tuple[int, int], goal: Tuple[int, int], 
               obstacle_coordinates: List[Tuple[int, int]], 
               step_size: int) -> List[Tuple[int, int]]:
        """A* 路徑規劃"""
        obstacle_radius = 50  # 動態調整障礙物半徑
        open_set = []
        closed_set: Set[Tuple[int, int]] = set()
        came_from = {}

        g_score = {start: 0}
        f_score = {start: heuristic_with_obstacles(start, goal, obstacle_coordinates, obstacle_radius)}
        heapq.heappush(open_set, (f_score[start], start))

        while open_set:
            current = heapq.heappop(open_set)[1]

            # 如果接近目標點，完成搜索
            if heuristic(current, goal) < step_size:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                path.append(goal)
                return path

            closed_set.add(current)
            
            for neighbor in get_neighbors(current, step_size, goal, obstacle_coordinates, obstacle_radius):
                if neighbor in closed_set or is_collision(neighbor, obstacle_coordinates, obstacle_radius):
                    continue

                tentative_g_score = g_score[current] + heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic_with_obstacles(neighbor, goal, obstacle_coordinates, obstacle_radius)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # 找不到路徑
    
    whole_paths=[]
    
    # 處理每對起點和終點
    for i in range(0, len(matched_target_and_array), 2):
        start_point = matched_target_and_array[i]
        target_center = matched_target_and_array[i + 1]

        # 使用 A* 尋找路徑
        path = a_star(start_point, target_center, obstacle_coordinates, step_size)

        whole_paths.append(path)
        
        if path:
            # 繪製路徑
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for j in range(len(path) - 1):
                cv2.line(image, path[j], path[j + 1], color, 2)
        else:
            print(f"無法找到從 {start_point} 到 {target_center} 的路徑")

    return image, whole_paths

# 主函數
def whole_step_6_draw_path(target_coordinate, light_image, size, obsticle_coordinates):
    matched_target_and_array=[]
    # 分配
    matched_target_and_array= assignment(target_coordinate,size)
    target_numbers = len(target_coordinate)
    ###設定光圖形最段距離移動步距， 與移動rate有關
    root = tk.Tk()
    root.withdraw()
    step_size =  simpledialog.askinteger("Input", "設定光圖形一幀移動的距離， 與光圖形移動rate有關 (integer):")
    step_size =int(step_size)
    
    # 繪製移動路徑
    image_with_paths, whole_paths = draw_and_get_paths(light_image, matched_target_and_array, obsticle_coordinates, step_size)
    scale_percent = 50  # 缩放比例
    width = int(image_with_paths.shape[1] * scale_percent / 100)
    height = int(image_with_paths.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_image_with_path= cv2.resize(image_with_paths, dim, interpolation=cv2.INTER_AREA)
    # 顯示影像
    cv2.imshow("Result of path", resized_image_with_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("C:/Users/Vivo/CGU/odep_cellarray/Cell_Manipulation_Simulation/virtual_test_image/path_image/0000.png", image_with_paths)
    
    return  whole_paths, step_size



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

#模擬移動
def simulate_movement(canvas, step_size, whole_paths, all_particle_coor, target_numbers, Rl, Rp, obstacle_coordinate):
    circles = all_particle_coor[target_numbers:]  # 静止的小圆坐标
    max_path_length = max(len(path) for path in whole_paths)  # 找到最长的路径长度

    # 填充路径使长度一致
    for i in range(len(whole_paths)):
        while len(whole_paths[i]) < max_path_length:
            whole_paths[i].append(whole_paths[i][-1])  # 用最后一个点填充

    # 设置视频输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outputpath = "movement_simulation_version2.mp4"
    out = cv2.VideoWriter(outputpath, fourcc, 20.0, (canvas.shape[1], canvas.shape[0]))

    all_particle_coor_list = [list(coord) for coord in all_particle_coor]  # 转为列表以便修改
    number=0
    for step in range(max_path_length):
        display_img = canvas.copy()  # 创建画布副本

        for obstacle in obstacle_coordinate:
            cv2.circle(display_img , obstacle, 15, (255, 0, 0), -1)
        
        # 绘制静止小圆
        for circle in circles:
            cv2.circle(display_img, circle, radius=Rp, color=(0, 0, 250), thickness=-1)

        # 绘制目标大圆和拖动的小圆
        for i, path in enumerate(whole_paths):
            
            if step < len(path):  # 确保不超出路径长度
                number+=1
                point = path[step]  # 当前大圆位置
                particle_coor = all_particle_coor_list[i]  # 当前小圆位置

                # 绘制小圆和大圆
                cv2.circle(display_img, tuple(particle_coor), Rp, (0, 0, 250), -1)
                cv2.circle(display_img, point, radius=Rl, color=(250, 250, 250), thickness=10)
                
                
                # 更新小圆位置
                unit_vector = find_unit_vector(point, particle_coor)
                particle_coor[0] += int(unit_vector[0] * step_size)
                particle_coor[1] += int(unit_vector[1] * step_size)

                # 确保小圆保持在大圆范围内
                dx = particle_coor[0] - point[0]
                dy = particle_coor[1] - point[1]
                distance_squared = dx**2 + dy**2
                if distance_squared > (Rl - Rp)**2:  # 超出大圆范围
                    distance = math.sqrt(distance_squared)
                    scale = (Rl - Rp) / distance
                    particle_coor[0] = point[0] + int(dx * scale)
                    particle_coor[1] = point[1] + int(dy * scale)


        # 缩放并显示
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
   
# stpe7 主函數    
def whole_step7_simulate_moving(size , target_numbers, whole_paths, all_sorted_coordinate, step_size, Rl, Rp, obstacle_coordinate):
    canvas = np.zeros((1220, 1814, 3), dtype=np.uint8)
    generate_array( canvas , size , target_numbers) 
    simulate_movement(canvas, step_size , whole_paths, all_sorted_coordinate , target_numbers , Rl, Rp,  obstacle_coordinate)



if __name__ == '__main__':
    obstacle_coordinate=[]
    target_coordinate=[]
    circle_in_array_not_traget=[]
    upadate_all_coordinate=[]
    size, target_numbers, arrayimage = wholestep3_draw_array_picture()
    target_coordinate, obstacle_coordinate, circle_in_array_not_traget, light_image, upadate_all_coordinate, all_sorted_coordinate = wholestep5_draw_light_image(arrayimage, target_numbers,size)
    whole_paths, step_size = whole_step_6_draw_path(target_coordinate, light_image, size, obstacle_coordinate)
    
    Rl, Rp= 20 ,8  # 光圈半徑和粒子半徑
    whole_step7_simulate_moving(size , target_numbers, whole_paths, all_sorted_coordinate, step_size, Rl, Rp, obstacle_coordinate)
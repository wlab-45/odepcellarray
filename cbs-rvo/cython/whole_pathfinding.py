import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog,simpledialog
import os
import math, random
import copy
import time
import math
import heapq
import math
from simulate_yolo import simulate_yolo
#from cbs import cbs_planning
from ORCA_RVO2 import orca_planning
from functions_cython import wholestep5_draw_light_image, whole_step_6_draw_path, assignment, whole_step7_simulate_moving, collision_or_not, convert_to_grid_coordinates

def create_canvas_and_draw_circles(output_folder, radius, length, width, size, file_i = 30):
    for i in range(file_i):
        # 創建一個黑色畫布
        canvas_width = 1814
        canvas_height = 1220
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        # 設定輸出文本檔案的資料夾路徑
        output_txt_folder = 'C:/Users/Vivo/CGU/odep_cellarray/Cell_Manipulation_Simulation/virtual_particle_points_txt'
        
        # 隨機生成 20 個點座標
        points = []
        while len(points) < 65:
            x = random.randint(0, canvas_width - 25)
            y = random.randint(0, canvas_height - 25)
            if not (x < length * size and y < width * size):
                if all(np.sqrt((x - px)**2 + (y - py)**2) >= 50 for px, py in points):
                    points.append((x, y))

        # 在畫布上畫紅色圓形
        for point in points:
            cv2.circle(canvas, point, radius, (0, 0, 255), -1)  # 使用紅色 (BGR)

        # 確保輸出資料夾存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 輸出結果至指定資料夾
        output_path = os.path.join(output_folder, f'{i}.png')
        cv2.imwrite(output_path, canvas)
        #print(f"結果已輸出至: {output_path}")


        
        # 將座標寫入文本檔案
        output_txt_path = os.path.join(output_txt_folder, f'{i}.txt')
        
        try:
            with open(output_txt_path, 'w') as f:
                for point in points:
                    f.write(f"{point}\n")
            #print(f"座標已輸出至: {output_txt_path}")
        except Exception as e:
            print(f"寫入文件時發生錯誤: {e}")

#step 3
'''def select_png_file():
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
        print('未選擇任何檔案')'''
        
def generate_array(image, size, length, width):
    for i in range(int(length)):
        for j in range(int(width)):
            top_left = (i * size, j * size)  # 左上角座標
            bottom_right = (top_left[0] + size, top_left[1] + size)  # 右下角座標
            # 繪製虛線正方形
            cv2.rectangle(image, top_left, bottom_right, (255, 191, 0), 2)
    return image 

def wholestep3_draw_array_picture(image, Rp):
    #set size of each square arraysize
    root = tk.Tk()
    root.withdraw()
    size = simpledialog.askinteger("Input", "Enter the length of each square in the array:")
    size=int(size)
    length = 9 #simpledialog.askinteger("Input", "Enter the number of columns in the array:")
    width = 5 #simpledialog.askinteger("Input", "Enter the number of rows in the array:")
    length = int(length)
    width = int(width)
    
    # create_canvas_and_draw_circles(output_folder, Rp, length, width, size, file_i = 30)
    #file_path=select_png_file()
    #image=cv2.imread(file_path)
    #file_name = os.path.basename(file_path)
    target_numbers = length * width
    arrayimage =generate_array(image , size, length, width)

    # cv2.imshow('Array of Squares', arrayimage )
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    file_name = 'test_image.png'
    save_arrayimage_directory="C:/Users/Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\array_image"
    array_image_path = os.path.join(save_arrayimage_directory, f'arrayimage_{file_name}')
    # cv2.imwrite(array_image_path, arrayimage)
    #print("array success")
    return size, target_numbers, arrayimage, file_name, length, width

def path_for_batch(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, size, image_width, image_height, step_size, Rl, walkablw_grid_np, obstacle_radius=20):
    start_time = time.time()
    grid_size = size
    # ORCA規劃並以直線距離作為參考路徑優先 (最快，適用於障礙物少的情況)
    final_paths, ORCA_STAIGHT_SUCCESS = orca_planning(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, grid_size, image_width, image_height, step_size, Rl, obstacle_radius, walkablw_grid_np,  mode = 'a_star') #"straight_path")
    if ORCA_STAIGHT_SUCCESS == False:
        print("ORCA規劃失敗，嘗試使用優先time-a*規劃")
        # A*規劃
        final_paths, Astar_SUCCESS = orca_planning(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, grid_size, image_width, image_height, step_size, Rl, obstacle_radius, walkablw_grid, mode = "a_star")
        
        if Astar_SUCCESS == False:
            raise ValueError("A*規劃失敗，無法找到路徑")
            return None
    
    if final_paths is None:
        print("無法找到路徑，請檢查參數或障礙物配置。")
        raise ValueError("無法找到路徑")  
    end_time = time.time()
    print(f"\n⏱️ 總耗時: {end_time - start_time:.2f} 秒")
    return final_paths

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
    
    all_coordinate, image, obstacle_coordinate = simulate_yolo()
    
    size, target_numbers, arrayimage, file_name, columns, rows = wholestep3_draw_array_picture(image, Rp)
    start_time = time.time()
    target_coordinate, light_image, all_sorted_coordinate = wholestep5_draw_light_image(arrayimage, target_numbers, size, file_name, columns, rows, Rl, all_coordinate)

    # 分區(L型)分配座標
    whole_paths = [[] for _ in range(len(target_coordinate))]
    sum_path_length = 0
    sum_path_counts = 0
    batch_size = 0
    matched_target_and_array = assignment(target_coordinate, size, columns, rows)
    for start in range(rows): # rows數= batch數
        obstacle_coordinate_changed_btbatch = copy.deepcopy(obstacle_coordinate)
        copyimage = light_image.copy()
        matched_target_and_array_batch = matched_target_and_array[start]  # 取出對應的批次
        batch_size = columns
        
        # #更新障礙物列表(非批次的粒子)
        before_start = matched_target_and_array[:start]  # start 之前的部分
        after_start = matched_target_and_array[start + 1:]  # start 之後的部分
        obstacle_coordinate_changed_btbatch.extend([destination for batch in before_start for _, destination in batch])
        obstacle_coordinate_changed_btbatch.extend([start_point for batch in after_start for start_point, _ in batch])

        walkablw_grid_np = convert_to_grid_coordinates(image_width,image_height, obstacle_coordinate_changed_btbatch, size, obstacle_radius=20)   #改用numpy且值為0/1
        whole_path_batch_astar= path_for_batch(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, size, image_width, image_height, step_size, Rl, walkablw_grid_np, obstacle_radius= 15)
        whole_step_6_draw_path(batch_size, copyimage, size, obstacle_coordinate_changed_btbatch, file_name, whole_path_batch_astar, step_size, walkablw_grid_np)
        

        max_path_length_in_batch = max(len(path) for path in whole_path_batch_astar)  # 找最長的路徑
        # 批次內的粒子先填充到一樣的長度
        for k in range(len(whole_path_batch_astar)):
            while 0 < len(whole_path_batch_astar[k]) < max_path_length_in_batch:
                whole_path_batch_astar[k].append(whole_path_batch_astar[k][-1])  # 最後一點填充(表一直在陣列終點)
          
        delay_factors = [3.5 , 3, 2.5, 2, 1.5, 1]
        collision = True
        for delay_factor in delay_factors:
            delay_batch_path = copy.deepcopy(whole_path_batch_astar)
            for i in range(len(delay_batch_path)): 
                for m in range( int(1/delay_factor * sum_path_length)):   #可以設定每批開始移動的間隔時間
                    delay_batch_path[i].insert(0, delay_batch_path[i][0])  # 填充第一個點來分批移動

                paths_to_check = []
                paths_to_check.extend(whole_paths) # 添加之前 Batch 的最終 Path (從未在測試迴圈中被修改)
                paths_to_check.extend(delay_batch_path) # 添加當前 Batch 帶 test_delay 的臨時 Pathindex =  sum_path_counts + i  

                    # 檢查是否有碰撞，如遇碰撞則 DELAY_FACTOR - 1
            collision = collision_or_not(paths_to_check, Rl)  
            if not collision:
                truedelay = delay_factor
                break  # 無碰撞，接受當前延遲
            print(f"延遲因子為{delay_factor}時，發生碰撞，嘗試減少延遲因子")
            max_path_length_in_batch = max(len(path) for path in whole_path_batch_astar)
        
        # 將當前批次的路徑添加到整體路徑中
        for i,path in enumerate(whole_path_batch_astar):
            final_path = list(path) # 複製 Path
            for _ in range(int(1/truedelay * sum_path_length)):
                final_path.insert(0, final_path[0])
            idx = sum_path_counts + i
            whole_paths[idx] = final_path
        # 更新總路徑長度和計數
        sum_path_length += max_path_length_in_batch
        sum_path_counts += batch_size

                                
    for i in range(len(whole_paths)):
        if len(whole_paths[i]) == 0:
            print(f"第 {i} 批次路徑為空，無法進行模擬")
            continue
        
    whole_step7_simulate_moving(size, target_numbers, whole_paths, all_sorted_coordinate, step_size, Rl, Rp, obstacle_coordinate, file_name, columns, rows, matched_target_and_array)

    print(f"總耗時: {time.time() - start_time:.2f} 秒")
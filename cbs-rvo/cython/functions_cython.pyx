# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=True
# cython: cdivision=True 

import numpy as np
cimport numpy as cnp
import math
import os
import cv2
import random
import time


from libc.math cimport sqrt
from libc.stdlib cimport rand, srand
from display import display_on_specific_monitor



############################## step3
cdef cnp.ndarray[cnp.uint8_t, ndim=3] generate_array(cnp.ndarray[cnp.uint8_t, ndim=3] image, int size, int length, int width):
    cdef int i, j
    for i in range(int(length)):
        for j in range(int(width)):
            top_left = (i * size, j * size)  # 左上角座標
            bottom_right = (top_left[0] + size, top_left[1] + size)  # 右下角座標
            # 繪製虛線正方形
            cv2.rectangle(image, top_left, bottom_right, (255, 191, 0), 2)
    return image 

############################## step5

cdef list picking_target(list all_sorted_coordinates,int columns,int rows):
    cdef list target_coordinate = []
    cdef int target_numbers = columns * rows

    sorted_available_coordinate = sorted(all_sorted_coordinates, key=get_distance)  # 依距離排序

    # 確保目標數量足夠
    if len(sorted_available_coordinate) >= target_numbers:
        target_coordinate = sorted_available_coordinate[:target_numbers]
    else:
        print(f"可移動粒子數量 {len(sorted_available_coordinate)} 不足，需要至少 {target_numbers} 個")
        target_coordinate = sorted_available_coordinate
    return target_coordinate

cdef cnp.ndarray[cnp.uint8_t, ndim=3] draw_light_image(cnp.ndarray[cnp.uint8_t, ndim=3] arrayimage, list target_coordinate):
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] light_image=arrayimage.copy()
    cdef int good=len(target_coordinate)
    cdef int i
    for i in range(good):
        cv2.circle(light_image, target_coordinate[i] , int(2*7+5),(250,250,255), 10) 
    return light_image    
        
cpdef tuple wholestep5_draw_light_image(cnp.ndarray[cnp.uint8_t, ndim=3] arrayimage, int target_numbers, int size, str file_name, int columns, int rows, int Rl, list all_coordinate):
    # generate 4 list
    cdef list all_sorted_coordinates
    cdef list target_coordinate=[]
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] light_image
    ##set coordinates file path

    #generating all_cooridinate
    all_sorted_coordinates = sorted(all_coordinate, key=get_distance)
    print(f'length of all_coordinate list={len(all_sorted_coordinates)}')
    
    #picking target
    target_coordinate = picking_target(all_sorted_coordinates, columns, rows)
    print(f'length of target_coordinate list={len(target_coordinate)}')
    
    #驗證用
    light_image = draw_light_image(arrayimage, target_coordinate)
    cdef int scale_percent = 70  # 縮放比例
    cdef int width = int(light_image.shape[1] * scale_percent / 100)
    cdef int height = int(light_image.shape[0] * scale_percent / 100)
    cdef tuple dim = (width, height)
    
    #resized_light_image= cv2.resize(light_image, dim, interpolation=cv2.INTER_AREA)
    #cv2.imshow('step5: processed image', resized_light_image )
    #light_image_directory="C:/Users/Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\lightimg_with_particle"
    #light_image_path = os.path.join(light_image_directory, f'light_image_{file_name}')
    #cv2.imwrite(light_image_path, light_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return target_coordinate, light_image, all_sorted_coordinates


############################## step6
# 以rows為子列表的方式組織array
cdef list structure_and_center_of_array (int columns, int rows, int size):
    cdef list array_layers = [[] for _ in range(rows)]
    cdef i,j,k
    for i in range(rows):  #假設每層3格(3*3矩陣)則標號為layer(row) 0,1,2 每層格編號(左至右)
        for j in range(1, columns*2, 2):
            k=i+1
            array_layers[i].append((size *j// 2, size *(2*k-1)//2))
    return array_layers

def get_distance(p):
    return sqrt(<double> p[0]**2 + p[1]**2)

def get_y(p):
    return p[1]

def get_x(p):
    return p[0]

# 分配座標
cpdef list assignment(list target_coordinate, int size, int columns, int rows):
    cdef list matched_target_and_array= [[] for _ in range (rows)] 
    cdef list target_coordinate_batches = [[] for _ in range (rows)]
    cdef int start, j
    
    array = structure_and_center_of_array(columns, rows, size) 
    for start in range(rows):
        if len(target_coordinate) < columns:
            print(f"剩餘 target_coordinate ({len(target_coordinate)}) 少於預期的 {columns} 個")
            break  
        
        #先依y座標排序，再分batch
        target_coordinate = sorted(target_coordinate, key=get_y)# 排序y座標
        target_coordinate_batch = target_coordinate[:columns]
        target_coordinate = target_coordinate[columns:]  # 扣除已經排序的點
        target_coordinate_batches[start]= sorted(target_coordinate_batch, key= get_x)# 排序x座標

        for j in range(columns):
            matched_target_and_array[start].append((target_coordinate_batches[start][j], array[start][j]))
    print(f'length of matched_target_and_array list={len(matched_target_and_array)}')
    return matched_target_and_array

# for a*
cpdef cnp.ndarray[cnp.uint8_t, ndim=2] convert_to_grid_coordinates(int image_width, int image_height, list obstacle_coordinates, int grid_size, int obstacle_radius=25):
    cdef Py_ssize_t num_rows, num_cols
    cdef int grid_y, grid_x

    # 計算網格的行數和列數
    num_rows = (image_height + grid_size - 1) // grid_size
    num_cols = (image_width + grid_size - 1) // grid_size

    # 建立網格，可通行的默認為 True
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] walkable_grid_np = np.ones((num_rows, num_cols), dtype=np.uint8)
    cdef cnp.uint8_t[:, :] walkable_mv = walkable_grid_np

    cdef int ox, oy, start_grid_x, end_grid_x, start_grid_y, end_grid_y
    cdef int grid_center_x, grid_center_y
    cdef int dx, dy

    # 檢查每個障礙物
    for obstacle in obstacle_coordinates:
        ox = obstacle[0] # 確保從 list 中取出的是 int
        oy = obstacle[1] # 確保從 list 中取出的是 int
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
                    walkable_mv[grid_y ,grid_x] = 0  # 0代表false
    
    if image_width % grid_size != 0:
        # 遍歷最右邊一列的所有行
        for grid_y in range(num_rows):
            walkable_mv[grid_y, num_cols - 1] = 0

    # 如果圖片高度不是網格大小的整數倍，將最下面一排標記為不可通行
    if image_height % grid_size != 0:
        # 遍歷最下面一排的所有列
        for grid_x in range(num_cols):
            walkable_mv[num_rows - 1, grid_x] = 0
    return walkable_grid_np

# 路徑尋找
cdef cnp.ndarray[cnp.uint8_t, ndim=3] draw_and_get_paths(cnp.ndarray[cnp.uint8_t, ndim=3] image, list whole_path_batch_astar, int size, cnp.ndarray[cnp.uint8_t, ndim=2] walkable_grid_np):
    cdef int grid_y, grid_x,top_left_x, top_left_y, bottom_right_x, bottom_right_y
    cdef Py_ssize_t y_pixel, x_pixel # 使用 Py_ssize_t 命名更明確是像素座標
    cdef Py_ssize_t grid_size = size # 使用 Py_ssize_t 以與 shape 相容
    cdef Py_ssize_t num_rows, num_cols # 獲取網格的行列數

    # 取得影像尺寸 (使用 Py_ssize_t)
    cdef Py_ssize_t image_height = image.shape[0]
    cdef Py_ssize_t image_width = image.shape[1]
    
    # 獲取網格的行列數 - 放在使用 num_rows 和 num_cols 之前
    num_rows = walkable_grid_np.shape[0]
    num_cols = walkable_grid_np.shape[1]
        
    cdef cnp.uint8_t[:, :] walkable_mv = walkable_grid_np
    
    # 繪製網格
    for y_pixel in range(0, image_height, grid_size):
        # cv2 函數通常可以接受整數座標，但如果擔心型別問題，可以加上 <int> 轉型
        cv2.line(image, (0, y_pixel), (image_width, y_pixel), (200, 200, 200), 1)
    for x_pixel in range(0, image_width, grid_size):
        cv2.line(image, (x_pixel, 0), (x_pixel, image_height), (200, 200, 200), 1)

    # 繪製障礙物（不可通行區域）
    for grid_y in range(num_rows):
        for grid_x in range(num_cols):
            if not walkable_mv[grid_y, grid_x]:  # 若為障礙物
                top_left_x = grid_x * grid_size
                top_left_y = grid_y * grid_size
                bottom_right_x = (grid_x + 1) * grid_size
                bottom_right_y = (grid_y + 1) * grid_size
                cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255,0), -1)  # 紅色填充不可通行區域

    # 繪製所有路徑
    for path in whole_path_batch_astar:
        if path:
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            for j in range(len(path) - 1):
                cv2.line(image, path[j], path[j + 1], color, 2)

    return image

# 主函數
cpdef void whole_step_6_draw_path(int batch_size, cnp.ndarray[cnp.uint8_t, ndim=3] copyimage, int size, list obstacle_coordinate_changed_btbatch, str file_name, list whole_path_batch_astar, int step_size, cnp.ndarray[cnp.uint8_t, ndim=2] walkable_grid_np):
    cdef int scale_percent, width, height
    cdef tuple dim 
    cdef str path_save_directory
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] image_with_paths

    # 繪製移動路徑                                                           
    image_with_paths = draw_and_get_paths(copyimage, whole_path_batch_astar, size, walkable_grid_np)
    scale_percent = 50  
    width = int(image_with_paths.shape[1] * scale_percent / 100)
    height = int(image_with_paths.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_image_with_path= cv2.resize(image_with_paths, dim, interpolation=cv2.INTER_AREA)
    # 顯示影像
    #cv2.imshow("Result of path", resized_image_with_path)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #path_save_directory = "C:/Users/Vivo\\CGU\\odep_cellarray\\Cell_Manipulation_Simulation\\virtual_test_image\\path_image"
    #path_image_path = os.path.join(path_save_directory, f'path_{file_name}')
    #cv2.imwrite(path_image_path, image_with_paths)
    


############################## step7

# 模擬移動 (修改版 - 光圈引導粒子拉動效果)
cdef void simulate_movement(cnp.ndarray[cnp.uint8_t, ndim=3] canvas, int step_size, list whole_paths, list all_particle_coor, int target_numbers, int Rl, int Rp, list obstacle_coordinate, str file_name, list matched_target_and_array):  
    cdef int max_path_length, k, step, i, i_subset, scale_percent,width, height
    cdef cnp.ndarray[cnp.float64_t, ndim=2] moving_particles_current_pos_np, light_targets_np, moving_particles_subset_np, vectors_to_light, unit_vectors, vectors_after_move
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] display_img
    cdef list light_targets, valid_indices, path
    cdef cnp.ndarray[cnp.float64_t, ndim=1] distances_to_light
    cdef tuple dim
    cdef double fps, frame_duration, start_time, elapsed, sleep_time

    # 靜止的粒子座標從 all_particle_coor 的 target_numbers 索引開始
    static_particles_coords = all_particle_coor[target_numbers:]

    max_path_length = 0
    if whole_paths: # 確保 whole_paths 不是空的列表
        max_path_length = max([len(path) for path in whole_paths])

    else:
        print("[警告] whole_paths 為空，無法進行模擬")
        return # 如果沒有路徑，提前結束

    # 因為影片時長由最長路徑決定
    for k in range(len(whole_paths)):
        while len(whole_paths[k]) < max_path_length:
            if whole_paths[k]:  # 確保原始路徑非空，可以複製最後一點
                whole_paths[k].append(whole_paths[k][-1])
            else:
                print(f"[警告] 第{k}個 whole_paths 在填充時為空。")
                pass # 不填充空路徑，依賴後續繪製檢查 path 是否非空

    # 設置影片輸出區
    fps = 30.0
    frame_duration = 1.0 / fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cdef str outputpath = f"C:/Users/Vivo/odepcellarray_fromlab/cbs-rvo/movement_simulation_{file_name}_rvo版.mp4"
    out = cv2.VideoWriter(outputpath, fourcc, 30.0, (canvas.shape[1], canvas.shape[0]))

    if not out.isOpened():
        print(f"Error: Could not open video writer for path {outputpath}")
        return

    # 初始化移動粒子的當前位置列表
    # all_particle_coor 包含了所有粒子的初始位置。
    # 移動粒子是前 target_numbers 個。
    # 複製這些位置，以便在模擬中更新。
    moving_particles_current_pos_np = np.array([list(particle) for batch in matched_target_and_array for (particle, array) in batch], dtype=np.float64)


    print(f"Starting simulation visualization for {len(whole_paths)} moving light circles over {max_path_length} frames.")

    # 模擬循環，遍歷每個影格
    for step in range(max_path_length):
        start_time = time.time()
        display_img = canvas.copy()

        # 繪製靜態元素 (可以保持原樣，或者如果數量巨大，考慮預先繪製)
        for obstacle in obstacle_coordinate:
            cv2.circle(display_img, obstacle, 15, (255, 0, 0), -1)
        for circle in static_particles_coords:
            cv2.circle(display_img, circle, radius=Rp, color=(0, 0, 250), thickness=-1)

        # whole_paths[i][step] 是第 i 個光圈在 step 的位置
        light_targets = []
        valid_indices = [] # 記錄哪些路徑是有效的 (非空且 step 在範圍內)
        for i, path in enumerate(whole_paths):
            if path and step < len(path):
                light_targets.append(path[step])
                valid_indices.append(i)
            elif path: # 如果路徑有效但 step 超出範圍，使用最後一個點
                light_targets.append(path[-1])
                valid_indices.append(i)
            # 如果 path 為空，則不添加到 light_targets，也不在後續處理中包含

        if not light_targets:
            print(f"[警告] 在步驟 {step} 沒有有效的光圈目標位置。")
            continue # 跳過移動物體相關的計算和繪製

        light_targets_np = np.array(light_targets, dtype=np.float64)
        # 只處理有效路徑對應的粒子位置
        moving_particles_subset_np = moving_particles_current_pos_np[valid_indices]

        # 計算從粒子到光圈的向量
        vectors_to_light = light_targets_np - moving_particles_subset_np
        distances_to_light = np.linalg.norm(vectors_to_light, axis=1) # 計算每個向量的長度

        # 避免除以零，處理距離為零的情況
        non_zero_dist_mask = distances_to_light > 1e-6
        unit_vectors = np.zeros_like(vectors_to_light)
        unit_vectors[non_zero_dist_mask] = vectors_to_light[non_zero_dist_mask] / distances_to_light[non_zero_dist_mask][:, np.newaxis] # 廣播除法

        # 計算移動距離
        # 粒子移動距離限制在 step_size 和到光圈的距離之間
        move_distances = np.minimum(distances_to_light, step_size)

        # 更新粒子位置 (使用浮點數)
        moving_particles_subset_np += unit_vectors * move_distances[:, np.newaxis]

        # 確保粒子在光圈影響範圍內 (Rl-Rp)
        max_pull_distance_from_light_center = Rl - Rp

        # 再次計算移動後粒子到光圈的距離
        vectors_after_move = moving_particles_subset_np - light_targets_np
        distances_after_move = np.linalg.norm(vectors_after_move, axis=1)

        # 找出超出範圍的粒子
        out_of_range_mask = distances_after_move > max_pull_distance_from_light_center

        # 對超出範圍的粒子進行拉回
        if np.any(out_of_range_mask):
            out_of_range_indices = np.where(out_of_range_mask)[0]
            # 避免除以零
            valid_pull_back_mask = distances_after_move[out_of_range_indices] > 1e-6
            valid_pull_back_indices = out_of_range_indices[valid_pull_back_mask]

            if valid_pull_back_indices.size > 0:
                scale = max_pull_distance_from_light_center / distances_after_move[valid_pull_back_indices]
                # 將粒子位置拉回到邊界上
                moving_particles_subset_np[valid_pull_back_indices] = light_targets_np[valid_pull_back_indices] + vectors_after_move[valid_pull_back_indices] * scale[:, np.newaxis]


        # 更新主粒子位置陣列
        moving_particles_current_pos_np[valid_indices] = moving_particles_subset_np

        # 繪製移動的光圈和粒子
        for i_subset, i_original in enumerate(valid_indices):
            light_coor = light_targets_np[i_subset]
            particle_coor = moving_particles_current_pos_np[i_original] # 使用原始索引獲取最新位置
            # 繪製光圈
            cv2.circle(display_img, (int(round(light_coor[0])), int(round(light_coor[1]))), radius=Rl, color=(250, 250, 250), thickness=10)
            # 繪製粒子
            #cv2.circle(display_img, (int(round(particle_coor[0])), int(round(particle_coor[1]))), Rp, (0, 0, 250), -1)


        #scale_percent = 50
        #width = int(display_img.shape[1] * scale_percent / 100)
        #height = int(display_img.shape[0] * scale_percent / 100)
        #dim = (width, height)
        #resized_image = cv2.resize(display_img, dim, interpolation=cv2.INTER_AREA)

        display_on_specific_monitor(display_img, target_monitor_index=1)
        #cv2.imshow('Movement Simulation', resized_image)
        out.write(display_img) # 注意這裡仍然寫入全尺寸圖像

        # 等待最小時間避免程序卡死且能捕捉鍵盤事件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Simulation interrupted by user.")
            break

        elapsed = time.time() - start_time
        sleep_time = frame_duration - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

    # 釋放影片寫入器並關閉視窗 (保持不變)
    out.release()
    cv2.destroyAllWindows()
    print(f"Simulation video saved to {outputpath}")


# step7 主函數    
cpdef void whole_step7_simulate_moving(int size, int target_numbers, list whole_paths, list all_sorted_coordinate, int step_size, int Rl, int Rp, list obstacle_coordinate, str file_name, int columns, int rows, list matched_target_and_array):
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] canvas, ar_img
    canvas = np.zeros((1220, 1814, 3), dtype=np.uint8)
    ar_img = generate_array(canvas, size, columns, rows)
    simulate_movement(ar_img, step_size, whole_paths, all_sorted_coordinate, target_numbers, Rl, Rp, obstacle_coordinate, file_name, matched_target_and_array)

#檢查現有路徑是否會有碰撞發生
cpdef bint collision_or_not(list whole_path, int Rl):
    cdef int max_path_length, step, idx, num_active
    cdef double collision_threshold_sq = (2*(Rl+5))**2.0
    cdef list active_points, path
    cdef cnp.ndarray[cnp.float64_t, ndim=2] active_points_np, sq_distances
    cdef object collision_mask 
    cdef cnp.ndarray[cnp.float64_t, ndim=3] vectors_diff

    max_path_length = max([len(path) for path in whole_path])

    for step in range(max_path_length):  
        active_points = []
        for idx, path in enumerate(whole_path):
            if step < len(path):  
                active_points.append(path[step])
            elif path: # If step is beyond path length, use the last point
                 active_points.append(path[-1])
        
        # 如果活動粒子少於 2 個，不需要檢查
        if len(active_points) < 2:
            print(f"第 {step} 時間步的粒子數量為 {len(active_points)}，不需要檢查")
            continue
        
        active_points_np = np.array(active_points, dtype=np.float64)
        vectors_diff = active_points_np[:, np.newaxis, :] - active_points_np[np.newaxis, :, :]
        sq_distances = np.sum(vectors_diff**2, axis=-1)
        collision_mask = sq_distances < collision_threshold_sq
        num_active = len(active_points_np)
        upper_triangle_indices = np.triu_indices(num_active, k=1)
        
        if np.any(collision_mask[upper_triangle_indices]):
            # 偵測到至少一對粒子碰撞 
            return True # 發生碰撞，立即返回 True
    return False  # 沒有碰撞




# run:  python setup.py build_ext --inplace

# cython: boundscheck=False, wraparound=True
# cython: cdivision=True 
# cython: nonecheck=False 
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free
import heapq

# for astar計算 (no 並行)
cdef double _g_scores(int current_x, int current_y, int neighbor_x, int neighbor_y):
    # 計算實際的 x 和 y 差距（絕對值）
    cdef double dx = fabs(current_x - neighbor_x) 
    cdef double dy = fabs(current_y - neighbor_y)
    if dx == dy: # 對角線移動
        return 14.0 * dx # 使用浮點數常數確保浮點計算
    else:
        return 10.0 * (dx + dy) # 使用浮點數常數確保浮點計算

cdef double heuristic_with_obstacles(int current_x, int current_y, int target_x, int target_y):
    cdef double dx = fabs(target_x - current_x)
    cdef double dy = fabs(target_y - current_y)
    # 使用 Euclidean 距離，與原 heuristic 函數邏輯一致
    cdef double base_distance = 10.0 * sqrt(dx*dx + dy*dy) # 使用浮點數常數確保浮點計算
    return base_distance

cpdef list a_star(int start_grid_x, int start_grid_y, int target_grid_x, int target_grid_y, cnp.ndarray[cnp.uint8_t, ndim=2] walkable_grid_np):
    # 回傳值為 int 的 NumPy 陣列 (M, 2)，存放網格座標點

    cdef Py_ssize_t num_rows = walkable_grid_np.shape[0]
    cdef Py_ssize_t num_cols = walkable_grid_np.shape[1]
    cdef cnp.uint8_t[:, :] walkable_mv = walkable_grid_np

    # 檢查邊界是否超出範圍
    if start_grid_x < 0 or start_grid_y < 0 or target_grid_x >= num_cols or target_grid_y >= num_rows:
        raise ValueError("Start or target grid out of bounds")

    # 檢查起始點和目標點是否在可通行區域內 (使用記憶體視圖存取)
    if not walkable_mv[start_grid_y, start_grid_x]:
        print("起始點不在可通行區域內，將直接設置為可行走區域")
        # 直接將起始點設置為可行 (當初始階段，障礙物與其起始點剛好在同一格)
        walkable_mv[start_grid_y, start_grid_x] = 1 # 初始起點皆設置為可行(避免發生障礙物與起點在同個grid的情況)
    if  not walkable_mv[target_grid_y, target_grid_x]:
        print(f"目標點網格座標({target_grid_x}, {target_grid_y})不在可通行區域內")
        return []  # 如果目標是障礙物，直接返回空路徑

    
    # G/F 分數可以使用 double 陣列，父節點使用一個 int 陣列來存儲前一個節點的「扁平化」索引 (y * num_cols + x)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] g_score_np = np.full((num_rows, num_cols), np.inf, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] f_score_np = np.full((num_rows, num_cols), np.inf, dtype=np.float64)
    
    # 存儲父節點的展平索引，-1 表示無父節點
    cdef cnp.ndarray[cnp.intp_t, ndim=2] came_from_np = np.full((num_rows, num_cols), -1, dtype=np.intp)

    # 改為記憶體視圖 (C 級別快速存取)
    cdef double[:, :] g_score_mv = g_score_np
    cdef double[:, :] f_score_mv = f_score_np
    cdef cnp.intp_t[:, :] came_from_mv = came_from_np

    # 設置起點分數
    g_score_mv[start_grid_y, start_grid_x] = 0.0
    f_score_mv[start_grid_y, start_grid_x] = heuristic_with_obstacles(start_grid_x, start_grid_y, target_grid_x, target_grid_y)

    # 使用計數器解決f_score相同的情況
    cdef int counter = 0

    # Open Set : Python 的 heapq，存放 (f_score, counter, x, y) 元組
    cdef list open_set = []
    # 將起點加入優先隊列 
    heapq.heappush(open_set, (f_score_mv[start_grid_y, start_grid_x], counter, start_grid_x, start_grid_y))
    counter += 1 # 第一次push後增加counter

    # Closed Set: 2D boolean陣列
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] closed_set_np = np.zeros((num_rows, num_cols), dtype=np.uint8)
    cdef cnp.uint8_t[:, :] closed_set_mv = closed_set_np

    # 定義相鄰節點的相對位置
    # 八個方向的相對位置
    cdef int[:] directions_x = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int32)
    cdef int[:] directions_y = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)

    cdef int current_x, current_y
    cdef double current_f, tentative_g_score, neighbor_heuristic
    cdef int neighbor_x, neighbor_y
    cdef Py_ssize_t k # 遍歷方向的索引
    cdef tuple current_item
    # 找到路徑，重建路徑
    cdef list path_grid_coords = [] # 存放網格座標元組
    cdef int path_x 
    cdef int path_y 
    cdef int parent_flat_idx
    cdef int parent_x, parent_y

    # A* 主迴圈
    while open_set:
        # 從優先隊列取出 f 分數最低的節點
        current_item = heapq.heappop(open_set)
        current_x = current_item[2]
        current_y = current_item[3]

        # 使用 memoryview 檢查 Closed Set (C 級別速度)
        if closed_set_mv[current_y, current_x]:
            continue

        # 將當前節點加入 Closed Set
        closed_set_mv[current_y, current_x] = 1

        # 檢查是否到達目標
        if current_x == target_grid_x and current_y == target_grid_y:
            path_x = current_x
            path_y = current_y
            # 從目標點回溯到起點
            while True: 
                path_grid_coords.append((path_x, path_y)) # 添加當前節點 (元組)
                parent_flat_idx = came_from_mv[path_y, path_x] # 讀取父節點的展平索引 ， 展平結構為(0,0), (0,1), ..., (0,C-1), (1,0), (1,1), ..., (1,C-1), ..., (R-1,0), ..., (R-1,C-1)     R= rows C=columns

                if parent_flat_idx == -1: # 如果是起點 (-1)
                    break

                # 從展平索引計算父節點的 x, y 座標
                parent_y = parent_flat_idx // num_cols # 使用 C 整數除法
                parent_x = parent_flat_idx % num_cols

                path_x = parent_x
                path_y = parent_y

            path_grid_coords.reverse() # 將路徑反轉 (Python 列表方法)
            return path_grid_coords

        # 遍歷鄰居節點 (8方向)
        for k in range(8): # 8 個方向
            neighbor_x = current_x + directions_x[k]
            neighbor_y = current_y + directions_y[k]

            # 檢查鄰居是否越界
            if neighbor_x < 0 or neighbor_y < 0 or neighbor_x >= num_cols or neighbor_y >= num_rows:
                 continue

            # 檢查鄰居是否已在 Closed Set (boolean array)
            if closed_set_mv[neighbor_y, neighbor_x]:
                 continue

            # 檢查鄰居是否可通行 
            if not walkable_mv[neighbor_y, neighbor_x]:
                 continue

            # 檢查斜向移動是否被阻擋 
            if directions_x[k] != 0 and directions_y[k] != 0:
                 # 檢查兩個直角鄰居是否都不可通行
                 if not walkable_mv[current_y, neighbor_x] or not walkable_mv[neighbor_y, current_x]:
                      continue # 如果兩側直向不可通行，這條斜向就不能走

            # 計算到鄰居的 G 分數
            tentative_g_score = g_score_mv[current_y, current_x] + _g_scores(current_x, current_y, neighbor_x, neighbor_y)

            # 比較 Tentative G 分數與已知的 G 分 
            if tentative_g_score < g_score_mv[neighbor_y, neighbor_x]:
                # 找到更好的路徑，更新信息
                came_from_mv[neighbor_y, neighbor_x] = current_y * num_cols + current_x # 記錄父節點的展平索引
                g_score_mv[neighbor_y, neighbor_x] = tentative_g_score # 更新 G 分數
                # 計算並更新 F 分數
                neighbor_heuristic = heuristic_with_obstacles(neighbor_x, neighbor_y, target_grid_x, target_grid_y)
                f_score_mv[neighbor_y, neighbor_x] = g_score_mv[neighbor_y, neighbor_x] + neighbor_heuristic
                heapq.heappush(open_set, (f_score_mv[neighbor_y, neighbor_x],counter, neighbor_x, neighbor_y))
                counter += 1
    # 若優先隊列為空且未找到路徑
    return []

cdef list convert_to_pixel_coordinates(int particle_x, int particle_y, int goal_x, int goal_y, list path, int grid_size, int step_size):
    cdef list actual_path = []
    cdef list trimmed_path, initial_segment, interpolated,final_segment
    cdef int first_grid_point_x, first_grid_point_y, start_index, grid_index, current_x, current_y, next_point_x, next_point_y,last_grid_point_x, last_grid_point_y

    if path:
        actual_path = []
        trimmed_path = path[:-1]  # 不包含最後一點
        
        # 處理起點到第一個網格點的路徑
        if len(trimmed_path) > 1:
            # **直接從起點插值到第二個網格的中心點**(避免路徑返回問題)
            first_grid_point_x = trimmed_path[1][0] * grid_size + grid_size//2
            first_grid_point_y = trimmed_path[1][1] * grid_size + grid_size//2
            initial_segment = interpolate_path(particle_x, particle_y, first_grid_point_x, first_grid_point_y , step_size)[:-1]  # 不包含终点，避免重复
            actual_path.extend(initial_segment)
            start_index = 1  # 確保後續從 path[1] 開始
                        
            # **處理中間路徑點**
            for grid_index in range(start_index, len(trimmed_path)-1):  # **從 start_index 開始，避免重複 trimmed_path[0]**
                current_x = trimmed_path[grid_index][0] * grid_size + grid_size//2
                current_y = trimmed_path[grid_index][1] * grid_size + grid_size//2
                next_point_x = trimmed_path[grid_index+1][0] * grid_size + grid_size//2
                next_point_y =  trimmed_path[grid_index+1][1] * grid_size + grid_size//2
                interpolated = interpolate_path(current_x,current_y, next_point_x, next_point_y, step_size)[:-1]  
                actual_path.extend(interpolated)
            
            # 處理最後一段到目標點的路徑
            last_grid_point_x = trimmed_path[-1][0] * grid_size + grid_size//2
            last_grid_point_y = trimmed_path[-1][1] * grid_size + grid_size//2
            final_segment = interpolate_path(last_grid_point_x, last_grid_point_y, goal_x, goal_y, step_size)
            actual_path.extend(final_segment)
            
        else:
            # **如果 trimmed_path 只有一個點，則只能插值到這個點** 
            actual_path = interpolate_path(particle_x, particle_y, goal_x, goal_y, step_size)
        return actual_path
    else:
        print(f"無法從粒子 {(particle_x, particle_y)} 到達目標 {(goal_x, goal_y)}")
        return []
    
# 網格轉step_size
cdef list interpolate_path(int start_x, int start_y, int end_x, int end_y, int step_size):
    cdef int dx = end_x - start_x
    cdef int dy = end_y - start_y
    cdef double all_length = sqrt(dx**2 + dy**2)
    cdef int step_idx

    if all_length == 0:  # 如果起點終點一樣
        return [(start_x, start_y)]
        
    cdef int step_count = <int>(all_length / step_size)
    if step_count == 0:  # 如果距離太短
        return [(start_x, start_y), (end_x, end_y)]
    
    interpolated_points = [(start_x, start_y)]
    cdef double step_x = dx / step_count
    cdef double step_y = dy / step_count
    
    for step_idx in range(1, step_count + 1):
        next_point_x = round(start_x + step_x * step_idx)
        next_point_y = round(start_y + step_y * step_idx)
        interpolated_points.append((next_point_x, next_point_y))
        
    if not (interpolated_points and interpolated_points[-1] == (end_x, end_y)):
        interpolated_points.append((end_x, end_y))
    return interpolated_points




cpdef list get_all_astar_path(double[:, :] START_POSITIONS, double[:, :] REAL_GOAL_POSITIONS, cnp.ndarray[cnp.uint8_t, ndim=2] walkable_grid_np, int grid_size):
    cdef Py_ssize_t i, num_of_agent = START_POSITIONS.shape[0]
    cdef int start_x, start_y, goal_x, goal_y
    cdef list whole_paths = []
    cdef list a_star_path, pixel_path
    if num_of_agent == REAL_GOAL_POSITIONS.shape[0]:
        for i in range (num_of_agent):
            start_grid_x = <int>START_POSITIONS[i, 0]//grid_size
            start_grid_y = <int>START_POSITIONS[i, 1]//grid_size
            goal_grid_x = <int>REAL_GOAL_POSITIONS[i, 0]//grid_size
            goal_grid_y = <int>REAL_GOAL_POSITIONS[i, 1]//grid_size
            a_star_path = a_star(start_grid_x, start_grid_y, goal_grid_x, goal_grid_y, walkable_grid_np)
            pixel_path = convert_to_pixel_coordinates(<int>START_POSITIONS[i,0], <int>START_POSITIONS[i,1], <int>REAL_GOAL_POSITIONS[i, 0], <int>REAL_GOAL_POSITIONS[i,1], a_star_path, grid_size, step_size=3)
            whole_paths.append(pixel_path)
    else:
        raise ValueError("a*輸入結構錯誤.")    
    return whole_paths

# run:  python setup.py build_ext --inplace
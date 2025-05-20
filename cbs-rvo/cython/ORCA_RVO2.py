import numpy as np
import rvo2
import time  
import math
import random
import cv2
import heapq
from astar_path import get_all_astar_path

def check_midpoint_and_obstacle_np(mid_point, OBSTACLE_CENTERS, grid_size):
    new_x, mid_y = mid_point
    dx = np.abs(OBSTACLE_CENTERS[:, 0] - new_x)
    dy = np.abs(OBSTACLE_CENTERS[:, 1] - mid_y)
    mask = (dx < grid_size) & (dy < grid_size)

    if np.any(mask):
        idx = np.argmax(mask)  # first match
        print(f"Obstacle {idx} overlaps with mid-point {mid_point}.")
        return OBSTACLE_CENTERS[idx]
    return None

def generate_mid_points_to_goal(OBSTACLE_CENTERS, REAL_GOAL_POSITIONS, grid_size):
    y_of_goal = REAL_GOAL_POSITIONS[0][1] # 因為是整rows，所以y值都是相同的
    mid_points =  [[] for _ in range(len(REAL_GOAL_POSITIONS))]
    max_iterations = 3
    iterations = 0
    for i in range(len(REAL_GOAL_POSITIONS)):
        mid_point = (30 + i* 80, y_of_goal + grid_size)  
        while True and iterations < max_iterations:
            iterations += 1
            obs = check_midpoint_and_obstacle_np(mid_point, OBSTACLE_CENTERS, grid_size)
            if obs is None:
                break
            else:
                if iterations ==1:
                    mid_point[1] = mid_point[1] - grid_size
                else: 
                    mid_point = REAL_GOAL_POSITIONS[i]
                continue
            
        mid_points[i] = mid_point
    return mid_points

def straight_path(current_point, target_center, step_size):
    #whole_paths = [[] for _ in range(len(matched_target_and_array))]
    path = []
    #for i, (current_point, target_center) in enumerate(matched_target_and_array):
    dx = target_center[0] - current_point[0]
    dy = target_center[1] - current_point[1]
    all_length = math.sqrt(dx**2 + dy**2)
    
    if all_length == 0:  # 如果起點終點一樣，直接連到終點
        path.append(current_point)
        return []

    step_count = int(all_length / step_size)  
    if step_count == 0:  # 如果距離太短，直接連到終點
        path.append(current_point)
        path.append(target_center)
        return path
    
    step_x = dx / step_count 
    step_y = dy / step_count  
    path.append(current_point)
    
    for step_idx in range(1, step_count + 1): 
        next_point = (
            round(current_point[0] + step_x * step_idx),
            round(current_point[1] + step_y * step_idx)
        )
        path.append(next_point)
    
    if path[-1] != target_center:
       path.append(target_center)    
    return path

def generate_circle_obstacles_np(centers, radius, num_points=16):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    unit_circle = np.stack((np.cos(angles), np.sin(angles)), axis=1)  # shape: (num_points, 2)

    obstacles = []
    for center in centers:
        shifted_circle = unit_circle * radius + center  # broadcasting
        vertices = shifted_circle.tolist()
        vertices.append(vertices[0])  # close the polygon
        obstacles.append(vertices)
    return obstacles

'''def a_star(start_grid, target_grid, walkable_grid, grid_size):
    
    if start_grid[0] < 0 or start_grid[1] < 0 or target_grid[0] >= len(walkable_grid[0]) or target_grid[1] >= len(walkable_grid):
        raise ValueError("Start or target grid out of bounds")
    
    #  檢查起始點和目標點是否在可通行區域內
    if not walkable_grid[start_grid[1]][start_grid[0]] or not walkable_grid[target_grid[1]][target_grid[0]]:
        print("起始點或目標點不在可通行區域內")
        walkable_grid[start_grid[1]][start_grid[0]] = True #初始起點皆設置為可行(避免發生障礙物與起點在同個grid的情況)
    
    #A* 路徑規劃 
    def g_scores(current, neighbor):
        # 計算實際的 x 和 y 差距
        dx = abs(current[0] - neighbor[0])
        dy = abs(current[1] - neighbor[1])
        if dx == dy:  # 對角線移動
            return 14 * dx  # 每個對角移動成本為 14
        else:
            return 10 * (dx + dy)  # 直線移動成本為 10

    def heuristic_with_obstacles(current, target):
        # h(n) 優化版本
        dx = abs(target[0] - current[0])
        dy = abs(target[1] - current[1])
        base_distance = 10 * math.sqrt(dx**2 + dy**2)  # 使用曼哈頓距離，與 g(n) 保持相似比例
        return base_distance 

    def get_neighbors(current_grid, walkable_grid):
        # 取得鄰居節點
        x, y = current_grid
        # 定義相鄰節點的相對位置
        directions = [
            (-1, 0), (-1, -1), (0, -1), (-1, 1),
            (0, 1), (1, 0), (1, -1), (1, 1)
        ]
        neighbors = []
        for dx, dy in directions:
            # 移除 grid_size 相乘
            neighbor_x = x + dx
            neighbor_y = y + dy
            # 確保鄰居在邊界內且是可通行的
            if not (0 <= neighbor_x < len(walkable_grid[0]) and 0 <= neighbor_y < len(walkable_grid)):
                continue
            # 檢查是否可通行
            if not walkable_grid[neighbor_y][neighbor_x]:
                continue
            # 如果是斜向移動，還要確認兩側直向是否也可通行
            if dx != 0 and dy != 0:
                if not walkable_grid[y][neighbor_x] or not walkable_grid[neighbor_y][x]:
                    continue  # 如果兩側直向不可通行，這條斜向就不能走
            neighbors.append((neighbor_x, neighbor_y))
        return neighbors        
        
    open_set = []
    closed_set = set()
    came_from = {}
    g_score = {start_grid: 0}
    h_score = heuristic_with_obstacles(start_grid, target_grid)
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
                f_score[neighbor] = g_score[neighbor] + heuristic_with_obstacles(neighbor, target_grid)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
    return []  # 若沒有找到路徑，返回空列表'''

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

'''def get_all_astar_path(START_POSITIONS,REAL_GOAL_POSITIONS,walkable_grid, grid_size):
    num_of_agent = len(START_POSITIONS)
    whole_paths = []

    if num_of_agent == len(REAL_GOAL_POSITIONS):
        for i in range (num_of_agent):
            print(f"Agent {i} start: {(START_POSITIONS[i][0]//grid_size, START_POSITIONS[i][1]//grid_size)}, goal: {(REAL_GOAL_POSITIONS[i][0]//grid_size, REAL_GOAL_POSITIONS[i][1]//grid_size)}")
            a_star_path = a_star((START_POSITIONS[i][0]//grid_size, START_POSITIONS[i][1]//grid_size), (REAL_GOAL_POSITIONS[i][0]//grid_size, REAL_GOAL_POSITIONS[i][1]//grid_size), walkable_grid, grid_size)
            pixel_path = convert_to_pixel_coordinates(START_POSITIONS[i],REAL_GOAL_POSITIONS[i], a_star_path, grid_size, step_size=3)
            whole_paths.append(pixel_path)
    else:
        raise ValueError("a*輸入結構錯誤.")    
    return whole_paths'''

def main(NUM_CIRCLES, TIME_STEP, NEIGHBOR_DIST, TIME_HORIZON, CIRCLE_RADIUS, MAX_SPEED, START_POSITIONS, OBSTACLE_CENTERS, OBSTACLE_RADIUS, SCENE_HEIGHT, SCENE_WIDTH, GOAL_POSITIONS, grid_size, walkable_grid_np, mode):
    all_agent_paths = [[] for _ in range(NUM_CIRCLES)]
    # 初始化 RVO2 模擬器
    sim = rvo2.PyRVOSimulator(TIME_STEP, NEIGHBOR_DIST, 5, TIME_HORIZON, 3, CIRCLE_RADIUS, MAX_SPEED) # 5: 一個代理在避碰計算時最多考慮的鄰居數量, timeHorizon: 代理間避碰的時間視界, timeHorizonObst (腳本中為 2): 代理與靜態障礙物避碰的時間視界
    SUCCESS = True
    
    # 添加光圈
    agents = []
    for pos in START_POSITIONS:   # 由左至右
        agent_id = sim.addAgent(tuple(pos))
        agents.append(agent_id)
    
    obstacles = generate_circle_obstacles_np(OBSTACLE_CENTERS, OBSTACLE_RADIUS)
    for poly in obstacles:
        sim.addObstacle(poly)
    
    # 將邊界全部設置為靜態障礙物 (不可超出邊界)
    # 左邊界
    sim.addObstacle([(0, 0), (0, SCENE_HEIGHT)])
    # 下邊界
    sim.addObstacle([(0, SCENE_HEIGHT), (SCENE_WIDTH, SCENE_HEIGHT)])
    # 右邊界
    sim.addObstacle([(SCENE_WIDTH, SCENE_HEIGHT), (SCENE_WIDTH, 0)])
    # 上邊界
    sim.addObstacle([(SCENE_WIDTH, 0), (0, 0)])
    sim.processObstacles()
    
    # 設置直線路徑作為參考路徑
    whole_paths = []
    if mode == "straight_path":
        for i, agent in enumerate(agents):
            start = START_POSITIONS[i]
            goal = GOAL_POSITIONS[i]
            pixel_path = straight_path(start, goal, step_size=3)
            whole_paths.append(pixel_path)
    elif mode == "a_star":
        start_pixel_positions_np = np.array(START_POSITIONS, dtype=np.float64)
        target_pixel_positions_np = np.array(GOAL_POSITIONS, dtype=np.float64) 
        whole_paths = get_all_astar_path(start_pixel_positions_np, target_pixel_positions_np, walkable_grid_np, grid_size)
        '''image = np.zeros((SCENE_HEIGHT, SCENE_WIDTH, 3), dtype=np.uint8)
            
        # 繪製網格
        for y in range(0, SCENE_HEIGHT, grid_size):
            cv2.line(image, (0, y), (SCENE_WIDTH, y), (200, 200, 200), 1)  # 橫線
        for x in range(0, SCENE_WIDTH, grid_size):
            cv2.line(image, (x, 0), (x, SCENE_HEIGHT), (200, 200, 200), 1)  # 縱線
        for path in whole_paths:
            
            if path:
                color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                # 確保每個點都是整數座標
                path = [tuple(map(int, point)) for point in path]
                for j in range(len(path) - 1):
                    cv2.line(image, (path[j]), path[j+1], color, 2)
                    
        scale_percent = 70  # 缩放比例
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        resized_image_with_path= cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        # 顯示影像
        cv2.imshow("A* Path", resized_image_with_path)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        agents_path_indices = [1] * NUM_CIRCLES
        
    agents_reached_final_goal = [False] * NUM_CIRCLES # 追蹤是否到達最終目標
    step = 0
    
    precise_arrival_threshold = 3
    # 模擬循環
    while any(np.linalg.norm(np.array(sim.getAgentPosition(agents[i])) - np.array(GOAL_POSITIONS[i])) >= precise_arrival_threshold for i in range(NUM_CIRCLES)) and step < 2000:
        for i, agent in enumerate(agents):
            pos = np.array(sim.getAgentPosition(agent))
            final_goal = np.array(GOAL_POSITIONS[i])
            dist_to_goal = np.linalg.norm(final_goal - pos) # 計算代理當前位置到最終目標的距離
            path = whole_paths[i]
            progress = step * TIME_STEP * MAX_SPEED * 1.5 / np.linalg.norm(np.array(path[-1]) - np.array(path[0]))  
            # "step * TIME_STEP * MAX_SPEED"這部分估計了在 step 個時間步內，如果代理一直以最大速度移動，它所能行駛的最大距離。
            # 1.2: 這是一個額外的係數，用於稍微加快進度的估計，可能為了讓代理更快地沿著參考路徑移動
            # progress: 通過將估計的已行駛距離除以總路徑長度，得到一個介於 0 和可能大於 1 的進度值，表示代理在參考路徑上的位置
            idx = min(int(progress * len(path)), len(path) - 1) # 這行程式碼根據計算出的 progress 來確定代理在參考路徑上應該朝向的目標點的索引
            target = np.array(path[idx]) # 最為目標參考點
            direction = target - pos # 計算當前代理位置到目標參考點的方向向量
            dist = np.linalg.norm(direction) + 1e-6 # 計算代理當前位置到目標點的距離
              
            arrival_threshold = 80
            arrival_threshold_slowdown = 20 # 更大的減速區域

            if dist_to_goal < arrival_threshold:
                sim.setAgentPrefVelocity(agent, (0, 0)) 
                agents_reached_final_goal[i] = True
            elif dist_to_goal < arrival_threshold_slowdown:
                # 在減速區域內，根據距離調整速度
                slowdown_factor = max(0.1, dist_to_goal / arrival_threshold_slowdown)
                desired_speed = min(MAX_SPEED, dist / TIME_STEP) * slowdown_factor
                velocity = direction / (dist) * desired_speed
                sim.setAgentPrefVelocity(agent, tuple(velocity))
            else:
                velocity = direction / dist * min(MAX_SPEED, dist / TIME_STEP) #  計算期望的速度大小。這個速度是代理希望朝向目標移動的速度，但不會超過其最大速度 (MAX_SPEED)。同時，它也考慮了代理在一個時間步內能夠行駛的最大距離 (dist / TIME_STEP)，以避免產生過大的速度。
                sim.setAgentPrefVelocity(agent, tuple(velocity))
        step += 1
        sim.doStep()

        # --- 在 sim.doStep() 之後，強制將已到達終點的粒子位置設定回精確終點 ---
        for i, agent_id in enumerate(agents):
            if agents_reached_final_goal[i] == True:
                rvo2_stop_pos = sim.getAgentPosition(agent_id) # 設定在進入到終點網格實踐暫停rvo2規劃，改用硬編碼(其他agent仍會根據手動編碼結果繞過)
                assert np.linalg.norm(np.array(rvo2_stop_pos) - np.array(GOAL_POSITIONS[i])) <= 130 , f"Agent {i} reached goal but position is not close to goal: {rvo2_stop_pos} vs {GOAL_POSITIONS[i]}" #int(grid_size//2 * 1.5)+1
                late_path = straight_path(rvo2_stop_pos, GOAL_POSITIONS[i], step_size = 2)
                if len(late_path) > 1:
                    sim.setAgentPosition(agent_id, late_path[1])
                    # else: 如果 late_path 是 None 或只有一個點，表示代理已經非常接近或就在中間目標點了，不做額外移動
                else:
                    sim.setAgentPosition(agent_id, tuple(GOAL_POSITIONS[i]))
                    
                    
        # 在每次 sim.doStep() 之後，記錄每個代理的當前位置
        for i, agent_id in enumerate(agents):
            current_pos = sim.getAgentPosition(agent_id)
            all_agent_paths[i].append(current_pos)
        
        # 最大次數限制
        if step >= 2000: 
            print("Simulation reached maximum steps without all agents reaching the goal.")
            SUCCESS = False
            break
        
    # 確定所有代理都抵達終點  
    for i, path in enumerate(all_agent_paths):
        if path[-1] != GOAL_POSITIONS[i]:
           all_agent_paths[i].append(GOAL_POSITIONS[i]) 
    
    return all_agent_paths, SUCCESS

def complete_path(all_agent_paths, REAL_GOAL_POSITIONS, OBSTACLE_CENTERS,obstacle_radius):
    
    def check_if_obstacles(OBSTACLE_CENTERS, left_upper_point, right_lower_point, obstacle_radius=15):
        x = OBSTACLE_CENTERS[:, 0]
        y = OBSTACLE_CENTERS[:, 1]

        in_x = (x >= 0) & (x <= right_lower_point[0] + obstacle_radius)
        in_y = (y >= left_upper_point[1] - obstacle_radius) & (y <= right_lower_point[1] + obstacle_radius)

        return np.any(in_x & in_y)

    if check_if_obstacles(OBSTACLE_CENTERS, all_agent_paths[0][-1], all_agent_paths[-1][-1], obstacle_radius)== True:
        print("Obstacle exists in the area.")
        return all_agent_paths
    else:        
        for i, path in enumerate(all_agent_paths): # 由左至右
            if len(path) > 0:
                last_point = path[-1]
                real_goal = REAL_GOAL_POSITIONS[i] # 由左至右
                mid_point = (real_goal[0], last_point[1])
                #path1 = straight_path(last_point, mid_point, step_size=3)
                path2 = straight_path(last_point, real_goal, step_size=3)
                if path2 is not None:
                    all_agent_paths[i] = all_agent_paths[i]+ path2
            else: 
                print(f"Agent {i} has no path.")
                raise ValueError("Agent{i}'s path is empty.")            
    return all_agent_paths

def orca_planning(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, grid_size, image_width, image_height, step_size, Rl, obstacle_radius, walkable_grid_np, mode):
    assert mode in ["straight_path", "a_star"], f"Invalid mode: {mode}. Choose 'straight_path' or 'a_star'." 
    # 場景參數
    SCENE_WIDTH = image_width  # 像素
    SCENE_HEIGHT = image_height
    NUM_CIRCLES = len(matched_target_and_array_batch)  # 光圈數量
    CIRCLE_RADIUS = Rl+10  # 光圈半徑（直徑 50 像素）
    OBSTACLE_RADIUS = 30 # obstacle_radius  # 障礙物半徑（直徑 30 像素）
    NEIGHBOR_DIST = 80 #(CIRCLE_RADIUS + 10) * 2  #80  # RVO2 鄰居距離 代理檢測其他代理的最大距離
    TIME_HORIZON = 1.5 # 代理人對其他代理人做出避碰行為時，考慮的未來時間範圍（秒數）
    TIME_STEP = 1 / 30  # 每幀 1/30 秒（30fps） 每個模擬步的時間長度
    MAX_SPEED = 1/(TIME_STEP) * step_size  #90  # 像素/秒  

    OBSTACLE_CENTERS = np.array(obstacle_coordinate_changed_btbatch, dtype=np.float64) # (x, y) 坐標
    REAL_GOAL_POSITIONS = [] # 由左至右
    START_POSITIONS = [] # 由左至右
    
    # 終點（左上角 橫排）
    for i, (start, goal) in enumerate(matched_target_and_array_batch):
        # matched_target_and_array (start point, target point)
        REAL_GOAL_POSITIONS.append(goal) # 由左至右
        START_POSITIONS.append(start)
    GOAL_POSITIONS = generate_mid_points_to_goal(OBSTACLE_CENTERS, REAL_GOAL_POSITIONS, grid_size)


    # start orca planning
    start_time = time.time()
    all_agent_paths, SUCCESS = main(NUM_CIRCLES, TIME_STEP, NEIGHBOR_DIST, TIME_HORIZON, CIRCLE_RADIUS, MAX_SPEED, START_POSITIONS, OBSTACLE_CENTERS, OBSTACLE_RADIUS, SCENE_HEIGHT, SCENE_WIDTH, GOAL_POSITIONS, grid_size, walkable_grid_np, mode)
    final_paths = complete_path(all_agent_paths, REAL_GOAL_POSITIONS, OBSTACLE_CENTERS, obstacle_radius)
    print(f"total time = {time.time()- start_time}")
    # 四捨五入 final_paths
    rounded_final_paths = []
    for path in final_paths:
        rounded_path = [(int(round(x)), int(round(y))) for x, y in path]
        rounded_final_paths.append(rounded_path)

    final_paths = rounded_final_paths
    # for path in final_paths:
    #     print(f"Final paths:{final_paths}")
    return final_paths, SUCCESS
    
    

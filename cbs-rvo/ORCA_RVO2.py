import numpy as np
import rvo2
import time  
import math

def check_midpoint_and_obstacle(mid_point, OBSTACLE_CENTERS, grid_size):
    new_x, mid_points_y = mid_point
    for j, obs in enumerate(OBSTACLE_CENTERS):
        obs_x, obs_y = obs
        if abs(obs_x - new_x) < grid_size and abs(obs_y - mid_points_y) < grid_size:
            print(f"Obstacle {j} overlaps with mid-point {mid_point}.")
            return obs 
    return None

def generate_mid_points_to_goal(OBSTACLE_CENTERS, REAL_GOAL_POSITIONS, grid_size):
    y_of_goal = REAL_GOAL_POSITIONS[0][1] # 因為是整rows，所以y值都是相同的
    mid_points =  [[] for _ in range(len(REAL_GOAL_POSITIONS))]
    # for i in range(0, len(REAL_GOAL_POSITIONS), 2):
    #     #print(f"real_gaol{REAL_GOAL_POSITIONS[i]}")
    #     mid_points_y_even = y_of_goal + 0 *grid_size # 偶數排上面
    #     mid_points_x_even = REAL_GOAL_POSITIONS[i][0]
    #     mid_point = (mid_points_x_even, mid_points_y_even) 
    #     #print(f"mid_point = {mid_point}")
    #     while True:
    #         obs = check_midpoint_and_obstacle(mid_point, OBSTACLE_CENTERS, grid_size)
    #         if obs is None:
    #             break
    #         else:
    #             mid_points_y_even = mid_points_y_even - grid_size
    #             mid_point = (mid_points_x_even, mid_points_y_even)
    #             continue
    #     assert check_midpoint_and_obstacle(mid_point, OBSTACLE_CENTERS, grid_size) is None, f"Mid-point {mid_point} overlaps with obstacle {obs}."
    #     mid_points[i] = (mid_points_x_even, mid_points_y_even) # 按照時序，先填入0,2,4,6.....
        
    #     if i+1 < len(REAL_GOAL_POSITIONS): # 奇數排下面    
    #         mid_points_y_odd = y_of_goal + 2 *grid_size # 奇數排下面
    #         mid_points_x_odd = REAL_GOAL_POSITIONS[i+1][0]
    #         mid_point = (mid_points_x_odd, mid_points_y_odd ) 
    #         while True:
    #             obs = check_midpoint_and_obstacle(mid_point, OBSTACLE_CENTERS, grid_size)
    #             if obs is None:
    #                 break
    #             else:
    #                 mid_points_y_odd = mid_points_y_odd + grid_size
    #                 mid_point = (mid_points_x_odd, mid_points_y_odd)
    #                 continue
    #         assert check_midpoint_and_obstacle(mid_point, OBSTACLE_CENTERS, grid_size) is None, f"Mid-point {mid_point} overlaps with obstacle {obs}."
    #         mid_points[i+1] = (mid_points_x_odd, mid_points_y_odd) # 再填入1,3,5,7.....
    for i in range(len(REAL_GOAL_POSITIONS)):
        mid_point = (30 + i* 80, y_of_goal + grid_size)  
        mid_points[i] = mid_point
        while True:
            obs = check_midpoint_and_obstacle(mid_point, OBSTACLE_CENTERS, grid_size)
            if obs is None:
                break
            else:
                mid_point[1] = mid_point[1] - grid_size
                continue
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

def main(NUM_CIRCLES, TIME_STEP, NEIGHBOR_DIST, TIME_HORIZON, CIRCLE_RADIUS, MAX_SPEED, START_POSITIONS, OBSTACLE_CENTERS, OBSTACLE_RADIUS, SCENE_HEIGHT, SCENE_WIDTH, GOAL_POSITIONS, grid_size):
    all_agent_paths = [[] for _ in range(NUM_CIRCLES)]
    # 初始化 RVO2 模擬器
    sim = rvo2.PyRVOSimulator(TIME_STEP, NEIGHBOR_DIST, 5, TIME_HORIZON, 3, CIRCLE_RADIUS, MAX_SPEED) # 5: 一個代理在避碰計算時最多考慮的鄰居數量, timeHorizon: 代理間避碰的時間視界, timeHorizonObst (腳本中為 2): 代理與靜態障礙物避碰的時間視界
    SUCCESS = True
    
    # 添加光圈
    agents = []
    for pos in START_POSITIONS:   # 由左至右
        agent_id = sim.addAgent(tuple(pos))
        agents.append(agent_id)
    
    # 添加障礙物
    for center in OBSTACLE_CENTERS:
        center = np.array(center)
        num_points = 16
        vertices = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + OBSTACLE_RADIUS * np.cos(angle)
            y = center[1] + OBSTACLE_RADIUS * np.sin(angle)
            vertices.append((x, y))
        vertices.append(vertices[0])
        sim.addObstacle(vertices)
    
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
    for i, agent in enumerate(agents):
        start = START_POSITIONS[i]
        # print(f"lenth of GOAL_POSITIONS = {len(GOAL_POSITIONS)}")
        # print(GOAL_POSITIONS)
        goal = GOAL_POSITIONS[i]
        pixel_path = straight_path(start, goal, step_size=3)
        #print(f"Agent {i} start: {start}, goal: {goal}, pixel path length: {len(pixel_path)}, path: {pixel_path[:5]}...")
        whole_paths.append(pixel_path)
    
    agents_reached_final_goal = [False] * NUM_CIRCLES # 追蹤是否到達最終目標
    step = 0
    
    precise_arrival_threshold = 3
    # 模擬循環
    while any(np.linalg.norm(np.array(sim.getAgentPosition(agents[i])) - np.array(GOAL_POSITIONS[i])) >= precise_arrival_threshold for i in range(NUM_CIRCLES)) and step < 2000:
        frame_start_time = time.perf_counter()
        for i, agent in enumerate(agents):
            pos = np.array(sim.getAgentPosition(agent))
            final_goal = np.array(GOAL_POSITIONS[i])
            dist_to_goal = np.linalg.norm(final_goal - pos) # 計算代理當前位置到最終目標的距離
            path = whole_paths[i]
            # 用網格計算其 
            goal_grid = (final_goal[0] // grid_size, final_goal[1] // grid_size)
            
            progress = step * TIME_STEP * MAX_SPEED * 1.5 / np.linalg.norm(np.array(path[-1]) - np.array(path[0]))  
            # "step * TIME_STEP * MAX_SPEED"這部分估計了在 step 個時間步內，如果代理一直以最大速度移動，它所能行駛的最大距離。
            # 1.2: 這是一個額外的係數，用於稍微加快進度的估計，可能為了讓代理更快地沿著參考路徑移動
            # progress: 通過將估計的已行駛距離除以總路徑長度，得到一個介於 0 和可能大於 1 的進度值，表示代理在參考路徑上的位置
            idx = min(int(progress * len(path)), len(path) - 1) # 這行程式碼根據計算出的 progress 來確定代理在參考路徑上應該朝向的目標點的索引
            target = path[idx] # 最為目標參考點
            direction = target - pos # 計算當前代理位置到目標參考點的方向向量
            dist = np.linalg.norm(direction) + 1e-6 # 計算代理當前位置到目標點的距離
                   
            arrival_threshold = 100
            arrival_threshold_slowdown = 20 # 更大的減速區域

            #if (pos[0]// grid_size, pos[1]// grid_size) == goal_grid:
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
             # 檢查這個粒子是否已經被標記為到達終點
            if agents_reached_final_goal[i] == True:
                rvo2_stop_pos = sim.getAgentPosition(agent_id) # 設定在進入到終點網格實踐暫停rvo2規劃，改用硬編碼(其他agent仍會根據手動編碼結果繞過)
                assert np.linalg.norm(np.array(rvo2_stop_pos) - np.array(GOAL_POSITIONS[i])) <= 100 , f"Agent {i} reached goal but position is not close to goal: {rvo2_stop_pos} vs {GOAL_POSITIONS[i]}" #int(grid_size//2 * 1.5)+1
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
    
    # 檢查到實際終點的路上是否有障礙物
    def check_if_obstacles(OBSTACLE_CENTERS, left_upper_point, right_lower_point, obstacle_radius = 15):   #檢查區域，傳入左上角與右下角
        for i, (obs_x, obs_y) in enumerate(OBSTACLE_CENTERS): 
            if 0 <= obs_x <= right_lower_point[0]+obstacle_radius and left_upper_point[1]- obstacle_radius <= obs_y <= right_lower_point[1]+ obstacle_radius:
                return True

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

def orca_planning(matched_target_and_array_batch, obstacle_coordinate_changed_btbatch, grid_size, image_width, image_height, step_size, Rl, obstacle_radius):
    # 場景參數
    SCENE_WIDTH = image_width  # 像素
    SCENE_HEIGHT = image_height
    NUM_CIRCLES = len(matched_target_and_array_batch)  # 光圈數量
    CIRCLE_RADIUS = Rl+10  # 光圈半徑（直徑 50 像素）
    OBSTACLE_RADIUS = 25 # obstacle_radius  # 障礙物半徑（直徑 30 像素）
    NEIGHBOR_DIST = 80 #(CIRCLE_RADIUS + 10) * 2  #80  # RVO2 鄰居距離 代理檢測其他代理的最大距離
    TIME_HORIZON = 1.5 # 代理人對其他代理人做出避碰行為時，考慮的未來時間範圍（秒數）
    TIME_STEP = 1 / 30  # 每幀 1/30 秒（30fps） 每個模擬步的時間長度
    MAX_SPEED = 1/(TIME_STEP) * step_size  #90  # 像素/秒  

    # 靜態障礙物 
    OBSTACLE_CENTERS = obstacle_coordinate_changed_btbatch # (x, y) 坐標
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
    all_agent_paths, SUCCESS = main(NUM_CIRCLES, TIME_STEP, NEIGHBOR_DIST, TIME_HORIZON, CIRCLE_RADIUS, MAX_SPEED, START_POSITIONS, OBSTACLE_CENTERS, OBSTACLE_RADIUS, SCENE_HEIGHT, SCENE_WIDTH, GOAL_POSITIONS, grid_size)
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
    
    

import multiprocessing
import time
from collections import deque

# --- 多程序規劃 ---
"""
多程序規劃如下:
PROCESS 1 感知 (Perception): 負責截圖與YOLO檢測並加上粒子編號，輸出形式為[(pos1, id1), (pos2, id2), ...]，並將結果發送到決策程序(Queue P -> D)
PROCESS 2 決策 (Decision): 負責接收感知結果，紀錄與維護個編號位置歷史。 從歷史中檢測是否有粒子暫停，判斷是否需要重新規劃。 IF YES: 1. 透過QUEUE1 D -> PP 發送暫停訊號給規劃程序，同時透過(QUEUE D->E)傳給執行程序， 2. 重新整理新的粒子列表與障礙物列表並發送給規劃程序(Queue2 D -> PP)
PROCESS 3 規劃 (Planning): 負責接收決策程序的規劃請求，計算路徑，並將結果發送給執行程序(Queue PP -> E)
PROCESS 4 執行 (Execution): 負責接收規劃結果與暫停訊號，生成投影影片並控制投影設備執行投影 (ONLY 接收)
"""

# 命名規則：process_name_task(接收的Queue, 發送的Queue, 配置參數)

def perception_process_task(output_queue, termination_event, config):
    """
    感知進程任務：捕捉影像 -> 偵測 -> 追蹤 -> 將帶ID的粒子位置和障礙物發送到 output_queue
    """
    print(f"Process Perception started with PID: {multiprocessing.current_process().pid}")

    while True:
        try:
            # 1. 不斷截圖


            # 2. 執行偵測與追蹤
            # yolo檢測implement here
            obstacle_list, particle_list_with_ids = yolo_detect_and_track  # particle_list_with_ids = [(pos1, id1), (pos2, id2), ...]


            # 3. 將最新結果發送到 Decision 程序
            try:
                output_queue.put_nowait((particle_list_with_ids, obstacle_list))  #Process D 取出為FIFO
            except multiprocessing.queues.Full:
                try:
                    # Queue 已滿，先嘗試取出最舊的數據騰出空間，然後再嘗試放入最新的數據
                    output_queue.get_nowait()
                    output_queue.put_nowait((particle_list_with_ids, obstacle_list))
                    print("Perception: Dropped oldest to put newest.")
                except (multiprocessing.queues.Full, multiprocessing.queues.Empty):
                    # 如果get之後又立即被填滿，或get時隊列並非真的Full而變空，則這次發送失敗
                    print("Perception: Failed to put newest after dropping oldest.")
                    pass # 這次最新的數據還是無法成功放入
            # 添加優雅退出機制 (例如檢查共享終止事件)
            if termination_event.is_set(): 
                break

        except Exception as e:
            print(f"Error in Perception process: {e}")
            break
    print("Process Perception exiting.")

def update_particle_states_and_history(particle_list_with_ids, particle_states):
    for pos, id in particle_list_with_ids:
        if id not in particle_states:
            history_length = config['history_length'] # 設定歷史要記錄的長度，其餘覆蓋或丟棄
            # 新粒子，初始化狀態
            particle_states[id] = {
                'history': deque(maxlen=history_length),  # 記錄最近10個位置
                'is_target': False,
                'is_stuck': False,
                'current_pos': pos,
                
            }
        # 更新粒子位置歷史
        particle_states[id]['history'].append(pos)
    return particle_states
   
def decision_process_task(perception_input_queue, stop_signal_output_queue, planning_planning_info_output_queue, termination_event, config):
    """
    決策進程任務：接收感知結果和執行狀態，判斷狀態，觸發規劃。
    這個進程是系統的大腦，以一定頻率運行決策循環。
    """
    print(f"Process Decision started with PID: {multiprocessing.current_process().pid}")
    
    # 創建歷史字典
    particle_states = {} # {id: {'history': deque, 'is_target': bool, 'is_stuck': bool, 'current_pos': (x, y)}}    
    particle_obstacles = [] # 被決策為"脫落"的粒子，會被加入到障礙物列表中
    replan_needed = False
    planning_request_counter = 0 # 用於標識規劃請求的序列號或 ID，幫助下游判斷哪個規劃是優先的
    decision_loop_period = 1.0 / config['decision_frequency'] # 基於時間的檢查頻率 (Hz)
    is_stuck_simulated = False  # 設置為當已經發出重新規劃訊號後，直到該訊號被接收且開始投影執行前都不再檢查 (否則會一直重新規劃) #true:不用檢查, false:要檢查
    
    iteration_count = 0 # 用於計算循環次數
    while True:
        loop_start_time = time.time()
        try:
            # 1. 獲取最新的感知數據
            latest_perception_data = None  #代表在此程序中取出的最新資訊
            while not perception_input_queue.empty():
                latest_perception_data = perception_input_queue.get_nowait()  # 從queue中取出最"舊"的感知數據 FIFO
                current_all_obstacles = [] # 每次取出最新的感知數據時，清空障礙物列表
                
                if latest_perception_data:
                    particle_list_with_ids, obstacle_list = latest_perception_data
                    # 更新粒子歷史
                    update_particle_states_and_history(particle_list_with_ids, particle_states) # 替換為你的更新邏輯
                    # 更新障礙物列表
                    if particle_obstacles:
                        current_all_obstacles = obstacle_list + particle_obstacles
                    else:
                        current_all_obstacles = obstacle_list
                
            if iteration_count != 0 and iteration_count % config['screen_shot_frequency'] == 0 and not is_stuck_simulated:
                # 2. 執行決策邏輯：檢查粒子是否「靜止」，是否需要重新規劃等
                replan_needed, update_stuck_particle_ids = check_replan_conditions(particle_states, current_commanded_light_pos, config)


            if not is_stuck_simulated and stop_signal_output_queue.empty(): # 確保沒有正在等待的規劃請求
                replan_needed = True
                particle_obstacles.extend(update_particle_obstacles(stuck_particle_ids, particle_states)) # 根據靜止粒子獲取障礙物列表
                # 如果需要重新規劃，將脫離的粒子加入臨時障礙物 (根據 particle_states)
                current_all_obstacles = obstacle_list + particle_obstacles
                print("Decision: Replan triggered!")


            # 3. 如果需要重新規劃，發送規劃請求給 Planning 進程
            if replan_needed:
                planning_request_counter += 1
                particle_coor = get_particle_coordinates(particle_states)
                # 準備規劃請求數據包
                request_data = {
                    'request_id': planning_request_counter,
                    'particle_coor': particle_states, # 粒子狀態 
                    'obstacles': current_all_obstacles
                }
                print(f"Decision: Sending planning request {planning_request_counter}")
                try:
                    planning_planning_info_output_queue.put_nowait(request_data) # 非阻塞發送請求
                    replan_needed = False # 請求已發送
                except multiprocessing.queues.Full:
                    # 規劃隊列滿了，規劃進程還沒處理完上一個請求，這次請求被跳過
                    print("Decision: Planning queue is full, cannot send request.")
                    pass # 繼續下一輪循環，可能在下一輪重試發送


            # 5. 檢查總體任務是否完成 (例如所有粒子都在目標位置)
            # task_completed = check_overall_task_completion_in_decision(...) # 替換為你的完成判斷邏輯
            # if task_completed:
            #      print("Decision: Task completed. Signaling termination.")
            #      # 發送終止信號給 Planning 和 Execution 進程 (需要額外的機制)
            #      planning_output_queue.put("TERMINATE") # Planning 需要能接收並處理
            #      # execution_input_queue 需要能接收並處理 (例如通過另一個 Queue)
            #      break # 退出主循環


            # 6. 控制循環頻率
            elapsed_time = time.time() - loop_start_time
            sleep_time = decision_loop_period - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            iteration_count += 1
        except Exception as e:
            print(f"Error in Decision process: {e}")
            break
    print("Process Decision exiting.")


def planning_process_task(request_input_queue, path_output_queue, config):
    """
    規劃進程任務：接收規劃請求，計算路徑，發送結果。
    這個進程阻塞等待請求，計算規劃，並支持中斷（可選）。
    """
    print(f"Process Planning started with PID: {multiprocessing.current_process().pid}")
    # 初始化規劃算法等
    # planning_algorithm = init_planning_algorithm(...)

    current_request_id = None # 當前正在處理的請求 ID

    while True:
        try:
            # 1. 等待並獲取規劃請求 (阻塞式獲取，沒有請求時等待)
            # 當獲取到請求時，表示 Decision 需要新的規劃
            print("Planning waiting for request...")
            request = request_input_queue.get() # 阻塞

            # 檢查終止信號
            if request == "TERMINATE":
                 break

            # 2. 解析請求
            # request_id = request['request_id']
            # planning_params = request['params'] # 包含粒子狀態、目標、障礙物等

            print(f"Planning received request {request['request_id']}, starting calculation.")
            current_request_id = request['request_id']

            # --- 3. 執行路徑規劃 (可能耗時) ---
            # 在這裡調用路徑規劃算法。如果算法支持中斷，可以在算法內部檢查中斷標誌
            # 例如：planned_path = run_planning_algorithm(planning_params, interrupt_check_function)

            # 模擬規劃計算，並加入中斷檢查 (這是需要修改你的規劃算法的地方)
            simulated_calculation_time = random.uniform(0.2, 1.5) # 模擬計算時間
            start_calc_time = time.time()
            calculation_step_time = 0.05 # 每計算一小步花的時間
            num_steps = int(simulated_calculation_time / calculation_step_time)
            calculated_path_segment = []

            for i in range(num_steps):
                # 模擬計算一小步
                time.sleep(calculation_step_time)
                calculated_path_segment.append((i*10, i*10)) # 模擬生成路徑點

                # --- 檢查是否有新的規劃請求到達 (中斷檢查) ---
                # 這是實現中斷的核心。如果在計算過程中收到新的請求，則放棄當前計算
                try:
                    next_request = request_input_queue.get_nowait()
                    print(f"Planning received NEW request {next_request['request_id']} while calculating {current_request_id}.")
                    # 判斷新請求是否比當前正在處理的舊請求新 (通過 request_id 或時間戳)
                    # if next_request['request_id'] > current_request_id: # 替換為你的邏輯
                    print(f"Planning abandoning calculation for {current_request_id} and starting new one.")
                    # 放棄當前計算，並處理新的請求 (跳出內層循環，讓外層循環獲取並處理新請求)
                    request_input_queue.put(next_request) # 將新請求放回隊列，讓外層循環獲取
                    raise StopIteration # 使用異常跳出內層循環 (或者使用 return, break + 標誌)

                except multiprocessing.queues.Empty:
                    pass # 沒有新的請求，繼續計算
                except StopIteration:
                    break # 中斷計算，跳出內層循環

            # --- 規劃完成 (如果沒有被中斷) ---
            print(f"Planning finished calculation for request {current_request_id}.")
            planned_path = calculated_path_segment # 替換為實際規劃結果

            # 4. 將規劃結果發送給 Execution 進程
            # 通常只需要發送最新的有效路徑
            try:
                 # 可選：清空隊列，確保 Execution 總是拿到最新的路徑
                 # while not path_output_queue.empty():
                 #     path_output_queue.get_nowait()
                 path_output_queue.put_nowait(planned_path) # 非阻塞發送路徑
                 # print("Planning sent path.")
            except multiprocessing.queues.Full:
                # 執行隊列滿了， Execution 還沒處理完上一個路徑，這個路徑被跳過
                print("Planning: Execution queue is full, cannot send path.")
                pass


        except StopIteration: # 捕獲到中斷信號觸發的異常，回到外層循環開頭處理新請求
            continue
        except Exception as e:
            print(f"Error in Planning process: {e}")
            break
    print("Process Planning exiting.")


def execution_process_task(path_input_queue, state_output_queue, config):
    """
    執行進程任務：接收規劃路徑，控制投影設備，並回饋狀態。
    這個進程是實時控制的核心執行者。
    """
    print(f"Process Execution started with PID: {multiprocessing.current_process().pid}")
    # 初始化投影設備接口
    projector = None # 替換為你的投影設備接口

    current_executing_path = None
    current_path_step = 0
    current_commanded_light_pos = None # 正在指令光圈去的位置

    execution_loop_period = 1.0 / config['control_frequency'] # 執行循環頻率

    while True:
        loop_start_time = time.time()
        try:
            # 1. 獲取最新的規劃路徑 (非阻塞，總是使用最新的路徑)
            latest_planned_path = None
            while not path_input_queue.empty():
                 latest_planned_path = path_input_queue.get_nowait()

            if latest_planned_path:
                # 收到新路徑，切換到執行這個新路徑
                current_executing_path = latest_planned_path
                current_path_step = 0 # 從新路徑的起點開始執行
                print(f"Execution received new path with {len(current_executing_path)} steps.")

            # 2. 根據當前執行的路徑，計算並發送下一個光圈指令
            if current_executing_path and current_path_step < len(current_executing_path):
                current_commanded_light_pos = current_executing_path[current_path_step]
                # 發送指令給投影設備
                # projector.move_to(current_commanded_light_pos) # 替換為你的投影指令
                # print(f"Execution moving light to step {current_path_step}/{len(current_executing_path)}")
                current_path_step += 1
            elif current_executing_path and current_path_step >= len(current_executing_path):
                 # 路徑已執行完畢，停留在最後一個點
                 current_commanded_light_pos = current_executing_path[-1]
                 # print("Execution: Current path finished, holding last position.")
                 pass # 等待 Decision 發送新的規劃請求或終止信號

            # 3. 將當前指令的光圈位置發送給 Decision 進程 (作為回饋)
            if current_commanded_light_pos is not None:
                try:
                    state_output_queue.put_nowait(current_commanded_light_pos) # 非阻塞發送狀態
                    # print("Execution sent light pos.")
                except multiprocessing.queues.Full:
                    # Decision 進程處理不過來，這次發送被跳過
                    # print("Execution queue is full, skipping state send.")
                    pass


            # 添加優雅退出機制
            # if termination_event.is_set(): break


            # 4. 控制循環頻率
            elapsed_time = time.time() - loop_start_time
            sleep_time = execution_loop_period - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Error in Execution process: {e}")
            break
    print("Process Execution exiting.")


# --- 主程式入口點 ---
if __name__ == '__main__':
    print("Starting multi-process system...")
    print("multiprcessing counts:", multiprocessing.cpu_count())
    
    # 設定共享隊列
    q_perception_to_decision = multiprocessing.Queue(maxsize=10) # 感知到決策  (Queue P -> D)
    q_decision_pp_oe_e_stop_signal = multiprocessing.Queue(maxsize=1)  # 決策到規劃 (Queue D -> PP/ E) 共用傳遞暫停訊號
    q_decision_to_planning_planning_info = multiprocessing.Queue(maxsize=1) # 決策到規劃 (Queue D -> PP)  傳遞規劃資訊(重新啟動規劃)
    q_planning_to_execution = multiprocessing.Queue(maxsize=1) # 規劃到執行 (Queue PP -> E) 傳遞規劃好的路徑列表供投影
    
    # 設定終止事件
    termination_event = multiprocessing.Event() # 用於退出的事件標誌，由決策進程設置

    # 設定系統配置參數
    config = {
        'size': 60, #單格陣列大小 (同時也是A*計算的網格大小) 
        'array_columns': 9,
        'array_rows': 5,
        'step_size': 3, # 預計每幀移動的步長 (PIXEL)
        'screen_shot_frequency': 10, # 捕捉影像頻率 (Hz)
        'screen_shot_area': (0, 0, 800, 600), # 捕捉影像的區域 (左上角 x, y, 寬度, 高度)
        'detection_frequency': 10, # 感知進程中YOLO偵測頻率 (Hz)
        'decision_frequency': 5,   # 決策進程根據agent位置歷史檢查的時機，代表每接收到5次感知結果就檢查一次 (Hz)
        'control_frequency': 30,   # 執行進程生成即時投影的影片幀率 (FPS)
        'stuck_pos_tolerance': 5,  # 判斷靜止的位置容忍度 (像素)，表示檢查根據歷史如果移動不到5即代表暫停 (現在影片幀率30fps, stepsize=3, 最大移動速率90pixel/s，yolo偵測頻率1秒檢查10次，表示發送的歷史位置每次最多移動9 pixel)
        'stuck_time_threshold': 3, # 判斷靜止的時間閾值 (秒)，表示如果在3秒內都沒有移動，則認為粒子靜止 #這是為了不要直接檢查歷史列表中的點，因為實際可能出現延遲、會者因queue滿而被丟棄
        'history_length': 10, # 粒子歷史記錄的長度 (幀數)，表示最多記錄10幀的歷史位置
    }

    # 創建並啟動多程序
    # process 1 感知 (Perception) 傳遞:1 接收:0
    p_perception = multiprocessing.Process(
        target=perception_process_task, 
        args=(q_perception_to_decision, termination_event, config),
        name="Perception_Process"
    )

    # process 2 決策 (Decision) 傳遞:3 接收:1
    p_decision = multiprocessing.Process(
        target=decision_process_task,
        args=(q_perception_to_decision, q_decision_pp_oe_e_stop_signal, q_decision_to_planning_planning_info, termination_event, config),
        name="Decision_Process"
    )

    # process 3 規劃 (Planning) 傳遞:1 接收:2
    p_planning = multiprocessing.Process(
        target=planning_process_task,
        args=(q_decision_pp_oe_e_stop_signal, q_decision_to_planning_planning_info, q_planning_to_execution, config),
        name="Planning_Process"
    )

    # process 4 執行 (Execution) 傳遞:0 接收:2
    p_execution = multiprocessing.Process(
        target=execution_process_task,
        args=(q_planning_to_execution, q_decision_pp_oe_e_stop_signal, config),
        name="Execution_Process"
    )

    # 啟動進程
    print("Starting all processes...")
    p_perception.start()
    p_decision.start()
    p_planning.start()
    p_execution.start()
    print("All processes started.")

    # --- 主進程管理與終止 ---
    # 主進程通常在這裡等待子進程結束，或者實現一個監控和優雅關閉的邏輯
    try:
        # 保持主進程運行，直到接收到終止信號 (例如 Ctrl+C)
        while True:
            time.sleep(1) # 讓主進程休眠，避免佔用 CPU
            # 在這裡可以添加檢查子進程狀態的邏輯

    except KeyboardInterrupt:
        print("Keyboard interrupt detected in main process.")

    finally:
        # --- 優雅關閉所有子進程 ---
        print("Attempting graceful shutdown...")
        # 通過向 Queue 發送特殊消息 (例如 "TERMINATE") 通知子進程退出循環
        # 需要在每個子進程的 Queue 讀取處檢查這個消息
        q_decision_to_planning.put("TERMINATE") # 通知 Planning 停止
        # Decision Process 也需要有接收到終止信號後向其下游發送終止信號的邏輯
        # 這裡簡單起見，假設 Decision 接收到 KeyboardInterrupt 後也會退出循環
        # 並且 Perception/Execution 進程也需要有自己的終止機制，例如檢查共享事件或Queue消息

        # 等待子進程結束
        p_perception.join()
        p_decision.join()
        p_planning.join()
        p_execution.join()

        print("All processes have finished.")

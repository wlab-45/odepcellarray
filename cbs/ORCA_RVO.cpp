#include <vector> 
#include <cmath> 
#include <iostream> 
#include <stdio.h>
#include <cassert> // 引入斷言函式庫，用於在程式碼中進行條件檢查
//#include <limits> // 引入用於數值極限的函式庫 (如 numeric_limits)
#include <RVO.h>

// 定義一個簡單的 Point 結構來表示二維座標
struct Point {
    double x, y; // x 和 y 座標

    // 建構函式
    Point(double x_ = 0.0, double y_ = 0.0) : x(x_), y(y_) {}

    // 基本的向量運算符重載
    Point operator+(const Point& other) const { return Point(x + other.x, y + other.y); } // 向量相加
    Point operator-(const Point& other) const { return Point(x - other.x, y - other.y); } // 向量相減
    Point operator*(double scalar) const { return Point(x * scalar, y * scalar); }       // 向量與純量相乘
    Point operator/(double scalar) const {                                             // 向量除以純量
        if (scalar == 0) return Point(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()); // 處理除以零的情況
        return Point(x / scalar, y / scalar);
    }

    // 計算向量的長度 (範數)
    double norm() const { return std::sqrt(x * x + y * y); }

    // 將自定義的 Point 結構轉換為 RVO2 函式庫使用的 RVO::Vector2
    RVO::Vector2 toRVOVector2() const { return RVO::Vector2(x, y); }
};

// 重載輸出串流運算符，以便可以直接列印 Point 物件
std::ostream& operator<<(std::ostream& os, const Point& p) {
    os << "(" << p.x << ", " << p.y << ")";
    return os;
}



Point* check_midpoint_and_obstacle(const Point& mid_point, const std::vector<Point>& OBSTACLE_CENTERS, int grid_size) {
    double new_x = mid_point.x;
    double mid_points_y = mid_point.y;

    for (int j = 0; j < OBSTACLE_CENTERS.size(); ++j) {
        const Point& obs = OBSTACLE_CENTERS[j];
        double obs_x = obs.x;
        double obs_y = obs.y;

        // 檢查中間點與障礙物中心在 X 和 Y 方向的距離是否都小於 grid_size (簡單的軸對齊邊界框檢查)
        if (std::abs(obs_x - new_x) < grid_size && std::abs(obs_y - mid_points_y) < grid_size) {
            printf("Obstacle %s overlaps with mid-point %s. \n", mid_point.toRVOVector2().toString().c_str(), obs.toRVOVector2().toString().c_str());
            // 回傳重疊的障礙物中心點的指標
            return const_cast<Point*>(&obs);
        }
    }
    return nullptr; // 沒有重疊，回傳空指標
}

// 函式：生成中間目標點
// OBSTACLE_CENTERS: 障礙物中心點列表 (雖然函式簽名有，但此版本實際未使用於生成邏輯)
// REAL_GOAL_POSITIONS: 最終真實目標點列表
// grid_size: 網格大小
// 回傳: 中間目標點列表，用於 RVO2 模擬階段
std::vector<Point> generate_mid_points_to_goal(const std::vector<Point>& OBSTACLE_CENTERS, const std::vector<Point>& REAL_GOAL_POSITIONS, double grid_size) {
    if (REAL_GOAL_POSITIONS.empty()) {
        return {}; // 如果真實目標列表為空，則回傳空列表
    }
    // 從第一個真實目標獲取 Y 座標，假設所有真實目標都在同一條水平線上
    double y_of_goal = REAL_GOAL_POSITIONS[0].y;
    std::vector<Point> mid_points(REAL_GOAL_POSITIONS.size());

    // 這個邏輯直接複製了 Python 程式碼中未被註解掉的部分
    // 生成的中間點座標是 (30 + i * 80, y_of_goal + grid_size)
    for (size_t i = 0; i < REAL_GOAL_POSITIONS.size(); ++i) {
        Point mid_point = Point(30.0 + i * 80.0, y_of_goal + grid_size);
        mid_points[i] = mid_point;
    }
    // Python 中註解掉的 while 迴圈檢查重疊並調整中間點的邏輯在此版本中未被使用。
    return mid_points;
}

// 函式：生成兩個點之間的直線路徑
// current_point: 起始點
// target_center: 終點
// step_size: 路徑點之間的間隔步長
// 回傳: 包含從起始點到終點的點列表的直線路徑
std::vector<Point> straight_path(const Point& current_point, const Point& target_center, double step_size) {
    std::vector<Point> path;
    Point direction_vec = target_center - current_point; // 計算方向向量
    double all_length = direction_vec.norm();          // 計算總長度

    if (all_length == 0) {
        path.push_back(current_point); // 如果起點終點相同，只包含起點
        return path; // Python 中返回空列表，這裡返回只含起點的列表以匹配後續使用邏輯
    }

    // 計算需要的步數
    int step_count = static_cast<int>(all_length / step_size);
    if (step_count == 0) {
        // 如果距離太短，只包含起點和終點
        path.push_back(current_point);
        path.push_back(target_center);
        return path;
    }

    Point step_vec = direction_vec / step_count; // 計算每一步的向量
    path.push_back(current_point);               // 添加起點

    // 添加中間點
    for (int step_idx = 1; step_idx <= step_count; ++step_idx) {
        Point next_point = current_point + step_vec * step_idx;
        // 座標四捨五入到最接近的整數，模仿 Python 行為
        path.push_back(Point(std::round(next_point.x), std::round(next_point.y)));
    }

    // 確保終點被包含在路徑中
    if (path.empty() || (path.back().x != target_center.x || path.back().y != target_center.y)) {
        path.push_back(target_center);
    }
    return path;
}

// 函式：執行 RVO2 模擬的主要邏輯
// 包含許多模擬參數和點位列表作為輸入
// 回傳: 一個 pair，包含所有代理在模擬過程中的路徑列表，以及一個表示模擬是否成功的布林值
std::pair<std::vector<std::vector<Point>>, bool> run_simulation(
    int NUM_CIRCLES, double TIME_STEP, double NEIGHBOR_DIST, double TIME_HORIZON,
    double CIRCLE_RADIUS, double MAX_SPEED, const std::vector<Point>& START_POSITIONS,
    const std::vector<Point>& OBSTACLE_CENTERS, double OBSTACLE_RADIUS,
    double SCENE_HEIGHT, double SCENE_WIDTH, const std::vector<Point>& GOAL_POSITIONS_MID, double grid_size) { // 注意這裡的目標點是中間目標點

    std::vector<std::vector<Point>> all_agent_paths(NUM_CIRCLES); // 用於記錄所有代理的路徑
    bool SUCCESS = true; // 模擬是否成功完成的標誌

    // 初始化 RVO2 模擬器
    // 參數: timeStep, neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed
    RVO::RVOSimulator simulator;
    simulator.setTimeStep(TIME_STEP); // 設定時間步長
    // 設定代理的預設參數。timeHorizonObst 使用了 Python 腳本中註解掉的 3.0
    simulator.setAgentDefaults(NEIGHBOR_DIST, 5, TIME_HORIZON, 3.0, CIRCLE_RADIUS, MAX_SPEED);

    // 添加代理
    std::vector<size_t> agents(NUM_CIRCLES); // RVO2 函式庫使用 size_t 作為代理 ID
    for (int i = 0; i < NUM_CIRCLES; ++i) {
        agents[i] = simulator.addAgent(START_POSITIONS[i].toRVOVector2()); // 添加代理並設定起始位置
    }

    // 添加圓形障礙物 (通過多邊形近似)
    for (const auto& center : OBSTACLE_CENTERS) {
        std::vector<RVO::Vector2> vertices;
        int num_points = 16; // 用 16 個點來近似圓形
        for (int i = 0; i < num_points; ++i) {
            double angle = 2 * M_PI * i / num_points; // 計算角度
            double x = center.x + OBSTACLE_RADIUS * std::cos(angle);
            double y = center.y + OBSTACLE_RADIUS * std::sin(angle);
            vertices.push_back(RVO::Vector2(x, y));
        }
        simulator.addObstacle(vertices); // 添加多邊形障礙物
    }

    // 添加邊界作為靜態障礙物 (防止代理走出邊界)
    std::vector<RVO::Vector2> boundary;
    // 左邊界
    boundary.clear();
    boundary.push_back(RVO::Vector2(0, 0));
    boundary.push_back(RVO::Vector2(0, SCENE_HEIGHT));
    simulator.addObstacle(boundary);
    // 下邊界
    boundary.clear();
    boundary.push_back(RVO::Vector2(0, SCENE_HEIGHT));
    boundary.push_back(RVO::Vector2(SCENE_WIDTH, SCENE_HEIGHT));
    simulator.addObstacle(boundary);
    // 右邊界
    boundary.clear();
    boundary.push_back(RVO::Vector2(SCENE_WIDTH, SCENE_HEIGHT));
    boundary.push_back(RVO::Vector2(SCENE_WIDTH, 0));
    simulator.addObstacle(boundary);
    // 上邊界
    boundary.clear();
    boundary.push_back(RVO::Vector2(SCENE_WIDTH, 0));
    boundary.push_back(RVO::Vector2(0, 0));
    simulator.addObstacle(boundary);

    simulator.processObstacles(); // 處理已添加的障礙物，準備用於模擬

    // 設定直線路徑作為代理追蹤的參考路徑，目標是中間目標點
    std::vector<std::vector<Point>> whole_paths(NUM_CIRCLES);
    for (int i = 0; i < NUM_CIRCLES; ++i) {
        whole_paths[i] = straight_path(START_POSITIONS[i], GOAL_POSITIONS_MID[i], 3.0); // 使用 step_size=3
    }

    std::vector<bool> agents_reached_mid_goal(NUM_CIRCLES, false); // 追蹤代理是否到達中間目標點
    int step = 0; // 模擬步數計數
    int max_steps = 2000; // 最大模擬步數限制
    double precise_arrival_threshold = 3.0; // 判斷到達中間目標點的距離閾值

    // 模擬迴圈
    bool all_reached_midpoint_goal = false; // 標記是否所有代理都已到達中間目標點
    while (!all_reached_midpoint_goal && step < max_steps) {
        //auto frame_start_time = std::chrono::high_resolution_clock::now(); // 可以用於計時單幀

        all_reached_midpoint_goal = true; // 假設所有代理都已到達，如果找到未到達的，則設定為 false

        for (int i = 0; i < NUM_CIRCLES; ++i) {
            RVO::Vector2 current_pos_rvo = simulator.getAgentPosition(agents[i]); // 獲取代理的當前位置 (RVO::Vector2 類型)
            Point current_pos(current_pos_rvo.x(), current_pos_rvo.y()); // 轉換為自定義的 Point 類型
            Point final_mid_goal = GOAL_POSITIONS_MID[i]; // 當前代理的中間目標點

            double dist_to_mid_goal = (final_mid_goal - current_pos).norm(); // 計算代理到中間目標點的距離

            // 檢查代理是否已到達中間目標點
            if (dist_to_mid_goal < precise_arrival_threshold) {
                agents_reached_mid_goal[i] = true; // 標記為已到達
                simulator.setAgentPrefVelocity(agents[i], RVO::Vector2(0, 0)); // 設定首選速度為零 (停止)
            } else {
                 all_reached_midpoint_goal = false; // 至少還有一個代理未到達中間目標
                std::vector<Point>& path = whole_paths[i]; // 當前代理的參考路徑
                if (path.empty()) {
                     simulator.setAgentPrefVelocity(agents[i], RVO::Vector2(0, 0)); // 如果參考路徑為空，則停止
                     continue; // 跳過當前代理後續的邏輯
                }

                // 計算代理在參考路徑上的"進度"
                // 這部分邏輯複製自 Python 腳本，基於時間步長和最大速度估計行駛距離
                 double progress_distance = step * TIME_STEP * MAX_SPEED * 1.5; // 估計已行駛距離
                 double path_length = (path.back() - path.front()).norm(); // 計算參考路徑總長度

                 // 計算進度比例 (0.0 到 1.0 或可能超過 1.0)
                 // 避免除以零
                 double progress = (path_length > 0) ? progress_distance / path_length : 1.0;

                 // 根據進度計算在參考路徑上應當追蹤的目標點索引
                 int idx = std::min(static_cast<int>(progress * path.size()), static_cast<int>(path.size()) - 1);
                 Point target = path[idx]; // 獲取參考路徑上的目標點

                 Point direction = target - current_pos; // 計算從當前位置到參考目標點的方向向量
                 double dist = direction.norm(); // 計算距離參考目標點的距離

                 double arrival_threshold = 100.0; // 用於判斷是否進入減速區域的外層閾值 (Python 代碼中未明確使用這個值來判斷是否停止，而是用 dist_to_mid_goal)
                 double arrival_threshold_slowdown = 20.0; // 用於判斷是否進入減速區域的內層閾值

                 RVO::Vector2 preferred_velocity; // 首選速度 (RVO::Vector2 類型)

                 // 根據距離中間目標點來設定首選速度
                 if (dist_to_mid_goal < arrival_threshold) {
                     // 靠近最終中間目標點，設定速度為零 (Python 代碼中是 <= 100，但在外層判斷是 <= 3)
                     // 這裡以靠近中間目標點的距離 dist_to_mid_goal 為主進行判斷
                      simulator.setAgentPrefVelocity(agents[i], RVO::Vector2(0, 0));
                 } else if (dist_to_mid_goal < arrival_threshold_slowdown) {
                     // 在減速區域內 (距離中間目標點 < 20)，根據距離調整速度
                     double slowdown_factor = std::max(0.1, dist_to_mid_goal / arrival_threshold_slowdown); // 減速因子，至少為 0.1
                     double desired_speed = std::min(MAX_SPEED, dist / TIME_STEP) * slowdown_factor; // 計算期望的速度大小
                     if (dist > 1e-6) { // 避免方向向量為零時除法
                          preferred_velocity = direction.toRVOVector2() / dist * desired_speed; // 計算首選速度向量
                     } else {
                         preferred_velocity = RVO::Vector2(0,0); // 已經非常接近參考目標點，停止
                     }
                     simulator.setAgentPrefVelocity(agents[i], preferred_velocity); // 設定代理的首選速度
                 } else {
                    // 在正常移動區域
                    double desired_speed = std::min(MAX_SPEED, dist / TIME_STEP); // 計算期望的速度大小 (不超過 MAX_SPEED)
                     if (dist > 1e-6) { // 避免方向向量為零時除法
                        preferred_velocity = direction.toRVOVector2() / dist * desired_speed; // 計算首選速度向量
                     } else {
                         preferred_velocity = RVO::Vector2(0,0); // 已經非常接近參考目標點，停止
                     }
                    simulator.setAgentPrefVelocity(agents[i], preferred_velocity); // 設定代理的首選速度
                 }
            }
        }

        simulator.doStep(); // 執行一步 RVO2 模擬，更新所有代理的位置和速度
        step++;           // 增加模擬步數

         // --- 在 sim.doStep() 之後，強制將已到達中間目標點的代理位置手動調整 ---
         // 這部分邏輯複製了 Python 腳本的行為，可能用於確保代理精確對齊中間目標
        for (int i = 0; i < NUM_CIRCLES; ++i) {
             // 檢查這個代理是否已經被標記為到達中間目標點
            if (agents_reached_mid_goal[i]) {
                 RVO::Vector2 current_pos_rvo = simulator.getAgentPosition(agents[i]); // 獲取代理當前 RVO 位置
                 Point current_pos(current_pos_rvo.x(), current_pos_rvo.y()); // 轉換為 Point
                 Point mid_goal = GOAL_POSITIONS_MID[i]; // 代理的中間目標點

                 // 再次計算從當前 RVO 位置到中間目標點的直線路徑 (步長為 2.0)
                 std::vector<Point> late_path = straight_path(current_pos, mid_goal, 2.0);

                 // 如果這條"後期路徑"點數大於 1，則將代理位置強制設定為這條路徑的**第二個點**
                 if (late_path.size() > 1) {
                     simulator.setAgentPosition(agents[i], late_path[1].toRVOVector2());
                 } else if (late_path.size() == 1) {
                    // 如果只有一個點，表示已經非常接近或就在中間目標點，則強制設定為中間目標點
                     simulator.setAgentPosition(agents[i], mid_goal.toRVOVector2());
                 }
             }
        }

        // 在每次 sim.doStep() (及可能的強制位置調整) 後，記錄每個代理的當前位置
        for (int i = 0; i < NUM_CIRCLES; ++i) {
            RVO::Vector2 current_pos_rvo = simulator.getAgentPosition(agents[i]);
            all_agent_paths[i].push_back(Point(current_pos_rvo.x(), current_pos_rvo.y())); // 添加當前位置到路徑列表
        }

        // 再次檢查是否所有代理都已到達中間目標點的足夠接近範圍，以決定是否結束模擬迴圈
        // (這個檢查與迴圈開始前的檢查類似，用於在當前步更新後判斷是否繼續下一輪)
        all_reached_midpoint_goal = true; // 再次假設全部到達
         for (int i = 0; i < NUM_CIRCLES; ++i) {
             RVO::Vector2 current_pos_rvo = simulator.getAgentPosition(agents[i]);
             Point current_pos(current_pos_rvo.x(), current_pos_rvo.y());
             Point final_mid_goal = GOAL_POSITIONS_MID[i];
             double dist_to_mid_goal = (final_mid_goal - current_pos).norm();

             if (dist_to_mid_goal >= precise_arrival_threshold) {
                 all_reached_midpoint_goal = false; // 找到未到達的，設定為 false
                 break; // 跳出檢查迴圈
             }
         }

        // 檢查是否達到最大步數限制
        if (step >= max_steps) {
            std::cout << "Simulation reached maximum steps without all agents reaching the intermediate goals." << std::endl;
            SUCCESS = false; // 標記模擬未完全成功到達中間目標
            break; // 跳出主模擬迴圈
        }
    } // 模擬迴圈結束

    // 確保在 all_agent_paths 中，每個代理的最後一個記錄點是其對應的**中間目標點**
     for (int i = 0; i < NUM_CIRCLES; ++i) {
         Point final_mid_goal = GOAL_POSITIONS_MID[i]; // 中間目標點
         if (!all_agent_paths[i].empty()) {
             const Point& last_recorded_point = all_agent_paths[i].back(); // 獲取最後一個記錄點
             // 如果最後一個記錄點與中間目標點不同 (距離大於一個很小的閾值)
             if ((last_recorded_point - final_mid_goal).norm() > 1e-6) {
                  all_agent_paths[i].push_back(final_mid_goal); // 將中間目標點添加到路徑末尾
             }
         } else {
              // 如果路徑是空的 (理論上不應該發生，除非模擬一開始就停止或出錯)
              // 這裡簡單地將中間目標點加入到路徑中 (行為可能需要根據實際需求調整)
              all_agent_paths[i].push_back(GOAL_POSITIONS_MID[i]);
         }
     }

    return {all_agent_paths, SUCCESS}; // 回傳 RVO 模擬階段的路徑和成功狀態
}

// 函式：完成最終路徑，從中間目標點延伸到真實目標點
// all_agent_paths: RVO 模擬結束後，到達中間目標點的路徑
// REAL_GOAL_POSITIONS: 最終真實目標點列表
// OBSTACLE_CENTERS: 障礙物中心點列表
// obstacle_radius: 障礙物半徑，用於後面的特定檢查
// 回傳: 完成後的最終路徑列表
std::vector<std::vector<Point>> complete_path(
    std::vector<std::vector<Point>>& all_agent_paths, // RVO階段生成並到達中間目標的路徑
    const std::vector<Point>& REAL_GOAL_POSITIONS, // 最終的真實目標
    const std::vector<Point>& OBSTACLE_CENTERS, double obstacle_radius) { // 障礙物資訊

    // 檢查在從中間目標到真實目標的"最終區域"內是否存在障礙物
    // *** 這部分的邏輯是複製 Python 腳本中一個比較特殊的檢查 ***
    // 它不是檢查實際的路徑線段，而是檢查一個由特定點位定義的矩形區域內是否有障礙物中心
    bool obstacles_in_final_area = false;
     if (!all_agent_paths.empty() && !all_agent_paths[0].empty() && !all_agent_paths.back().empty()) {
         // Python 代碼似乎是使用第一個和最後一個代理在 RVO 結束時的最終位置 (即中間目標點)
         // 來定義一個檢查區域的邊界
         Point first_agent_last_pos = all_agent_paths[0].back(); // 理應是第一個代理的中間目標點
         Point last_agent_last_pos = all_agent_paths.back().back(); // 理應是最後一個代理的中間目標點

         // 定義檢查區域的邊界
         // check_min_x 固定為 0 (可能是場景邊界)
         // check_max_x 是最後一個代理的中間目標點的 X 座標 + 障礙物半徑
         // check_min_y 是第一個代理的中間目標點的 Y 座標 - 障礙物半徑
         // check_max_y 是最後一個代理的中間目標點的 Y 座標 + 障礙物半徑
         double check_min_x = 0;
         double check_max_x = last_agent_last_pos.x + obstacle_radius;
         double check_min_y = first_agent_last_pos.y - obstacle_radius;
         double check_max_y = last_agent_last_pos.y + obstacle_radius;

         // 遍歷所有障礙物中心，檢查它們是否落入這個特定區域
         for (const auto& obs : OBSTACLE_CENTERS) {
             if (obs.x >= check_min_x && obs.x <= check_max_x && obs.y >= check_min_y && obs.y <= check_max_y) {
                 obstacles_in_final_area = true; // 發現障礙物
                 break; // 跳出障礙物檢查迴圈
             }
         }
     }

    // 根據障礙物檢查結果決定如何處理路徑
    if (obstacles_in_final_area) {
        std::cout << "Obstacle exists in the final path area (based on specific check)." << std::endl;
        // 如果特定區域有障礙物，則回傳 RVO 模擬階段的路徑 (路徑只到中間目標點)
        return all_agent_paths;
    } else {
        // 如果特定區域沒有障礙物，則將路徑從中間目標點延伸到真實目標點
        for (size_t i = 0; i < all_agent_paths.size(); ++i) {
            if (!all_agent_paths[i].empty()) {
                Point last_point = all_agent_paths[i].back(); // 獲取代理在 RVO 模擬結束時的位置 (中間目標點)
                Point real_goal = REAL_GOAL_POSITIONS[i]; // 該代理的真實最終目標點

                // 生成從中間目標點到真實目標點的直線路徑段
                std::vector<Point> path_segment = straight_path(last_point, real_goal, 3.0); // 使用 step_size=3

                // 將新生成的路徑段添加到原有路徑的末尾
                if (!path_segment.empty()) {
                    // 檢查路徑段的第一個點是否與原有路徑的最後一個點重複，避免添加重複點
                     if (!all_agent_paths[i].empty() && (path_segment[0].x == all_agent_paths[i].back().x && path_segment[0].y == all_agent_paths[i].back().y)) {
                         all_agent_paths[i].insert(all_agent_paths[i].end(), path_segment.begin() + 1, path_segment.end()); // 從路徑段的第二個點開始添加
                     } else {
                         all_agent_paths[i].insert(all_agent_paths[i].end(), path_segment.begin(), path_segment.end()); // 添加整個路徑段
                     }
                 }
            } else {
                std::cerr << "Agent " << i << " has no initial path from RVO simulation." << std::endl;
                // 如果代理在 RVO 模擬後沒有生成路徑 (例如，一開始就失敗了)
                // 這裡簡單地將真實目標點添加到路徑中 (這可能不是期望的行為，需要根據需求調整)
                 all_agent_paths[i].push_back(REAL_GOAL_POSITIONS[i]);
            }
        }
    }
    return all_agent_paths; // 回傳完成後的最終路徑
}


// 函式：設置參數並呼叫 ORCA 規劃流程的頂層函式
// matched_target_and_array_batch: 包含起始點和真實目標點對的列表 (pair of (start, goal))
// obstacle_coordinate_changed_btbatch: 障礙物中心點列表
// grid_size, image_width, image_height, step_size, Rl, obstacle_radius: 規劃所需的各類參數
// 回傳: 一個 pair，包含所有代理的最終路徑列表，以及一個表示規劃是否成功的布林值
std::pair<std::vector<std::vector<Point>>, bool> orca_planning_cpp(
    const std::vector<std::pair<Point, Point>>& matched_target_and_array_batch, // (起始點, 真實目標點) 對的列表
    const std::vector<Point>& obstacle_coordinate_changed_btbatch, // 障礙物中心點列表
    double grid_size, double image_width, double image_height, double step_size,
    double Rl, double obstacle_radius) {

    // 場景參數
    double SCENE_WIDTH = image_width;
    double SCENE_HEIGHT = image_height;
    int NUM_CIRCLES = matched_target_and_array_batch.size(); // 代理數量

    // RVO2 參數 (這些參數影響避碰行為)
    double CIRCLE_RADIUS = Rl + 10; // 代理的半徑 (RVO2 使用)
    double OBSTACLE_RADIUS_RVO = 25; // RVO2 內部使用的障礙物半徑 (Python 代碼中設定為 25)
    double NEIGHBOR_DIST = 80.0; // 代理尋找附近其他代理的最大距離
    double TIME_HORIZON = 1.5; // 代理預測其他代理未來位置的時間視界
    double TIME_STEP = 1.0 / 30.0; // 每個模擬步的時間長度 (例如 30fps)
    // 代理的最大移動速度 (像素/秒)。這裡根據 step_size 和 TIME_STEP 計算，模仿 Python 邏輯
    double MAX_SPEED = (1.0 / TIME_STEP) * step_size;

    // 靜態障礙物中心點列表
    std::vector<Point> OBSTACLE_CENTERS = obstacle_coordinate_changed_btbatch;
    std::vector<Point> REAL_GOAL_POSITIONS; // 儲存真實最終目標點
    std::vector<Point> START_POSITIONS; // 儲存起始點

    // 將輸入的 (start, goal) 對分離到各自的列表中
    for (const auto& item : matched_target_and_array_batch) {
        REAL_GOAL_POSITIONS.push_back(item.second); // item.second 是 goal
        START_POSITIONS.push_back(item.first);    // item.first 是 start
    }

    // 生成用於 RVO2 模擬階段的中間目標點
    std::vector<Point> GOAL_POSITIONS_MID = generate_mid_points_to_goal(OBSTACLE_CENTERS, REAL_GOAL_POSITIONS, grid_size);

    // 開始執行 RVO 規劃的主體模擬
    auto start_time = std::chrono::high_resolution_clock::now(); // 記錄開始時間
    auto result = run_simulation(
        NUM_CIRCLES, TIME_STEP, NEIGHBOR_DIST, TIME_HORIZON, CIRCLE_RADIUS,
        MAX_SPEED, START_POSITIONS, OBSTACLE_CENTERS, OBSTACLE_RADIUS_RVO, // 注意這裡傳入的是 OBSTACLE_RADIUS_RVO
        SCENE_HEIGHT, SCENE_WIDTH, GOAL_POSITIONS_MID, grid_size); // 注意這裡傳入的是中間目標點
    std::vector<std::vector<Point>> all_agent_paths_to_mid_goals = result.first; // RVO 模擬到中間目標的路徑
    bool SUCCESS = result.second; // RVO 模擬是否成功到達中間目標

    // 完成最終路徑，從中間目標延伸到真實目標
    std::vector<std::vector<Point>> final_paths = complete_path(
        all_agent_paths_to_mid_goals, REAL_GOAL_POSITIONS, OBSTACLE_CENTERS, obstacle_radius); // 注意這裡傳入的是 REAL_GOAL_POSITIONS 和原始 obstacle_radius

    auto end_time = std::chrono::high_resolution_clock::now(); // 記錄結束時間
    std::chrono::duration<double> elapsed_time = end_time - start_time; // 計算總耗時
    std::cout << "Total planning time = " << elapsed_time.count() << " seconds" << std::endl;

    // 對最終路徑的所有點座標進行四捨五入取整
    std::vector<std::vector<Point>> rounded_final_paths;
    for (const auto& path : final_paths) {
        std::vector<Point> rounded_path;
        for (const auto& p : path) {
            rounded_path.push_back(Point(std::round(p.x), std::round(p.y)));
        }
        rounded_final_paths.push_back(rounded_path);
    }

    return {rounded_final_paths, SUCCESS}; // 回傳四捨五入後的最終路徑和成功狀態
}

// C++ 程式的入口函式
int main() {
    // --- 定義範例參數和輸入數據 ---
    // 您需要將這些替換為您的實際數據
    double grid_size = 20.0;
    double image_width = 640.0;
    double image_height = 480.0;
    double step_size = 5.0; // 例如，代理每一步嘗試移動的像素距離，影響 MAX_SPEED
    double Rl = 15.0; // 例如，代理的邏輯半徑，用於計算 RVO 的 CIRCLE_RADIUS (Rl + 10)
    double obstacle_radius = 25.0; // 例如，障礙物的實際半徑，用於 complete_path 中的檢查

    // 範例的起始點和真實最終目標點列表 ( pair<起始點, 真實目標點> )
    // 例如：代理 0 從 (50, 400) 移動到 (50, 50)
    std::vector<std::pair<Point, Point>> matched_target_and_array_batch = {
        {Point(50, 400), Point(50, 50)},
        {Point(150, 400), Point(150, 50)},
        {Point(250, 400), Point(250, 50)}
        // 添加更多代理的起始點和目標點...
    };

    // 範例的障礙物中心點列表
    std::vector<Point> obstacle_coordinate_changed_btbatch = {
        {Point(300, 200)},
        {Point(400, 300)}
        // 添加更多障礙物中心點...
    };

    // 呼叫 ORCA 規劃函式執行規劃
    auto result = orca_planning_cpp(
        matched_target_and_array_batch,
        obstacle_coordinate_changed_btbatch,
        grid_size, image_width, image_height, step_size,
        Rl, obstacle_radius
    );

    std::vector<std::vector<Point>> final_paths = result.first; // 獲取最終路徑
    bool success = result.second; // 獲取規劃是否成功到達中間目標的狀態

    // 列印規劃結果 (最終路徑)
    if (success) {
        std::cout << "\nORCA Planning Successful! (Reached intermediate goals)" << std::endl;
        for (size_t i = 0; i < final_paths.size(); ++i) {
            std::cout << "Agent " << i << " Final Path:" << std::endl;
            for (const auto& p : final_paths[i]) {
                std::cout << p << " -> ";
            }
            std::cout << "END" << std::endl;
        }
    } else {
        std::cout << "\nORCA Planning Failed or did not reach all intermediate goals within max steps." << std::endl;
         // 即使未完全成功，也可能生成了部分路徑，可以選擇列印出來
         std::cout << "Partial paths generated:" << std::endl;
         for (size_t i = 0; i < final_paths.size(); ++i) {
            std::cout << "Agent " << i << " Partial Path:" << std::endl;
            for (const auto& p : final_paths[i]) {
                std::cout << p << " -> ";
            }
            std::cout << "END" << std::endl;
        }
    }

    return 0; // 程式成功結束
}

/*
如何編譯和執行：

1.  **安裝 RVO2 C++ 函式庫：** 您需要先下載 RVO2 的 C++ 版本的原始碼，並根據其提供的說明進行編譯和安裝。通常使用 CMake 進行編譯。

2.  **保存程式碼：** 將上述 C++ 程式碼保存為一個 `.cpp` 檔案，例如 `orca_simulation.cpp`。

3.  **編譯：** 使用 C++ 編譯器 (如 g++) 進行編譯。您需要提供 RVO2 函式庫頭文件的路徑 (`-I`) 和函式庫檔案的路徑 (`-L`)，並連結 RVO 函式庫 (`-lRVO`)。
    請替換 `/您的/RVO2/函式庫/include路徑` 和 `/您的/RVO2/函式庫/lib路徑` 為您實際安裝 RVO2 函式庫的位置。

    ```bash
    g++ orca_simulation.cpp -o orca_simulation -I/您的/RVO2/函式庫/include路徑 -L/您的/RVO2/函式庫/lib路徑 -lRVO -std=c++11 -Wall -Wextra
    ```
    * `-o orca_simulation`: 指定輸出執行檔的名稱。
    * `-I...`: 指定頭文件搜索路徑。
    * `-L...`: 指定函式庫搜索路徑。
    * `-lRVO`: 連結 RVO 函式庫。
    * `-std=c++11`: 使用 C++11 或更高版本標準。
    * `-Wall -Wextra`: 啟用常用的編譯警告。

4.  **執行：** 編譯成功後，在終端中執行生成的執行檔：

    ```bash
    ./orca_simulation
    ```

程式將會執行模擬並列印出每個代理計算出的路徑點列表。
*/
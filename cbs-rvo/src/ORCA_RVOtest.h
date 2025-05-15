// src/orca_simulation.h
#ifndef ORCA_SIMULATION_H // include guards，防止這個頭文件被重複包含
#define ORCA_SIMULATION_H

#include <vector>    // 引入 std::vector 容器
#include <utility>   // 引入 std::pair
#include <iostream>  // 引入 std::ostream，用於 Point 的輸出運算符重載
#include <RVO.h>     // 引入 RVO2 函式庫的頭文件，因為 Point 結構需要轉換為 RVO::Vector2

// 定義一個簡單的 Point 結構來表示二維座標
struct Point {
    double x, y; // x 和 y 座標

    // 建構函式宣告
    Point(double x_ = 0.0, double y_ = 0.0);

    // 基本的向量運算符宣告 (const 表示這些操作不會修改 Point 物件)
    Point operator+(const Point& other) const; // 向量相加
    Point operator-(const Point& other) const; // 向量相減
    Point operator*(double scalar) const;       // 向量與純量相乘
    Point operator/(double scalar) const;       // 向量除以純量

    // 計算向量的長度 (範數) 宣告
    double norm() const;

    // 將自定義的 Point 結構轉換為 RVO2 函式庫使用的 RVO::Vector2 宣告
    RVO::Vector2 toRVOVector2() const;
};

// 重載輸出串流運算符宣告，以便可以直接列印 Point 物件
std::ostream& operator<<(std::ostream& os, const Point& p);

// 函式宣告：檢查中間點是否與障礙物重疊
// 回傳: 如果重疊，回傳重疊的障礙物中心點指標；否則回傳 nullptr
Point* check_midpoint_and_obstacle(const Point& mid_point, const std::vector<Point>& OBSTACLE_CENTERS, double grid_size);

// 函式宣告：生成中間目標點列表，用於 RVO2 模擬階段
std::vector<Point> generate_mid_points_to_goal(const std::vector<Point>& OBSTACLE_CENTERS, const std::vector<Point>& REAL_GOAL_POSITIONS, double grid_size);

// 函式宣告：生成兩個點之間的直線路徑
std::vector<Point> straight_path(const Point& current_point, const Point& target_center, double step_size);

// 函式宣告：執行 RVO2 模擬的核心函式 (到中間目標點)
// 回傳: 一個 pair，包含所有代理在模擬過程中的路徑列表，以及一個表示模擬是否成功到達中間目標的布林值
std::pair<std::vector<std::vector<Point>>, bool> run_simulation(
    int NUM_CIRCLES, double TIME_STEP, double NEIGHBOR_DIST, double TIME_HORIZON,
    double CIRCLE_RADIUS, double MAX_SPEED, const std::vector<Point>& START_POSITIONS,
    const std::vector<Point>& OBSTACLE_CENTERS, double OBSTACLE_RADIUS,
    double SCENE_HEIGHT, double SCENE_WIDTH, const std::vector<Point>& GOAL_POSITIONS_MID, double grid_size);

// 函式宣告：完成最終路徑，從中間目標點延伸到真實目標點
std::vector<std::vector<Point>> complete_path(
    std::vector<std::vector<Point>>& all_agent_paths, // 注意這裡是非 const 引用，因為這個函式可能會修改傳入的路徑列表
    const std::vector<Point>& REAL_GOAL_POSITIONS,
    const std::vector<Point>& OBSTACLE_CENTERS, double obstacle_radius);

// 主要規劃入口函式宣告 (這是您將通過 pybind11 呼叫的函式)
// 回傳: 一個 pair，包含所有代理的最終路徑列表，以及一個表示模擬是否成功到達中間目標的布林值
std::pair<std::vector<std::vector<Point>>, bool> orca_planning_cpp(
    const std::vector<std::pair<Point, Point>>& matched_target_and_array_batch, // 包含起始點和真實目標點對的列表
    const std::vector<Point>& obstacle_coordinate_changed_btbatch, // 障礙物中心點列表
    double grid_size, double image_width, double image_height, double step_size,
    double Rl, double obstacle_radius);

// 注意：獨立測試用的 main 函式不應該放在頭文件中，它只屬於一個 .cpp 文件。

#endif // ORCA_SIMULATION_H
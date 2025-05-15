#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // 包含對 STL 容器 (如 vector, pair) 的支援
#include <pybind11/complex.h> // 根據需要包含其他型別的支援

// 引入您前面寫好的 C++ 模擬程式碼的頭文件
// 假設您將前面提供的 C++ 程式碼保存為一個或多個 .h/.hpp 和 .cpp 文件
// 如果您目前所有程式碼都在一個 .cpp 文件，可能需要將需要暴露的部分（如 Point 結構和 orca_planning_cpp 函式）
// 提取到一個 .h/.hpp 頭文件中，或者直接 #include 這個 .cpp 文件（不推薦，但可行）
#include "ORCA_RVOtest.h" // 請替換為您實際的頭文件名

namespace py = pybind11;

// PYBIND11_MODULE 宏定義了一個 Python 模組
// 第一個參數是模組名稱（當您在 Python 中 import 時使用的名稱）
// 第二個參數是一個 py::module 物件，用於定義模組的內容
PYBIND11_MODULE(my_orca_module, m) { // 例如，將模組命名為 my_orca_module
    m.doc() = "Pybind11 binding for ORCA RVO2 C++ simulation"; // 模組的文檔字符串

    // 暴露您的 Point 結構給 Python
    // 定義一個 Python 中的 Point 類別
    py::class_<Point>(m, "Point")
        // 暴露 Point 的建構函式
        .def(py::init<double, double>(), py::arg("x") = 0.0, py::arg("y") = 0.0)
        // 暴露 x 和 y 成員變數，允許讀寫
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y);
        // 如果有其他方法，也可以在這裡暴露，例如 .def("norm", &Point::norm)

    // 暴露 orca_planning_cpp 函式給 Python
    // m.def 函數用於暴露 C++ 函式
    m.def("orca_planning", &orca_planning_cpp,
          "Run ORCA planning simulation", // 函式的文檔字符串
          // 定義函式參數名稱，方便 Python 呼叫時使用關鍵字參數
          py::arg("matched_target_and_array_batch"),
          py::arg("obstacle_coordinate_changed_btbatch"),
          py::arg("grid_size"),
          py::arg("image_width"),
          py::arg("image_height"),
          py::arg("step_size"),
          py::arg("Rl"),
          py::arg("obstacle_radius"));

    // Pybind11 會自動處理 std::vector, std::pair, double, bool 的轉換
    // 前提是您包含了 <pybind11/stl.h>
}
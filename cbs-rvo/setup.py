# setup.py
import setuptools
from setuptools import setup, Extension
import pybind11
import os
import platform # 引入 platform 模組來判斷作業系統


# --- 您需要根據您的環境修改以下路徑 ---
# 1. RVO2 函式庫的 include 路徑
#    這應該是指向 RVO2 函式庫安裝後，包含 RVO.h 的資料夾
#    例如：'/usr/local/include/RVO' 或 '/opt/rvo2/include'
#    如果 RVO.h 在 /opt/rvo2/include/RVO/RVO.h，那麼這裡可能是 '/opt/rvo2/include'
#    **請務必修改為您實際的路徑！**
RVO2_INCLUDE_DIR = "C:/Users/Vivo/odepcellarray_fromlab/RVO2-main/src"

# 2. RVO2 函式庫的 lib 路徑
#    這應該是指向 RVO2 函式庫編譯生成的 .a, .so, 或 .lib 檔案所在的資料夾
#    例如：'/usr/local/lib' 或 '/opt/rvo2/lib'
#    **請務必修改為您實際的路徑！**
RVO2_LIB_DIR = "C:/Users/Vivo/odepcellarray_fromlab/RVO2-main/build/src/Debug"

# 3. RVO2 函式庫的名稱
#    在 Linux/macOS 上通常是 'RVO' (對應 libRVO.so 或 libRVO.a)
#    在 Windows 上通常是 'RVO' 或 'RVO2' (對應 RVO.lib 或 RVO2.lib)
#    請檢查您的 RVO2 安裝資料夾來確定正確的名稱
#    **請務必修改為您實際的名稱！**
RVO2_LIB_NAME = 'RVO'
# --- 標準設定，通常不需要修改 ---
# pybind11 的 include 路徑
PYBIND11_INCLUDE_DIR = pybind11.get_include()

# 您的 C++ 源碼資料夾路徑
# 這裡假設您的 C++ 源碼 (.cpp, .h, bind.cpp) 都放在 ./src 資料夾中
SRC_DIR = 'src'

# --- 新增 RVO2 建構目錄 include 路徑 ---
# 請修改為您實際找到 Export.h 的資料夾路徑！
RVO2_BUILD_INCLUDE_DIR = r"C:\Users\Vivo\odepcellarray_fromlab\RVO2-main\build\src" # <-- 請將此路徑替換為 Export.h 所在的資料夾路徑


# 定義要編譯和連結的 C++ 源文件列表
SOURCES = [
    os.path.join(SRC_DIR, 'ORCA_RVOtest.cpp'), # 您的核心 C++ 邏輯實現
    os.path.join(SRC_DIR, 'bind.cpp'), # pybind11 繫結程式碼
]

# 根據不同的作業系統設定不同的編譯參數
compile_args = ['-O3'] # 通用優化參數，可選
link_args = [] # 通用連結參數，可選

if platform.system() == "Windows":
    # 在 Windows 上使用 MSVC
    # /std:c++17 是一個不錯的選擇
    # /O2 是優化旗標
    compile_args = ['/std:c++17', '/O2']
    # 在 Windows 上連結 .lib 文件時，只需要指定庫名稱和庫路徑，setuptools 會處理
    # 如果 RVO2 構建的是 DLL，可能需要額外的處理，但通常 .lib 是夠的
else:
    # 在 Linux 或 macOS 上使用 GCC 或 Clang
    compile_args = ['-std=c++11', '-fPIC', '-O3'] # 可以改為 c++14 或 c++17
    # 在 Linux/macOS 上連結共享庫時，如果庫不在標準路徑，可能需要指定運行時庫路徑
    # link_args = [f"-Wl,-rpath,{RVO2_LIB_DIR}"] # 這種方式可以幫助找到運行時庫

# 定義 C++ 擴展模組
ext_module = Extension(
    'my_orca_module',  # **您希望在 Python 中使用的模組名稱 (例如: import my_orca_module)**
                       # 請注意這個名稱與 bind.cpp 中的 PYBIND11_MODULE 名稱一致
    sources=SOURCES,    # 要編譯的源文件列表
    include_dirs=[
        PYBIND11_INCLUDE_DIR, # 包含 pybind11 的頭文件路徑
        RVO2_INCLUDE_DIR,# 包含 RVO2 函式庫的頭文件路徑
        RVO2_BUILD_INCLUDE_DIR,
        SRC_DIR               # 包含您自己的頭文件 (orca_simulation.h) 的路徑
    ],
    library_dirs=[RVO2_LIB_DIR], # RVO2 函式庫文件所在的資料夾路徑
    libraries=[RVO2_LIB_NAME], # 要連結的 RVO2 函式庫的名稱 (不帶 .lib/.so/.dll 副檔名)
    language='c++', # 指定源文件是 C++ 語言
    extra_compile_args=compile_args, # 使用上面根據編譯器設定的參數
    extra_link_args=link_args, # 使用上面設定的連結參數
    # runtime_library_dirs=[RVO2_LIB_DIR] # 如果 RVO2 是共享庫且不在系統標準路徑，可能需要添加
)

# 呼叫 setuptools 的 setup 函式來定義專案
setup(
    name='my_orca_module_project', # **您的整個 Python 專案名稱** (這個名稱可以與模組名稱不同)
    version='1.0',        # 專案版本
    description='ORCA RVO2 C++ simulation wrapped with Pybind11', # 專案描述
    ext_modules=[ext_module], # 包含您定義的擴展模組列表
    # 其他 metadata (作者, 授權等) 可以根據需要添加
    # install_requires=['pybind11>=2.6'] # 如果需要指定 pybind11 為依賴
)
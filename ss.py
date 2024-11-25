import pyautogui
import time
import os
from datetime import datetime
from tkinter import Tk
from tkinter.filedialog import askdirectory

# 啟動 tkinter 根視窗 (避免顯示空白視窗)
root = Tk()
root.withdraw()  # 隱藏主視窗

# 使用文件選擇對話框來選擇儲存的資料夾
screenshot_directory = askdirectory(title="選擇儲存截圖的資料夾")
if not screenshot_directory:
    print("未選擇資料夾，程序將終止。")
    exit()

# 確保所選資料夾存在
if not os.path.exists(screenshot_directory):
    os.makedirs(screenshot_directory)

# 設定幀率
frame_rate = 3  # 每秒截圖數量

# 設定截圖範圍 (左上角 x, y 右下角 x, y)
left, top, right, bottom = 250, 45, 2325, 1435  # 這裡設置你想要的區域
region = (left, top, right - left, bottom - top)

# 設定開始時間
start_time = time.time()

# 設定日期資料夾
def get_today_folder():
    today = datetime.today().strftime('%Y-%m-%d')  # 取得今天的日期
    folder_path = os.path.join(screenshot_directory, today)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# 獲取今天資料夾內的最高編號
def get_starting_counter(today_folder):
    max_counter = 0
    for filename in os.listdir(today_folder):
        # 檢查文件是否符合日期_數字.png 格式
        if filename.startswith(datetime.today().strftime('%Y-%m-%d')) and filename.endswith(".png"):
            try:
                count = int(filename.split('_')[-1].split('.')[0])
                if count > max_counter:
                    max_counter = count
            except ValueError:
                continue
    return max_counter + 1

# 開始截圖
try:
    today_folder = get_today_folder()
    counter = get_starting_counter(today_folder)
    initial_counter = counter

    while True:
        # 計算當前時間與起始時間的差，並判斷是否達到幀率要求
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= 1 / frame_rate:  # 每秒幾幀
            # 進行截圖
            screenshot = pyautogui.screenshot(region=region)
            
            # 檔案名稱格式：YYYY-MM-DD_第n張.png
            screenshot_filename = f"{datetime.today().strftime('%Y-%m-%d')}_{counter}.png"
            
            # 保存截圖
            screenshot.save(os.path.join(today_folder, screenshot_filename))
            
            # 更新截圖計數器和時間
            counter += 1
            start_time = time.time()  # 重置時間

            print(f"Screenshot saved as {os.path.join(today_folder, screenshot_filename)}")
            
        # 可以根據需要做延遲，這裡設置為小延遲防止過度佔用 CPU
        time.sleep(0.01)

except KeyboardInterrupt:
    print(f"截圖已停止。今天已拍了 {counter - initial_counter} 張照片。")

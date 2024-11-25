import mss
import time
import os
from datetime import datetime
from tkinter import Tk, Label, Entry, Button
from tkinter.filedialog import askdirectory
import keyboard  # 用於鍵盤事件監聽
from PIL import Image

# 啟動 tkinter 根視窗 (避免顯示空白視窗)
root = Tk()
root.withdraw()  # 隱藏主視窗

# 使用文件選擇對話框來選擇儲存的資料夾
screenshot_directory = askdirectory(title="選擇儲存截圖的資料夾")
if not screenshot_directory:
    print("未選擇資料夾，程序將終止。")
    exit()

# 設定日期資料夾
def get_today_folder():
    today = datetime.today().strftime('%Y.%m.%d')  # 取得今天的日期
    folder_path = os.path.join(screenshot_directory, today)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# 獲取今天資料夾內的最高編號
def get_starting_counter(today_folder):
    max_counter = 0
    for filename in os.listdir(today_folder):
        # 檢查文件是否符合日期_數字.png 格式
        if filename.startswith(datetime.today().strftime('%Y.%m.%d')) and filename.endswith(".png"):
            try:
                count = int(filename.split('_')[-1].split('.')[0])
                if count > max_counter:
                    max_counter = count
            except ValueError:
                continue
    return max_counter + 1

# 視窗讓使用者輸入幀率
def get_frame_rate():
    # 創建一個共享變數來存儲幀率
    frame_rate_var = {'frame_rate': 0}

    def on_submit():
        try:
            frame_rate_var['frame_rate'] = int(entry.get())
            print(f"Frame rate set to: {frame_rate_var['frame_rate']}")
            window.quit()  # 關閉視窗
        except ValueError:
            label_error.config(text="請輸入有效的數字")

    window = Tk()
    window.title("設定幀率")

    label = Label(window, text="請輸入幀率 (每秒截圖數量):")
    label.pack()

    entry = Entry(window)
    entry.pack()

    label_error = Label(window, text="", fg="red")
    label_error.pack()

    submit_button = Button(window, text="確定", command=on_submit)
    submit_button.pack()

    window.mainloop()

    return frame_rate_var['frame_rate']

# 開始截圖的主程式
def start_screenshotting(frame_rate):
    # 開始截圖
    today_folder = get_today_folder()
    counter = get_starting_counter(today_folder)
    initial_counter = counter
    start_time = time.time()

    print(f"按下空白鍵開始截圖，按下 End 鍵結束截圖...")

    # 使用 mss 來截圖
    with mss.mss() as sct:
        # 定義截圖區域
        region = (280, 153, 2094, 1373)    #ccd圖像fit in windows (ctrl+0) size: 3088*2076

        # 等待空白鍵開始
        while True:
            if keyboard.is_pressed('space'):  # 按下空白鍵開始截圖
                print("開始截圖...")
                break
            time.sleep(0.1)

        try:
            while True:
                # 計算當前時間與起始時間的差，並判斷是否達到幀率要求
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if elapsed_time >= 1 / frame_rate:  # 每秒幾幀
                    # 進行截圖
                    screenshot = sct.grab(region)
                    img = Image.frombytes('RGB', (screenshot.width, screenshot.height), screenshot.rgb)

                    # 檔案名稱格式：YYYY.MM.DD_001.png，001、002、003 等
                    screenshot_filename = f"{datetime.today().strftime('%Y.%m.%d')}_{str(counter).zfill(3)}.png"
                    
                    # 保存截圖
                    screenshot_path = os.path.join(today_folder, screenshot_filename)
                    img.save(screenshot_path)
                    
                    # 更新截圖計數器和時間
                    counter += 1
                    start_time = time.time()  # 重置時間

                    print(f"Screenshot saved as {screenshot_path}")

                # 檢查是否按下 End 鍵結束截圖
                if keyboard.is_pressed('end'):
                    print(f"截圖已停止。今天已拍了 {counter - initial_counter} 張照片。")
                    break

                # 可以根據需要做延遲，這裡設置為小延遲防止過度佔用 CPU
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print(f"截圖已停止。今天已拍了 {counter} 張照片。")

# 主程序
frame_rate = get_frame_rate()
start_screenshotting(frame_rate)

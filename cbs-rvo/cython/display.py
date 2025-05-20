import cv2
import numpy as np
from screeninfo import get_monitors

# 全局變數，用於追蹤視窗是否已經初始化
_display_window_initialized = False
_display_window_name = 'Movement Simulation' # 視窗名稱，保持一致

def display_on_specific_monitor(image_to_display: np.ndarray, target_monitor_index: int = 2, center_position: tuple = None, interpolation_method=cv2.INTER_AREA):
    """
    在指定螢幕上顯示圖片，並顯示等比例縮放的圖片

    Args:
        image_to_display (np.ndarray): 要顯示的圖片 (OpenCV 圖像格式)。
        target_monitor_index (int): 目標螢幕的索引 (0-indexed)。預設為 2 (即第三個螢幕: 投影機)。
    """
    global _display_window_initialized # 宣告使用全局變數
    global _display_window_name # 宣告使用全局變數
    
    # 根據投影機設定，視窗height=130，寬度等比例縮放(四捨五入)
    display_height = 130

    # 1.縮小圖片
    original_h, original_w = image_to_display.shape[:2]
    # 避免除以零或無意義的縮放
    if original_h == 0 or display_height <= 0:
        print("警告: 原始圖片高度為零或 display_height 無效。無法縮放。")
        resized_image = image_to_display # 回退到使用原始圖片
    else:
        scale = display_height / original_h
        target_w = round(original_w * scale)
        # 確保縮放後的寬度至少為 1 像素，避免錯誤
        if target_w == 0:
            target_w = 1 
        resized_image = cv2.resize(image_to_display, (target_w, display_height), interpolation=interpolation_method)
        window_width, window_height = resized_image.shape[1], display_height
        
    if not _display_window_initialized:
        # 只在第一次呼叫時執行視窗初始化和設定
        cv2.namedWindow(_display_window_name, cv2.WINDOW_NORMAL)

        monitors = get_monitors()
        target_monitor = None

        if len(monitors) > target_monitor_index:
            target_monitor = monitors[target_monitor_index]
            print(f"Found target monitor (Index {target_monitor_index}): {target_monitor.name} at ({target_monitor.x}, {target_monitor.y}) with size {target_monitor.width}x{target_monitor.height}")
        elif len(monitors) > 0:
            print(f"Warning: Less than {target_monitor_index + 1} monitors found. Using the primary monitor instead.")
            for m in monitors:
                if m.is_primary:
                    target_monitor = m
                    break
            if target_monitor is None: # 如果連主螢幕都找不到，就用第一個螢幕
                target_monitor = monitors[0]
                print("Warning: Primary monitor not found. Using the first available monitor.")
        else:
            print("Error: No monitors found. Cannot set window position or fullscreen mode. Displaying on default monitor.")
            # 如果沒有螢幕，則不進行任何特定視窗設定，讓 imshow 使用預設行為
            target_monitor = None

        if target_monitor:
            if center_position:
                center_x, center_y = center_position
            else: #使用預設
                center_x = 499
                center_y = 355

            top_left_x = int(target_monitor.x + center_x - window_width / 2)
            top_left_y = int(target_monitor.y + center_y - window_height / 2)
            
            cv2.resizeWindow(_display_window_name, target_w, window_height)
            cv2.moveWindow(_display_window_name, top_left_x, top_left_y)

        _display_window_initialized = True # 標記為已初始化

    # 顯示圖片到已初始化和設定好的視窗
    cv2.imshow(_display_window_name, resized_image)
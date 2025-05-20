import cv2
import numpy as np
from screeninfo import get_monitors

# 全局變數，用於追蹤視窗是否已經初始化
_display_window_initialized = False
_display_window_name = 'Movement Simulation' # 視窗名稱，保持一致

def display_on_specific_monitor(image_to_display: np.ndarray, target_monitor_index: int = 2):
    """
    在指定螢幕上顯示圖片，並將視窗設定為全螢幕。

    Args:
        image_to_display (np.ndarray): 要顯示的圖片 (OpenCV 圖像格式)。
        target_monitor_index (int): 目標螢幕的索引 (0-indexed)。
                                   預設為 2 (即第三個螢幕)。
    """
    global _display_window_initialized # 宣告使用全局變數

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
            # 將視窗移動到目標螢幕的左上角
            cv2.moveWindow(_display_window_name, target_monitor.x, target_monitor.y)

            # 設定視窗為全螢幕模式
            cv2.setWindowProperty(_display_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print(f"Window '{_display_window_name}' set to fullscreen on monitor: {target_monitor.name}")

        _display_window_initialized = True # 標記為已初始化

    # 顯示圖片到已初始化和設定好的視窗
    cv2.imshow(_display_window_name, image_to_display)
import cv2
import numpy as np
from collections import deque

# def process_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (9, 9), 2)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     kernel = np.ones((5, 5), np.uint8)
#     binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#     binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel)

#     contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     min_area = np.pi * (12 ** 2)
#     max_area = np.pi * (20 ** 2)
#     result_centers = []

#     for contour in contours:
#         area = cv2.contourArea(contour)
#         perimeter = cv2.arcLength(contour, True)
#         if perimeter == 0:
#             continue
#         circularity = 4 * np.pi * (area / (perimeter ** 2))
#         if min_area < area < max_area and 0.8 < circularity < 1.2:
#             M = cv2.moments(contour)
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 result_centers.append((cx, cy))

#     return result_centers

# def extract_and_process_realtime(video_path, target_fps=10, buffer_size=100):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("無法開啟影片:", video_path)
#         return

#     original_fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_interval = int(original_fps / target_fps)
#     print(f"原始 FPS: {original_fps}")
#     print(f"每 {frame_interval} 幀擷取並處理一張圖片")
#     print(f"環形緩衝區大小: {buffer_size} 張圖片")

#     frame_count = 0
#     buffer = deque(maxlen=buffer_size)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % frame_interval == 0:
#             buffer.append(frame.copy())
#             print(f"[Frame {frame_count}] 加入 buffer（目前長度: {len(buffer)}）")

#             # 即時處理最新的一張 frame
#             centers = process_frame(frame)
#             print(f"→ 偵測到 {len(centers)} 個符合條件的 contour 中心點: {centers}")

#         frame_count += 1

#         #若要即時顯示，可取消註解以下這行
#         cv2.imshow("Live Frame", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("影片處理完成")


# # 使用範例
# video_file = r"C:\Users\Vivo\CGU\odep_cellarray\Testimage\test_video.mp4"
# extract_and_process_realtime(video_file, target_fps=5, buffer_size=50)


import os 
import cv2
import numpy as np
import cv2
import os
from collections import deque
import math

def process_image(image_path):
    rawimage = cv2.imread(image_path)
    new_width = 1840
    new_height = 1220
    resized_image = cv2.resize(rawimage, (new_width, new_height))
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_mask = np.zeros_like(gray)
    min_area = np.pi * (1 ** 2)
    max_area = np.pi * (15 ** 2)
    centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if min_area < area < max_area and 0.8 < circularity < 1.2:
            cv2.drawContours(result_mask, [contour], -1, 255, -1)
            # 計算中心點
            # 改用包圓法找中心點
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            centers.append(center)
            cv2.circle(resized_image, center, 3, (0, 255, 0), -1)
                # 可選：在原圖上畫出中心點
                # cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
    # scale_percent = 50  # 缩放比例
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)

    # resized_image= cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # resized_image_mask= cv2.resize(result_mask, dim, interpolation=cv2.INTER_AREA)
    # # 顯示結果（可選）
    # cv2.imshow("Result Mask", resized_image_mask)
    # cv2.imshow("Detected Centers", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return centers, resized_image

def simulate_yolo():
    image_path = r"C:\Users\Vivo\CGU\odep_cellarray\Testimage\test.JPG"
    obsticle_coordinates = []
    centers, resized_image = process_image(image_path)
    print(f"偵測到 {len(centers)} 個符合條件的 contour 中心點: {centers}")
    for i in range(5):
        while True:
            x, y = np.random.randint(60*int(math.sqrt(64)), 1814), np.random.randint(60*int(math.sqrt(64)), 1220)
            if all(math.sqrt((x - cx)**2 + (y - cy)**2) > 4 * 20 for cx, cy in centers):
                obsticle_coordinates.append((x, y))
                break
    return centers, resized_image, obsticle_coordinates
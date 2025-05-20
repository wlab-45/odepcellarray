import os
from PIL import Image

def flip_and_rename_images(folder_path):
    # 取得資料夾內所有 JPG 檔案並排序
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

    # 確保資料夾內有圖片
    if not images:
        print("資料夾中沒有 JPG 圖片。")
        return
    
    # 找到最後一個圖片的數字編號
    last_image = images[-1]
    last_number = int(last_image.split('_')[-1].split('.')[0])
    
    # 翻轉圖片並重新命名
    new_number = last_number + 1
    for image_name in images:
        # 開啟圖片並進行左右翻轉
        image_path = os.path.join(folder_path, image_name)
        img = Image.open(image_path)
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 生成新的檔名
        date_part = image_name.split('_')[0]  # 提取日期部分
        new_image_name = f"{date_part}_{new_number:03d}.jpg"
        new_image_path = os.path.join(folder_path, new_image_name)
        
        # 儲存翻轉後的圖片
        flipped_img.save(new_image_path)
        print(f"已儲存: {new_image_name}")
        
        new_number += 1

# 使用範例（請將 "your_folder_path" 替換為你的資料夾路徑）
folder_path = "C:/Users/Wu Lab/Desktop/sam2/dataset/images"
flip_and_rename_images(folder_path)

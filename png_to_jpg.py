import os
from PIL import Image

def convert_png_to_jpg(folder_path):
    # 遍歷資料夾內的所有檔案
    for filename in os.listdir(folder_path):
        # 確認檔案是否為 .png 格式
        if filename.endswith('.png'):
            # 取得完整檔案路徑
            png_path = os.path.join(folder_path, filename)
            
            # 開啟圖片
            with Image.open(png_path) as img:
                # 取得不含副檔名的檔案名稱
                base_filename = os.path.splitext(filename)[0]
                # 設定輸出的 .jpg 檔案路徑
                jpg_path = os.path.join(folder_path, base_filename + '.jpg')
                
                # 轉換為 RGB 模式並儲存為 .jpg 格式
                img.convert('RGB').save(jpg_path, 'JPEG')
                print(f"已轉換: {filename} -> {base_filename}.jpg")

# 指定你的資料夾路徑
folder_path = 'C:/Users/Wu Lab/Desktop/sam2/dataset/masks'  # 替換成你的實際路徑
convert_png_to_jpg(folder_path)

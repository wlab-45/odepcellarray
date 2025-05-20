
from tkinter import Tk 
from tkinter.filedialog import askdirectory 
import os

def rename_png_files(directory): 
    # 獲取所有 PNG 文件的列表 
    files = [f for f in os.listdir(directory) if f.endswith(".png")] 
    files.sort()  # 按照字母順序排序文件 
    File_number = 0 

    # 依次重命名文件 

    for index, filename in enumerate(files):  # 使用 enumerate 獲取索引和文件名 
        File_number += 1 
        # 分析文件名以提取日期部分 

        basefilename, _ = os.path.splitext(filename)  # 去掉擴展名 
        parts = basefilename.split("_")  # 假設文件名格式為 “YYYYMMDD_任意名稱” 

        if len(parts) > 1:  # 確保有足夠的部分 
            file_date = parts[0]  # 假設第一部分是日期 
            new_filename = f"{file_date}_{File_number:03d}.jpg"  # 使用四位數格式化序號 
            old_file = os.path.join(directory, filename)  # 獲取舊文件的完整路徑 
            new_file = os.path.join(directory, new_filename)  # 獲取新文件的完整路徑 

            try: 
                os.rename(old_file, new_file)  # 重命名文件 
                print(f"Renamed'{filename}'to'{new_filename}'") 
            except Exception as e: 
                print(f"Error renaming '{filename}': {e}") 
        else: 
            print(f"Filename '{filename}' does not match expected format.")
    print("Renaming complete.") 

 

def main(): 
    # 隱藏主 tkinter 視窗 
    Tk().withdraw() 
    # 打開文件對話框以選擇資料夾 
    user_input_directory = askdirectory(title="選擇重新命名的資料夾") 
    if not user_input_directory: 
        print("未選擇資料夾，程式結束。") 
        return
    rename_png_files(user_input_directory) 

 

if __name__ == "__main__": 
    main() 
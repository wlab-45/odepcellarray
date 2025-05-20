
主要執行程式為 whole_pathfinding.py
1. 在main下方有一行是調用simulate_yolo函數，那是為了測試使用，需要將其註解(import也要註解)。

2. 剩餘的規劃函數由wholestep3_draw_array_picture開始， 期望收到的至少有粒子座標列表: all_coordinate= [(x1,y1), (x2,y2), .....]
還有障礙物列表(先使用generate_obstacle_coordinates)隨機生成，之後會根據yolo的結果以及拉不動的粒子生成

3. 至少傳入一張image(無論甚麼image都可以，原本是為了要確認debug用)，現在我把所有imshow跟imwrite都註解掉了(只有影片會顯示)，所以不用管image是啥，也不用管路徑設定(確定可以使用我再改掉他們)

4. orca_rvo2.py 是路徑規畫的主程式，會輸出的是路徑列表 (現在不須更改)

5. cython有兩個檔案，是為了加速使用的。 第一個是functions_cython包含主要程式會用到的函數(非必要盡量避免更改，格式跟python稍有不同，是c+python的結合版)。  如果真的有更改，在兩個cython檔案最後一行有一個run: ，必須進到cython檔案所在位置執行才可編譯(編譯設定在setup.py，非必要避免更改)

6. display.py負責resize圖片以及控制投影。 裡面有center_x與center_y可以更動投影的中間點
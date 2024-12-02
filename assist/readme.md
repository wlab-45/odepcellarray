input_path = 原始圖像的資料夾

mask_path = 會根據opencv偵測到的東西生成初步的mask， 但是有可能會有無法辨識的點， 因此要再用labelme 重新檢查標記

最後會把 mask圖 再轉回json， 然後輸出回到與原始圖片同一個資料夾   (一定要同一個!!!!!)

接著用labelme 打開含有json與原始圖片的資料夾， 再微調即可
from PIL import Image, ExifTags
import os

# 設定輸入與輸出資料夾
input_folder = r'D:\Original\Fire_hydrant'
output_folder = r'E:\Machine Learning\Midterm_Project\train\Fire_hydrant'
resize_width = 180
resize_height = 240

# 建立輸出資料夾（如果還沒存在）
os.makedirs(output_folder, exist_ok=True)

# 支援的圖片副檔名
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 處理每一張圖片，並重新命名
for idx, filename in enumerate(os.listdir(input_folder), start=1):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_folder, filename)

        # 生成新的檔案名稱 (speed-camera-0001.jpg, speed-camera-0002.jpg, ...)
        new_filename = f"Speed_camera-{idx:04d}{os.path.splitext(filename)[1]}"
        output_path = os.path.join(output_folder, new_filename)

        # 開啟圖片並修正方向
        with Image.open(input_path) as img:
            # 處理 EXIF 資訊，修正圖片的旋轉
            try:
                exif = img._getexif()
                if exif is not None:
                    for tag, value in exif.items():
                        if ExifTags.TAGS.get(tag, tag) == 'Orientation':
                            if value == 3:
                                img = img.rotate(180, expand=True)
                            elif value == 6:
                                img = img.rotate(270, expand=True)
                            elif value == 8:
                                img = img.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # 沒有 EXIF 資訊，或無法修正，忽略
                pass

            # 縮放圖片
            resized_img = img.resize((resize_width, resize_height), Image.LANCZOS)
            resized_img.save(output_path)
            print(f'已處理並重新命名：{new_filename}')

print('所有圖片已縮放並重新命名完成！')

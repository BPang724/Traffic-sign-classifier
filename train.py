# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# import os

# # 參數設定
# IMG_SIZE = (128, 128)
# BATCH_SIZE = 32
# EPOCHS = 20

# # 資料夾結構需為：
# # dataset/
# # ├── train/
# # │   ├── stop_sign/
# # │   ├── speed_limit/
# # │   ├── ...
# # └── test/
# #     ├── stop_sign/
# #     ├── speed_limit/
# #     ├── ...

# # 資料前處理與增強
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_data = train_datagen.flow_from_directory(
#     "dataset/train",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )
# test_data = test_datagen.flow_from_directory(
#     "dataset/test",
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# # 建立 CNN 模型
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(train_data.num_classes, activation='softmax')
# ])

# # 編譯模型
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # 模型訓練
# history = model.fit(
#     train_data,
#     epochs=EPOCHS,
#     validation_data=test_data
# )

# # 模型評估
# loss, acc = model.evaluate(test_data)
# print(f"測試準確率: {acc:.2%}")

# # 儲存模型
# model.save("traffic_sign_classifier.h5")
# print("模型已儲存為 traffic_sign_classifier.h5")
























import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# 參數設定
IMG_SIZE = (180, 240)  # 更新為 180x240
BATCH_SIZE = 32
EPOCHS = 20

# 絕對路徑設定
TRAIN_DIR = r"E:\Machine Learning\Midterm_Project\dataset\train"  # 訓練資料夾的絕對路徑
TEST_DIR = r"E:\Machine Learning\Midterm_Project\dataset\test"    # 測試資料夾的絕對路徑

# 資料前處理與增強
train_datagen = ImageDataGenerator(
    rescale=1./255,          # 將圖片像素值正規化到 [0, 1]
    rotation_range=20,      # 隨機旋轉圖片
    zoom_range=0.2,         # 隨機縮放圖片
    horizontal_flip=True    # 隨機水平翻轉圖片
)
test_datagen = ImageDataGenerator(rescale=1./255)  # 測試資料只進行正規化

# 訓練資料與測試資料讀取
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,  # 訓練資料的資料夾路徑
    target_size=IMG_SIZE,  # 將圖片尺寸統一為 180x240
    batch_size=BATCH_SIZE,  # 每批次的圖片數量
    class_mode='categorical'  # 分類模式（假設為多類別分類）
)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,  # 測試資料的資料夾路徑
    target_size=IMG_SIZE,  # 將圖片尺寸統一為 180x240
    batch_size=BATCH_SIZE,  # 每批次的圖片數量
    class_mode='categorical'  # 分類模式
)

# 建立 CNN 模型
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # 第一層卷積層
    MaxPooling2D(2,2),  # 最大池化層
    Conv2D(64, (3,3), activation='relu'),  # 第二層卷積層
    MaxPooling2D(2,2),  # 最大池化層
    Conv2D(128, (3,3), activation='relu'),  # 第三層卷積層
    MaxPooling2D(2,2),  # 最大池化層
    Flatten(),  # 扁平化層
    Dense(128, activation='relu'),  # 全連接層
    Dropout(0.5),  # Dropout層，用來防止過擬合
    Dense(train_data.num_classes, activation='softmax')  # 輸出層，對應到不同類別
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data  # 驗證資料集
)

# 評估模型
loss, acc = model.evaluate(test_data)
print(f"測試準確率: {acc:.2%}")

# 儲存模型
model.save("traffic_sign_classifier.h5")
print("模型已儲存為 traffic_sign_classifier.h5")


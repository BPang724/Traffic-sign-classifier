import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 參數設定
IMG_SIZE = (180, 240)  # 更新為 180x240
BATCH_SIZE = 32
EPOCHS = 50

# 絕對路徑設定
TRAIN_DIR = r"E:\Machine Learning\Midterm_Project\dataset\train"  # 訓練資料夾的絕對路徑
TEST_DIR = r"E:\Machine Learning\Midterm_Project\dataset\test"    # 測試資料夾的絕對路徑

# 資料前處理與增強
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 將圖片像素值正規化到 [0, 1]
    validation_split=0.2,   # 20% 資料切成 validation
    rotation_range=20,      # 隨機旋轉圖片
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,         # 隨機縮放圖片
    fill_mode='nearest',    
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # 增加亮度範圍
    shear_range=0.1,  # 隨機剪切
    channel_shift_range=20.0  # 隨機改變顏色通道
)

# 訓練資料與驗證資料讀取
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,  # 訓練資料的資料夾路徑
    target_size=IMG_SIZE,  # 將圖片尺寸統一為 180x240
    batch_size=BATCH_SIZE,  # 每批次的圖片數量
    class_mode='categorical',  # 分類模式（假設為多類別分類）
    subset='training'  # 這部分是訓練集
)

val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,  # 訓練資料的資料夾路徑
    target_size=IMG_SIZE,  # 將圖片尺寸統一為 180x240
    batch_size=BATCH_SIZE,  # 每批次的圖片數量
    class_mode='categorical',  # 分類模式
    subset='validation'  # 這部分是驗證集
)

# 測試資料只進行正規化
test_datagen = ImageDataGenerator(rescale=1./255)  
test_data = test_datagen.flow_from_directory(
    TEST_DIR,  # 測試資料的資料夾路徑
    target_size=IMG_SIZE,  # 將圖片尺寸統一為 180x240
    batch_size=BATCH_SIZE,  # 每批次的圖片數量
    class_mode='categorical',  # 分類模式
    shuffle=False  # 評估時不打亂順序
)

# 建立 CNN 模型
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),  # 第一層卷積層
    MaxPooling2D(2,2),  # 最大池化層
    Conv2D(32, (3,3), activation='relu'),  # 第二層卷積層
    MaxPooling2D(2,2),  # 最大池化層
    Conv2D(64, (3,3), activation='relu'),  # 第三層卷積層
    MaxPooling2D(2,2),  # 最大池化層
    Flatten(),  # 扁平化層
    Dense(128, activation='relu'),  # 全連接層
    Dropout(0.5),  # Dropout層，用來防止過擬合
    Dense(train_data.num_classes, activation='softmax')  # 輸出層，對應到不同類別
])

from tensorflow.keras.optimizers import Adam
# 設定學習率
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)

# 編譯模型時使用這個優化器
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data  # 驗證資料集
)

# 評估模型
loss, acc = model.evaluate(test_data)
print(f"最終測試準確率: {acc:.2%}")

# 儲存模型
model.save("traffic_sign_classifier.h5")
print("模型已儲存為 traffic_sign_classifier.h5")

# 繪製訓練過程中的 Accuracy 和 Loss 圖表
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 5))

    # Accuracy 圖
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss 圖
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# 預測測試資料
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 顯示分類報告
report = classification_report(y_true, y_pred, target_names=class_labels)
print("分類報告:")
print(report)

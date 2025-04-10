import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 載入模型
model = load_model("traffic_sign_classifier.h5")

# 圖片大小必須與訓練時一致
IMG_SIZE = (180, 240)

# 這裡要跟訓練集的 class 順序對應（使用訓練好的資料建立一次以取得 class_labels）
from tensorflow.keras.preprocessing.image import ImageDataGenerator
temp_gen = ImageDataGenerator(rescale=1./255)
temp_data = temp_gen.flow_from_directory(
    r"E:\Machine Learning\Midterm_Project\dataset\train",  # 訓練資料的絕對路徑
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='categorical'
)
class_labels = list(temp_data.class_indices.keys())

# 要預測的圖片資料夾路徑
test_folder = r"E:\Machine Learning\Midterm_Project\dataset\predict"  # 預測資料夾的絕對路徑

# 預測並顯示圖片與結果
for filename in os.listdir(test_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 只處理圖片檔案
        img_path = os.path.join(test_folder, filename)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 預測
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]

        # 顯示圖片與預測結果
        plt.imshow(img)
        plt.title(f"預測類別: {predicted_label}")
        plt.axis('off')
        plt.show()

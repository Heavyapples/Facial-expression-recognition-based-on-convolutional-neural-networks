import cv2
import numpy as np
import pandas as pd
import os

def load_and_preprocess_image(image_path):
    # 加载图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 灰度化
    img = cv2.resize(img, (48, 48))  # 调整大小
    img = img / 255.0  # 归一化
    return img

# 遍历所有图片并进行预处理
images = []
labels = []
for subject in range(1, 45):  # 44个subject
    subject_path = f"E:/project2_database/enterface database/subject {subject}"
    for emotion in ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']:
        emotion_path = os.path.join(subject_path, emotion)
        if os.path.exists(emotion_path):
            for sentence in range(1, 6):  # 5个sentence
                sentence_path = os.path.join(emotion_path, f"sentence {sentence}")
                if os.path.exists(sentence_path):
                    for frame in os.listdir(sentence_path):
                        if frame.endswith('.jpg'):
                            frame_path = os.path.join(sentence_path, frame)
                            img = load_and_preprocess_image(frame_path)
                            images.append(img)
                            labels.append(emotion)

# 转换为numpy数组
images = np.array(images)
labels = np.array(labels)

# 将二维的图像数据展平为一维
flattened_images = images.reshape(len(images), -1)

# 创建一个pandas DataFrame，然后保存到CSV文件
df = pd.DataFrame(flattened_images)
df['label'] = labels
df.to_csv('preprocessed_data.csv', index=False)

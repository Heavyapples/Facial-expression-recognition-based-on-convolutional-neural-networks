from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('preprocessed_data.csv')

# 提取标签并进行编码
labels = df['label'].values
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# 提取并恢复图像形状
images = df.drop(columns='label').values
images = images.reshape(len(images), 48, 48, 1)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 读取模型
model = load_model('emotion_model.h5')

# 预测测试集
y_pred = model.predict(X_test)

# 转换预测结果为标签形式
y_pred = np.argmax(y_pred, axis=1)

# 转换测试集标签为标签形状
y_true = np.argmax(y_test, axis=1)

# 打印各类别的分类报告
print(classification_report(y_true, y_pred, target_names=le.classes_))

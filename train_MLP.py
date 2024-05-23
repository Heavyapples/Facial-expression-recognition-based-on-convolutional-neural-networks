from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取数据
df = pd.read_csv('preprocessed_data.csv')

# 提取标签并进行编码
labels = df['label'].values
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# 提取并恢复图像形状
images = df.drop(columns='label').values
images = images.reshape(len(images), 48*48)  # 对于MLP,我们不需要保留二维结构

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 创建模型
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(48*48,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# 编译模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# 保存模型
model.save('mlp_emotion_model.h5')

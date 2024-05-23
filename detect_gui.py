import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image, ImageTk

# 加载模型
model = load_model('emotion_model.h5')

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((250, 250), Image.LANCZOS)
        image = ImageTk.PhotoImage(image)
        label_image.config(image=image)
        label_image.image = image
        label_image.file_path = file_path

def predict_emotion():
    if hasattr(label_image, 'file_path'):
        img = cv2.imread(label_image.file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = img.reshape(1, 48, 48, 1)

        prediction = model.predict(img)
        emotion = np.argmax(prediction)
        emotion_dict = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Surprise'}
        label_result.config(text=f"情感: {emotion_dict[emotion]}")

root = tk.Tk()

# 创建画布
canvas = tk.Canvas(root, width=500, height=500)
canvas.pack()

# 创建标签用于显示图像和结果
label_image = tk.Label(root)
label_image.place(x=125, y=50)

label_result = tk.Label(root, font=("Arial", 14))
label_result.place(x=180, y=320)

# 创建按钮
button_load = tk.Button(root, text="导入图像", command=load_image, width=20, height=2, font=("Arial", 14))
button_load.place(x=136, y=350)

button_predict = tk.Button(root, text="检测情感", command=predict_emotion, width=20, height=2, font=("Arial", 14))
button_predict.place(x=136, y=420)

root.mainloop()

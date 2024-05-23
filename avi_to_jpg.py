import os
import cv2

# 遍历所有subject的文件夹
for subject in range(1, 45):  # 44个subject
    subject_path = f"E:/project2_database/enterface database/subject {subject}"
    print(f"Checking {subject_path}...")

    # 情绪的缩写字典
    emotion_dict = {'anger': 'an', 'disgust': 'di', 'fear': 'fe', 'happiness': 'ha', 'sadness': 'sa', 'surprise': 'su'}

    # 遍历每个subject文件夹中的每个情绪
    for emotion, emotion_abbr in emotion_dict.items():
        emotion_path = os.path.join(subject_path, emotion)
        print(f"  Checking {emotion_path}...")

        # 检查情绪文件夹是否存在
        if os.path.exists(emotion_path):
            # 遍历每个情绪文件夹中的每个sentence
            for sentence in range(1, 6):  # 5个sentence
                sentence_path = os.path.join(emotion_path, f"sentence {sentence}")
                print(f"    Checking {sentence_path}...")

                # 检查sentence文件夹是否存在
                if os.path.exists(sentence_path):
                    # 视频文件的路径
                    video_path = os.path.join(sentence_path, f's{subject}_{emotion_abbr}_{sentence}.avi')
                    print(f"      Checking {video_path}...")

                    # 检查视频文件是否存在
                    if os.path.exists(video_path):
                        # 使用cv2读取视频文件
                        vidcap = cv2.VideoCapture(video_path)

                        if vidcap.isOpened():
                            # 初始化帧计数器
                            count = 0

                            # 循环读取视频的每一帧
                            while True:
                                # 读取一帧
                                success, image = vidcap.read()

                                # 如果读取成功，保存该帧为图像文件
                                if success:
                                    # 图像文件的路径
                                    frame_path = os.path.join(sentence_path, f'frame{count}.jpg')

                                    # 保存图像文件
                                    write_success = cv2.imwrite(frame_path, image)
                                    if write_success:
                                        print(f"        Saved frame {count} to {frame_path}")
                                    else:
                                        print(f"        Failed to save frame {count} to {frame_path}")

                                    # 帧计数器加1
                                    count += 1
                                else:
                                    # 如果读取不成功（例如已经到了视频的末尾），则跳出循环
                                    print(f"        Finished reading frames from {video_path}")
                                    break
                        else:
                            print(f"      Failed to open {video_path}")
                    else:
                        print(f"      {video_path} does not exist")
                else:
                    print(f"    {sentence_path} does not exist")
        else:
            print(f"  {emotion_path} does not exist")

import cv2
import os
import numpy as np

def draw_dota_annotations(image_path, label_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        return

    # 读取标签文件
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # 遍历标签文件中的每一行
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10:
            continue

        # 提取坐标和类别
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[0:8])
        cls = parts[8]
        difficulty = parts[9]

        # 绘制多边形
        points = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
        pts = cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

        # 显示类别
        cv2.putText(image, cls, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存标注后的图像
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")


# 指定图像和标签路径
# image_path = 'path/to/your/image.jpg'
# image_path = '../../data/trainval/images/P2781__1024__2531___0.png'
image_path = '/mnt/mydisk/code/First_Ablation_Experiment/data/train/images/P0001__1024__0___2560.png'
label_path = '/mnt/mydisk/code/First_Ablation_Experiment/data/train/annfiles/P0001__1024__0___2560.txt'
output_path = '../../data/out_demo/annotated_image.jpg'

# 绘制标注
draw_dota_annotations(image_path, label_path, output_path)

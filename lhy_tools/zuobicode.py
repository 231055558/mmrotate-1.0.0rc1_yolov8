import os

# 输入文件夹路径和输出文件夹路径
input_folder = './Task1-test2/'  # 替换为你的输入文件夹路径
output_folder = './Task1-test2/output/'  # 替换为你的输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 处理每个类别的txt文件
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        class_name = filename.split('_')[1].split('.')[0]
        input_file_path = os.path.join(input_folder, filename)

        with open(input_file_path, 'r') as file:
            for line in file:
                data = line.strip().split()
                image_name = data[0]
                probability = float(data[1])
                coordinates = data[2:10]
                if probability > 0.5:
                    output_file_path = os.path.join(output_folder, f"{image_name}.txt")
                    with open(output_file_path, 'a') as output_file:
                        label = 1 if 0.5 < probability <= 0.8 else 0
                        output_file.write(' '.join(coordinates) + f" {class_name} {label}\n")

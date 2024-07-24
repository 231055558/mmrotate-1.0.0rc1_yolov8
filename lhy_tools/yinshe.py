import os
import cv2
import numpy as np

def read_positions(position_folder):
    positions = {}
    for position_file in os.listdir(position_folder):
        if position_file.endswith(".txt"):
            file_path = os.path.join(position_folder, position_file)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    image_id = parts[0]
                    coords = list(map(float, parts[1:]))
                    if image_id not in positions:
                        positions[image_id] = []
                    positions[image_id].append(coords)
    return positions

def draw_polygon(image, coords, color=(0, 255, 0), thickness=2):
    points = np.array(coords[1::], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

def process_images(image_folder, position_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    positions = read_positions(position_folder)

    for image_file in os.listdir(image_folder):
        if image_file.endswith(".png"):
            image_id = os.path.splitext(image_file)[0]
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)

            if image_id in positions:
                for coords in positions[image_id]:
                    draw_polygon(image, coords)

                output_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_path, image)
                print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    image_folder = "../../data/test/images"
    position_folder = "../../task"
    output_folder = "../../data/out_show"

    process_images(image_folder, position_folder, output_folder)

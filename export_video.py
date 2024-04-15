import imageio
import os
import cv2

# image path
image_path = [f"./stylize_result/catstatue_rgba_{i*100}.png" for i in range(15)]
img = cv2.imread(image_path[0])
H, W, _ = img.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./stylize_result/steps.mp4', fourcc, 30, (W, H))
for img_path in image_path:
    img = cv2.imread(img_path)
    out.write(img)
out.release()

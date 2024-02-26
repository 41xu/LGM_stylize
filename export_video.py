import imageio
import os
import cv2

# image path
image_path = [f"./ip2p_test/seed_{i}.png" for i in range(100)]
img = cv2.imread(image_path[0])
H, W, _ = img.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./ip2p_test/seed_step100.mp4', fourcc, 30, (W, H))
for img_path in image_path:
    img = cv2.imread(img_path)
    out.write(img)
out.release()

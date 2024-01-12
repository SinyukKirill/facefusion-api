from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load a model
model = YOLO(model='models/yolov8n-face.pt', task='detect')  # pretrained YOLOv8n model

# Run batched inference on an image
input_size = (640, 640)

img = cv2.imread('test.png')
# img = cv2.resize(img, input_size)
# img = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=input_size, swapRB=True)

results = model.predict(source=img, mode='predict', device=torch.device('cpu'), classes=[0])[0]

print(results)
# Apply gaussian blur to the detected ellipses
print(f'boxes: {results.boxes}')
if results.boxes is not None:
    for box in results.boxes.xyxy:
        # Extract bbox coordinates
        x1, y1, x2, y2 = map(int, box[:4])
        width = x2 - x1
        height = y2 - y1
        center = (x1 + width // 2, y1 + height // 2)

        # Create an elliptical mask
        mask = np.zeros_like(img)
        cv2.ellipse(mask, center, (width // 2, height // 2), 0, 0, 360, (255, 255, 255), -1)
        masked_img = cv2.bitwise_and(img, mask)

        # Crop the region of interest and apply Gaussian Blur
        roi = masked_img[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)

        # Place the blurred region back into the original image
        img[y1:y2, x1:x2][mask[y1:y2, x1:x2] == 255] = blurred_roi[mask[y1:y2, x1:x2] == 255]

bbox_list = []
kps_list = []
for detection in results:
    if detection.boxes is not None:
        for box in detection.boxes.xyxy:
            x1, y1, x2, y2 = box[:4]
            bbox_list.append(np.array(
            [
                x1,
                y1,
                x2,
                y2
            ]))
if results.keypoints is not None:
    print(results.keypoints)
    if results.keypoints is not None:
        for keypoints in results.keypoints.xy:
            kps_list.append(np.array(keypoints))

print(f'\nbbox_list: {bbox_list}, \nkps_list: {kps_list}')

# Optionally save or display the modified image
cv2.imwrite('blurred_ellipses.png', img)
# cv2.imshow('Blurred Ellipses', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

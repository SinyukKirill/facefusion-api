import onnxruntime
import numpy
import cv2
import time
import threading

from collections import namedtuple
from typing import Any

Bbox = numpy.ndarray[Any, Any]
Kps = numpy.ndarray[Any, Any]
Score = float
Face = namedtuple('Face',
[
	'bbox',
	'kps',
	'score',
])

score_threshold = 0.25
iou_threshold = 0.4

model = onnxruntime.InferenceSession('./models/yolov8n-face-dynamic.onnx')
pre_images = []
image1 = cv2.imread('./images/1.jpg')
image2 = cv2.imread('./images/2.jpg')
image3 = cv2.imread('./images/3.jpg')
image4 = cv2.imread('./images/4.jpg')
image5 = cv2.imread('./images/5.jpg')
image6 = cv2.imread('./images/6.jpg')

pre_images.append(numpy.array(image1))
pre_images.append(numpy.array(image2))
pre_images.append(numpy.array(image3))
pre_images.append(numpy.array(image4))
pre_images.append(numpy.array(image5))
pre_images.append(numpy.array(image6))

images = []
ratios = numpy.zeros(len(pre_images))
dws = numpy.zeros(len(pre_images))
dhs = numpy.zeros(len(pre_images))
for i, image in enumerate(pre_images):
	input_size = (640, 640)
	shape = image.shape[:2]
	ratio = min(input_size[0] / shape[0], input_size[1] / shape[1])
	new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
	dw, dh = (input_size[1] - new_unpad[0]) / 2, (input_size[0] - new_unpad[1]) / 2
	image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = round(dh - 0.1), round(dh + 0.1)
	left, right = round(dw - 0.1), round(dw + 0.1)
	image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
	image = image.astype(numpy.float32) / 255.0
	image = image[..., ::-1].transpose((2, 0, 1))
	images.append(image)
	ratios[i], dws[i], dhs[i] = ratio, dw, dh

images = numpy.array(images)
images = numpy.ascontiguousarray(images)

inferencetime = time.time()
with threading.Semaphore():
	predictions = model.run(None, {model.get_inputs()[0].name: images})[0]
print(f'inference time: {time.time() - inferencetime}')

predictions = numpy.transpose(predictions, (0, 2, 1))
predictions = numpy.ascontiguousarray(predictions)

FACES = []
for i, pred in enumerate(predictions):
	bbox, score, kps = numpy.split(pred, [4, 5], axis=1)
	ratio, dw, dh = ratios[i], dws[i], dhs[i]

	new_ratio = 1/ratio
	x_center, y_center, width, height = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
	x_min = (x_center - (width / 2) - dw) * new_ratio
	y_min = (y_center - (height / 2) - dh) * new_ratio
	x_max = (x_center + (width / 2) - dw) * new_ratio
	y_max = (y_center + (height / 2) - dh) * new_ratio
	bbox = numpy.stack((x_min, y_min, x_max, y_max), axis=1)
	for i in range(kps.shape[1] // 3):
		kps[:, i * 3] = (kps[:, i * 3] - dw) * new_ratio
		kps[:, i * 3 + 1] = (kps[:, i * 3 + 1] - dh) * new_ratio

	indices_above_threshold = numpy.where(score > score_threshold)[0]
	bbox = bbox[indices_above_threshold]
	score = score[indices_above_threshold]
	kps = kps[indices_above_threshold]

	nms_indices = cv2.dnn.NMSBoxes(bbox.tolist(), score.ravel().tolist(), score_threshold, iou_threshold)
	bbox = bbox[nms_indices]
	score = score[nms_indices]
	kps = kps[nms_indices]

	bbox_list = []
	for box in bbox:
		bbox_list.append(numpy.array(
		[
			box[0],
			box[1],
			box[2],
			box[3],
		]))
	score_list = score.ravel().tolist()
	kps_list = []
	for keypoints in kps:
		kps_xy = []
		for i in range(0, len(keypoints), 3):
			kps_xy.append([keypoints[i], keypoints[i+1]])
		kps_list.append(numpy.array(kps_xy))

	faces = []
	for index in range(len(bbox_list)):
		bbox = bbox_list[index]
		kps = kps_list[index]
		score = score_list[index]
		faces.append(Face(
			bbox = bbox,
			kps = kps,
			score = score,
		))
	FACES.append(faces)

print(f'FACES: {FACES}')

for i, faces in enumerate(FACES):
	frame = pre_images[i]
	bbox_list = []
	kps_list = []
	for face in faces:
		bbox_list.append(face.bbox)
		kps_list.append(face.kps)
	print(f'bbox_list: {bbox_list}')
	print(f'kps_list: {kps_list}')
	for bbox, keypoints in zip(bbox_list, kps_list):
		start_point = (int(bbox[0]), int(bbox[1]))
		end_point = (int(bbox[2]), int(bbox[3]))
		frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
		for kp in keypoints:
			x, y = int(kp[0]), int(kp[1])
			frame = cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
	cv2.imshow(f'frame{i}', frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
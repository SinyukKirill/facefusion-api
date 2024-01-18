import onnxruntime
import numpy
import cv2
import time
import threading
from collections import namedtuple

from collections import namedtuple
from typing import Any, List

Session = onnxruntime.InferenceSession
Image = numpy.ndarray[Any, Any]
Prediction = numpy.ndarray[Any, Any]
MetaData = dict[str, list[int]]
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


def load_model(model_path: str) -> Session:
	global MODEL
	if MODEL is None:
		with threading.Lock():
			MODEL = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
	return MODEL


def load_images(image_paths: List[str]) -> List[Image]:
    return [cv2.imread(path) for path in image_paths]


def preprocess_images(images: List[Image], input_size=(640, 640)) -> tuple[numpy.ndarray[Any, Any], MetaData]:
	preprocessed = []
	meta_data = {'ratios': [], 'dws': [], 'dhs': []}
	for image in images:
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
		preprocessed.append(image)
		meta_data['ratios'].append(ratio)
		meta_data['dws'].append(dw)
		meta_data['dhs'].append(dh)
	return numpy.ascontiguousarray(numpy.array(preprocessed)), meta_data


def predict(model: Session, images: Image) -> Prediction:
	with threading.Semaphore():
		start_time = time.time()
		predictions = model.run(None, {model.get_inputs()[0].name: images})[0]
		print(f'inference time: {time.time() - start_time}')
	return predictions


def postprocess_predictions(predictions: Prediction, meta_data: MetaData, score_threshold=0.25, iou_threshold=0.4) -> List[Face]:
	predictions = numpy.transpose(predictions, (0, 2, 1))
	predictions = numpy.ascontiguousarray(predictions)
	batch_faces = []
	for i, pred in enumerate(predictions):
		bbox, score, kps = numpy.split(pred, [4, 5], axis=1)
		ratio, dw, dh = meta_data['ratios'][i], meta_data['dws'][i], meta_data['dhs'][i]

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

		batch_faces.append([Face(bbox=bbox, kps=kps, score=score) for bbox, kps, score in zip(bbox_list, kps_list, score_list)])
	return batch_faces


def draw_results(images: List[Image], batch_faces: List[Face]):
	for i, (image, faces) in enumerate(zip(images, batch_faces)):
		bbox_list = []
		kps_list = []
		for face in faces:
			bbox_list.append(face.bbox)
			kps_list.append(face.kps)
		for bbox, keypoints in zip(bbox_list, kps_list):
			start_point = (int(bbox[0]), int(bbox[1]))
			end_point = (int(bbox[2]), int(bbox[3]))
			image = cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
			for kp in keypoints:
				x, y = int(kp[0]), int(kp[1])
				image = cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
		cv2.imshow(f'image{i}', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


def main():
	image_paths = ['./images/1.jpg', './images/2.jpg', './images/3.jpg', './images/4.jpg', './images/5.jpg', './images/6.jpg']
	model = load_model('./models/yolov8n-face-dynamic.onnx')
	pre_images = load_images(image_paths)
	images, meta_data = preprocess_images(pre_images)
	predictions = predict(model, images)
	batch_faces = postprocess_predictions(predictions, meta_data)
	draw_results(pre_images, batch_faces)


if __name__ == '__main__':
	MODEL = None
	main()
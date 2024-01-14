from typing import Any, Optional, List, Dict, Tuple
import threading
import cv2
import numpy
import onnxruntime
from ultralytics import YOLO
import torch

import facefusion.globals
from facefusion.face_cache import get_faces_cache, set_faces_cache
from facefusion.face_helper import warp_face, create_static_anchors, distance_to_kps, distance_to_bbox, apply_nms
from facefusion.typing import Frame, Face, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, ModelValue, Bbox, Kps, Score, Embedding
from facefusion.utilities import resolve_relative_path, conditional_download
from facefusion.vision import resize_frame_dimension

FACE_ANALYSER = None
THREAD_SEMAPHORE : threading.Semaphore = threading.Semaphore()
THREAD_LOCK : threading.Lock = threading.Lock()
MODELS : Dict[str, ModelValue] =\
{
	# TODO: githubのurl変える
	'face_detector_yolo_face_pt':
	{
		'url': '',
		'path': resolve_relative_path('../.assets/models/yolov8n-face.pt')
	},
	'face_detector_yolo_face_onnx':
	{
		'url': '',
		'path': resolve_relative_path('../.assets/models/yolov8n-face.onnx')
	},
	'face_detector_retinaface':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/retinaface_10g.onnx',
		'path': resolve_relative_path('../.assets/models/retinaface_10g.onnx')
	},
	'face_detector_yunet':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/yunet_2023mar.onnx',
		'path': resolve_relative_path('../.assets/models/yunet_2023mar.onnx')
	},
	'face_recognizer_arcface_blendface':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.onnx')
	},
	'face_recognizer_arcface_inswapper':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_w600k_r50.onnx')
	},
	'face_recognizer_arcface_simswap':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_simswap.onnx',
		'path': resolve_relative_path('../.assets/models/arcface_simswap.onnx')
	},
	'gender_age':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gender_age.onnx',
		'path': resolve_relative_path('../.assets/models/gender_age.onnx')
	}
}


def get_face_analyser() -> Any:
	global FACE_ANALYSER

	with THREAD_LOCK:
		if FACE_ANALYSER is None:
			if facefusion.globals.face_detector_model == 'retinaface':
				face_detector = onnxruntime.InferenceSession(MODELS.get('face_detector_retinaface').get('path'), providers = facefusion.globals.execution_providers)
			if facefusion.globals.face_detector_model == 'yunet':
				face_detector = cv2.FaceDetectorYN.create(MODELS.get('face_detector_yunet').get('path'), '', (0, 0))
			if facefusion.globals.face_detector_model == 'yolo_face_pt':
				face_detector = YOLO(MODELS.get('face_detector_yolo_face_pt').get('path'))
			if facefusion.globals.face_detector_model == 'yolo_face_onnx':
				face_detector = onnxruntime.InferenceSession((MODELS.get('face_detector_yolo_face_onnx').get('path')), providers = facefusion.globals.execution_providers)
			if facefusion.globals.face_recognizer_model == 'arcface_blendface':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_blendface').get('path'), providers = facefusion.globals.execution_providers)
			if facefusion.globals.face_recognizer_model == 'arcface_inswapper':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_inswapper').get('path'), providers = facefusion.globals.execution_providers)
			if facefusion.globals.face_recognizer_model == 'arcface_simswap':
				face_recognizer = onnxruntime.InferenceSession(MODELS.get('face_recognizer_arcface_simswap').get('path'), providers = facefusion.globals.execution_providers)
			gender_age = onnxruntime.InferenceSession(MODELS.get('gender_age').get('path'), providers = facefusion.globals.execution_providers)			
			FACE_ANALYSER =\
			{
				'face_detector': face_detector,
				'face_recognizer': face_recognizer,
				'gender_age': gender_age
			}
	return FACE_ANALYSER


def clear_face_analyser() -> Any:
	global FACE_ANALYSER

	FACE_ANALYSER = None


def pre_check() -> bool:
	if not facefusion.globals.skip_download:
		download_directory_path = resolve_relative_path('../.assets/models')
		model_urls =\
		[
			MODELS.get('face_detector_retinaface').get('url'),
			MODELS.get('face_detector_yunet').get('url'),
			MODELS.get('face_recognizer_arcface_inswapper').get('url'),
			MODELS.get('face_recognizer_arcface_simswap').get('url'),
			MODELS.get('gender_age').get('url')
		]
		conditional_download(download_directory_path, model_urls)
	return True


def extract_faces(frame: Frame) -> List[Face]:
	face_detector_width, face_detector_height = map(int, facefusion.globals.face_detector_size.split('x'))
	frame_height, frame_width, _ = frame.shape
	temp_frame = resize_frame_dimension(frame, face_detector_width, face_detector_height)
	temp_frame_height, temp_frame_width, _ = temp_frame.shape
	ratio_height = frame_height / temp_frame_height
	ratio_width = frame_width / temp_frame_width
	if facefusion.globals.face_detector_model == 'retinaface':
		bbox_list, kps_list, score_list = detect_with_retinaface(temp_frame, temp_frame_height, temp_frame_width, face_detector_height, face_detector_width, ratio_height, ratio_width)
		return create_faces(frame, bbox_list, kps_list, score_list)
	elif facefusion.globals.face_detector_model == 'yunet':
		bbox_list, kps_list, score_list = detect_with_yunet(temp_frame, temp_frame_height, temp_frame_width, ratio_height, ratio_width)
		return create_faces(frame, bbox_list, kps_list, score_list)
	elif facefusion.globals.face_detector_model == 'yolo_face_pt':
		bbox_list, kps_list = detect_with_yolo_face_pt(temp_frame, ratio_height, ratio_width)
		return create_faces_yolo_pt(bbox_list, kps_list)
	elif facefusion.globals.face_detector_model == 'yolo_face_onnx':
		bbox_list, kps_list, score_list = detect_with_yolo_face_onnx(frame)
		return create_faces(frame, bbox_list, kps_list, score_list)
	return []


def detect_with_yolo_face_pt(temp_frame: Frame, ratio_height: float, ratio_width: float) -> Tuple[List[Bbox], List[Kps]]:
	face_detector = get_face_analyser().get('face_detector')
	bbox_list = []
	kps_list = []
	with THREAD_SEMAPHORE:
		detections = face_detector.predict(source=temp_frame, mode='predict', device=torch.device('cpu'), classes=[0])
	for detection in detections:
		if detection.boxes is not None:
			for box in detection.boxes.xyxy:
				x1, y1, x2, y2 = map(float, box[:4])
				bbox_list.append(numpy.array(
				[
					x1 * ratio_width,
					y1 * ratio_height,
					x2 * ratio_width,
					y2 * ratio_height
				]))
		if detection.keypoints is not None:
			for keypoints_tensor in detection.keypoints.xy:
				keypoints_numpy = keypoints_tensor.numpy() if isinstance(keypoints_tensor, torch.Tensor) else keypoints_tensor
				scaled_keypoints = []
				for kp in keypoints_numpy:
					scaled_kp = [kp[0] * ratio_width, kp[1] * ratio_height]
					scaled_keypoints.append(scaled_kp)
				kps_list.append(numpy.array(scaled_keypoints))
	return bbox_list, kps_list


def detect_with_yolo_face_onnx(temp_frame: Frame) -> Tuple[List[Bbox], List[Kps], List[Score]]:
	face_detector = get_face_analyser().get('face_detector')

	input_size = (640, 640)
	# 新しいサイズに合わせてアスペクト比を保持するために画像をリサイズ
	shape = temp_frame.shape[:2]  # 現在の画像サイズを取得
	ratio = min(input_size[0] / shape[0], input_size[1] / shape[1])
	new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
	dw, dh = input_size[1] - new_unpad[0], input_size[0] - new_unpad[1]  # パディング幅を計算
	dw /= 2  # 左右のパディング
	dh /= 2  # 上下のパディング

	# BGRからRGBへの変換
	temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
	temp_frame = cv2.resize(temp_frame, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = round(dh - 0.1), round(dh + 0.1)
	left, right = round(dw - 0.1), round(dw + 0.1)
	temp_frame = cv2.copyMakeBorder(temp_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # パディング適用
	temp_frame = temp_frame.astype(numpy.float32) / 255.0

	# BCHW形式に変換
	temp_frame = numpy.transpose(temp_frame, (2, 0, 1))
	temp_frame = numpy.expand_dims(temp_frame, axis=0)

	# 推論実行
	with THREAD_SEMAPHORE:
		pred = face_detector.run(None, {face_detector.get_inputs()[0].name: temp_frame})

	pred = numpy.squeeze(pred)
	pred = numpy.transpose(pred, (1, 0))
	bbox, score, kps = numpy.split(pred, [4, 5], axis=1)

	new_ratio = 1/ratio
	x_center, y_center, width, height = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
	# スケール変換の修正
	x_min = (x_center - (width / 2) - dw) * new_ratio
	y_min = (y_center - (height / 2) - dh) * new_ratio
	x_max = (x_center + (width / 2) - dw) * new_ratio
	y_max = (y_center + (height / 2) - dh) * new_ratio
	bbox = numpy.stack((x_min, y_min, x_max, y_max), axis=1)

	# キーポイント変換の修正
	for i in range(kps.shape[1] // 3):
		kps[:, i * 3] = (kps[:, i * 3] - dw) * new_ratio
		kps[:, i * 3 + 1] = (kps[:, i * 3 + 1] - dh) * new_ratio

	# スコアの閾値設定
	indices_above_threshold = numpy.where(score > facefusion.globals.face_detector_score)[0]

	bbox = bbox[indices_above_threshold]
	score = score[indices_above_threshold]
	kps = kps[indices_above_threshold]

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

	return bbox_list, kps_list, score_list


def detect_with_retinaface(temp_frame : Frame, temp_frame_height : int, temp_frame_width : int, face_detector_height : int, face_detector_width : int, ratio_height : float, ratio_width : float) -> Tuple[List[Bbox], List[Kps], List[Score]]:
	face_detector = get_face_analyser().get('face_detector')
	bbox_list = []
	kps_list = []
	score_list = []
	feature_strides = [ 8, 16, 32 ]
	feature_map_channel = 3
	anchor_total = 2
	prepare_frame = numpy.zeros((face_detector_height, face_detector_width, 3))
	prepare_frame[:temp_frame_height, :temp_frame_width, :] = temp_frame
	temp_frame = (prepare_frame - 127.5) / 128.0
	temp_frame = numpy.expand_dims(temp_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	with THREAD_SEMAPHORE:
		detections = face_detector.run(None,
		{
			face_detector.get_inputs()[0].name: temp_frame
		})
	for index, feature_stride in enumerate(feature_strides):
		keep_indices = numpy.where(detections[index] >= facefusion.globals.face_detector_score)[0]
		if keep_indices.any():
			stride_height = face_detector_height // feature_stride
			stride_width = face_detector_width // feature_stride
			anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
			bbox_raw = (detections[index + feature_map_channel] * feature_stride)
			kps_raw = detections[index + feature_map_channel * 2] * feature_stride
			for bbox in distance_to_bbox(anchors, bbox_raw)[keep_indices]:
				bbox_list.append(numpy.array(
				[
					bbox[0] * ratio_width,
					bbox[1] * ratio_height,
					bbox[2] * ratio_width,
					bbox[3] * ratio_height
				]))
			for kps in distance_to_kps(anchors, kps_raw)[keep_indices]:
				kps_list.append(kps * [ ratio_width, ratio_height ])
			for score in detections[index][keep_indices]:
				score_list.append(score[0])
	return bbox_list, kps_list, score_list


def detect_with_yunet(temp_frame : Frame, temp_frame_height : int, temp_frame_width : int, ratio_height : float, ratio_width : float) -> Tuple[List[Bbox], List[Kps], List[Score]]:
	face_detector = get_face_analyser().get('face_detector')
	face_detector.setInputSize((temp_frame_width, temp_frame_height))
	face_detector.setScoreThreshold(facefusion.globals.face_detector_score)
	bbox_list = []
	kps_list = []
	score_list = []
	with THREAD_SEMAPHORE:
		_, detections = face_detector.detect(temp_frame)
	if detections.any():
		for detection in detections:
			bbox_list.append(numpy.array(
			[
				detection[0] * ratio_width,
				detection[1] * ratio_height,
				(detection[0] + detection[2]) * ratio_width,
				(detection[1] + detection[3]) * ratio_height
			]))
			kps_list.append(detection[4:14].reshape((5, 2)) * [ ratio_width, ratio_height])
			score_list.append(detection[14])
	return bbox_list, kps_list, score_list


def create_faces(frame : Frame, bbox_list : List[Bbox], kps_list : List[Kps], score_list : List[Score]) -> List[Face] :
	faces : List[Face] = []
	# print(f'bbox_list: {bbox_list}')
	if facefusion.globals.face_detector_score > 0:
		keep_indices = apply_nms(bbox_list, facefusion.globals.face_detector_iou)
		# print(f'keep_indices: {keep_indices}')
		for index in keep_indices:
			bbox = bbox_list[index]
			kps = kps_list[index]
			score = score_list[index]
			embedding, normed_embedding = calc_embedding(frame, kps)
			gender, age = detect_gender_age(frame, kps)
			faces.append(Face(
				bbox = bbox,
				kps = kps,
				score = score,
				embedding = embedding,
				normed_embedding = normed_embedding,
				gender = gender,
				age = age
			))
	# print(f'faces: {faces}')
	bbox_list = []
	kps_list = []
	for face in faces:
		bbox_list.append(face.bbox)
		kps_list.append(face.kps)
	# print(f'bbox_list: {bbox_list}')
	# print(f'kps_list: {kps_list}')
	# for bbox, keypoints in zip(bbox_list, kps_list):
	# 	start_point = (int(bbox[0]), int(bbox[1]))
	# 	end_point = (int(bbox[2]), int(bbox[3]))
	# 	frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
	# 	for kp in keypoints:
	# 		x, y = int(kp[0]), int(kp[1])
	# 		frame = cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
	return faces


def create_faces_yolo_pt(bbox_list : List[Bbox], kps_list : List[Kps]) -> List[Face] :
	faces : List[Face] = []
	if facefusion.globals.face_detector_score > 0:
		keep_indices = apply_nms(bbox_list, 0.4)
		for index in keep_indices:
			bbox = bbox_list[index]
			kps = kps_list[index]
			faces.append(Face(
				bbox = bbox,
				kps = kps,
				score = None,
				embedding = None,
				normed_embedding = None,
				gender = None,
				age = None
			))
	return faces


def calc_embedding(temp_frame : Frame, kps : Kps) -> Tuple[Embedding, Embedding]:
	face_recognizer = get_face_analyser().get('face_recognizer')
	crop_frame, matrix = warp_face(temp_frame, kps, 'arcface_v2', (112, 112))
	crop_frame = crop_frame.astype(numpy.float32) / 127.5 - 1
	crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
	crop_frame = numpy.expand_dims(crop_frame, axis = 0)
	embedding = face_recognizer.run(None,
	{
		face_recognizer.get_inputs()[0].name: crop_frame
	})[0]
	embedding = embedding.ravel()
	normed_embedding = embedding / numpy.linalg.norm(embedding)
	return embedding, normed_embedding


def detect_gender_age(frame : Frame, kps : Kps) -> Tuple[int, int]:
	gender_age = get_face_analyser().get('gender_age')
	crop_frame, affine_matrix = warp_face(frame, kps, 'arcface_v2', (96, 96))
	crop_frame = numpy.expand_dims(crop_frame, axis = 0).transpose(0, 3, 1, 2).astype(numpy.float32)
	prediction = gender_age.run(None,
	{
		gender_age.get_inputs()[0].name: crop_frame
	})[0][0]
	gender = int(numpy.argmax(prediction[:2]))
	age = int(numpy.round(prediction[2] * 100))
	return gender, age


def get_one_face(frame : Frame, position : int = 0) -> Optional[Face]:
	many_faces = get_many_faces(frame)
	if many_faces:
		try:
			return many_faces[position]
		except IndexError:
			return many_faces[-1]
	return None


def get_many_faces(frame : Frame) -> List[Face]:
	try:
		faces_cache = get_faces_cache(frame)
		if faces_cache:
			faces = faces_cache
		else:
			faces = extract_faces(frame)
			set_faces_cache(frame, faces)
		if facefusion.globals.face_analyser_order:
			faces = sort_by_order(faces, facefusion.globals.face_analyser_order)
		if facefusion.globals.face_analyser_age:
			faces = filter_by_age(faces, facefusion.globals.face_analyser_age)
		if facefusion.globals.face_analyser_gender:
			faces = filter_by_gender(faces, facefusion.globals.face_analyser_gender)
		return faces
	except (AttributeError, ValueError):
		return []


def find_similar_faces(frame : Frame, reference_face : Face, face_distance : float) -> List[Face]:
	many_faces = get_many_faces(frame)
	similar_faces = []
	if many_faces:
		for face in many_faces:
			if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
				current_face_distance = 1 - numpy.dot(face.normed_embedding, reference_face.normed_embedding)
				if current_face_distance < face_distance:
					similar_faces.append(face)
	return similar_faces


def sort_by_order(faces : List[Face], order : FaceAnalyserOrder) -> List[Face]:
	if order == 'left-right':
		return sorted(faces, key = lambda face: face.bbox[0])
	if order == 'right-left':
		return sorted(faces, key = lambda face: face.bbox[0], reverse = True)
	if order == 'top-bottom':
		return sorted(faces, key = lambda face: face.bbox[1])
	if order == 'bottom-top':
		return sorted(faces, key = lambda face: face.bbox[1], reverse = True)
	if order == 'small-large':
		return sorted(faces, key = lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
	if order == 'large-small':
		return sorted(faces, key = lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse = True)
	if order == 'best-worst':
		return sorted(faces, key = lambda face: face.score, reverse = True)
	if order == 'worst-best':
		return sorted(faces, key = lambda face: face.score)
	return faces


def filter_by_age(faces : List[Face], age : FaceAnalyserAge) -> List[Face]:
	filter_faces = []
	for face in faces:
		if face.age < 13 and age == 'child':
			filter_faces.append(face)
		elif face.age < 19 and age == 'teen':
			filter_faces.append(face)
		elif face.age < 60 and age == 'adult':
			filter_faces.append(face)
		elif face.age > 59 and age == 'senior':
			filter_faces.append(face)
	return filter_faces


def filter_by_gender(faces : List[Face], gender : FaceAnalyserGender) -> List[Face]:
	filter_faces = []
	for face in faces:
		if face.gender == 0 and gender == 'female':
			filter_faces.append(face)
		if face.gender == 1 and gender == 'male':
			filter_faces.append(face)
	return filter_faces

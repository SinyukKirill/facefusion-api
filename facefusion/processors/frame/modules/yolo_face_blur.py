from typing import Any, List, Dict, Literal, Optional
from argparse import ArgumentParser
import cv2
import threading
import numpy
import onnxruntime

import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import wording
from facefusion.face_analyser import get_many_faces, clear_face_analyser
from facefusion.face_helper import warp_face, paste_back_ellipse
from facefusion.content_analyser import clear_content_analyser
from facefusion.typing import Face, Frame, Update_Process, ProcessMode, ModelValue, OptionsWithModel
from facefusion.utilities import conditional_download, resolve_relative_path, is_image, is_video, is_file, is_download_done, create_metavar, update_status
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame import choices as frame_processors_choices

FRAME_PROCESSOR = None
THREAD_SEMAPHORE : threading.Semaphore = threading.Semaphore()
THREAD_LOCK : threading.Lock = threading.Lock()
NAME = 'FACEFUSION.FRAME_PROCESSOR.FACE_BLUR'
MODELS : Dict[str, ModelValue] =\
{
	'codeformer':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx',
		'path': resolve_relative_path('../.assets/models/codeformer.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'gfpgan_1.2':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.2.onnx',
		'path': resolve_relative_path('../.assets/models/gfpgan_1.2.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'gfpgan_1.3':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.3.onnx',
		'path': resolve_relative_path('../.assets/models/gfpgan_1.3.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'gfpgan_1.4':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx',
		'path': resolve_relative_path('../.assets/models/gfpgan_1.4.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'gpen_bfr_256':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_256.onnx',
		'path': resolve_relative_path('../.assets/models/gpen_bfr_256.onnx'),
		'template': 'arcface_v2',
		'size': (128, 256)
	},
	'gpen_bfr_512':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_512.onnx',
		'path': resolve_relative_path('../.assets/models/gpen_bfr_512.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'restoreformer':
	{
		'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/restoreformer.onnx',
		'path': resolve_relative_path('../.assets/models/restoreformer.onnx'),
		'template': 'ffhq',
		'size': (512, 512)
	},
	'yolo_face':
	{
		'template': 'ffhq',
		'size': (512, 512)
	}
}
OPTIONS : Optional[OptionsWithModel] = None


def get_frame_processor() -> Any:
	global FRAME_PROCESSOR

	with THREAD_LOCK:
		if FRAME_PROCESSOR is None:
			model_path = get_options('model').get('path')
			FRAME_PROCESSOR = onnxruntime.InferenceSession(model_path, providers = facefusion.globals.execution_providers)
	return FRAME_PROCESSOR


def clear_frame_processor() -> None:
	global FRAME_PROCESSOR

	FRAME_PROCESSOR = None


def get_options(key : Literal['model']) -> Any:
	global OPTIONS

	if OPTIONS is None:
		OPTIONS =\
		{
			'model': MODELS[frame_processors_globals.face_blur_model]
		}
	return OPTIONS.get(key)


def set_options(key : Literal['model'], value : Any) -> None:
	global OPTIONS

	OPTIONS[key] = value


def register_args(program : ArgumentParser) -> None:
	program.add_argument('--face-blur-model', help = wording.get('frame_processor_model_help'), dest = 'face_blur_model', default = 'yolo_face', choices = frame_processors_choices.face_blur_models)
	program.add_argument('--face-blur-blend', help = wording.get('frame_processor_blend_help'), dest = 'face_blur_blend', type = int, default = 40, choices = frame_processors_choices.face_blur_blend_range, metavar = create_metavar(frame_processors_choices.face_blur_blend_range))


def apply_args(program : ArgumentParser) -> None:
	args = program.parse_args()
	frame_processors_globals.face_blur_model = args.face_blur_model
	frame_processors_globals.face_blur_blend = args.face_blur_blend


def pre_check() -> bool:
	# if not facefusion.globals.skip_download:
	# 	download_directory_path = resolve_relative_path('../.assets/models')
	# 	model_url = get_options('model').get('url')
	# 	conditional_download(download_directory_path, [ model_url ])
	return True


def pre_process(mode : ProcessMode) -> bool:
	# model_url = get_options('model').get('url')
	# model_path = get_options('model').get('path')
	# if not facefusion.globals.skip_download and not is_download_done(model_url, model_path):
	# 	update_status(wording.get('model_download_not_done') + wording.get('exclamation_mark'), NAME)
	# 	return False
	# elif not is_file(model_path):
	# 	update_status(wording.get('model_file_not_present') + wording.get('exclamation_mark'), NAME)
	# 	return False
	# if mode in [ 'output', 'preview' ] and not is_image(facefusion.globals.target_path) and not is_video(facefusion.globals.target_path):
	# 	update_status(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
	# 	return False
	# if mode == 'output' and not facefusion.globals.output_path:
	# 	update_status(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
	# 	return False
	return True


def post_process() -> None:
	clear_frame_processor()
	clear_face_analyser()
	clear_content_analyser()
	read_static_image.cache_clear()


# def blur_face(target_face: Face, temp_frame: Frame) -> Frame:
# 	model_template = get_options('model').get('template')
# 	model_size = get_options('model').get('size')
# 	crop_frame, affine_matrix = warp_face(temp_frame, target_face.kps, model_template, model_size)
# 	crop_frame = prepare_crop_frame(crop_frame)
# 	# blur = frame_processors_globals.face_blur_blend
# 	# blurred_face = cv2.GaussianBlur(crop_frame, (blur, blur), 0)
# 	blurred_face = cv2.GaussianBlur(crop_frame, (45, 45), 0)

# 	temp_frame = paste_back_ellipse(temp_frame, blurred_face, affine_matrix, facefusion.globals.face_mask_blur, facefusion.globals.face_mask_padding)
# 	return temp_frame

def blur_face(target_face: Face, temp_frame: Frame) -> Frame:
    model_template = get_options('model').get('template')
    model_size = get_options('model').get('size')
    crop_frame, affine_matrix = warp_face(temp_frame, target_face.kps, model_template, model_size)
    crop_frame = prepare_crop_frame(crop_frame)

    # crop_frameのデバッグ情報を出力
    print("crop_frame shape:", crop_frame.shape)
    print("crop_frame dtype:", crop_frame.dtype)

    # crop_frameが期待される形式であるか確認
    if crop_frame.ndim == 3 and crop_frame.shape[2] == 3:  # 通常、画像は高さx幅xチャネルの形式です。
        # 画像をグレースケールに変換（必要に応じて）
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
        
        # ガウスぼかしを適用
        blurred_face = cv2.GaussianBlur(crop_frame, (45, 45), 0)
    else:
        print("crop_frame is not in expected format.")
        # エラー処理またはデフォルトのアクションをここに実装する
        blurred_face = crop_frame  # この行は適切なエラー処理に置き換えてください。

    temp_frame = paste_back_ellipse(temp_frame, blurred_face, affine_matrix, facefusion.globals.face_mask_blur, facefusion.globals.face_mask_padding)
    return temp_frame


def prepare_crop_frame(crop_frame : Frame) -> Frame:
	crop_frame = crop_frame[:, :, ::-1] / 255.0
	crop_frame = (crop_frame - 0.5) / 0.5
	crop_frame = numpy.expand_dims(crop_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return crop_frame


def normalize_crop_frame(crop_frame : Frame) -> Frame:
	crop_frame = numpy.clip(crop_frame, -1, 1)
	crop_frame = (crop_frame + 1) / 2
	crop_frame = crop_frame.transpose(1, 2, 0)
	crop_frame = (crop_frame * 255.0).round()
	crop_frame = crop_frame.astype(numpy.uint8)[:, :, ::-1]
	return crop_frame


# def blend_frame(temp_frame : Frame, paste_frame : Frame) -> Frame:
# 	face_blur_blend = 1 - (frame_processors_globals.face_blur_blend / 100)
# 	temp_frame = cv2.addWeighted(temp_frame, face_blur_blend, paste_frame, 1 - face_blur_blend, 0)
# 	return temp_frame


def process_frame(source_face : Face, reference_face : Face, temp_frame : Frame) -> Frame:
	many_faces = get_many_faces(temp_frame)
	if many_faces:
		for target_face in many_faces:
			temp_frame = blur_face(target_face, temp_frame)
	else:
		import time
		print('No face detected in this frame.')
		write_image(f'temp/error/no_face_detected-{time.time()}.jpg', temp_frame)
	return temp_frame


def process_frames(source_path : str, temp_frame_paths : List[str], update_progress : Update_Process) -> None:
	for temp_frame_path in temp_frame_paths:
		temp_frame = read_image(temp_frame_path)
		result_frame = process_frame(None, None, temp_frame)
		write_image(temp_frame_path, result_frame)
		update_progress()


def process_image(source_path : str, target_path : str, output_path : str) -> None:
	target_frame = read_static_image(target_path)
	result_frame = process_frame(None, None, target_frame)
	write_image(output_path, result_frame)


def process_video(source_path : str, temp_frame_paths : List[str]) -> None:
	frame_processors.multi_process_frames(None, temp_frame_paths, process_frames)

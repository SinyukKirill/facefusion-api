from typing import Any, List, Dict, Literal, Optional
from argparse import ArgumentParser
import threading
import numpy
import onnx
import onnxruntime
from onnx import numpy_helper
import cv2
import time


import facefusion.globals
import facefusion.processors.frame.core as frame_processors
from facefusion import wording
from facefusion.face_analyser import get_one_face, get_many_faces, find_similar_faces, clear_face_analyser
from facefusion.face_helper import warp_face, paste_back_ellipse
from facefusion.face_reference import get_face_reference
from facefusion.content_analyser import clear_content_analyser
from facefusion.typing import Face, Frame, Update_Process, ProcessMode, ModelValue, OptionsWithModel, Embedding
from facefusion.utilities import conditional_download, resolve_relative_path, is_image, is_video, is_file, is_download_done, update_status
from facefusion.vision import read_image, read_static_image, write_image
from facefusion.processors.frame import globals as frame_processors_globals
from facefusion.processors.frame import choices as frame_processors_choices

FRAME_PROCESSOR = None
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
	}
}
OPTIONS : Optional[OptionsWithModel] = None

def get_frame_processor() -> Any:
	pass


def clear_frame_processor() -> None:
	pass


def get_options(key : Literal['model']) -> Any:
	global OPTIONS

	if OPTIONS is None:
		OPTIONS =\
		{
			'model': MODELS[frame_processors_globals.face_enhancer_model]
		}
	return OPTIONS.get(key)


def set_options(key : Literal['model'], value : Any) -> None:
	pass


def register_args(program : ArgumentParser) -> None:
    pass


def apply_args(program : ArgumentParser) -> None:
	pass


def pre_check() -> bool:
	return True


def pre_process(mode : ProcessMode) -> bool:
	if mode in [ 'output', 'preview' ] and not is_image(facefusion.globals.target_path) and not is_video(facefusion.globals.target_path):
		update_status(wording.get('select_image_or_video_target') + wording.get('exclamation_mark'), NAME)
		return False
	if mode == 'output' and not facefusion.globals.output_path:
		update_status(wording.get('select_file_or_directory_output') + wording.get('exclamation_mark'), NAME)
		return False
	return True


def post_process() -> None:
	clear_face_analyser()
	clear_content_analyser()
	read_static_image.cache_clear()


def apply_blur_to_face(target_face: Face, temp_frame: Frame) -> Frame:
    model_template = get_options('model').get('template')
    model_size = get_options('model').get('size')
    crop_frame, affine_matrix = warp_face(temp_frame, target_face.kps, model_template, model_size)

    blurred_face = apply_blur(crop_frame)
    temp_frame = paste_back_ellipse(temp_frame, blurred_face, affine_matrix, facefusion.globals.face_mask_blur, facefusion.globals.face_mask_padding)

    return temp_frame


def apply_blur(crop_frame: Frame) -> Frame:
    blurred_frame = cv2.GaussianBlur(crop_frame, (301, 301), 0)
    return blurred_frame


def prepare_crop_frame(crop_frame : Frame) -> Frame:
	model_mean = get_options('model').get('mean')
	model_standard_deviation = get_options('model').get('standard_deviation')
	crop_frame = crop_frame[:, :, ::-1] / 255.0
	crop_frame = (crop_frame - model_mean) / model_standard_deviation
	crop_frame = crop_frame.transpose(2, 0, 1)
	crop_frame = numpy.expand_dims(crop_frame, axis = 0).astype(numpy.float32)
	return crop_frame


def normalize_crop_frame(crop_frame : Frame) -> Frame:
	crop_frame = crop_frame.transpose(1, 2, 0)
	crop_frame = (crop_frame * 255.0).round()
	crop_frame = crop_frame[:, :, ::-1].astype(numpy.uint8)
	return crop_frame


def process_frame(temp_frame: Frame) -> Frame:
	many_faces = get_many_faces(temp_frame)
	if len(many_faces) > 1:
		print('more than one face detected in this frame.')
		write_image(f'temp/manyfaces/more_than_one_face_detected-{time.time()}.jpg', temp_frame)
	if many_faces:
		for target_face in many_faces:
			temp_frame = apply_blur_to_face(target_face, temp_frame)
	else:
		print('No face detected in this frame.')
		write_image(f'temp/error/no_face_detected-{time.time()}.jpg', temp_frame)
	return temp_frame


def process_frames(source_path : str, temp_frame_paths : List[str], update_progress : Update_Process) -> None:
	for temp_frame_path in temp_frame_paths:
		temp_frame = read_image(temp_frame_path)
		result_frame = process_frame(temp_frame)
		write_image(temp_frame_path, result_frame)
		update_progress()


def process_image(source_path : str, target_path : str, output_path : str) -> None:
	target_frame = read_static_image(target_path)
	result_frame = process_frame(target_frame)
	write_image(output_path, result_frame)


def process_video(source_path : str, temp_frame_paths : List[str]) -> None:
	frame_processors.multi_process_frames(source_path, temp_frame_paths, process_frames)

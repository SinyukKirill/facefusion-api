from pydantic import BaseModel
from typing import Optional, List
from facefusion.typing import FaceSelectorMode, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, OutputVideoEncoder, FaceDetectorModel, FaceRecognizerModel, TempFrameFormat, Padding
from facefusion.processors.frame.typings import FaceSwapperModel, FaceEnhancerModel, FrameEnhancerModel, FaceDebuggerItem

class Params(BaseModel):
    user_id : str
    source : Optional[str] = None
    target : str
    source_type: Optional[str] = None
    target_type: str

    # execution
    execution_providers: Optional[List[str]] = ['CPUExecutionProvider']
    execution_thread_count: Optional[int] = 4
    execution_queue_count: Optional[int] = 1
    max_memory: Optional[int] = 0

    # face analyser
    face_analyser_order: Optional[FaceAnalyserOrder] = 'left-right'
    face_analyser_age: Optional[FaceAnalyserAge] = None
    face_analyser_gender: Optional[FaceAnalyserGender] = None
    face_detector_model: Optional[FaceDetectorModel] = 'retinaface'
    face_detector_size: Optional[str] = '640x640'
    face_detector_score: Optional[float] = 0.5
    face_recognizer_model: Optional[FaceRecognizerModel] = 'arcface_inswapper'

    # face selector
    face_selector_mode: Optional[FaceSelectorMode] = 'reference'
    reference_face_position: Optional[int] = 0
    reference_face_distance: Optional[float] = 0.6
    reference_frame_number: Optional[int] = 0

    # face mask
    face_mask_blur: Optional[float] = 0.3
    face_mask_padding: Optional[Padding] = (0, 0, 0, 0)

    # frame extraction
    trim_frame_start: Optional[int] = None
    trim_frame_end: Optional[int] = None
    temp_frame_format: Optional[TempFrameFormat] = 'jpg'
    temp_frame_quality: Optional[int] = 100
    keep_temp: Optional[bool] = False

    # output creation
    output_image_quality: Optional[int] = 80
    output_video_encoder: Optional[OutputVideoEncoder] = 'libx264'
    output_video_quality: Optional[int] = 80
    keep_fps: Optional[bool] = False
    skip_audio: Optional[bool] = False

    # frame processors
    frame_processors: List[str] = ['face_blur']

    face_swapper_model: Optional[FaceSwapperModel] = 'inswapper_128'
    face_enhancer_model: Optional[FaceEnhancerModel] = 'gfpgan_1.4'
    face_enhancer_blend: Optional[int] = 80
    frame_enhancer_model: Optional[FrameEnhancerModel] = 'real_esrgan_x2plus'
    frame_enhancer_blend: Optional[int] = 80
    face_debugger_items: Optional[List[FaceDebuggerItem]] = ['kps', 'face-mask']


import facefusion.globals as globals
import facefusion.processors.frame.globals as frame_processors_globals
def print_globals():
    print(f'execution_providers: {globals.execution_providers}')
    print(f'execution_thread_count: {globals.execution_thread_count}')
    print(f'execution_queue_count: {globals.execution_queue_count}')
    print(f'max_memory: {globals.max_memory}')
    print(f'face_analyser_order: {globals.face_analyser_order}')
    print(f'face_analyser_age: {globals.face_analyser_age}')
    print(f'face_analyser_gender: {globals.face_analyser_gender}')
    print(f'face_detector_model: {globals.face_detector_model}')
    print(f'face_detector_size: {globals.face_detector_size}')
    print(f'face_detector_score: {globals.face_detector_score}')
    print(f'face_recognizer_model: {globals.face_recognizer_model}')
    print(f'face_selector_mode: {globals.face_selector_mode}')
    print(f'reference_face_position: {globals.reference_face_position}')
    print(f'reference_face_distance: {globals.reference_face_distance}')
    print(f'reference_frame_number: {globals.reference_frame_number}')
    print(f'face_mask_blur: {globals.face_mask_blur}')
    print(f'face_mask_padding: {globals.face_mask_padding}')
    print(f'trim_frame_start: {globals.trim_frame_start}')
    print(f'trim_frame_end: {globals.trim_frame_end}')
    print(f'temp_frame_format: {globals.temp_frame_format}')
    print(f'temp_frame_quality: {globals.temp_frame_quality}')
    print(f'keep_temp: {globals.keep_temp}')
    print(f'output_image_quality: {globals.output_image_quality}')
    print(f'output_video_encoder: {globals.output_video_encoder}')
    print(f'output_video_quality: {globals.output_video_quality}')
    print(f'keep_fps: {globals.keep_fps}')
    print(f'skip_audio: {globals.skip_audio}')
    print(f'frame_processors: {globals.frame_processors}')
    print(f'face_swapper_model: {frame_processors_globals.face_swapper_model}')
    print(f'face_enhancer_model: {frame_processors_globals.face_enhancer_model}')
    print(f'face_enhancer_blend: {frame_processors_globals.face_enhancer_blend}')
    print(f'frame_enhancer_model: {frame_processors_globals.frame_enhancer_model}')
    print(f'frame_enhancer_blend: {frame_processors_globals.frame_enhancer_blend}')
    print(f'face_debugger_items: {frame_processors_globals.face_debugger_items}')
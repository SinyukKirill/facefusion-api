API mode
========

Run the command:

```
python run.py --api
```

Request body:

```
user_id : str
# source image
source : str
# target image
target : str
# 'image' or 'video'
type: str

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
frame_processors: List[str] = ['face_swapper']

face_swapper_model: Optional[FaceSwapperModel] = 'inswapper_128'
face_enhancer_model: Optional[FaceEnhancerModel] = 'gfpgan_1.4'
face_enhancer_blend: Optional[int] = 80
frame_enhancer_model: Optional[FrameEnhancerModel] = 'real_esrgan_x2plus'
frame_enhancer_blend: Optional[int] = 80
face_debugger_items: Optional[List[FaceDebuggerItem]] = ['kps', 'face-mask']
```

Types:
```
Padding = Tuple[int, int, int, int]

FaceSelectorMode = Literal['reference', 'one', 'many']
FaceAnalyserOrder = Literal['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']
FaceAnalyserAge = Literal['child', 'teen', 'adult', 'senior']
FaceAnalyserGender = Literal['male', 'female']
FaceDetectorModel = Literal['retinaface', 'yunet']
FaceRecognizerModel = Literal['arcface_blendface', 'arcface_inswapper', 'arcface_simswap']
TempFrameFormat = Literal['jpg', 'png']
OutputVideoEncoder = Literal['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']

FaceSwapperModel = Literal['blendface_256', 'inswapper_128', 'inswapper_128_fp16', 'simswap_256', 'simswap_512_unofficial']
FaceEnhancerModel = Literal['codeformer', 'gfpgan_1.2', 'gfpgan_1.3', 'gfpgan_1.4', 'gpen_bfr_256', 'gpen_bfr_512', 'restoreformer']
FrameEnhancerModel = Literal['real_esrgan_x2plus', 'real_esrgan_x4plus', 'real_esrnet_x4plus']

FaceDebuggerItem = Literal['bbox', 'kps', 'face-mask', 'score']
```

FaceFusion
==========

> Next generation face swapper and enhancer.

[![Build Status](https://img.shields.io/github/actions/workflow/status/facefusion/facefusion/ci.yml.svg?branch=master)](https://github.com/facefusion/facefusion/actions?query=workflow:ci)
![License](https://img.shields.io/badge/license-MIT-green)


Preview
-------

![Preview](https://raw.githubusercontent.com/facefusion/facefusion/master/.github/preview.png?sanitize=true)


Installation
------------

Be aware, the installation needs technical skills and is not for beginners. Please do not open platform and installation related issues on GitHub. We have a very helpful [Discord](https://join.facefusion.io) community that will guide you to complete the installation.

Get started with the [installation](https://docs.facefusion.io/installation) guide.


Usage
-----

Run the command:

```
python run.py [options]

options:
  -h, --help                                                                                                         show this help message and exit
  -s SOURCE_PATH, --source SOURCE_PATH                                                                               select a source image
  -t TARGET_PATH, --target TARGET_PATH                                                                               select a target image or video
  -o OUTPUT_PATH, --output OUTPUT_PATH                                                                               specify the output file or directory
  -v, --version                                                                                                      show program's version number and exit

misc:
  --skip-download                                                                                                    omit automate downloads and lookups
  --headless                                                                                                         run the program in headless mode

execution:
  --execution-providers {cpu} [{cpu} ...]                                                                            choose from the available execution providers
  --execution-thread-count [1-128]                                                                                   specify the number of execution threads
  --execution-queue-count [1-32]                                                                                     specify the number of execution queries
  --max-memory [0-128]                                                                                               specify the maximum amount of ram to be used (in gb)

face analyser:
  --face-analyser-order {left-right,right-left,top-bottom,bottom-top,small-large,large-small,best-worst,worst-best}  specify the order used for the face analyser
  --face-analyser-age {child,teen,adult,senior}                                                                      specify the age used for the face analyser
  --face-analyser-gender {male,female}                                                                               specify the gender used for the face analyser
  --face-detector-model {retinaface,yunet}                                                                           specify the model used for the face detector
  --face-detector-size {160x160,320x320,480x480,512x512,640x640,768x768,960x960,1024x1024}                           specify the size threshold used for the face detector
  --face-detector-score [0.0-1.0]                                                                                    specify the score threshold used for the face detector

face selector:
  --face-selector-mode {reference,one,many}                                                                          specify the mode for the face selector
  --reference-face-position REFERENCE_FACE_POSITION                                                                  specify the position of the reference face
  --reference-face-distance [0.0-1.5]                                                                                specify the distance between the reference face and the target face
  --reference-frame-number REFERENCE_FRAME_NUMBER                                                                    specify the number of the reference frame

face mask:
  --face-mask-blur [0.0-1.0]                                                                                         specify the blur amount for face mask
  --face-mask-padding FACE_MASK_PADDING [FACE_MASK_PADDING ...]                                                      specify the face mask padding (top, right, bottom, left) in percent

frame extraction:
  --trim-frame-start TRIM_FRAME_START                                                                                specify the start frame for extraction
  --trim-frame-end TRIM_FRAME_END                                                                                    specify the end frame for extraction
  --temp-frame-format {jpg,png}                                                                                      specify the image format used for frame extraction
  --temp-frame-quality [0-100]                                                                                       specify the image quality used for frame extraction
  --keep-temp                                                                                                        retain temporary frames after processing

output creation:
  --output-image-quality [0-100]                                                                                     specify the quality used for the output image
  --output-video-encoder {libx264,libx265,libvpx-vp9,h264_nvenc,hevc_nvenc}                                          specify the encoder used for the output video
  --output-video-quality [0-100]                                                                                     specify the quality used for the output video
  --keep-fps                                                                                                         preserve the frames per second (fps) of the target
  --skip-audio                                                                                                       omit audio from the target

frame processors:
  --frame-processors FRAME_PROCESSORS [FRAME_PROCESSORS ...]                                                         choose from the available frame processors (choices: face_debugger, face_enhancer, face_swapper, frame_enhancer, ...)
  --face-debugger-items {bbox,kps,face-mask,score} [{bbox,kps,face-mask,score} ...]                                  specify the face debugger items
  --face-enhancer-model {codeformer,gfpgan_1.2,gfpgan_1.3,gfpgan_1.4,gpen_bfr_256,gpen_bfr_512,restoreformer}        choose the model for the frame processor
  --face-enhancer-blend [0-100]                                                                                      specify the blend factor for the frame processor
  --face-swapper-model {blendface_256,inswapper_128,inswapper_128_fp16,simswap_256,simswap_512_unofficial}           choose the model for the frame processor
  --frame-enhancer-model {real_esrgan_x2plus,real_esrgan_x4plus,real_esrnet_x4plus}                                  choose the model for the frame processor
  --frame-enhancer-blend [0-100]                                                                                     specify the blend factor for the frame processor

uis:
  --ui-layouts UI_LAYOUTS [UI_LAYOUTS ...]                                                                           choose from the available ui layouts (choices: benchmark, webcam, default, ...)
```


Documentation
-------------

Read the [documentation](https://docs.facefusion.io) for a deep dive.

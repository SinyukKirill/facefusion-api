from typing import List, Optional

from facefusion.processors.frame.typings import FaceSwapperModel, FaceEnhancerModel, FrameEnhancerModel, FaceDebuggerItem, FaceBlurModel

face_swapper_model : Optional[FaceSwapperModel] = None
face_enhancer_model : Optional[FaceEnhancerModel] = None
face_enhancer_blend : Optional[int] = None
frame_enhancer_model : Optional[FrameEnhancerModel] = None
frame_enhancer_blend : Optional[int] = None
face_debugger_items : Optional[List[FaceDebuggerItem]] = None

face_blur_model: Optional[FaceBlurModel] = None
face_blur_blend : Optional[int] = None


import base64
import requests
import time

def image_to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

# source_image_path = 'source.jpg'
target_image_path = 'manyface.png'
# target_image_path = 'demo-2.mp4'
# target_image_path = 'error-faces.jpg'

# source_str = image_to_base64_str(source_image_path)
target_str = image_to_base64_str(target_image_path)

params = {
    'user_id': 'test',
    'target': target_str,
    # 'target_type': 'mp4',
    'target_type': 'png',
    'frame_processors': ['face_blur'],
    'face_mask_blur': 0.5,
    'face_mask_padding': [5, 5, 5, 5],
    # 'keep_fps': True,
    'execution_thread_count': 100,
    'execution_queue_count': 4,
    'face_detector_score': 0.25,
    'face_detector_iou': 0.4,
    'face_detector_model': 'yolo_face_onnx',
}

url = 'http://0.0.0.0:8000/'
response = requests.post(url, json=params)

print("Status Code:", response.status_code)
# print("Response Body:", response.text)

if response.status_code == 200:
    output_data = base64.b64decode(response.json()['output'])
    with open(f'output/{int(time.time())}.jpg', 'wb') as f:
        f.write(output_data)
else:
    print("Error: The request did not succeed.")

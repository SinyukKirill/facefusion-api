import base64
import requests
import time

def image_to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

source_image_path = 'source.jpg'
target_image_path = 'test.jpg'

source_str = image_to_base64_str(source_image_path)
target_str = image_to_base64_str(target_image_path)

params = {
    'user_id': 'test',
    'source': source_str,
    'target': target_str,
    'source_type': 'jpg',
    'target_type': 'jpg',
    'frame_processors': ['face_swapper'],
    # 'face_mask_blur': 0.5,
    # 'face_mask_padding': [5, 5, 5, 5],
    # 'keep_fps': True,
    # 'execution_thread_count': 40,
    # 'face_selector_mode': 'one',
}

url = 'https://e082-221-103-234-215.ngrok-free.app/'
response = requests.post(url, json=params)

print("Status Code:", response.status_code)
# print("Response Body:", response.text)

if response.status_code == 200:
    output_data = base64.b64decode(response.json()['output'])
    with open(f'output/{int(time.time())}.jpg', 'wb') as f:
        f.write(output_data)
else:
    print("Error: The request did not succeed.")

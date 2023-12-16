import base64
import requests
import time

def image_to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

source_image_path = 'source.jpg'
target_image_path = 'target.jpg'

source_str = image_to_base64_str(source_image_path)
target_str = image_to_base64_str(target_image_path)

params = {
    'user_id': 'test',
    'source': source_str,
    'target': target_str,
    'source_type': 'jpg',
    'target_type': 'jpg',
}

url = 'https://3499-2400-4050-b6e0-1600-8c32-63d1-26ff-e1f3.ngrok-free.app/'
response = requests.post(url, json=params)

# ステータスコードとレスポンスの内容を確認
print("Status Code:", response.status_code)
print("Response Body:", response.text)

# ステータスコードが200の場合のみ処理を進める
if response.status_code == 200:
    output_data = base64.b64decode(response.json()['output'])
    with open(f'output/{int(time.time())}.jpg', 'wb') as f:
        f.write(output_data)
else:
    print("Error: The request did not succeed.")

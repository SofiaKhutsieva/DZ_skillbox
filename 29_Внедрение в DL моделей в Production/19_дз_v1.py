import sys
import json
import requests
import numpy as np

from PIL import Image

# Подготовка входных данных для нейронной сети
inp = sys.argv[1]
pic = Image.open(inp)
pic = pic.resize((150, 150), Image.ANTIALIAS)
pix = np.array(pic.getdata()).reshape(1, pic.size[0], pic.size[1], 3) / 255.


# Подготовка данных для HTTP запроса
request_data = json.dumps({
    "signature_name": "serving_default",
    "instances": pix.tolist()
})
headers = {"content-type": "application/json"}


# HTTP запрос на сервер
json_response = requests.post(
    'http://localhost:8501/v1/models/saved_model/versions/1:predict',
    data=request_data, headers=headers)

# Обработка JSON ответа
predictions = json.loads(json_response.text)
print(predictions)
predictions = json.loads(json_response.text)['predictions']
print(predictions[0][0])
Result1 = 1 if predictions[0][0] >= 0.5 else 0
print(f'Result1={Result1}')

# HTTP запрос на сервер
json_response = requests.post(
    'http://localhost:8501/v1/models/saved_model/versions/1:predict_classes',
    data=request_data, headers=headers)

# Обработка JSON ответа
predictions = json.loads(json_response.text)
print(predictions)
predictions = json.loads(json_response.text)#['predictions']
Result2 = np.argmax(predictions)
print(f'Result2={Result2}')

dict = {0: 'кошка', 1: 'собака'}

# Печать результата распознавания
print(f'На картинке изображена {dict[Result1]}')

# tensorflow_model_server --rest_api_port=8501 --model_name=saved_model --model_base_path="/mnt/s/P/skillbox/Data_Scientist_ML_Средний_уровень_нейронные_сети/19_Внедрение в DL моделей в Production/дз/saved_model/"
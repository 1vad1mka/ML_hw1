# Для создания приложения
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import io
# Сервер
import uvicorn
# Валидация данных
from pydantic import BaseModel
from typing import List
# Сериализация модели
import pickle
# Для обработки данных
import pandas as pd
import numpy as np


# Функция для первичной обработке данных в приложении
def initial_preprocessing(data):

    # Функция для выявления единиц измерения
    def torque_cat(x):
        if not isinstance(x, str):
            return np.nan
        elif 'nm' in x:
            return 'nm'
        elif 'kgm' in x:
            return 'kgm'
        else:
            return 'unknown'


    # Функция для выделения torque
    def get_torque(x):
        if isinstance(x, list):
            return x[0]
        else:
            return x

    # Функция выделения rpm
    def get_max_rpm(x):
        if isinstance(x, list):
            if len(x) > 1:
                el = x[1]
                if '-' in el:
                    return el.split('-')[1]
                return el
            elif len(x) == 1:
                return np.nan
        else:
            return x
            
    # Функция для конвертации torque из kgm в nm
    def convert_torque(row):
        G = 9.80665
        
        if row['torque_cat'] == 'kgm':
            return row['torque'] * G
    
        else:
            return row['torque']
    

    # Убираем единицы измерения, приводим к float
    data['mileage'] = data['mileage'].str.replace('[kmpl|km/kg]', '', regex=True).astype('float')
    data['engine'] = data['engine'].str.replace('CC', '').astype('float')
    data['max_power'] = data['max_power'].str.replace('bhp', '').replace('^\s*$', np.nan, regex=True).astype('float')


    # Преобразовываем torque
    # Приведем в нижний регистр
    data['torque'] = data['torque'].str.lower()
    # Запонминаем единицы измерения для будущей переменной torque
    data['torque_cat'] = data['torque'].apply(torque_cat)
    
    # Паттерн для разделения данных
    pattern = r'(?:nm@|kgm at|@|nm at|/|nm)'
    
    # Создаем новые признаки в тренировочных данных
    data['torque'] = data['torque'].str.replace('\(.*\)', '', regex=True).str.strip(' rpm')\
                                           .str.replace('~', '-')\
                                           .str.replace(',', '')\
                                           .str.split(pattern, regex=True)
    
    data['max_torque_rpm'] = data['torque'].apply(get_max_rpm).str.extract(r'(\d+)', expand=False).astype('float')
    data['torque_new'] = data['torque'].apply(get_torque).str.replace('[a-zA-Z]', '', regex=True).astype('float')

    data = data.drop(['torque'], axis=1).rename({'torque_new': 'torque'}, axis=1)
    

    # Конвертируем kgm в трейне
    data['torque'] = data.apply(convert_torque, axis=1)
    data = data.rename({'torque': 'torque_nm'}, axis=1)
    data = data.drop(['torque_cat'], axis=1)

    # Преобразовываем название машины
    data['name'] = data['name'].apply(lambda x: x.split()[0])

    return data 


# Загружаем модель
with open('elastic_regressor.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Инициализируем модель данных для одного наблюдения
class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


# Создаем приложение
app = FastAPI()

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item = item.dict()
    item = pd.DataFrame(item, index=[0])
    item = initial_preprocessing(item)

    predictions = model.predict(item)[0]

    return predictions


@app.post("/predict_items")
def predict_items(file: UploadFile):
    data = pd.read_csv(file.file)
    data = initial_preprocessing(data)

    predictions = pd.Series(model.predict(data))

    result = pd.concat([data, predictions], axis=1)
    result.columns = list(data.columns) + ['selling_price_pred']
    stream = io.StringIO()
    result.to_csv(stream, index=False)
    # Возвращаем курсор в начало потока
    stream.seek(0)
    
    return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=response.csv"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




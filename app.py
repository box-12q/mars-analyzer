import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import warnings

warnings.filterwarnings("ignore") # передаём бибилиотеке аргумент, чтобы при работе с программой если будет какая-то ошибка ой её скроет

app = Flask(__name__) # создаём экземаляр класса для сайта (встроенная перемена name)
app.secret_key = "dev" # получаем доступ для шифрования защиты сайта

# Папки
UPLOAD_FOLDER = 'uploads' # создаём переменную для папки
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # создаём папку в нашем проекте

# создаём функцию для обнаружения шапок
def detect_polar_caps(image_path):
    """Обнаруживает полярные шапки на марсианских снимках"""
    try:
        # Загружаем изображение
        img = cv2.imread(image_path) # пытаемся подрузить фотографию из переменной image_path
        if img is None: # проверяем если не находят фото
            return None, "Ошибка загрузки изображения" # возвращаем предупреждение

        # Конвертируем в HSV для лучшего выделения льда
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Создаем маски (границы цветовых изображений) для белых/голубых областей (лед)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])

        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])

        # Применяем маски
        mask_white = cv2.inRange(hsv, lower_white, upper_white) # проверяем каждый пиксель попадёт ли наш диапазон, если не опадает - делает чёрным
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) # тоже самое для голубого, если пиксель не подходит - становится чёрным

        # Комбинируем маски
        ice_mask = cv2.bitwise_or(mask_white, mask_blue) # склеиваем все макси воедино (bitwise_or - функция объединения наших масок)

        # Морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8) # создаём масив с единичками  5х5 (использыем для обработки маски )
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, kernel) # убираем малкие чёёрные точки на белом фоне (увеличиваем края белых пиквелей и потом опять сужаем)
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, kernel) # убираем мелкие одиночные точки на чёрном фоне(аналогично ^)

        # Находим контуры льда
        contours, _ = cv2.findContours(ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # находим контур наших шапок

        # Фильтруем контуры по размеру
        min_area = 500 # минимальный допустимый размер шляпы
        ice_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Создаем изображение с выделенными шапками
        result_img = img.copy()

        # Рисуем контуры полярных шапок
        cv2.drawContours(result_img, ice_contours, -1, (0, 255, 255), 3)

        # Вычисляем статистику
        total_ice_area = sum(cv2.contourArea(cnt) for cnt in ice_contours)
        total_image_area = img.shape[0] * img.shape[1]
        ice_percentage = (total_ice_area / total_image_area) * 100

        # Добавляем информацию на изображение
        info_text = f"Polar caps: {len(ice_contours)} areas, {ice_percentage:.1f}%"
        cv2.putText(result_img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Определяем наличие значительных полярных шапок
        has_significant_caps = ice_percentage > 1.0 and len(ice_contours) > 0

        analysis_result = {
            'ice_contours': ice_contours,
            'ice_area': total_ice_area,
            'ice_percentage': ice_percentage,
            'cap_count': len(ice_contours),
            'has_caps': has_significant_caps,
            'message': f"Обнаружено {len(ice_contours)} полярных шапок ({ice_percentage:.1f}% площади)"
        }

        return result_img, analysis_result

    except Exception as e:
        return None, f"Ошибка анализа: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        flash('Выберите файл с изображением Марса')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('Выберите файл')
        return redirect(request.url)

    # Сохраняем файл
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Анализ полярных шапок
        result_img, analysis_result = detect_polar_caps(filepath)

        if result_img is None:
            flash(f'Ошибка анализа: {analysis_result}')
            return redirect(request.url)

        # Сохраняем результат
        result_filename = f"result_{filename}"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv2.imwrite(result_path, result_img)

        return render_template('result.html',
                               has_caps=analysis_result['has_caps'],
                               cap_count=analysis_result['cap_count'],
                               ice_percentage=analysis_result['ice_percentage'],
                               message=analysis_result['message'],
                               img_url=url_for('uploaded_file', filename=result_filename))

    except Exception as e:
        flash(f'Ошибка обработки: {str(e)}')
        return redirect(request.url)


if __name__ == '__main__':
    print("=" * 50)
    print("🔍 Анализатор полярных шапок Марса")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000)
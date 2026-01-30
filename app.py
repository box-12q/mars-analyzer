import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "dev"  # Ключ для работы сессий
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Ограничение загрузки: 16 МБ

# Папки
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Разрешенные расширения файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_polar_caps(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, "Ошибка загрузки изображения"

        # Конвертируем в HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Маски для белого и голубого (лед)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 50, 255])
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])

        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        ice_mask = cv2.bitwise_or(mask_white, mask_blue)

        # Морфология
        kernel = np.ones((5, 5), np.uint8)
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, kernel)
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, kernel)

        # Контуры
        contours, _ = cv2.findContours(ice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 500
        ice_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Рисуем контуры
        result_img = img.copy()
        cv2.drawContours(result_img, ice_contours, -1, (0, 255, 255), 3)

        # Статистика
        total_ice_area = sum(cv2.contourArea(cnt) for cnt in ice_contours)
        total_image_area = img.shape[0] * img.shape[1]
        ice_percentage = (total_ice_area / total_image_area) * 100

        info_text = f"Polar caps: {len(ice_contours)} areas, {ice_percentage:.1f}%"
        cv2.putText(result_img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        has_significant_caps = ice_percentage > 1.0 and len(ice_contours) > 0

        # ВАЖНО: Для сохранения в сессию преобразуем numpy-типы в стандартные python-типы (float, int)
        analysis_result = {
            'ice_area': float(total_ice_area),
            'ice_percentage': float(ice_percentage),
            'cap_count': int(len(ice_contours)),
            'has_caps': bool(has_significant_caps),
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
    # Проверяем наличие файловой части
    if 'file' not in request.files:
        flash('Файлы не выбраны')
        return redirect(request.url)

    files = request.files.getlist('file')  # Получаем список всех файлов

    if not files or files[0].filename == '':
        flash('Файлы не выбраны')
        return redirect(request.url)

    processed_data = []  # Список для хранения результатов всех файлов

    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Генерируем уникальное имя
                filename = f"{uuid.uuid4().hex}_{file.filename}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                # Анализируем
                result_img, analysis_result = detect_polar_caps(filepath)

                if result_img is None:
                    continue  # Пропускаем битые файлы

                # Сохраняем результат обработки
                result_filename = f"result_{filename}"
                result_path = os.path.join(UPLOAD_FOLDER, result_filename)
                cv2.imwrite(result_path, result_img)

                # Добавляем данные в список (для сессии)
                processed_data.append({
                    'original_name': file.filename,
                    'result_filename': result_filename,
                    'stats': analysis_result
                })

            except Exception as e:
                print(f"Ошибка при обработке {file.filename}: {e}")
                continue

    if not processed_data:
        flash('Не удалось обработать ни одного изображения (проверьте формат).')
        return redirect(url_for('index'))

    # Сохраняем результаты в сессию браузера
    session['results'] = processed_data

    # Переходим к просмотру первого результата (индекс 0)
    return redirect(url_for('show_result', index=0))


@app.route('/result/<int:index>')
def show_result(index):
    # Получаем данные из сессии
    results = session.get('results', [])

    # Если данных нет или индекс выходит за границы
    if not results or index < 0 or index >= len(results):
        flash('Результаты устарели или не найдены. Попробуйте снова.')
        return redirect(url_for('index'))

    current_item = results[index]
    total_count = len(results)

    return render_template('result.html',
                           data=current_item,
                           index=index,
                           total=total_count)


if __name__ == '__main__':
    print("=" * 50)
    print("🔍 Мульти-анализатор полярных шапок Марса запущен")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)

# KV_FaceRecognition

KV_FaceRecognition — это модель компьютерного зрения, разработанная для распознавания эмоций человека на основе изображений лиц. Проект использует сверточную нейронную сеть (CNN), обученную на наборе данных изображений эмоций, и поддерживает прогнозирование в реальном времени с использованием веб-камеры. Модель определяет шесть основных эмоций: **Angry**, **Fearful**, **Happy**, **Sad**, **Surprised**, и **Neutral**.

## Основные особенности:
- **Сверточная нейронная сеть (CNN)**: Модель построена с использованием нескольких свёрточных и пулинговых слоёв для извлечения признаков с изображений лиц и последующей классификации эмоций.
- **Реализация в реальном времени**: Используется классификатор Хаара для обнаружения лиц в потоке с камеры, после чего предсказание эмоции выводится на изображение в реальном времени.
- **Ранняя остановка**: В процессе обучения используется техника ранней остановки, чтобы избежать переобучения и сохранить лучшие веса модели.
- **L2-регуляризация и адаптивное уменьшение скорости обучения**: Оптимизация модели проводится с использованием регуляризации и адаптивного планировщика скорости обучения для достижения лучших результатов.

## Основные технологии:
- **PyTorch**: Используется для разработки и обучения модели CNN.
- **OpenCV**: Применяется для захвата изображений с веб-камеры и обнаружения лиц.
- **Torchvision**: Для трансформации и предобработки изображений.
- **Трансформации данных**: Изображения нормализуются и изменяются по размеру перед подачей в модель.

## Применение:
Модель может использоваться для:
- Систем наблюдения
- Интерфейсов для мониторинга настроений пользователей
- Других приложений, где необходимо распознавание эмоций по лицам

## Примечание:
Для использования KV_FaceRecognition_V4.ipynb требуется NVIDEA CUDA 12.3. Поэтому я создал KV_Model_Test.ipynb, где можно протестировать модель в режиме реального времени(Будеть работать на CPU, а не GPU). 
Весы модели можно взять по ссылке: https://drive.google.com/drive/folders/1cDgMqHmqHf20Gv1rU29wXEwzlXrtIeAD?usp=drive_link
Ну или же скачать KV_Model_Test.exe(Будеть работать на CPU, а не GPU).

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from cnn_model import EmotionRecognitionCNN

# Загрузка обученной модели
model = EmotionRecognitionCNN(num_classes=6)
model.load_state_dict(torch.load('emotion_recognition_model_V4++.pth'))
model.eval()

# Загрузка классификатора Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Преобразования для предобработки изображения
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Используем три канала
])

# Метки эмоций
emotions = ['Angry', 'Fearful', 'Sad', 'Happy', 'Surprised', 'Disgt']

# Запуск камеры
cap = cv2.VideoCapture(0)

while True:
    # Захват изображения с камеры
    ret, frame = cap.read()
    if not ret:
        break

    # Обнаружение лиц на изображении
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Извлечение лица
        face = frame[y:y + h, x:x + w]

        # Преобразование для модели
        face_img = transform(face).unsqueeze(0)  # Добавляем размер батча

        # Предсказание эмоции
        with torch.no_grad():
            output = model(face_img)
            emotion_prediction = F.softmax(output, dim=1)
            emotion_label = torch.argmax(emotion_prediction).item()

        # Метка предсказанной эмоции
        emotion_text = emotions[emotion_label]

        # Отображение предсказанной эмоции на изображении
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Показ изображения
    cv2.imshow('Emotion Recognition', frame)

    # Прерывание по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение камеры и закрытие окон
cap.release()
cv2.destroyAllWindows()

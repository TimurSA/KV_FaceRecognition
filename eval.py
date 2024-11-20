from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
from cnn_model import EmotionRecognitionCNN
from data import dt


def evaluate(model, dataloader, class_names, device):
    # Перенос модели на устройство
    model = model.to(device)
    model.eval()  # Переключаем модель в режим оценки
    all_preds = []
    all_labels = []

    # Сбор предсказаний и истинных меток
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Вычисляем матрицу ошибок
    cm = confusion_matrix(all_labels, all_preds)

    # Отображаем матрицу ошибок
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    # Выводим отчет классификации
    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)


if __name__ == '__main__':
    # Проверка доступности GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Загрузка обученной модели
    model = EmotionRecognitionCNN(num_classes=6)
    model.load_state_dict(torch.load('emotion_recognition_model_V4++.pth'))

    evaluate(model, dt.val_loader, dt.class_names, device)

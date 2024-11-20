from tqdm import tqdm
import time
from cnn_model import EarlyStopping
from utils import plot_metrics
from cnn_model import EmotionRecognitionCNN
from data import dt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model_with_metrics(model, criterion, optimizer, scheduler, num_epochs=25, model_save_path='best_model.pth',
                             patience=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Списки для записи метрик
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            dataloader = dt.dataloaders[phase]
            dataset_size = dt.dataset_sizes[phase]

            for inputs, labels in tqdm(dataloader, desc=f"{phase.capitalize()} Progress", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                early_stopping(epoch_loss, model, model_save_path)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    model.load_state_dict(torch.load(model_save_path))
                    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
                    return model

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    model.load_state_dict(torch.load(model_save_path))

    # Построим графики метрик
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    return model


if __name__ == '__main__':
    # Проверка доступности GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Инициализируем модель
    num_classes = len(dt.class_names)
    model = EmotionRecognitionCNN(num_classes=num_classes)
    model = model.to(device)

    # Определим функцию потерь и оптимизатор с L2-регуляризацией
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,
                           weight_decay=1e-4)  # Добавляем L2-регуляризацию через weight_decay

    # Планировщик для уменьшения скорости обучения при остановке улучшений
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Обучаем модель с ранней остановкой и сохранением лучших весов
    model_save_path = 'emotion_recognition_model_V4++.pth'
    model = train_model_with_metrics(model, criterion, optimizer, exp_lr_scheduler, num_epochs=20,
                                     model_save_path=model_save_path, patience=5)
    torch.save(model.state_dict(), model_save_path)

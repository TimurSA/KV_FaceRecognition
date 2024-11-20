from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Data_Transformer():
    # Задаем размер входного изображения
    input_size = 48

    def __init__(self):
        self.data_transforms = transforms.Compose([
            transforms.Resize((Data_Transformer.input_size, Data_Transformer.input_size)),  # Изменение размера
            transforms.ToTensor(),  # Преобразование в тензор
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализация
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((Data_Transformer.input_size, Data_Transformer.input_size)),  # Изменение размера
            transforms.ToTensor(),  # Преобразование в тензор
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Нормализация
        ])
        # Загрузка данных из разных папок
        self.train_dir = r"C:\Users\Omen\Desktop\Datasets_NEW\train"  # Папка с тренировочными данными
        self.val_dir = r"C:\Users\Omen\Desktop\Datasets_NEW\test"  # Папка с валидационными данными

        # Создаем наборы данных с использованием трансформаций
        self.train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.data_transforms)
        self.val_dataset = datasets.ImageFolder(root=self.val_dir, transform=self.val_transforms)

        # DataLoader для загрузки данных в батчах
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)

        # Словарь для DataLoader'ов и размеров наборов данных
        self.dataloaders = {'train': self.train_loader, 'val': self.val_loader}
        self.dataset_sizes = {'train': len(self.train_dataset), 'val': len(self.val_dataset)}
        self.class_names = self.train_dataset.classes  # Можно использовать только тренировочные классы, если они одинаковы


dt = Data_Transformer()

if __name__ == '__main__':
    print(f"Train dataset size: {len(dt.train_dataset)}")
    print(f"Val dataset size: {len(dt.val_dataset)}")
    print(f"Classes: {dt.class_names}")

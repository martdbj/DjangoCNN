# tu_app/ai_logic.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
from django.conf import settings


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear((128 * 16 * 16), 64)  # Asegúrate de que sea 64
        self.output = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.pooling(self.bn1(self.conv1(x))))
        x = self.relu(self.pooling(self.bn2(self.conv2(x))))
        x = self.relu(self.pooling(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.linear(x))
        x = self.output(x)
        return x


class AiInitializer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['cat', 'horse']

        model_path = os.path.join(settings.BASE_DIR, 'djangoproject', 'ai', 'animal_classifier.pth')

        self.model = Net()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image)
            prob = F.softmax(output, dim=1)
            value, predicted = torch.max(prob, 1)
            return self.classes[predicted.item()], f"{value.item() * 100:.2f}%"
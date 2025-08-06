import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

DATA_DIR = r'data/chest_xray/train'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_raw_images(folder):
    X, y = [], []
    label_names = os.listdir(folder)

    for label in label_names:
        path = os.path.join(folder, label)
        if not os.path.isdir(path):
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    return np.array(X), np.array(y)

#CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        sample_input = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
        x = self.pool(torch.relu(self.conv1(sample_input)))
        self.flattened_size = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(self.flattened_size, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        return x

def train_cnn(X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) / 255.0
    y = torch.tensor(y, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "model_raw.pth")
    print("Model saved to model_raw.pth")

    return model, le

def evaluate_cnn(model, X, y, le):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) / 255.0
    y_true = le.transform(y)
    with torch.no_grad():
        preds = model(X.to(DEVICE))
        preds = torch.argmax(preds, dim=1).cpu().numpy()
    print("Evaluating CNN...")
    print(classification_report(y_true, preds))

#RANDOM FOREST
def train_random_forest(X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    return clf, le

#MAIN
if __name__ == "__main__":
    print("Loading raw images (no preprocessing)...")
    X_raw, y = load_raw_images(DATA_DIR)

    print("Training CNN on raw data...")
    cnn_model, cnn_le = train_cnn(X_raw, y)
    evaluate_cnn(cnn_model, X_raw, y, cnn_le)

    print("Training Random Forest on raw flattened data...")
    flat_X = X_raw.reshape(X_raw.shape[0], -1)
    X_train, X_test, y_train, y_test = train_test_split(flat_X, y, test_size=0.2, random_state=42)
    rf_model, rf_le = train_random_forest(X_train, y_train)

    print("Evaluating Random Forest...")
    y_pred = rf_model.predict(X_test)
    y_test_encoded = rf_le.transform(y_test)
    print(classification_report(y_test_encoded, y_pred))

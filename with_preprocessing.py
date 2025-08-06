import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

DATA_DIR = r'data/chest_xray/train'
IMG_SIZE = 128  # Resize all images to this size
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#PREPROCESSING

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #linear Filtering (Gaussian Blur)
    linear = cv2.GaussianBlur(gray, (5, 5), 0)

    #fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)

    #geometric Transform (Rotation)
    rows, cols = gray.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)  # Rotate by 15 degrees
    rotated = cv2.warpAffine(gray, M, (cols, rows))

    #point Operator (Histogram Equalization)
    point_op = cv2.equalizeHist(gray)

    #non-linear Filtering (Median)
    non_linear = cv2.medianBlur(gray, 5)

    #pyramids
    pyr_down = cv2.pyrDown(gray)


    return cv2.resize(linear, (IMG_SIZE, IMG_SIZE)), cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

#LOAD IMAGES
def load_images_from_folder(folder):
    X_raw, X_proc, y = [], [], []
    label_names = os.listdir(folder)

    for label in label_names:
        path = os.path.join(folder, label)
        if not os.path.isdir(path):  #Skip if not a directory
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            #skip non-image files
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            if not os.path.isfile(img_path):
                continue
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                proc, raw = preprocess_image(img)
                X_proc.append(proc)
                X_raw.append(raw)
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    return np.array(X_raw), np.array(X_proc), np.array(y)

#DIMENSIONALITY REDUCTION
def apply_pca_svd(X):
    flat_X = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(flat_X)

    svd = TruncatedSVD(n_components=100)
    X_svd = svd.fit_transform(flat_X)

    return X_pca, X_svd

#CNN MODEL
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)   # Output: (16, 126, 126)
        self.pool = nn.MaxPool2d(2, 2)     # Output: (16, 63, 63)
        
        # Automatically compute the flattened size
        sample_input = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
        x = self.pool(torch.relu(self.conv1(sample_input)))
        self.flattened_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        return x
#TRAIN CNN

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
    
    #SAVE MODEL
    torch.save(model.state_dict(), "model.pth")  # Save to file in current directory
    print("Model saved to model.pth")

    return model, le


#EVALUATE CNN
def evaluate_cnn(model, X, y, le):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) / 255.0
    y_true = le.transform(y)
    with torch.no_grad():
        preds = model(X.to(DEVICE))
        preds = torch.argmax(preds, dim=1).cpu().numpy()
    print("Evaluating CNN...")
    print(classification_report(y_true, preds))


#TRADITIONAL ML MODEL
def train_random_forest(X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    return clf, le

from sklearn.model_selection import train_test_split  #ADD THIS IMPORT AT TOP

#MAIN PIPELINE
if __name__ == "__main__":
    print("Loading images...")
    X_raw, X_proc, y = load_images_from_folder(DATA_DIR)

    print("Applying PCA and SVD...")
    X_pca, X_svd = apply_pca_svd(X_proc)

    print("Training CNN...")
    cnn_model, cnn_le = train_cnn(X_proc, y)
    evaluate_cnn(cnn_model, X_proc, y, cnn_le)  #EVALUATE CNN

    print("Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    rf_model, rf_le = train_random_forest(X_train, y_train)

    print("Evaluating Random Forest...")
    y_pred = rf_model.predict(X_test)
    y_test_encoded = rf_le.transform(y_test)
    print(classification_report(y_test_encoded, y_pred))

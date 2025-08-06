import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

IMG_SIZE = 128 

#trained model 
MODEL_PATH = 'model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

model = CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


#load model
model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

#preprocessing functions
def apply_all_preprocessing(img):
    results = {}

    #original image
    results['Original'] = img

    #linear Filtering (Gaussian Blur)
    results['Gaussian Blur'] = cv2.GaussianBlur(img, (5, 5), 0)

    #fourier Transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    results['Fourier Transform'] = magnitude_spectrum

    #geometric Transform (Rotation)
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    results['Rotation'] = cv2.warpAffine(img, M, (cols, rows))

    #point Operator (Histogram Equalization)
    results['Histogram Equalization'] = cv2.equalizeHist(img)

    #non-linear Filtering (Median Blur)
    results['Median Filtering'] = cv2.medianBlur(img, 5)

    #pyramids
    results['Pyramid Down'] = cv2.pyrDown(img)
    
    return results

#visualization Function
def visualize_preprocessing(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed_images = apply_all_preprocessing(img)

    plt.figure(figsize=(16, 10))
    for i, (title, img) in enumerate(processed_images.items()):
        plt.subplot(2, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

#inference Function
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = 'Pneumonia' if predicted.item() == 1 else 'Normal'
    
    print(f"Prediction for {image_path}: {label}")
    return label

#DEMO USAGE
if __name__ == "__main__":
    img_path = 'data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg'  
    visualize_preprocessing(img_path)
    predict_image(img_path)

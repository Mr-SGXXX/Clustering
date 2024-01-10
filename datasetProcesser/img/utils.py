import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import numpy as np
from skimage.feature import hog
from skimage import color
import cv2 as cv

from utils import config



# Define Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name == "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class ResNet50Extractor(nn.Module):
    def __init__(self, data, cfg:config):
        super(ResNet50Extractor, self).__init__()
        self.device = cfg.get("global", "device")    
        extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
        model = models.resnet50(pretrained=True).to(self.device)
        self.model = FeatureExtractor(model, extract_list)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        data = torch.Tensor(data) / 255
        dataset = TensorDataset(data)  
        self.dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    def forward(self):
        features = []
        with tqdm(self.dataloader, desc="Extracting features", dynamic_ncols=True, leave=False) as dataloader:
            for img, in dataloader:
                if img.size(1) == 1:
                    img = img.repeat(1, 3, 1, 1)
                img = self.transform(img)
                img = img.to(self.device)
                feature = self.model(img)[3]
                feature = feature.view(feature.size(0), -1)
                feature = feature.cpu().detach().numpy()   
                features.append(feature)
        features = np.concatenate(features, axis=0)
        return features
    
# Define a function to extract HOG features
def extract_hog_features(image):
    # Convert the image to grayscale
    grayscale = color.rgb2gray(image)
    # Extract HOG features
    features, _ = hog(grayscale, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
    return features

# Define a function to extract 8x8 color map features
def extract_color_map_features(image, size=(8, 8)):
    # Resize the image to 8x8
    resized_image = cv.resize(image, size)
    # Flatten the color map
    color_map = resized_image.flatten()
    return color_map

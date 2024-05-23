import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from io import BytesIO

# Configure Gemini API
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load a pre-trained ResNet model for example purposes
model = models.resnet50(pretrained=True)
model.eval()

# Define a transform to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image):
    img_t = preprocess(image)
    img_t = torch.unsqueeze(img_t, 0)  # Create a mini-batch as expected by the model
    with torch.no_grad():
        features = model(img_t)
    return features.squeeze().numpy()



# Additional color histogram analysis
def color_histogram(image):
    
    # Convert image to RGB mode
    img = image.convert('RGB')
    hist = img.histogram()
    return np.array(hist).reshape(-1, 256).sum(axis=0)
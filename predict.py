from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
import io
import requests 
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
Class = ['Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Astma_weed', 'Avacado', 'Badipala', 'Bamboo', 'Basale', 'Beans', 'Betel', 'Betel_Nut', 'Bhrami', 'Brahmi', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry_Leaf', 'Doddapatre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Gauva', 'Geranium', 'Ginger', 'Globe Amarnath', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemon_grass', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Nagadali', 'Neem', 'Nelavembu', 'Nerale', 'Nithyapushpa', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Pappaya', 'Parijatha', 'Pea', 'Pepper', 'Pomegranate', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Raktachandini', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulasi', 'Tulsi', 'Turmeric', 'Wood_sorel', 'camphor', 'kamakasturi', 'kepala']


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
num_classes = len(Class)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
model.load_state_dict(torch.load('mobilenet_v3_plant_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0) 
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return Class[preds.item()]

@app.route('/predict', methods=['POST'])
def predict_plant():
    data = request.get_json()
    
    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400
    
    image_url = data['image_url']
    
    # Fetch the image from the URL
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    # Predict the class of the plant image
    prediction = predict(image)
    
    return jsonify({"prediction": prediction,})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

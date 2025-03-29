from PIL import Image
import torch
from torchvision import models, transforms

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

num_classes = 93  # Set this to the number of plant classes in your dataset
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
model.load_state_dict(torch.load('mobilenet_v3_plant_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# Image transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names= ['Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Astma_weed', 'Avacado', 'Badipala', 'Bamboo', 'Basale', 'Beans', 'Betel', 'Betel_Nut', 'Bhrami', 'Brahmi', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry_Leaf', 'Doddapatre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Gauva', 'Geranium', 'Ginger', 'Globe Amarnath', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemon_grass', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Nagadali', 'Neem', 'Nelavembu', 'Nerale', 'Nithyapushpa', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Pappaya', 'Parijatha', 'Pea', 'Pepper', 'Pomegranate', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Raktachandini', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulasi', 'Tulsi', 'Turmeric', 'Wood_sorel', 'camphor', 'kamakasturi', 'kepala']


# Prediction function
def predict(image_path):
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return class_names[preds.item()]

# Example usage
if __name__ == '__main__':
    image_path = 'D:\\SEM8Major\Datasets\Test\Aloevera\\5.jpg'# Replace with your image path
    prediction = predict(image_path)
    print(f'The predicted class is: {prediction}')

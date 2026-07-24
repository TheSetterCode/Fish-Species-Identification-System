import torch
from PIL import Image
from torchvision import transforms
from model import load_model


model = load_model()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


classes = [
    "Catfish",
"Climbing Perch",
"Fourfinger Threadfin",
"Freshwater Eel",
"Glass Perchlet",
"Goby",
"Knifefish",
"Mudfish",
"Mullet",
"Perch",
"Silver Perch",
"Snakehead",
"Tenpounder",
"Tilapia",
]


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)


    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            1
        )


    fish_name = classes[predicted.item()]

    confidence = confidence.item()*100


    return fish_name, confidence
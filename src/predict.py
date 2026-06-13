import torch
from PIL import Image
from torchvision import transforms

from model import get_model

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    "models/trash_classifier.pth",
    map_location=device
)

classes = checkpoint["classes"]

model = get_model(len(classes))

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(
    "test.jpg"
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

with torch.no_grad():

    output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    )

print(
    classes[prediction.item()]
)

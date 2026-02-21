from fastapi import FastAPI,File,UploadFile
from inference.model_loader import load_model
import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app=FastAPI()

test_transforms=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124),
                                 std=(0.24703233,0.24348505, 0.26158768))
        ])

model= load_model("models/resnet_cifar10.pth")

classes = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

@app.get("/")
def read_root():
    return {"message":"Model API is running"}

@app.post("/predict")
async def predict(img:UploadFile= File(...)):
    image=Image.open(img.file).convert("RGB")
    x=test_transforms(image).unsqueeze(0)

    with torch.inference_mode():
        output=model(x).to(device)
        pred=torch.argmax(output,dim=1).item()
    
    return {"prediction":classes[pred]}
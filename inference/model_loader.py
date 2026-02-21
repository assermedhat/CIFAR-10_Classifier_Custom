from src.model import ResNet18
import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(path):
    model=ResNet18(pretrained=False)
    model.load_state_dict(torch.load(path,map_location=device))
    model.eval()
    return model